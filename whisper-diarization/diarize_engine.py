#!/usr/bin/env python3
"""
diarize_engine.py — Reusable, in-process diarization engine.

Loads Whisper, CTC aligner, punctuation (when available), and NeMo MSDD **once**,
and exposes `process_one(audio_path, dest_dir)` to diarize a single file.

Outputs:
- <dest_dir>/<stem>.txt
- <dest_dir>/<stem>.srt
- Moves the (possibly stemmed) input copy to <dest_dir>/<stem>.wav on success
  (the original source file is removed by the caller).
- On failure, caller can move/copy the original to an error folder.

Notes:
- We reuse a per-process temp root: temp_outputs_<pid>/.
- For each file we allocate a unique subfolder to avoid clashes.
- Concurrency: use a single engine per process (one file at a time).
"""

import os
import re
import shutil
import uuid
import torch
import torchaudio
import faster_whisper

from pathlib import Path

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel

from helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,  # not used directly but kept for compatibility
    write_srt,
)

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)

mtypes = {"cpu": "int8", "cuda": "float16"}

class DiarizationEngine:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        whisper_model_name: str = "medium.en",
        batch_size: int = 8,
        suppress_numerals: bool = False,
        enable_stemming: bool = True,
    ):
        self.device = device
        self.whisper_model_name = whisper_model_name
        self.batch_size = batch_size
        self.suppress_numerals = suppress_numerals
        self.enable_stemming = enable_stemming

        # temp root per-process
        self.pid = os.getpid()
        self.temp_root = Path(f"temp_outputs_{self.pid}")
        self.temp_root.mkdir(parents=True, exist_ok=True)

        # Load Whisper once
        self.whisper_model = faster_whisper.WhisperModel(
            self.whisper_model_name, device=self.device, compute_type=mtypes[self.device]
        )
        self.whisper_pipeline = faster_whisper.BatchedInferencePipeline(self.whisper_model)

        # Load CTC aligner once
        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # NeMo MSDD model will be instantiated per file with its temp dir config.

        # Punctuation model (lazy; load when language supports it)
        self.punct_model = None

    def _stem_if_needed(self, audio_path: Path, work_dir: Path) -> Path:
        """
        If enable_stemming is True, try to separate vocals with Demucs.
        Fall back to original audio on failure. Returns the path to the audio to use.
        """
        if not self.enable_stemming:
            return audio_path

        # htdemucs output path:
        # work_dir/htdemucs/<stem>/vocals.wav
        cmd = (
            f'python -m demucs.separate -n htdemucs --two-stems=vocals '
            f'"{audio_path}" -o "{work_dir}" --device "{self.device}"'
        )
        rc = os.system(cmd)
        if rc != 0:
            return audio_path

        vocals = work_dir / "htdemucs" / audio_path.stem / "vocals.wav"
        return vocals if vocals.exists() else audio_path

    def _load_audio(self, path: Path):
        return faster_whisper.decode_audio(str(path))

    def _ensure_punct_model(self):
        if self.punct_model is None:
            self.punct_model = PunctuationModel(model="kredor/punctuate-all")

    def process_one(self, audio_path: Path, dest_dir: Path) -> None:
        """
        Process a single audio file and write outputs to dest_dir.
        On success writes:
            dest_dir/<stem>.txt
            dest_dir/<stem>.srt
            dest_dir/<stem>.wav  (a copy/moved of original or stemmed input)
        Raises Exception on failure.
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Per-file temp subdir
        work_dir = self.temp_root / f"job_{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1) Language (None -> autodetect); keep same logic as original
            language = process_language_arg(None, self.whisper_model_name)

            # 2) Optional source separation (Demucs)
            diar_audio_path = self._stem_if_needed(Path(audio_path), work_dir)

            # 3) Transcribe
            audio_waveform = self._load_audio(diar_audio_path)
            suppress_tokens = (
                find_numeral_symbol_tokens(self.whisper_model.hf_tokenizer)
                if self.suppress_numerals
                else [-1]
            )

            if self.batch_size > 0:
                transcript_segments, info = self.whisper_pipeline.transcribe(
                    audio_waveform,
                    language,
                    suppress_tokens=suppress_tokens,
                    batch_size=self.batch_size,
                )
            else:
                transcript_segments, info = self.whisper_model.transcribe(
                    audio_waveform,
                    language,
                    suppress_tokens=suppress_tokens,
                    vad_filter=True,
                )
            full_transcript = "".join(seg.text for seg in transcript_segments)

            # 4) Forced alignment (CTC)
            emissions, stride = generate_emissions(
                self.alignment_model,
                torch.from_numpy(audio_waveform)
                .to(self.alignment_model.dtype)
                .to(self.alignment_model.device),
                batch_size=self.batch_size,
            )

            tokens_starred, text_starred = preprocess_text(
                full_transcript,
                romanize=True,
                language=langs_to_iso[info.language],
            )
            segments, scores, blank_token = get_alignments(
                emissions, tokens_starred, self.alignment_tokenizer
            )
            spans = get_spans(tokens_starred, segments, blank_token)
            word_timestamps = postprocess_results(text_starred, spans, stride, scores)

            # 5) Convert to mono wav for NeMo
            mono_wav = work_dir / "mono_file.wav"
            torchaudio.save(
                str(mono_wav),
                torch.from_numpy(audio_waveform).unsqueeze(0).float(),
                16000,
                channels_first=True,
            )

            # 6) NeMo diarization (per-job model w/ per-job config & temp path)
            msdd_model = NeuralDiarizer(cfg=create_config(str(work_dir))).to(self.device)
            msdd_model.diarize()
            del msdd_model
            torch.cuda.empty_cache()

            # 7) Speaker mapping
            rttm = work_dir / "pred_rttms" / "mono_file.rttm"
            speaker_ts = []
            with open(rttm, "r") as f:
                for line in f:
                    parts = line.split(" ")
                    s = int(float(parts[5]) * 1000)
                    e = s + int(float(parts[8]) * 1000)
                    speaker_ts.append([s, e, int(parts[11].split("_")[-1])])

            wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

            # 8) Optional punctuation
            if info.language in punct_model_langs:
                self._ensure_punct_model()
                words_list = list(map(lambda x: x["word"], wsm))
                labeled_words = self.punct_model.predict(words_list, chunk_size=230)
                ending_puncts = ".?!"
                model_puncts = ".,;:!?"
                is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

                for word_dict, labeled_tuple in zip(wsm, labeled_words):
                    word = word_dict["word"]
                    if (
                        word
                        and labeled_tuple[1] in ending_puncts
                        and (word[-1] not in model_puncts or is_acronym(word))
                    ):
                        word2 = word + labeled_tuple[1]
                        if word2.endswith(".."):
                            word2 = word2.rstrip(".")
                        word_dict["word"] = word2
            # else: keep original punctuation

            wsm = get_realigned_ws_mapping_with_punctuation(wsm)
            ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

            # 9) Write outputs to dest_dir
            stem = Path(audio_path).stem
            out_txt = dest_dir / f"{stem}.txt"
            out_srt = dest_dir / f"{stem}.srt"
            with open(out_txt, "w", encoding="utf-8-sig") as f:
                get_speaker_aware_transcript(ssm, f)
            with open(out_srt, "w", encoding="utf-8-sig") as srt:
                write_srt(ssm, srt)

            # 10) Copy the audio we actually diarized (original or vocals) into dest
            #     (normalize to <stem>.wav for consistency)
            out_wav = dest_dir / f"{stem}.wav"
            if diar_audio_path.suffix.lower() == ".wav" and Path(diar_audio_path).exists():
                # If the diarized path is a temp vocals.wav, copy it; else copy original
                src = diar_audio_path if Path(diar_audio_path).is_file() else audio_path
            else:
                src = audio_path
            try:
                # best-effort copy of the source audio so outputs stay together
                shutil.copy2(src, out_wav)
            except Exception:
                # non-fatal if copying fails
                pass

        finally:
            # Clean the per-file temp folder
            cleanup(str(work_dir))
