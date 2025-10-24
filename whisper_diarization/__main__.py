from diarize import diarize
import argparse
import helpers
import torch
from pathlib import Path
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="name of the target audio file", required=True)
    parser.add_argument("--txt", help="path of the .txt output file", required=True)
    parser.add_argument("--srt", help="path of the .srt output file", required=True)
    parser.add_argument("--no-stem", action="store_false", dest="stemming", default=True, help="Disables source separation. This helps with long files that don't contain a lot of music.")
    parser.add_argument("--suppress_numerals", action="store_true", dest="suppress_numerals", default=False, help="Suppresses Numerical Digits. This helps the diarization accuracy but converts all digits into written text.")
    parser.add_argument("--model", dest="model_name", default="medium.en", help="name of the Whisper model to use")
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=8, help="Batch size for batched inference, reduce if you run out of memory, set to 0 for original whisper longform inference")
    parser.add_argument("--language", type=str, default=None, choices=helpers.whisper_langs, help="Language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--device", dest="device", default="cuda" if torch.cuda.is_available() else "cpu", help="if you have a GPU use 'cuda', otherwise 'cpu'")
    args = parser.parse_args()

    language = helpers.process_language_arg(args.language, args.model_name)
    tmp_dir = Path("_tmp") / f"temp_outputs_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    diarize(
        helpers.get_vocal_path(Path(args.audio), tmp_dir, args.stemming, args.device),
        Path(args.txt),
        Path(args.srt),
        tmp_dir,
        args.suppress_numerals,
        args.model_name,
        args.batch_size,
        args.language,
        args.device
    )