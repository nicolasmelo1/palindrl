from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import tiktoken

OPENAI_GPT2_BPE_URLS: dict[str, str] = {
    "encoder.json": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
    "vocab.bpe": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
}
DEFAULT_TOKENIZER_DIR = Path("assets/tokenizers/openai-gpt2")


class OpenAIBPETokenizer:
    """Thin wrapper around OpenAI's GPT-2 BPE via tiktoken."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.encoding.decode(token_ids)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _download_file(url: str, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return

    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())


def download_openai_gpt2_bpe(out_dir: Path, overwrite: bool = False) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths: list[Path] = []
    for name, url in OPENAI_GPT2_BPE_URLS.items():
        destination = out_dir / name
        _download_file(url=url, path=destination, overwrite=overwrite)
        downloaded_paths.append(destination)
    return downloaded_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI BPE tokenizer helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download", help="Download OpenAI GPT-2 BPE assets"
    )
    download_parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_TOKENIZER_DIR,
        help="Relative to repo root unless absolute.",
    )
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if files exist.",
    )

    encode_parser = subparsers.add_parser(
        "encode", help="Encode text with OpenAI GPT-2 BPE"
    )
    encode_parser.add_argument("text", type=str)

    decode_parser = subparsers.add_parser(
        "decode", help="Decode comma-separated token IDs"
    )
    decode_parser.add_argument("tokens", type=str, help='Example: "123,456,789"')

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "download":
        out_dir = args.out_dir
        if not out_dir.is_absolute():
            out_dir = _project_root() / out_dir
        paths = download_openai_gpt2_bpe(out_dir=out_dir, overwrite=args.overwrite)
        print("Downloaded OpenAI GPT-2 BPE files:")
        for path in paths:
            print(f"- {path}")
        return

    tokenizer = OpenAIBPETokenizer()
    if args.command == "encode":
        print(json.dumps(tokenizer.encode(args.text)))
        return

    token_ids = [
        int(token.strip()) for token in args.tokens.split(",") if token.strip()
    ]
    print(tokenizer.decode(token_ids))


if __name__ == "__main__":
    main()
