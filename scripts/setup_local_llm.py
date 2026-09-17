"""Download a verified portable Ollama runtime on Windows; no global installation."""

import hashlib
from pathlib import Path
import zipfile

import httpx

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.34.1"
URL = f"https://github.com/ollama/ollama/releases/download/v{VERSION}/ollama-windows-amd64.zip"
EXPECTED_SHA256 = "428c94622a04764b318ddf13a061898edf69e32ffa896f638ed6015fd3f33288"


def main():
    directory = ROOT / ".tools" / "ollama"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory.parent / "ollama-windows-amd64.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with httpx.stream("GET", URL, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)
        temporary.replace(archive)
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError("Ollama archive checksum mismatch; refusing extraction.")
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            target = (directory / entry.filename).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError("Unsafe archive path.")
        bundle.extractall(directory)
    print(f"Verified Ollama {VERSION} is ready in {directory}")


if __name__ == "__main__":
    main()
