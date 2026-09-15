"""Package or restore the portable DVC remote without committing dataset bytes."""

import argparse
import hashlib
import io
import json
import stat
import urllib.request
import zipfile
from pathlib import Path

from src.config import ROOT, load_config, project_path

MANIFEST = ROOT / "configs" / "data_release.json"
REMOTE = ROOT / ".dvc-remote"


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def download(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("Data downloads require HTTPS.")
    request = urllib.request.Request(url, headers={"User-Agent": "ames-housing-mlops/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def acquire_source() -> None:
    config = load_config()
    content = download(config["data"]["source_url"])
    if digest(content) != config["data"]["source_sha256"]:
        raise ValueError("The source file changed. Review it before adopting a new dataset version.")
    path = project_path(config["data"]["raw_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    print(f"Verified original source and saved {path.relative_to(ROOT)}")


def package_remote(repository: str) -> None:
    from src.data import dataset_hash

    config = load_config()
    version = dataset_hash(config)
    relative = f"files/md5/{version[:2]}/{version[2:]}"
    object_path = REMOTE / relative
    if not object_path.is_file():
        raise FileNotFoundError("Local DVC remote is empty. Run dvc push before packaging.")
    content = object_path.read_bytes()
    if digest(content) != config["data"]["source_sha256"]:
        raise ValueError("Remote object does not match the approved source checksum.")
    output = ROOT / "dist" / "ames-dvc-remote-v1.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(info, content)
    manifest = {
        "release_url": f"https://github.com/{repository}/releases/download/data-v1/{output.name}",
        "archive_sha256": digest(output.read_bytes()),
        "source_url": config["data"]["source_url"],
        "source_sha256": config["data"]["source_sha256"],
        "dvc_md5": version,
        "objects": {relative: {"sha256": digest(content), "size": len(content)}},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Packaged {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes)")
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")


def restore_remote(content: bytes, manifest: dict, destination: Path = REMOTE) -> None:
    if digest(content) != manifest["archive_sha256"]:
        raise ValueError("Release archive checksum mismatch; nothing was extracted.")
    root = destination.resolve()
    verified = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) != len(manifest["objects"]) or set(archive.namelist()) != set(manifest["objects"]):
            raise ValueError("Archive contents differ from the approved object manifest.")
        for member in members:
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root) or target == root:
                raise ValueError("Archive entry escapes the intended storage directory.")
            if stat.S_ISLNK(member.external_attr >> 16) or member.is_dir():
                raise ValueError("Archive entries must be regular data files.")
            expected = manifest["objects"][member.filename]
            if member.file_size != expected["size"]:
                raise ValueError("Data object size differs from the manifest.")
            data = archive.read(member)
            if digest(data) != expected["sha256"]:
                raise ValueError("Data object checksum mismatch.")
            verified.append((target, data))
    # Verify the entire small bundle before writing any objects.
    for target, data in verified:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def bootstrap(archive_path: str | None = None) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    content = Path(archive_path).read_bytes() if archive_path else download(manifest["release_url"])
    restore_remote(content, manifest)
    print("DVC remote restored and SHA-256 verified. Run: dvc pull")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("acquire", help="Download and verify the original dataset (maintainer setup).")
    package = commands.add_parser("package", help="Package the populated local DVC remote (maintainer setup).")
    package.add_argument("--repository", required=True, help="GitHub owner/repository.")
    restore = commands.add_parser("bootstrap", help="Restore the local DVC remote from the release attachment.")
    restore.add_argument("--archive", help="Use an already downloaded archive instead of the release URL.")
    args = parser.parse_args()
    if args.command == "acquire":
        acquire_source()
    elif args.command == "package":
        package_remote(args.repository)
    else:
        bootstrap(args.archive)


if __name__ == "__main__":
    main()
