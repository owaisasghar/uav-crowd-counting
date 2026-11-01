"""Download UP-COUNT-TRACK (CC BY-NC 4.0, Zenodo record 13829572) labels and selected test videos.

The test videos come as one 19.7 GB zip. Its table of contents is read with HTTP range requests and only the chosen
members are downloaded, so a few videos cost a few GB. Standard library only:

    .venv/bin/python scripts/download_upcount_track.py --list
    .venv/bin/python scripts/download_upcount_track.py --videos 0011 0034 --out data/raw/UP-COUNT-TRACK

Without --videos only the label zips (MOT-style tracks, per-frame GPS/altitude, DroneCrowd-style points) are fetched.
"""
from __future__ import annotations

import argparse
import shutil
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

RECORD = "https://zenodo.org/records/13829572/files/{}?download=1"
VIDEOS = "test_videos.zip"
LABELS = ["mot_style_test_labels.zip", "test_gps.zip", "dronecrowd_style_labels.zip"]
CHUNK = 8 * 2**20


class RemoteFile:
    """Read-only, seekable view of a remote file that fetches each read with an HTTP range request."""

    def __init__(self, url: str) -> None:
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = (0, self.position, self.size)[whence] + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.position + size, self.size)
        if end <= self.position:
            return b""
        data = self.range(self.position, end)
        self.position += len(data)
        return data

    def range(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(request) as response:
            return response.read()


def extract_member(remote: RemoteFile, info: zipfile.ZipInfo, target: Path) -> None:
    """Stream one member's bytes straight from the archive, inflating if needed."""
    # The local header repeats the name and may carry a different extra field, so read its lengths first.
    name_length, extra_length = struct.unpack("<HH", remote.range(info.header_offset + 26, info.header_offset + 30))
    start = info.header_offset + 30 + name_length + extra_length
    request = urllib.request.Request(remote.url, headers={"Range": f"bytes={start}-{start + info.compress_size - 1}"})
    inflate = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        while chunk := response.read(CHUNK):
            out.write(inflate.decompress(chunk) if inflate else chunk)
        if inflate:
            out.write(inflate.flush())
    if partial.stat().st_size != info.file_size:
        raise RuntimeError(f"{info.filename}: got {partial.stat().st_size} bytes, expected {info.file_size}")
    partial.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list the videos in the archive and exit")
    parser.add_argument("--videos", nargs="*", default=[], help="video names (or name prefixes) to download")
    parser.add_argument("--out", type=Path, default=Path("data/raw/UP-COUNT-TRACK"))
    args = parser.parse_args()

    remote = RemoteFile(RECORD.format(VIDEOS))
    archive = zipfile.ZipFile(remote)
    members = [info for info in archive.infolist() if not info.is_dir()]
    if args.list:
        for info in members:
            print(f"{info.filename:<60}{info.file_size / 2**30:8.2f} GB")
        print(f"{len(members)} files, {sum(i.file_size for i in members) / 2**30:.1f} GB")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    for name in LABELS:
        target = args.out / name
        if not target.exists():
            with urllib.request.urlopen(RECORD.format(name)) as response, target.open("wb") as out:
                shutil.copyfileobj(response, out, CHUNK)
        with zipfile.ZipFile(target) as labels:
            labels.extractall(args.out)
    # Keep the archive's folder names: the tracker finds a video's altitude log by swapping test_videos for test_gps.
    for wanted in args.videos:
        for info in [i for i in members if Path(i.filename).name.startswith(wanted)]:
            target = args.out / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            print(f"downloading {info.filename} ({info.file_size / 2**30:.2f} GB)", flush=True)
            extract_member(remote, info, target)


if __name__ == "__main__":
    main()
