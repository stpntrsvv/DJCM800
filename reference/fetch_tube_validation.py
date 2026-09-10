"""Восстановить только первичные данные отдельного лампового опыта."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "tube_validation_sources.json").read_text())
    dest = ROOT / "originals"
    dest.mkdir(exist_ok=True)
    for source in manifest["downloads"]:
        path = dest / source["file"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "JCM800-research/0.2"})
            data = urllib.request.urlopen(request, timeout=60).read()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                path.with_suffix(path.suffix+".changed").write_bytes(data)
                raise RuntimeError(f"Source changed: {source['url']}")
            path.write_bytes(data)
        print("Verified:", path.name)
    library = dest / "reefman/TubeLib.inc"
    digest = manifest["library_sha256"]
    if not library.exists() or hashlib.sha256(library.read_bytes()).hexdigest() != digest:
        archiver = shutil.which("7zz") or shutil.which("7z")
        if not archiver:
            raise RuntimeError("Install 7zz/7z to extract the author's MSI without running it")
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run([archiver, "x", str(dest / "ExtractModel4.msi"), f"-o{temp}", "-y"], check=True, stdout=subprocess.DEVNULL)
            data = (Path(temp) / "TubeLib.inc").read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("Unexpected library in author archive")
            library.parent.mkdir(exist_ok=True)
            library.write_bytes(data)
    print("Verified: reefman/TubeLib.inc")


if __name__ == "__main__":
    main()
