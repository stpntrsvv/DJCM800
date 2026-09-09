"""Download source artifacts to ignored originals/, verifying recorded SHA-256."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    dest = ROOT / "originals"
    dest.mkdir(exist_ok=True)
    for source in manifest["sources"]:
        path = dest / source["file"]
        expected = source["sha256"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print("Present:", path.name)
            continue
        request = urllib.request.Request(source["url"], headers={"User-Agent": "JCM800-research/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            changed = path.with_suffix(path.suffix + ".changed")
            changed.write_bytes(data)
            raise RuntimeError(f"Source changed: {source['url']}; saved separately: {changed}")
        path.write_bytes(data)
        print("Downloaded:", path.name)
    archive = dest / "Tubemods.zip"
    with zipfile.ZipFile(archive) as zipped:
        target = (dest / "koren").resolve()
        for info in zipped.infolist():
            (target / info.filename).resolve().relative_to(target)
        zipped.extractall(target)


if __name__ == "__main__":
    main()
