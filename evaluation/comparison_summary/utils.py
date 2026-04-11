import json
import csv
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_json(path):
    if path is None:
        return None
    if isinstance(path, str) and not path.strip():
        return None
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def zip_directory(directory_path, zip_path):
    directory_path = Path(directory_path)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(directory_path.parent))
    return zip_path.as_posix()

def copy_artifact(source, destination_dir):
    if not source:
        return None
    source = Path(source)
    if not source.exists():
        return None
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_dir / source.name)
    return (destination_dir / source.name).as_posix()
