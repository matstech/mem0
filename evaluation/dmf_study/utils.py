import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from time import strftime


def timestamp_slug():
    return strftime("%Y%m%d_%H%M%S")


def slugify(value):
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return slug or "artifact"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def zip_directory(directory, zip_path):
    directory = Path(directory)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = zip_path.with_suffix("")
    archive_path = shutil.make_archive(base_name.as_posix(), "zip", root_dir=directory.parent, base_dir=directory.name)
    return Path(archive_path)


def run_command(cmd, workdir):
    return subprocess.run(cmd, cwd=workdir, check=True)


def replace_config_values(config_text, overrides):
    updated_text = config_text
    for key, value in overrides.items():
        if isinstance(value, bool):
            replacement = "true" if value else "false"
        elif isinstance(value, (int, float)):
            replacement = str(value)
        else:
            replacement = f'"{value}"'
        updated_text = re.sub(
            rf"^{key}\s*=\s*.*$",
            f"{key} = {replacement}",
            updated_text,
            flags=re.MULTILINE,
        )
    return updated_text


def materialize_config(template_path, output_path, overrides):
    template_path = Path(template_path)
    output_path = Path(output_path)
    config_text = template_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(replace_config_values(config_text, overrides), encoding="utf-8")
    return output_path


def make_variant_name(overrides):
    budget = overrides.get("token_budget", "na")
    window = overrides.get("window_size", "na")
    pruning = overrides.get("pruning_frequency_x", "na")
    recall = overrides.get("recall_limit", "na")
    threshold = str(overrides.get("distance_threshold", "na")).replace(".", "")
    return f"dmf_b{budget}_w{window}_p{pruning}_r{recall}_d{threshold}"


def python_executable():
    return sys.executable or "python3"
