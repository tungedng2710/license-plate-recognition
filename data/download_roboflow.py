import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from roboflow import Roboflow

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_INTERVAL_SECONDS = 600
DEFAULT_EXPORT_FORMAT = "yolov8"


def load_env_file(path: Path) -> Dict[str, str]:
    """Populate os.environ with values from a simple KEY=VALUE .env file."""
    if not path.is_file():
        return {}

    parsed: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        parsed[key.strip()] = value.strip()

    for key, value in parsed.items():
        os.environ.setdefault(key, value)
    return parsed


def update_env_value(path: Path, key: str, value: str) -> None:
    """Update or append KEY=VALUE in the .env file."""
    lines = []
    found = False

    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle.read().splitlines():
                if raw_line.strip().startswith("#") or "=" not in raw_line:
                    lines.append(raw_line)
                    continue
                current_key = raw_line.split("=", 1)[0].strip()
                if current_key == key:
                    lines.append(f"{key}={value}")
                    found = True
                else:
                    lines.append(raw_line)

    if not found:
        lines.append(f"{key}={value}")

    # Ensure trailing newline
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def parse_int(value: str, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Environment variable {key} must be an integer, got {value!r}") from exc


def attempt_download(version_number: int, export_format: str) -> None:
    api_key = require_env("ROBOFLOW_API_KEY")
    workspace = require_env("ROBOFLOW_WORKSPACE")
    project_name = require_env("ROBOFLOW_PROJECT")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    version.download(export_format)


def main() -> None:
    load_env_file(ENV_PATH)

    poll_interval_seconds = os.environ.get("ROBOFLOW_POLL_INTERVAL")
    interval = (
        parse_int(poll_interval_seconds, "ROBOFLOW_POLL_INTERVAL")
        if poll_interval_seconds
        else DEFAULT_INTERVAL_SECONDS
    )
    export_format = os.environ.get("ROBOFLOW_EXPORT_FORMAT", DEFAULT_EXPORT_FORMAT)

    print(
        f"[roboflow] Polling every {interval} seconds using format '{export_format}' "
        f"and configuration from {ENV_PATH}"
    )

    while True:
        version_str: Optional[str] = os.environ.get("ROBOFLOW_VERSION")
        if version_str is None:
            print("[roboflow] ROBOFLOW_VERSION not set; aborting.", file=sys.stderr)
            sys.exit(1)

        try:
            version_number = parse_int(version_str, "ROBOFLOW_VERSION")
        except RuntimeError as exc:
            print(f"[roboflow] {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"[roboflow] Attempting download for version {version_number}")
        try:
            attempt_download(version_number, export_format)
        except Exception as exc:  # broad catch to keep loop alive
            print(f"[roboflow] Download failed for version {version_number}: {exc}", file=sys.stderr)
        else:
            next_version = version_number + 1
            update_env_value(ENV_PATH, "ROBOFLOW_VERSION", str(next_version))
            os.environ["ROBOFLOW_VERSION"] = str(next_version)
            print(f"[roboflow] Download successful. Next version set to {next_version}.")

        time.sleep(interval)


if __name__ == "__main__":
    main()
