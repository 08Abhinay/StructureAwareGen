#!/usr/bin/env python3
import sys
import os
import platform
import json
from datetime import datetime

def main():
    # Basic environment info
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }

    # Collect installed packages (pip-style metadata)
    try:
        # Python 3.8+
        from importlib.metadata import distributions
        pkgs = []
        for d in distributions():
            name = (d.metadata.get("Name") or d.metadata.get("Summary") or d._name or "").strip()
            version = (d.version or "").strip()
            if name:
                pkgs.append({"name": name, "version": version})
        pkgs.sort(key=lambda x: x["name"].lower())
    except Exception as e:
        pkgs = []
        info["error_collecting_packages"] = repr(e)

    # Print to stdout
    print("=== ENV INFO ===")
    for k, v in info.items():
        print(f"{k}: {v}")

    print("\n=== PACKAGES (name==version) ===")
    for p in pkgs:
        print(f"{p['name']}=={p['version']}")

    # Save to JSON + TXT
    out_base = f"env_packages_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_path = out_base + ".json"
    txt_path = out_base + ".txt"

    with open(json_path, "w") as f:
        json.dump({"info": info, "packages": pkgs}, f, indent=2)

    with open(txt_path, "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        for p in pkgs:
            f.write(f"{p['name']}=={p['version']}\n")

    print(f"\nSaved:")
    print(f"  {json_path}")
    print(f"  {txt_path}")

if __name__ == "__main__":
    main()
