from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from .utils import REPO_ROOT, sha256_file, utc_now


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    return {"git_sha": run("rev-parse", "HEAD"),
            "git_status": run("status", "--porcelain", "--untracked-files=normal")}


def runtime_record(config_path: Path, config: dict) -> dict:
    git = git_state()
    if not git["git_sha"] or git["git_status"]:
        raise RuntimeError("Official execution requires a clean committed Git revision")
    if not torch.cuda.is_available():
        raise RuntimeError("Official execution requires a CUDA GPU")
    return {**git, "config_sha256": sha256_file(config_path),
            "sobol_sha256": {str(s): config["data"]["frozen_sobol_assets"][s]["sha256"] for s in (0,1,2)},
            "rff_sha256": config["model"]["rff"]["matrix_sha256"],
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "accelerator": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda_device_count": torch.cuda.device_count(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "captured_at_utc": utc_now()}
