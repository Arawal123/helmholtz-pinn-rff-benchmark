"""One-time asset generation. Never called by the protected runner."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.data import generate_sobol_points
from src.utils import sha256_file


def main() -> None:
    config_path = ROOT / "configs/frozen.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for seed in (0, 1, 2):
        record = config["data"]["frozen_sobol_assets"][seed]
        path = ROOT / record["path"]
        if path.exists():
            raise FileExistsError(f"Refusing to replace frozen asset: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, generate_sobol_points(seed))
        record["sha256"] = sha256_file(path)
    # Independent CPU torch.Generator, seeded once; row-major normal draw.
    generator = torch.Generator(device="cpu").manual_seed(2026)
    matrix = (3.0 * torch.randn((32, 2), generator=generator, dtype=torch.float32)).numpy()
    record = config["model"]["rff"]
    path = ROOT / record["matrix_path"]
    if path.exists():
        raise FileExistsError(f"Refusing to replace frozen asset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    record["matrix_sha256"] = sha256_file(path)
    (path.parent / "B_seed_2026.csv").write_text(
        "b_x,b_y\n" + "".join(f"{row[0]:.9g},{row[1]:.9g}\n" for row in matrix), encoding="utf-8"
    )
    (ROOT / "data/frozen/generation.json").write_text(json.dumps({
        "sobol": "torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed).draw(2048); float32 map 2*x-1",
        "sobol_seeds": [0, 1, 2], "torch_version": torch.__version__,
        "rff": "torch.Generator(device='cpu').manual_seed(2026); float32 3*torch.randn((32,2), generator=g)",
        "rff_seed": 2026, "asset_hashes": {
            str(seed): config["data"]["frozen_sobol_assets"][seed]["sha256"] for seed in (0,1,2)
        }, "rff_sha256": record["matrix_sha256"]
    }, indent=2) + "\n", encoding="utf-8")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
