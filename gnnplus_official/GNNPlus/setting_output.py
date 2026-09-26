"""Deterministic setting directory and independent seed/split run directories."""
import hashlib
from pathlib import Path
import re
import yaml


def setting_directory(cfg):
    root=Path(cfg.out_dir)
    setting=yaml.safe_load(cfg.dump())
    # Seeds/splits are run coordinates, not distinct model settings.
    for key in ("out_dir","run_dir","run_id","seed","params"):
        setting.pop(key,None)
    setting.get("dataset",{}).pop("split_index",None)
    setting.get("train",{}).pop("auto_resume",None)
    canonical=yaml.safe_dump(setting,sort_keys=True)
    digest=hashlib.sha256(canonical.encode()).hexdigest()
    endpoint=cfg.posenc_Endpoint
    mode=f"endpoint-{endpoint.injection}-q{endpoint.qsteps}" if endpoint.enable else "rwse-baseline"
    if cfg.posenc_StructuralEdge.enable:
        param = cfg.posenc_StructuralEdge
        mode = f"{param.injection}-{param.route}-d{param.width}"
    label=f"{cfg.dataset.name}-{cfg.gnn.layer_type}-{mode}"
    label=re.sub(r"[^a-zA-Z0-9_.-]+","-",label)[:100]
    directory=root/f"{label}-{digest[:16]}"
    directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/"setting.yaml"
    try:
        with manifest.open("x") as f:
            f.write(canonical)
    except FileExistsError:
        if manifest.read_text()!=canonical:
            raise RuntimeError("Setting hash collision or conflicting manifest")
    return directory


def set_run_directory(cfg,setting,seed,split):
    run=setting/f"seed-{seed}_split-{split}"
    # Never erase an earlier run. Reuse requires explicit auto_resume.
    run.mkdir(parents=False,exist_ok=bool(cfg.train.auto_resume))
    cfg.out_dir=str(setting)
    cfg.run_dir=str(run)
