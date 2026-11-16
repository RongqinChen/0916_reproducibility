import json
from types import SimpleNamespace

import pytest

from rebuttal_experiments.common import sha256
from rebuttal_experiments.protein_report import (
    load_shards,
    record_stem,
    report,
)


CONFIG = {
    "dataset": "ENZYMES",
    "encodings": ["endpoint", "joint"],
    "poly_dims": [8],
    "folds": [0],
    "primary_comparison": ["joint", "endpoint"],
    "reference_commit": "deadbeef",
    "evaluation": "synthetic",
}

PAIR = {
    "initial_state_sha256": "init",
    "parameter_count": 10,
    "train_graph_ids": [0, 1],
    "selection_and_scoring_graph_ids": [2],
    "initialization_seed": 1000,
    "raw_dataset_sha256": "data",
    "native_model_settings": {"hidden_dim": 2},
    "native_training_settings": {"batch_size": 1},
    "environment": {"torch": "test"},
}


def write_shard(root, encoding, accuracy, duplicate=False):
    root.mkdir(parents=True)
    identity = "identity"
    (root / "manifest.json").write_text(json.dumps({
        "status": "execution_finished",
        "identity": {"configuration": CONFIG},
        "identity_sha256": identity,
    }))
    stem = record_stem(encoding, 8, 0)
    (root / "predictions").mkdir()
    (root / "native").mkdir()
    (root / "predictions" / f"{stem}.npz").write_bytes(encoding.encode())
    checkpoint = root / "native" / "best.ckpt"
    checkpoint.write_bytes(encoding.encode())
    record = {
        "status": "complete",
        "identity_sha256": identity,
        **PAIR,
        "dataset": "ENZYMES",
        "encoding": encoding,
        "poly_dim": 8,
        "fold": 0,
        "checkpoint": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256(checkpoint),
        "predictions": f"predictions/{stem}.npz",
        "predictions_sha256": sha256(root / "predictions" / f"{stem}.npz"),
        "metrics": {"test_acc": accuracy},
    }
    (root / "records").mkdir()
    target = root / "records" / f"{stem}.json"
    target.write_text(json.dumps(record))
    if duplicate:
        duplicate_root = root.parent / (root.name + "_duplicate")
        duplicate_root.mkdir()
        (duplicate_root / "manifest.json").write_text(
            (root / "manifest.json").read_text()
        )
        (duplicate_root / "records").mkdir()
        (duplicate_root / "records" / target.name).write_text(json.dumps(record))
        for name in ("predictions", "native"):
            (duplicate_root / name).mkdir()
        pinned = duplicate_root / record["checkpoint"]
        pinned.write_bytes(checkpoint.read_bytes())
        prediction = duplicate_root / record["predictions"]
        prediction.write_bytes((root / record["predictions"]).read_bytes())
        return duplicate_root
    return root


def test_report_merges_verified_encoding_shards(tmp_path):
    first = write_shard(tmp_path / "endpoint", "endpoint", 0.5)
    second = write_shard(tmp_path / "joint", "joint", 0.75)
    config, identity, records, roots = load_shards([first, second])
    assert config == CONFIG
    assert identity == "identity"
    assert len(records) == 2
    assert roots == [first.resolve(), second.resolve()]

    report(SimpleNamespace(out=tmp_path / "report", shards=[first, second]))
    result = json.loads((tmp_path / "report" / "protein_report.json").read_text())
    primary = next(
        row for row in result["paired_comparisons"] if row["primary"]
    )
    assert result["status"] == "complete"
    assert primary["comparison"] == "joint_minus_endpoint"
    assert primary["mean_difference_pp"] == pytest.approx(25.0)


def test_shard_merge_rejects_duplicate_and_missing_records(tmp_path):
    first = write_shard(tmp_path / "endpoint", "endpoint", 0.5)
    duplicate = write_shard(
        tmp_path / "endpoint_again", "endpoint", 0.5, duplicate=True
    )
    with pytest.raises(ValueError, match="Duplicate shard record"):
        load_shards([first, duplicate])

    second = write_shard(tmp_path / "joint", "joint", 0.75)
    with pytest.raises(ValueError, match="Shard grid mismatch"):
        load_shards([first])
