#!/usr/bin/env python3
"""Verify the packaged, reviewed semantic-existence dataset."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/release/semantic_existence_v1"


def load(release, name):
    return [json.loads(s) for s in (release / name).read_text().splitlines() if s]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=RELEASE)
    args = parser.parse_args()
    release = args.release
    splits = {name: load(release, name + ".jsonl") for name in ("train", "val", "test")}
    inventory = json.loads((release / "semantic_inventory.json").read_text())
    held_actions = set(inventory["heldout_actions"])
    held_pairs = set(inventory["heldout_compositions"])
    seen_pairs = set(inventory["action_object_compositions"])
    assert held_actions.isdisjoint(inventory["actions"])
    assert held_pairs.isdisjoint(seen_pairs)
    assert all(action in inventory["actions"] and obj in inventory["objects"]
               for action, obj in (p.split("|", 1) for p in held_pairs))
    ids = set()
    videos = {}
    for name, rows in splits.items():
        videos[name] = {r["video_id"] for r in rows}
        for row in rows:
            assert row["qid"] not in ids, row["qid"]
            ids.add(row["qid"])
            assert row["vid"] == row["video_id"]
            assert bool(row["relevant_windows"]) == bool(row["exist_label"])
            assert row["partition"][0] == ("U" if row["semantic_status"] == "unseen" else "S")
            assert row["partition"][1] == ("+" if row["exist_label"] else "-")
            graph = row["semantic_graph"]
            event = graph["action"] + "|" + graph["object"]
            if row["semantic_status"] == "unseen":
                assert graph["action"] in held_actions or event in held_pairs, row["qid"]
            else:
                assert event in seen_pairs, row["qid"]
            if name == "train":
                assert graph["action"] not in held_actions and event not in held_pairs
            if row["exist_label"] == 0:
                assert row["verification_status"] in {"user_attested_video_review", "manual_video_review_confirmed"}
            assert (graph["action"], graph["object"]) != ("dress", "front")
    assert videos["train"].isdisjoint(videos["val"])
    assert videos["train"].isdisjoint(videos["test"])
    assert videos["val"].isdisjoint(videos["test"])
    assert all(Counter(r["partition"] for r in splits["test"])[p] for p in ("S+", "S-", "U+", "U-"))
    positives = {r["qid"]: r for r in splits["test"] if r["partition"] == "U+"}
    negatives = {r["qid"]: r for r in splits["test"] if r["partition"] == "U-"}
    pairs = load(release, "matched_u_pairs.jsonl")
    matched_rows = load(release, "test_matched_u.jsonl")
    assert len(matched_rows) == 2 * len(pairs)
    assert len({r["positive_qid"] for r in pairs}) == len(pairs)
    for pair in pairs:
        pos, neg = positives[pair["positive_qid"]], negatives[pair["negative_qid"]]
        assert pos["video_id"] == neg["video_id"] and neg["source_qid"] == pos["qid"]
        assert pos["novelty_type"] == neg["novelty_type"]
    manifest = json.loads((release / "manifest.json").read_text())
    assert not manifest["missing_release_videos_in_archive"]
    for name, digest in manifest["release_sha256"].items():
        assert sha256(release / name) == digest, name
    print(json.dumps({"split_counts": {k: dict(Counter(r["partition"] for r in v)) for k, v in splits.items()},
                      "matched_u_pairs": len(pairs), "video_split_overlap": 0,
                      "release_sha256_verified": len(manifest["release_sha256"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
