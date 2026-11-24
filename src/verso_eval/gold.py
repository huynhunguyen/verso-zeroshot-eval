"""Read evaluation dataset questions, answers, manifest IDs and precomputed strata."""

import csv
import json
from functools import lru_cache

from . import paths


@lru_cache(maxsize=1)
def dataset_items():
    """{question id: dataset record} of circuits_full.json."""
    return {x["id"]: x for x in json.loads(paths.DATASET_JSON.read_text(encoding="utf-8"))}


@lru_cache(maxsize=1)
def evaluation_ids():
    """Ids of the evaluation questions (the 1,765 in-scope questions listed in manifest.csv), sorted."""
    with open(paths.MANIFEST_CSV, encoding="utf-8", newline="") as f:
        return tuple(sorted(qid for r in csv.DictReader(f) for qid in r["question_ids"].split()))


def image_path(item):
    """Path of the diagram of a dataset question."""
    return paths.IMAGES_DIR / item["id_image"]


@lru_cache(maxsize=1)
def text_drawing_mismatch():
    """{question id: what disagrees} of the questions whose text and drawing disagree."""
    with open(paths.TEXT_DRAWING_CSV, encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r["chi_tiet"] for r in csv.DictReader(f)}


@lru_cache(maxsize=1)
def circuit_strata():
    """{question id: 'uses circuit' or 'formula only'}, precomputed from the main VERSO repository."""
    return json.loads(paths.CIRCUIT_STRATA_JSON.read_text(encoding="utf-8"))
