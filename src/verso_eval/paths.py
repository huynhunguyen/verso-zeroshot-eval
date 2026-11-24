"""Locations of the standalone zero-shot evaluation repository's data."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "datasets" / "circuit-dataset"
DATASET_JSON = DATASET_DIR / "circuits_full.json"
TEXT_DRAWING_CSV = DATASET_DIR / "loi_de_lech_hinh.csv"
CIRCUIT_STRATA_JSON = DATASET_DIR / "circuit_strata.json"
IMAGES_DIR = DATASET_DIR / "images"

MANIFEST_CSV = ROOT / "datasets" / "circuit-netlists" / "manifest.csv"

RUNS_DIR = ROOT / "runs"
LLM_CACHE = ROOT / ".cache" / "llm.sqlite"
MANUAL_GRADES_DIR = RUNS_DIR / "manual_grades"
MODEL_PROFILES_DIR = ROOT / "configs" / "models"
PROMPTS_DIR = ROOT / "configs" / "prompts"
