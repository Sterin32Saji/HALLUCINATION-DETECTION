from pathlib import Path

RAGAS_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = 0.75

PROJECT_ROOT = Path(__file__).resolve().parent


def _find_workspace_root() -> Path:
	for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
		if (candidate / "README.md").exists() and (candidate / "notebooks").exists():
			return candidate
	raise FileNotFoundError("Could not locate workspace root from backend/config.py")


WORKSPACE_ROOT = _find_workspace_root()

MODEL_PATH = WORKSPACE_ROOT / "notebooks" / "final_all_dataset" / "hybrid_detector_model.pkl"

DEFAULT_BASELINE_FILE = WORKSPACE_ROOT / "notebooks" / "final_all_dataset" / "150_filtered" / "baseline_score_filtered.jsonl"
DEFAULT_RAGAS_FILE = WORKSPACE_ROOT / "notebooks" / "final_all_dataset" / "150_filtered" / "ragas_hallucination_results_filtered.jsonl"
DEFAULT_SELFCHECK_FILE = WORKSPACE_ROOT / "notebooks" / "final_all_dataset" / "150_filtered" / "selfcheck_dataset_filtered.jsonl"
DEFAULT_SIMILARITY_FILE = WORKSPACE_ROOT / "notebooks" / "final_all_dataset" / "150_filtered" / "similarity_dataset_filtered.jsonl"
DEFAULT_HUMAN_FILE = WORKSPACE_ROOT / "notebooks" / "final_all_dataset" / "150_filtered" / "human_label.jsonl"

RAW_BASELINE_RESULTS_FILE = WORKSPACE_ROOT / "notebooks" / "files-distributed-based-on-difficulty" / "hallucination_dataset.jsonl"
RAW_RAGAS_RESULTS_FILE = WORKSPACE_ROOT / "notebooks" / "files-distributed-based-on-difficulty" / "ragas_hallucination_results.jsonl"
RAW_SELFCHECK_RESULTS_FILE = WORKSPACE_ROOT / "notebooks" / "files-distributed-based-on-difficulty" / "selfcheck_dataset.jsonl"
RAW_SIMILARITY_RESULTS_FILE = WORKSPACE_ROOT / "notebooks" / "files-distributed-based-on-difficulty" / "similarity_dataset.jsonl"

MODEL_FEATURES = ["baseline", "ragas", "selfcheck", "similarity"]
