from __future__ import annotations

import joblib
import pandas as pd

from config import MODEL_FEATURES, MODEL_PATH


class HybridModelService:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)

    def predict_records(self, records: list[dict]) -> list[dict]:
        if not records:
            return []

        df = pd.DataFrame(records)
        missing = [col for col in MODEL_FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features for prediction: {missing}")

        x = df[MODEL_FEATURES]
        predictions = self.model.predict(x)

        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(x)

        output = []
        for idx, row in df.iterrows():
            result = {
                "id": row.get("id"),
                "question": row.get("question"),
                "baseline": int(row["baseline"]),
                "ragas": int(row["ragas"]),
                "selfcheck": int(row["selfcheck"]),
                "similarity": int(row["similarity"]),
                "hybrid_prediction": int(predictions[idx]),
            }
            if probabilities is not None:
                result["hybrid_probability_not_hallucinated"] = float(probabilities[idx][0])
                result["hybrid_probability_hallucinated"] = float(probabilities[idx][1])
            output.append(result)
        return output
