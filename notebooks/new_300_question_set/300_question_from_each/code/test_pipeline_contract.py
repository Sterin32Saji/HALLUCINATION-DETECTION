from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import pipeline


class FeatureInputContractTest(unittest.TestCase):
    def test_custom_features_use_generated_answer(self) -> None:
        captured = []

        class RecordingExtractor:
            def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
                captured.append(frame.copy())
                return pd.DataFrame({"id": frame["id"], "example_feature": [1.0]})

        raw = pd.DataFrame([{
            "id": "example-1",
            "question": "What is the port?",
            "question_type": "factual",
            "difficulty": "easy",
            "answer": "The reference says 443.",
            "rag_answer": "The generated answer says 9999.",
            "evidence": "The documented port is 443.",
        }])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(pipeline, "CustomFeatureExtractor", RecordingExtractor), patch.object(
                pipeline, "FEATURES_PATH", Path(directory) / "features.csv"
            ):
                pipeline.build_features(raw)

        self.assertEqual(captured[0].loc[0, "answer"], raw.loc[0, "rag_answer"])
        self.assertNotIn("rag_answer", captured[0].columns)
        self.assertNotEqual(captured[0].loc[0, "answer"], raw.loc[0, "answer"])

    def test_missing_generated_answer_is_rejected(self) -> None:
        raw = pd.DataFrame([{
            "id": "example-1",
            "question": "What is the port?",
            "question_type": "factual",
            "difficulty": "easy",
            "rag_answer": "",
            "evidence": "The documented port is 443.",
        }])
        with patch.object(pipeline, "CustomFeatureExtractor", lambda: None):
            with self.assertRaisesRegex(ValueError, "generated RAG answer"):
                pipeline.build_features(raw)


if __name__ == "__main__":
    unittest.main()
