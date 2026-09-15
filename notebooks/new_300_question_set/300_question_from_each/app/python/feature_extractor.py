from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/\.]+")
DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)


@dataclass
class ExtractedSignals:
    entities: set[str]
    numbers: set[str]
    dates: set[str]
    technical_terms: set[str]


class CustomFeatureExtractor:
    """Independent feature extractor for RAG hallucination detection.

    The extractor is intentionally detector-agnostic and does not rely on
    baseline/RAGAS/SelfCheck/similarity detector outputs.
    """

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self._spacy_nlp = self._load_spacy()

    @staticmethod
    def _load_spacy():
        try:
            import spacy

            return spacy.load("en_core_web_sm")
        except Exception:
            import spacy

            return spacy.blank("en")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in TOKEN_PATTERN.findall(text)]

    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        if den == 0:
            return 0.0
        return float(num / den)

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0
        va = self.embedding_model.encode(text_a, normalize_embeddings=True)
        vb = self.embedding_model.encode(text_b, normalize_embeddings=True)
        score = float(np.dot(va, vb))
        return max(min(score, 1.0), -1.0)

    def _extract_generic_technical_terms(self, text: str) -> set[str]:
        patterns = [
            r"\bRFC[\s\-]?\d{3,5}\b",
            r"\bNIST(?:\s+(?:SP|IR|CSWP))?[\s\-\.]?\d{2,4}(?:\.\d+)*(?:r\d+)?\b",
            r"\b(?:AWS|Amazon)\s+[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,2}\b",
            r"\b[1-5]\d{2}\b",
            r"\bv?\d+(?:\.\d+){1,3}\b",
            r"\b[A-Z]{2,}(?:\/[0-9.]+)?\b",
            r"\b[A-Z]{2,}(?:-[A-Z0-9]{2,})+\b",
        ]

        signals: set[str] = set()
        for pat in patterns:
            for match in re.findall(pat, text):
                signals.add(str(match).strip().lower())
        return signals

    def _extract_spacy_entities(self, text: str) -> set[str]:
        if not text.strip():
            return set()
        doc = self._spacy_nlp(text)
        ent_labels = {"ORG", "PRODUCT", "GPE", "PERSON", "EVENT", "LAW", "WORK_OF_ART", "NORP"}
        ents = {ent.text.strip().lower() for ent in getattr(doc, "ents", []) if ent.label_ in ent_labels}
        return {e for e in ents if e}

    def _extract_numbers(self, text: str) -> set[str]:
        values = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        return {v.strip().lower() for v in values if v.strip()}

    def _extract_dates(self, text: str) -> set[str]:
        values = set(m.group(0).strip().lower() for m in DATE_PATTERN.finditer(text))
        if text.strip():
            doc = self._spacy_nlp(text)
            values |= {ent.text.strip().lower() for ent in getattr(doc, "ents", []) if ent.label_ == "DATE"}
        return {d for d in values if d}

    def _extract_signals(self, text: str) -> ExtractedSignals:
        entities = self._extract_spacy_entities(text)
        numbers = self._extract_numbers(text)
        dates = self._extract_dates(text)
        technical_terms = self._extract_generic_technical_terms(text)
        return ExtractedSignals(
            entities=entities,
            numbers=numbers,
            dates=dates,
            technical_terms=technical_terms,
        )

    def _keyword_overlap(self, reference_text: str, rag_text: str, top_k: int = 12) -> float:
        if not reference_text.strip() or not rag_text.strip():
            return 0.0

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([reference_text, rag_text])
        vocab = np.array(vectorizer.get_feature_names_out())

        ref_scores = matrix[0].toarray().ravel()
        rag_scores = matrix[1].toarray().ravel()

        ref_top_idx = np.argsort(ref_scores)[-top_k:]
        rag_top_idx = np.argsort(rag_scores)[-top_k:]

        ref_keywords = set(vocab[ref_top_idx])
        rag_keywords = set(vocab[rag_top_idx])

        if not ref_keywords:
            return 0.0
        return len(ref_keywords & rag_keywords) / len(ref_keywords)

    def _pair_lexical_features(self, source_text: str, target_text: str, prefix: str) -> dict[str, float]:
        source_tokens = [t for t in self._tokenize(source_text) if t not in ENGLISH_STOP_WORDS]
        target_tokens = [t for t in self._tokenize(target_text) if t not in ENGLISH_STOP_WORDS]

        source_set = set(source_tokens)
        target_set = set(target_tokens)
        common = source_set & target_set

        return {
            f"{prefix}_keyword_overlap": self._keyword_overlap(source_text, target_text),
            f"{prefix}_jaccard": self._jaccard(source_set, target_set),
            f"{prefix}_token_overlap": self._safe_ratio(len(common), max(len(source_set), 1)),
            f"{prefix}_source_coverage": self._safe_ratio(len(common), max(len(source_set), 1)),
            f"{prefix}_target_coverage": self._safe_ratio(len(common), max(len(target_set), 1)),
        }

    def _lexical_features(self, answer: str, evidence: str) -> dict[str, float]:
        answer_tokens = [t for t in self._tokenize(answer) if t not in ENGLISH_STOP_WORDS]
        evidence_tokens = [t for t in self._tokenize(evidence) if t not in ENGLISH_STOP_WORDS]

        answer_set = set(answer_tokens)
        evidence_set = set(evidence_tokens)

        common = answer_set & evidence_set
        answer_extra = answer_set - evidence_set

        answer_len = len(answer_tokens)
        evidence_len = len(evidence_tokens)

        return {
            "answer_evidence_keyword_overlap": self._keyword_overlap(answer, evidence),
            "answer_evidence_jaccard": self._jaccard(answer_set, evidence_set),
            "answer_evidence_token_overlap": self._safe_ratio(len(common), max(len(answer_set), 1)),
            "answer_evidence_coverage": self._safe_ratio(len(common), max(len(evidence_set), 1)),
            "answer_unsupported_word_ratio": self._safe_ratio(len(answer_extra), max(len(answer_set), 1)),
            "answer_evidence_length_ratio": self._safe_ratio(answer_len, max(evidence_len, 1)),
            "answer_token_count": float(len(self._tokenize(answer))),
            "evidence_token_count": float(len(self._tokenize(evidence))),
        }

    @staticmethod
    def _entity_comparison(reference: set[str], rag: set[str]) -> tuple[float, float, float]:
        overlap = len(reference & rag)
        missing = len(reference - rag)
        extra = len(rag - reference)
        denom = max(len(reference), 1)
        return overlap / denom, missing / denom, extra / max(len(rag), 1)

    def _technical_features(self, answer: str, evidence: str) -> dict[str, float]:
        ans = self._extract_signals(answer)
        evd = self._extract_signals(evidence)

        ent_overlap, ent_missing, ent_extra = self._entity_comparison(evd.entities, ans.entities)
        num_overlap, num_missing, num_extra = self._entity_comparison(evd.numbers, ans.numbers)
        term_overlap, term_missing, term_extra = self._entity_comparison(evd.technical_terms, ans.technical_terms)
        date_overlap, date_missing, date_extra = self._entity_comparison(evd.dates, ans.dates)

        mismatches = 0.0
        if evd.numbers:
            mismatches += num_missing
        if evd.technical_terms:
            mismatches += term_missing
        if evd.dates:
            mismatches += date_missing

        return {
            "entity_overlap": ent_overlap,
            "entity_missing": ent_missing,
            "entity_extra": ent_extra,
            "number_match": num_overlap,
            "number_mismatch": num_missing,
            "number_extra": num_extra,
            "date_match": date_overlap,
            "date_mismatch": date_missing,
            "date_extra": date_extra,
            "technical_term_overlap": term_overlap,
            "technical_term_missing": term_missing,
            "technical_term_extra": term_extra,
            "technical_consistency_mismatch": mismatches,
            "evidence_entity_count": float(len(evd.entities)),
            "answer_entity_count": float(len(ans.entities)),
            "evidence_number_count": float(len(evd.numbers)),
            "answer_number_count": float(len(ans.numbers)),
        }

    def build_row_features(self, row: pd.Series) -> dict[str, float | str]:
        question = str(row.get("question", ""))
        answer = str(row.get("answer", ""))
        evidence = str(row.get("evidence", ""))

        semantic = {
            "semantic_question_evidence": self._cosine_similarity(question, evidence),
            "semantic_question_answer": self._cosine_similarity(question, answer),
            "semantic_answer_evidence": self._cosine_similarity(answer, evidence),
        }

        lexical = self._lexical_features(answer=answer, evidence=evidence)
        question_evidence_lexical = self._pair_lexical_features(
            source_text=question,
            target_text=evidence,
            prefix="question_evidence",
        )
        technical = self._technical_features(answer=answer, evidence=evidence)

        question_signals = self._extract_signals(question)
        evidence_signals = self._extract_signals(evidence)
        qe_entity_overlap, _, _ = self._entity_comparison(question_signals.entities, evidence_signals.entities)
        qe_number_overlap, _, _ = self._entity_comparison(question_signals.numbers, evidence_signals.numbers)
        qe_term_overlap, _, _ = self._entity_comparison(
            question_signals.technical_terms,
            evidence_signals.technical_terms,
        )
        question_evidence_technical = {
            "question_evidence_entity_overlap": qe_entity_overlap,
            "question_evidence_number_overlap": qe_number_overlap,
            "question_evidence_term_overlap": qe_term_overlap,
        }

        return {
            "id": str(row.get("id", "")),
            "question_type": str(row.get("question_type", "")),
            "difficulty": str(row.get("difficulty", "")),
            **semantic,
            **lexical,
            **question_evidence_lexical,
            **technical,
            **question_evidence_technical,
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = [self.build_row_features(row) for _, row in df.iterrows()]
        return pd.DataFrame(rows)
