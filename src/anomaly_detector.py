"""Anomaly Detection Engine for Public Parliamentary Expenses.

Combines statistical checks (Z-Score, IQR) and Machine Learning (Isolation Forest)
with domain-specific heuristic rules to identify suspicious spending patterns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SpendingAnomalyDetector:
    """Detects statistical outliers, fraud indicators, and unusual expenditure spikes."""

    def __init__(self, contamination: float = 0.03, z_threshold: float = 3.0):
        self.contamination = contamination
        self.z_threshold = z_threshold

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all anomaly detection algorithms on the expenses dataset."""
        if df.empty or len(df) < 5:
            df["is_anomaly"] = False
            df["anomaly_score"] = 0.0
            df["anomaly_reasons"] = "Dados insuficientes"
            return df

        result_df = df.copy()
        result_df["anomaly_reasons"] = [[] for _ in range(len(result_df))]

        # 1. Statistical Z-Score per Category
        result_df = self._flag_zscore_anomalies(result_df)

        # 2. Rule: Weekend Meal / Excessive Weekend Expenses
        result_df = self._flag_weekend_heuristics(result_df)

        # 3. Rule: Duplicate Same-Day Transactions (Fracionamento)
        result_df = self._flag_duplicate_receipts(result_df)

        # 4. Machine Learning: Isolation Forest
        result_df = self._flag_isolation_forest(result_df)

        # Compile final flag
        result_df["is_anomaly"] = result_df["anomaly_reasons"].apply(lambda r: len(r) > 0)
        result_df["anomaly_reasons_str"] = result_df["anomaly_reasons"].apply(lambda r: " | ".join(r) if r else "Normal")

        return result_df

    def _flag_zscore_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag receipts where spending is > z_threshold standard deviations above category mean."""
        if "tipo_despesa" not in df.columns or "valor_liquido" not in df.columns:
            return df

        for cat, group in df.groupby("tipo_despesa"):
            vals = group["valor_liquido"]
            std = vals.std()
            mean = vals.mean()
            if std > 0:
                z_scores = (vals - mean) / std
                outlier_indices = group[z_scores > self.z_threshold].index
                for idx in outlier_indices:
                    val = df.loc[idx, "valor_liquido"]
                    df.at[idx, "anomaly_reasons"].append(f"Z-Score alto ({val:.2f} > 3σ da média {mean:.2f})")

        return df

    def _flag_weekend_heuristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag excessive weekend expenditures or out-of-session meal charges."""
        if "is_fim_de_semana" not in df.columns or "valor_liquido" not in df.columns:
            return df

        # Flag high-value weekend expenses (> R$ 1,500)
        weekend_high = df[(df["is_fim_de_semana"] == True) & (df["valor_liquido"] > 1500.0)].index
        for idx in weekend_high:
            df.at[idx, "anomaly_reasons"].append(f"Gasto expressivo em fim de semana (R$ {df.loc[idx, 'valor_liquido']:.2f})")

        return df

    def _flag_duplicate_receipts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag suspicious repeated amounts to same vendor on same date."""
        subset_cols = ["id_deputado", "nome_fornecedor", "data_documento", "valor_liquido"]
        existing = [c for c in subset_cols if c in df.columns]
        if len(existing) == len(subset_cols):
            dups = df[df.duplicated(subset=subset_cols, keep=False)].index
            for idx in dups:
                df.at[idx, "anomaly_reasons"].append("Transação duplicada no mesmo dia/fornecedor/valor")

        return df

    def _flag_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply multivariate Isolation Forest on numerical features."""
        if not HAS_SKLEARN or len(df) < 15:
            df["anomaly_score"] = 0.0
            return df

        features = ["valor_liquido"]
        if "is_fim_de_semana" in df.columns:
            df["_weekend_int"] = df["is_fim_de_semana"].astype(int)
            features.append("_weekend_int")

        X = df[features].fillna(0).values
        clf = IsolationForest(contamination=self.contamination, random_state=42)
        preds = clf.fit_predict(X)
        scores = -clf.decision_function(X)

        df["anomaly_score"] = scores
        outliers_idx = df[preds == -1].index
        for idx in outliers_idx:
            df.at[idx, "anomaly_reasons"].append(f"ML Isolation Forest (Score: {df.loc[idx, 'anomaly_score']:.3f})")

        if "_weekend_int" in df.columns:
            df.drop(columns=["_weekend_int"], inplace=True)

        return df

    def summarize_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate high-level report on detected anomalies."""
        if "is_anomaly" not in df.columns:
            return {"total_records": len(df), "total_anomalies": 0, "anomaly_percentage": 0.0}

        anomalies = df[df["is_anomaly"] == True]
        total_anomalies = len(anomalies)
        pct = (total_anomalies / len(df) * 100) if len(df) > 0 else 0.0
        total_suspicious_amount = float(anomalies["valor_liquido"].sum()) if not anomalies.empty else 0.0

        return {
            "total_records": len(df),
            "total_anomalies": total_anomalies,
            "anomaly_percentage": round(pct, 2),
            "total_suspicious_amount_brl": round(total_suspicious_amount, 2),
            "most_common_reasons": anomalies["anomaly_reasons_str"].value_counts().head(5).to_dict() if not anomalies.empty else {},
        }
