"""Advanced Machine Learning & Forensic Intelligence Engine for Public Parliamentary Expenses (CEAP).

Implements:
1. Multi-Model Ensemble Anomaly Detection (Isolation Forest, Local Outlier Factor, Elliptic Envelope, Z-Score, Heuristics).
2. Benford's Law (First-Digit Forensic Analysis & Chi-Square Goodness-of-Fit).
3. Deputy Behavioral Clustering (K-Means with Multi-Dimensional Spending Profiling).
4. Predictive Spending Forecasting (Time Series Trend & Seasonality Projection).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from scipy.stats import chisquare
    HAS_ML = True
except ImportError as err:
    logger.warning("ML libraries missing: %s", err)
    HAS_ML = False


class PublicSpendingMLEngine:
    """Comprehensive AI/ML engine for public spending forensic auditing."""

    def __init__(self, contamination: float = 0.03):
        self.contamination = contamination

    # =========================================================================
    # 1. MULTI-MODEL ENSEMBLE ANOMALY DETECTION
    # =========================================================================
    def detect_ensemble_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run ensemble of Isolation Forest, LOF, Elliptic Envelope, Z-Scores and heuristics."""
        if df.empty or len(df) < 10:
            df["is_anomaly"] = False
            df["risk_score"] = 0.0
            df["risk_level"] = "NORMAL"
            df["models_flagged"] = [[] for _ in range(len(df))]
            df["anomaly_reasons_str"] = "Dados insuficientes"
            return df

        result = df.copy()
        result["models_flagged"] = [[] for _ in range(len(result))]
        result["risk_score"] = 0.0

        # Features for numerical outlier models
        features = ["valor_liquido"]
        if "is_fim_de_semana" in result.columns:
            result["_weekend_int"] = result["is_fim_de_semana"].astype(int)
            features.append("_weekend_int")

        X = result[features].fillna(0).values

        if HAS_ML and len(result) >= 20:
            # Model 1: Isolation Forest
            try:
                if_model = IsolationForest(contamination=self.contamination, random_state=42)
                if_preds = if_model.fit_predict(X)
                for idx in result[if_preds == -1].index:
                    result.at[idx, "models_flagged"].append("Isolation Forest")
                    result.at[idx, "risk_score"] += 25.0
            except Exception as e:
                logger.debug("Isolation Forest error: %s", e)

            # Model 2: Local Outlier Factor (LOF)
            try:
                lof_model = LocalOutlierFactor(n_neighbors=min(20, len(result) - 1), contamination=self.contamination)
                lof_preds = lof_model.fit_predict(X)
                for idx in result[lof_preds == -1].index:
                    result.at[idx, "models_flagged"].append("Local Outlier Factor (LOF)")
                    result.at[idx, "risk_score"] += 25.0
            except Exception as e:
                logger.debug("LOF error: %s", e)

            # Model 3: Elliptic Envelope (Mahalanobis Distance)
            try:
                ee_model = EllipticEnvelope(contamination=self.contamination, random_state=42)
                ee_preds = ee_model.fit_predict(X)
                for idx in result[ee_preds == -1].index:
                    result.at[idx, "models_flagged"].append("Elliptic Envelope")
                    result.at[idx, "risk_score"] += 20.0
            except Exception as e:
                logger.debug("Elliptic Envelope error: %s", e)

        # Statistical: Category-specific Z-Score (> 3 standard deviations)
        if "tipo_despesa" in result.columns and "valor_liquido" in result.columns:
            for cat, group in result.groupby("tipo_despesa"):
                std = group["valor_liquido"].std()
                mean = group["valor_liquido"].mean()
                if std > 0:
                    z_scores = (group["valor_liquido"] - mean) / std
                    for idx in group[z_scores > 3.0].index:
                        result.at[idx, "models_flagged"].append(f"Z-Score 3σ ({cat[:20]})")
                        result.at[idx, "risk_score"] += 20.0

        # Heuristic 1: Weekend high expenditure (> R$ 1.500)
        if "is_fim_de_semana" in result.columns:
            wknd_idx = result[(result["is_fim_de_semana"] == True) & (result["valor_liquido"] > 1500.0)].index
            for idx in wknd_idx:
                result.at[idx, "models_flagged"].append("Gasto Atípico de Fim de Semana")
                result.at[idx, "risk_score"] += 15.0

        # Heuristic 2: Duplicate transaction (same deputy, same vendor, same date, same amount)
        subset_cols = ["id_deputado", "nome_fornecedor", "data_documento", "valor_liquido"]
        existing = [c for c in subset_cols if c in result.columns]
        if len(existing) == len(subset_cols):
            dup_idx = result[result.duplicated(subset=subset_cols, keep=False)].index
            for idx in dup_idx:
                result.at[idx, "models_flagged"].append("Recibo Duplicado (Mesmo Dia/Fornecedor)")
                result.at[idx, "risk_score"] += 15.0

        # Clean helper column
        if "_weekend_int" in result.columns:
            result.drop(columns=["_weekend_int"], inplace=True)

        # Normalize risk score to max 100
        result["risk_score"] = result["risk_score"].clip(upper=100.0)

        # Classify Risk Level
        def classify_risk(score: float) -> str:
            if score >= 60.0:
                return "CRÍTICO"
            if score >= 40.0:
                return "ALTO"
            if score >= 20.0:
                return "MÉDIO"
            return "NORMAL"

        result["risk_level"] = result["risk_score"].apply(classify_risk)
        result["is_anomaly"] = result["risk_score"] >= 20.0
        result["anomaly_reasons_str"] = result["models_flagged"].apply(
            lambda flags: " • ".join(flags) if flags else "Normal"
        )

        return result

    # =========================================================================
    # 2. BENFORD'S LAW FORENSIC AUDIT
    # =========================================================================
    def audit_benfords_law(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform first-digit Benford's Law analysis and Chi-Square test."""
        if df.empty or "valor_liquido" not in df.columns:
            return {}

        positive_vals = df[df["valor_liquido"] > 0]["valor_liquido"].astype(str)
        # Extract first non-zero digit
        first_digits = positive_vals.str.replace(r"^[0\.]+", "", regex=True).str[0]
        valid_digits = first_digits[first_digits.isin([str(i) for i in range(1, 10)])].astype(int)

        total_obs = len(valid_digits)
        if total_obs < 30:
            return {"status": "INSUFFICIENT_DATA", "total_receipts": total_obs}

        digits = list(range(1, 10))
        counts = valid_digits.value_counts()
        observed_counts = [int(counts.get(d, 0)) for d in digits]
        observed_pct = [round((c / total_obs) * 100, 2) for c in observed_counts]

        # Theoretical Benford distribution: log10(1 + 1/d)
        expected_prob = [math.log10(1 + 1 / d) for d in digits]
        expected_counts = [p * total_obs for p in expected_prob]
        expected_pct = [round(p * 100, 2) for p in expected_prob]

        # Mean Absolute Deviation (MAD)
        mad = float(np.mean([abs(obs / 100 - exp) for obs, exp in zip(observed_pct, expected_prob)]))

        # Chi-Square Goodness-of-Fit
        if HAS_ML:
            chi2_stat, p_val = chisquare(observed_counts, f_exp=expected_counts)
        else:
            chi2_stat = float(sum((o - e) ** 2 / e for o, e in zip(observed_counts, expected_counts)))
            p_val = 0.05

        # Conformity rating according to forensic accounting standards (Nigrini, 2012)
        if mad <= 0.006:
            conformity = "CONFORMIDADE ESTREITA (PADRÃO NATURAL)"
            risk_level = "BAIXO"
        elif mad <= 0.012:
            conformity = "CONFORMIDADE ACEITÁVEL"
            risk_level = "MODERADO"
        elif mad <= 0.015:
            conformity = "CONFORMIDADE MARGINAL"
            risk_level = "ATENÇÃO"
        else:
            conformity = "NÃO-CONFORMIDADE (DESVIO ESTRUTURAL DETECTADO)"
            risk_level = "ALTO"

        return {
            "total_receipts_evaluated": total_obs,
            "digits": digits,
            "observed_pct": observed_pct,
            "expected_pct": expected_pct,
            "chi2_stat": round(float(chi2_stat), 2),
            "p_value": round(float(p_val), 4),
            "mad": round(mad, 5),
            "conformity_status": conformity,
            "forensic_risk": risk_level,
            "interpretation": (
                "A Lei de Newcomb-Benford mede a frequência natural do primeiro dígito (1 a 9) em dados financeiros. "
                "Desvios anormais apontam possíveis notas fracionadas ou arredondamentos manuais deliberados."
            ),
        }

    # =========================================================================
    # 3. K-MEANS BEHAVIORAL CLUSTERING OF DEPUTIES
    # =========================================================================
    def cluster_deputy_behavior(self, df: pd.DataFrame, n_clusters: int = 4) -> Dict[str, Any]:
        """Segment deputies into behavioral spending clusters using K-Means."""
        if not HAS_ML or df.empty or "nome_deputado" not in df.columns:
            return {"clusters": []}

        # Aggregate metrics per deputy
        dep_groups = df.groupby(["nome_deputado", "partido", "uf"])

        records = []
        for (nome, partido, uf), group in dep_groups:
            tot = float(group["valor_liquido"].sum())
            cnt = len(group)
            avg = float(group["valor_liquido"].mean()) if cnt > 0 else 0.0
            wknd_pct = float(group.get("is_fim_de_semana", pd.Series([False])).sum() / cnt * 100) if cnt > 0 else 0.0
            unique_vendors = int(group["nome_fornecedor"].nunique()) if "nome_fornecedor" in group.columns else 1

            # Category shares
            cat_sums = group.groupby("macro_categoria")["valor_liquido"].sum() if "macro_categoria" in group.columns else {}
            transp_pct = float(cat_sums.get("TRANSPORTE & LOGÍSTICA", 0.0) / tot * 100) if tot > 0 else 0.0
            consult_pct = float(cat_sums.get("CONSULTORIA & SERVIÇOS TÉCNICOS", 0.0) / tot * 100) if tot > 0 else 0.0
            comunic_pct = float(cat_sums.get("COMUNICAÇÃO & MARKETING", 0.0) / tot * 100) if tot > 0 else 0.0

            records.append({
                "nome": nome,
                "partido": partido,
                "uf": uf,
                "total_gasto": tot,
                "qtd_recibos": cnt,
                "ticket_medio": avg,
                "pct_fim_de_semana": round(wknd_pct, 1),
                "pct_transporte": round(transp_pct, 1),
                "pct_consultoria": round(consult_pct, 1),
                "pct_comunicacao": round(comunic_pct, 1),
                "fornecedores_unicos": unique_vendors,
            })

        dep_df = pd.DataFrame(records)
        if len(dep_df) < n_clusters:
            return {"clusters": [], "deputies": records}

        # Features for clustering
        feature_cols = [
            "total_gasto",
            "ticket_medio",
            "pct_fim_de_semana",
            "pct_transporte",
            "pct_consultoria",
            "pct_comunicacao",
        ]
        X = dep_df[feature_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        dep_df["cluster_id"] = labels

        # Semantic cluster profiles
        cluster_info = []
        cluster_names = [
            ("Logística & Deslocamento Intensivo", "Despesas focadas predominantemente em passagens aéreas e locação de veículos.", "cyan"),
            ("Consultoria & Serviços Especializados", "Ticket médio elevado com contratos pontuais de assessoria e pesquisa.", "indigo"),
            ("Comunicação & Divulgação Parlamentar", "Foco em marketing institucional, publicidade e materiais impressos.", "amber"),
            ("Perfil Econômico & Baixo Volume", "Uso contido da cota parlamentar com valores globais abaixo da média geral.", "emerald"),
        ]

        for cid in range(n_clusters):
            c_members = dep_df[dep_df["cluster_id"] == cid]
            name, desc, color = cluster_names[cid % len(cluster_names)]
            cluster_info.append({
                "cluster_id": cid,
                "name": name,
                "description": desc,
                "color": color,
                "total_deputies": len(c_members),
                "avg_spending": round(float(c_members["total_gasto"].mean()), 2) if not c_members.empty else 0.0,
                "avg_ticket": round(float(c_members["ticket_medio"].mean()), 2) if not c_members.empty else 0.0,
                "top_members": c_members.sort_values("total_gasto", ascending=False)[["nome", "partido", "uf", "total_gasto"]].head(5).to_dict(orient="records"),
            })

        return {
            "total_evaluated_deputies": len(dep_df),
            "clusters": cluster_info,
            "deputies_clustered": dep_df[["nome", "partido", "uf", "cluster_id", "total_gasto", "ticket_medio"]].to_dict(orient="records"),
        }

    # =========================================================================
    # 4. PREDICTIVE SPENDING FORECASTING (TIME SERIES)
    # =========================================================================
    def forecast_spending(self, monthly_timeline_df: pd.DataFrame, forecast_months: int = 6) -> Dict[str, Any]:
        """Project future spending for the next N months using regression with seasonality."""
        if monthly_timeline_df.empty or len(monthly_timeline_df) < 12:
            return {"historical": [], "forecast": [], "trend": "INSUFFICIENT_DATA"}

        df = monthly_timeline_df.copy().sort_values(["ano", "mes"]).reset_index(drop=True)
        df["time_idx"] = np.arange(len(df))
        df["month_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

        X = df[["time_idx", "month_sin", "month_cos"]].values
        y = df["total_valor"].values

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        # Predict future months
        last_time = df["time_idx"].iloc[-1]
        last_year = int(df["ano"].iloc[-1])
        last_month = int(df["mes"].iloc[-1])

        future_records = []
        cur_year = last_year
        cur_month = last_month

        # Estimate residual standard deviation for confidence bounds
        y_pred_hist = model.predict(X)
        residuals_std = float(np.std(y - y_pred_hist))

        for step in range(1, forecast_months + 1):
            cur_month += 1
            if cur_month > 12:
                cur_month = 1
                cur_year += 1

            future_time = last_time + step
            feat = np.array([[future_time, np.sin(2 * np.pi * cur_month / 12), np.cos(2 * np.pi * cur_month / 12)]])
            pred_val = float(max(0.0, model.predict(feat)[0]))

            future_records.append({
                "ano": cur_year,
                "mes": cur_month,
                "label": f"{cur_month:02d}/{cur_year}",
                "predicted_valor": round(pred_val, 2),
                "lower_bound": round(max(0.0, pred_val - 1.96 * residuals_std), 2),
                "upper_bound": round(pred_val + 1.96 * residuals_std, 2),
            })

        # Calculate slope/trend
        slope = float(model.coef_[0])
        trend_label = "TENDÊNCIA DE ALTA" if slope > 5000 else "TENDÊNCIA DE QUEDA" if slope < -5000 else "ESTÁVEL"

        historical = [
            {"label": f"{int(r['mes']):02d}/{int(r['ano'])}", "ano": int(r["ano"]), "mes": int(r["mes"]), "valor": round(float(r["total_valor"]), 2)}
            for r in df.to_dict(orient="records")[-24:]  # Last 24 months
        ]

        return {
            "trend": trend_label,
            "monthly_slope_brl": round(slope, 2),
            "historical": historical,
            "forecast": future_records,
        }
