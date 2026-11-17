"""Unit tests for the Anomaly Detector."""

import pytest
import pandas as pd
from src.anomaly_detector import SpendingAnomalyDetector


@pytest.fixture
def detector():
    return SpendingAnomalyDetector()


def test_detector_empty_dataframe(detector):
    df = pd.DataFrame()
    res = detector.analyze_dataframe(df)
    assert res.empty


def test_detector_flags_weekend_high_spend(detector):
    data = [
        {"id_deputado": 1, "tipo_despesa": "COMBUSTÍVEL", "data_documento": "2024-01-01", "valor_liquido": 100.0, "is_fim_de_semana": False, "nome_fornecedor": "POSTO A"},
        {"id_deputado": 1, "tipo_despesa": "COMBUSTÍVEL", "data_documento": "2024-01-02", "valor_liquido": 120.0, "is_fim_de_semana": False, "nome_fornecedor": "POSTO A"},
        {"id_deputado": 1, "tipo_despesa": "COMBUSTÍVEL", "data_documento": "2024-01-03", "valor_liquido": 110.0, "is_fim_de_semana": False, "nome_fornecedor": "POSTO A"},
        {"id_deputado": 1, "tipo_despesa": "COMBUSTÍVEL", "data_documento": "2024-01-04", "valor_liquido": 105.0, "is_fim_de_semana": False, "nome_fornecedor": "POSTO A"},
        {"id_deputado": 1, "tipo_despesa": "COMBUSTÍVEL", "data_documento": "2024-01-06", "valor_liquido": 3500.0, "is_fim_de_semana": True, "nome_fornecedor": "POSTO B"},
    ]
    df = pd.DataFrame(data)
    analyzed = detector.analyze_dataframe(df)
    assert "is_anomaly" in analyzed.columns

    weekend_outlier = analyzed[analyzed["valor_liquido"] == 3500.0]
    assert weekend_outlier["is_anomaly"].iloc[0] == True
    assert "fim de semana" in weekend_outlier["anomaly_reasons_str"].iloc[0]
