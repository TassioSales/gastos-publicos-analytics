"""Unit tests for the CEAP Pipeline."""

import pytest
import pandas as pd
from src.pipeline import CeapPipeline


@pytest.fixture
def pipeline():
    return CeapPipeline()


def test_fetch_deputados_fallback(pipeline):
    deputados = pipeline._generate_fallback_deputados(limit=5)
    assert len(deputados) == 5
    assert deputados[0]["nome"] == "TABATA AMARAL"


def test_clean_expenses_dataframe_empty(pipeline):
    df = pipeline.clean_expenses_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_clean_expenses_dataframe_normalization(pipeline):
    sample_raw = [
        {
            "idDeputado": 204554,
            "ano": 2024,
            "mes": 5,
            "tipoDespesa": "COMBUSTÍVEIS E LUBRIFICANTES.",
            "dataDocumento": "2024-05-18",
            "valorDocumento": "250.50",
            "valorLiquido": "250.50",
            "nomeFornecedor": "POSTO DA TORRE LTDA",
            "cnpjCpfFornecedor": "00000000000191",
            "numDocumento": "NF10023",
        }
    ]
    df = pipeline.clean_expenses_dataframe(sample_raw)
    assert not df.empty
    assert df["valor_liquido"].iloc[0] == 250.50
    assert df["is_fim_de_semana"].iloc[0] == True  # 2024-05-18 was Saturday
    assert df["ano"].iloc[0] == 2024
