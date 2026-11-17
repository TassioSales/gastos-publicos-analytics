"""Unit tests for the Data Warehouse."""

import os
import pytest
import pandas as pd
from src.warehouse import SpendingWarehouse


@pytest.fixture
def temp_warehouse(tmp_path):
    db_file = str(tmp_path / "test_dw.duckdb")
    return SpendingWarehouse(db_path=db_file)


def test_warehouse_initialization(temp_warehouse):
    kpis = temp_warehouse.get_summary_kpis()
    assert kpis.get("total_transacoes", 0) == 0


def test_warehouse_load_and_kpis(temp_warehouse):
    sample_df = pd.DataFrame([
        {
            "id_deputado": 100,
            "ano": 2024,
            "mes": 1,
            "tipo_despesa": "PASSAGEM AÉREA",
            "data_documento": "2024-01-10",
            "valor_documento": 1200.0,
            "valor_liquido": 1200.0,
            "nome_fornecedor": "LATAM AIRLINES",
            "cnpjCpfFornecedor": "02012862000160",
            "cnpj_cpf_fornecedor": "02012862000160",
            "num_documento": "BILHETE123",
            "is_fim_de_semana": False,
        }
    ])
    metrics = temp_warehouse.load_dimensions_and_facts(sample_df)
    assert metrics["fact_rows"] == 1

    kpis = temp_warehouse.get_summary_kpis()
    assert kpis["total_transacoes"] == 1
    assert kpis["total_gasto"] == 1200.0
