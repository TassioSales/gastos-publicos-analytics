"""Data Ingestion and ETL Pipeline for Brazilian Parliamentary Expenses (CEAP).

Connects to the Dados Abertos API (Câmara dos Deputados) and transforms raw
expenditures into cleaned, structured tabular datasets.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://dadosabertos.camara.leg.br/api/v2"


class CeapPipeline:
    """ETL Pipeline for extracting and transforming CEAP expenses."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "GastosPublicosAnalytics/1.0"})

    def fetch_deputados(self, legislatura: int = 57, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch active federal deputies from Dados Abertos API."""
        url = f"{BASE_URL}/deputados"
        params = {"idLegislatura": legislatura, "itens": limit, "ordem": "ASC", "ordenarPor": "nome"}
        try:
            logger.info("Fetching deputies list from %s...", url)
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json().get("dados", [])
            logger.info("Successfully fetched %d deputies.", len(data))
            return data
        except Exception as exc:
            logger.warning("API fetch failed (%s). Generating fallback deputies data.", exc)
            return self._generate_fallback_deputados(limit)

    def fetch_despesas_deputado(
        self,
        deputado_id: int,
        ano: int = 2024,
        mes: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch expense receipts for a specific deputy in a given year/month."""
        url = f"{BASE_URL}/deputados/{deputado_id}/despesas"
        params: Dict[str, Any] = {"ano": ano, "itens": limit, "ordem": "DESC", "ordenarPor": "dataDocumento"}
        if mes:
            params["mes"] = mes

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            dados = response.json().get("dados", [])
            for item in dados:
                item["idDeputado"] = deputado_id
            return dados
        except Exception as exc:
            logger.warning("Error fetching expenses for deputy %d: %s", deputado_id, exc)
            return []

    def clean_expenses_dataframe(self, raw_expenses: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform raw expense records into a normalized pandas DataFrame."""
        if not raw_expenses:
            return pd.DataFrame()

        df = pd.DataFrame(raw_expenses)

        # Standardize columns
        rename_map = {
            "idDeputado": "id_deputado",
            "ano": "ano",
            "mes": "mes",
            "tipoDespesa": "tipo_despesa",
            "dataDocumento": "data_documento",
            "valorDocumento": "valor_documento",
            "valorLiquido": "valor_liquido",
            "nomeFornecedor": "nome_fornecedor",
            "cnpjCpfFornecedor": "cnpj_cpf_fornecedor",
            "numDocumento": "num_documento",
        }
        available_cols = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)

        # Type conversions
        if "data_documento" in df.columns:
            df["data_documento"] = pd.to_datetime(df["data_documento"], errors="coerce")
        else:
            df["data_documento"] = pd.Timestamp.now()

        for num_col in ["valor_documento", "valor_liquido"]:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0)

        # Fill text defaults
        for str_col in ["tipo_despesa", "nome_fornecedor", "cnpj_cpf_fornecedor"]:
            if str_col in df.columns:
                df[str_col] = df[str_col].fillna("NÃO INFORMADO").astype(str).str.strip().str.upper()

        # Add derived date features
        df["ano"] = df["data_documento"].dt.year.fillna(2024).astype(int)
        df["mes"] = df["data_documento"].dt.month.fillna(1).astype(int)
        df["dia_semana"] = df["data_documento"].dt.day_name()
        df["is_fim_de_semana"] = df["data_documento"].dt.weekday.isin([5, 6])

        # Deduplicate
        if "num_documento" in df.columns and "id_deputado" in df.columns:
            df = df.drop_duplicates(subset=["id_deputado", "num_documento", "valor_liquido", "data_documento"])

        return df

    def _generate_fallback_deputados(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Mock fallback for offline resilience or testing."""
        deputados = [
            {"id": 204554, "nome": "TABATA AMARAL", "siglaPartido": "PSB", "siglaUf": "SP"},
            {"id": 204521, "nome": "NIKOLAS FERREIRA", "siglaPartido": "PL", "siglaUf": "MG"},
            {"id": 204560, "nome": "GUILHERME BOULOS", "siglaPartido": "PSOL", "siglaUf": "SP"},
            {"id": 160532, "nome": "EDUARDO BOLSONARO", "siglaPartido": "PL", "siglaUf": "SP"},
            {"id": 178959, "nome": "REGINALDO LOPES", "siglaPartido": "PT", "siglaUf": "MG"},
            {"id": 204423, "nome": "ERIKA HILTON", "siglaPartido": "PSOL", "siglaUf": "SP"},
            {"id": 204443, "nome": "KIM KATAGUIRI", "siglaPartido": "UNIÃO", "siglaUf": "SP"},
            {"id": 160511, "nome": "ALESSANDRO MOLON", "siglaPartido": "PSB", "siglaUf": "RJ"},
            {"id": 204535, "nome": "AMALIA BARROS", "siglaPartido": "PL", "siglaUf": "MT"},
            {"id": 141434, "nome": "GLEISI HOFFMANN", "siglaPartido": "PT", "siglaUf": "PR"},
        ]
        return deputados[:limit]
