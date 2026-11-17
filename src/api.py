"""FastAPI Analytics REST API for Brazilian Parliamentary Expenses.

Exposes REST endpoints to query KPIs, warehouse aggregations, and detected anomalies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd

from src.pipeline import CeapPipeline
from src.warehouse import SpendingWarehouse
from src.anomaly_detector import SpendingAnomalyDetector

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gastos Públicos Analytics API",
    description="REST API for exploring Brazilian Parliamentary Spending (CEAP) & Anomaly Detection",
    version="1.0.0",
)

warehouse = SpendingWarehouse()
pipeline = CeapPipeline()
detector = SpendingAnomalyDetector()


class KPIResponse(BaseModel):
    total_transacoes: int
    total_gasto: float
    ticket_medio: float
    maior_despesa: float
    total_deputados: int
    total_fornecedores: int


class CategoryResponse(BaseModel):
    tipo_despesa: str
    macro_categoria: str
    qtd_recibos: int
    total_valor: float


@app.get("/api/v1/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "Gastos Públicos Analytics API", "version": "1.0.0"}


@app.get("/api/v1/kpis")
def get_kpis() -> Dict[str, Any]:
    """Retrieve macro KPIs from the data warehouse."""
    kpis = warehouse.get_summary_kpis()
    if not kpis:
        return {
            "total_transacoes": 0,
            "total_gasto": 0.0,
            "ticket_medio": 0.0,
            "maior_despesa": 0.0,
            "total_deputados": 0,
            "total_fornecedores": 0,
        }
    return kpis


@app.get("/api/v1/top-categories", response_model=List[CategoryResponse])
def get_top_categories(limit: int = Query(default=10, ge=1, le=50)) -> List[Dict[str, Any]]:
    """Retrieve top expenditure categories sorted by total value."""
    df = warehouse.get_top_spending_categories(limit=limit)
    if df.empty:
        return []
    return df.to_dict(orient="records")


@app.post("/api/v1/etl/run")
def trigger_etl(deputados_limit: int = Query(default=5, ge=1, le=20), ano: int = 2024) -> Dict[str, Any]:
    """Trigger on-demand ETL to ingest and load fresh data."""
    deputados = pipeline.fetch_deputados(limit=deputados_limit)
    all_expenses = []

    for dep in deputados:
        dep_id = dep.get("id")
        if dep_id:
            exp = pipeline.fetch_despesas_deputado(deputado_id=dep_id, ano=ano, limit=50)
            all_expenses.extend(exp)

    cleaned_df = pipeline.clean_expenses_dataframe(all_expenses)
    load_metrics = warehouse.load_dimensions_and_facts(cleaned_df, deputados_list=deputados)

    return {
        "status": "success",
        "deputados_processed": len(deputados),
        "total_receipts_ingested": len(cleaned_df),
        "warehouse_metrics": load_metrics,
    }
