"""FastAPI Analytics REST API & Modern Enterprise Dashboard for Brazilian Parliamentary Expenses (CEAP).

Exposes high-performance analytical endpoints and serves an interactive, responsive Bloomberg/Stark-style Web UI.
Supports granular multi-year filtering (2019 - 2026).
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.pipeline import CeapPipeline
from src.warehouse import SpendingWarehouse
from src.anomaly_detector import SpendingAnomalyDetector
from src.ml_engine import PublicSpendingMLEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gastos Públicos Analytics API & Dashboard",
    description="Enterprise Analytics Engine for Brazilian Parliamentary Spending (CEAP) & Machine Learning Anomaly Detection",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

warehouse = SpendingWarehouse(read_only=True)
pipeline = CeapPipeline()
detector = SpendingAnomalyDetector()
ml_engine = PublicSpendingMLEngine()


@app.get("/api/v1/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "Gastos Públicos Analytics Engine v2.1", "engine": "DuckDB + FastAPI"}


@app.get("/api/v1/kpis")
def get_kpis(ano: Optional[int] = Query(default=None)) -> Dict[str, Any]:
    kpis = warehouse.get_summary_kpis(ano=ano)
    if not kpis:
        return {
            "total_transacoes": 0,
            "total_gasto": 0.0,
            "ticket_medio": 0.0,
            "maior_despesa": 0.0,
            "total_deputados": 0,
            "total_fornecedores": 0,
        }
    return {
        "total_transacoes": int(kpis.get("total_transacoes", 0)),
        "total_gasto": float(kpis.get("total_gasto", 0.0)),
        "ticket_medio": float(kpis.get("ticket_medio", 0.0)),
        "maior_despesa": float(kpis.get("maior_despesa", 0.0)),
        "total_deputados": int(kpis.get("total_deputados", 0)),
        "total_fornecedores": int(kpis.get("total_fornecedores", 0)),
    }


@app.get("/api/v1/years")
def get_years_summary() -> List[Dict[str, Any]]:
    df = warehouse.get_spending_by_year()
    if df.empty:
        return []
    records = []
    for r in df.to_dict(orient="records"):
        records.append({
            "ano": int(r.get("ano", 0)),
            "qtd_recibos": int(r.get("qtd_recibos", 0)),
            "total_valor": float(r.get("total_valor", 0.0)),
            "ticket_medio": float(r.get("ticket_medio", 0.0)),
        })
    return records


@app.get("/api/v1/parties")
def get_parties_summary(
    ano: Optional[int] = Query(default=None),
    limit: int = Query(default=15, ge=1, le=50)
) -> List[Dict[str, Any]]:
    df = warehouse.get_spending_by_party(ano=ano, limit=limit)
    if df.empty:
        return []
    records = []
    for r in df.to_dict(orient="records"):
        records.append({
            "partido": str(r.get("partido", "OUTROS")),
            "total_deputados": int(r.get("total_deputados", 0)),
            "qtd_recibos": int(r.get("qtd_recibos", 0)),
            "total_valor": float(r.get("total_valor", 0.0)),
            "ticket_medio": float(r.get("ticket_medio", 0.0)),
        })
    return records


@app.get("/api/v1/states")
def get_states_summary(
    ano: Optional[int] = Query(default=None),
    limit: int = Query(default=27, ge=1, le=27)
) -> List[Dict[str, Any]]:
    df = warehouse.get_spending_by_uf(ano=ano, limit=limit)
    if df.empty:
        return []
    records = []
    for r in df.to_dict(orient="records"):
        records.append({
            "uf": str(r.get("uf", "BR")),
            "total_deputados": int(r.get("total_deputados", 0)),
            "qtd_recibos": int(r.get("qtd_recibos", 0)),
            "total_valor": float(r.get("total_valor", 0.0)),
        })
    return records


@app.get("/api/v1/top-categories")
def get_top_categories(
    ano: Optional[int] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50)
) -> List[Dict[str, Any]]:
    df = warehouse.get_top_spending_categories(ano=ano, limit=limit)
    if df.empty:
        return []
    records = []
    for r in df.to_dict(orient="records"):
        records.append({
            "tipo_despesa": str(r.get("tipo_despesa", "")),
            "macro_categoria": str(r.get("macro_categoria", "OUTROS")),
            "qtd_recibos": int(r.get("qtd_recibos", 0)),
            "total_valor": float(r.get("total_valor", 0.0)),
        })
    return records


@app.get("/api/v1/suppliers")
def get_top_suppliers(
    ano: Optional[int] = Query(default=None),
    limit: int = Query(default=15, ge=1, le=50)
) -> List[Dict[str, Any]]:
    df = warehouse.get_top_suppliers(ano=ano, limit=limit)
    if df.empty:
        return []
    records = []
    for r in df.to_dict(orient="records"):
        records.append({
            "nome_fornecedor": str(r.get("nome_fornecedor", "")),
            "cnpj_cpf": str(r.get("cnpj_cpf", "")),
            "qtd_recibos": int(r.get("qtd_recibos", 0)),
            "total_valor": float(r.get("total_valor", 0.0)),
            "ticket_medio": float(r.get("ticket_medio", 0.0)),
            "maior_recibo": float(r.get("maior_recibo", 0.0)),
        })
    return records


@app.get("/api/v1/deputies")
def get_deputies_ranking(
    ano: Optional[int] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100)
) -> List[Dict[str, Any]]:
    df = warehouse.get_deputy_ranking(ano=ano, limit=limit)
    if df.empty:
        return []
    records = []
    for r in df.to_dict(orient="records"):
        records.append({
            "nome": str(r.get("nome", "")),
            "partido": str(r.get("partido", "")),
            "uf": str(r.get("uf", "")),
            "qtd_recibos": int(r.get("qtd_recibos", 0)),
            "total_gasto": float(r.get("total_gasto", 0.0)),
            "ticket_medio": float(r.get("ticket_medio", 0.0)),
            "maior_gasto": float(r.get("maior_gasto", 0.0)),
        })
    return records


@app.get("/api/v1/deputies/list")
def get_deputies_list() -> List[Dict[str, Any]]:
    """Return distinct list of deputies for filter dropdowns."""
    return warehouse.get_all_deputies_list()


@app.get("/api/v1/expenses")
def get_detailed_expenses(
    ano: Optional[int] = Query(default=None),
    deputado: Optional[str] = Query(default=None),
    fornecedor: Optional[str] = Query(default=None),
    categoria: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="valor_desc"),
) -> Dict[str, Any]:
    """Return granular itemized expenses: who spent, what they bought, and where (supplier/CNPJ)."""
    res = warehouse.get_detailed_expenses(
        ano=ano,
        deputado=deputado,
        fornecedor=fornecedor,
        categoria=categoria,
        search=search,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
    )
    cleaned_expenses = []
    for r in res.get("expenses", []):
        cleaned_expenses.append({
            "despesa_id": str(r.get("despesa_id", "")),
            "id_deputado": int(r.get("id_deputado") or 0),
            "deputado_nome": str(r.get("deputado_nome") or "PARLAMENTAR"),
            "deputado_partido": str(r.get("deputado_partido") or "OUTROS"),
            "deputado_uf": str(r.get("deputado_uf") or "BR"),
            "categoria": str(r.get("categoria") or ""),
            "macro_categoria": str(r.get("macro_categoria") or "OUTROS"),
            "fornecedor_nome": str(r.get("fornecedor_nome") or ""),
            "fornecedor_cnpj": str(r.get("fornecedor_cnpj") or ""),
            "data_documento": str(r.get("data_documento") or "")[:10],
            "ano": int(r.get("ano") or 0),
            "mes": int(r.get("mes") or 0),
            "valor_liquido": float(r.get("valor_liquido") or 0.0),
            "is_fim_de_semana": bool(r.get("is_fim_de_semana")),
            "num_documento": str(r.get("num_documento") or ""),
        })
    return {
        "total_matching": res.get("total_matching", 0),
        "limit": limit,
        "offset": offset,
        "expenses": cleaned_expenses,
    }


@app.get("/api/v1/deputies/{deputado_nome}/profile")
def get_deputy_profile(
    deputado_nome: str,
    ano: Optional[int] = Query(default=None),
) -> Dict[str, Any]:
    """Return comprehensive breakdown/dossier for a specific deputy (who, what, where)."""
    profile = warehouse.get_deputy_profile(deputado_nome, ano=ano)
    if not profile:
        raise HTTPException(status_code=404, detail="Parlamentar não encontrado no Data Warehouse")
    return profile


@app.get("/api/v1/timeline")
def get_timeline(ano: Optional[int] = Query(default=None)) -> List[Dict[str, Any]]:
    where_clause = f"WHERE f.ano = {int(ano)}" if ano else "WHERE f.ano IS NOT NULL"
    query = f"""
        SELECT
            f.ano,
            f.mes,
            COUNT(*) as qtd_recibos,
            SUM(f.valor_liquido) as total_valor
        FROM fact_despesas f
        {where_clause}
        GROUP BY f.ano, f.mes
        ORDER BY f.ano ASC, f.mes ASC
    """
    if warehouse.use_duckdb:
        df = warehouse.conn.execute(query).df()
    else:
        df = pd.read_sql_query(query, warehouse.conn)

    if df.empty:
        return []
    records = []
    for r in df.to_dict(orient="records"):
        records.append({
            "ano": int(r.get("ano", 0)),
            "mes": int(r.get("mes", 1)),
            "qtd_recibos": int(r.get("qtd_recibos", 0)),
            "total_valor": float(r.get("total_valor", 0.0)),
        })
    return records


@app.get("/api/v1/ml/benford")
def get_benford_audit(ano: Optional[int] = Query(default=None)) -> Dict[str, Any]:
    """Execute Benford's Law forensic analysis and Chi-Square Goodness-of-Fit."""
    where_clause = f"WHERE f.ano = {int(ano)}" if ano else ""
    query = f"""
        SELECT f.valor_liquido
        FROM fact_despesas f
        {where_clause}
    """
    if warehouse.use_duckdb:
        df = warehouse.conn.execute(query).df()
    else:
        df = pd.read_sql_query(query, warehouse.conn)
    return ml_engine.audit_benfords_law(df)


@app.get("/api/v1/ml/clusters")
def get_deputy_clusters(ano: Optional[int] = Query(default=None)) -> Dict[str, Any]:
    """Segment deputies into behavioral spending clusters using K-Means."""
    where_clause = f"WHERE f.ano = {int(ano)}" if ano else ""
    query = f"""
        SELECT f.despesa_id, f.valor_liquido, f.is_fim_de_semana,
               d.nome as nome_deputado, d.partido, d.uf,
               c.macro_categoria, forn.nome_fornecedor
        FROM fact_despesas f
        JOIN dim_deputado d ON f.deputado_key = d.deputado_key
        JOIN dim_categoria c ON f.categoria_key = c.categoria_key
        JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
        {where_clause}
    """
    if warehouse.use_duckdb:
        df = warehouse.conn.execute(query).df()
    else:
        df = pd.read_sql_query(query, warehouse.conn)
    return ml_engine.cluster_deputy_behavior(df)


@app.get("/api/v1/ml/forecast")
def get_spending_forecast(months: int = Query(default=6, ge=1, le=12)) -> Dict[str, Any]:
    """Forecast future monthly spending using time series regression with seasonality."""
    query = """
        SELECT ano, mes, SUM(valor_liquido) as total_valor
        FROM fact_despesas
        WHERE ano IS NOT NULL
        GROUP BY ano, mes
        ORDER BY ano ASC, mes ASC
    """
    if warehouse.use_duckdb:
        df = warehouse.conn.execute(query).df()
    else:
        df = pd.read_sql_query(query, warehouse.conn)
    return ml_engine.forecast_spending(df, forecast_months=months)


@app.get("/api/v1/anomalies")
def get_anomalies(
    ano: Optional[int] = Query(default=None),
    risk_level: Optional[str] = Query(default=None),
    limit: int = Query(default=250, ge=1, le=1000)
) -> Dict[str, Any]:
    """Return multi-model ensemble anomalies (Isolation Forest + LOF + Elliptic Envelope + Z-Score)."""
    where_clause = f"WHERE f.ano = {int(ano)}" if ano else ""
    query = f"""
        SELECT f.despesa_id, f.data_documento, f.ano, f.mes, f.valor_liquido,
               f.is_fim_de_semana, f.num_documento,
               d.nome as nome_deputado, d.partido, d.uf,
               c.tipo_despesa, c.macro_categoria,
               forn.nome_fornecedor, forn.cnpj_cpf
        FROM fact_despesas f
        LEFT JOIN dim_deputado d ON f.deputado_key = d.deputado_key
        LEFT JOIN dim_categoria c ON f.categoria_key = c.categoria_key
        LEFT JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
        {where_clause}
        ORDER BY f.valor_liquido DESC
        LIMIT 5000
    """
    if warehouse.use_duckdb:
        df = warehouse.conn.execute(query).df()
    else:
        df = pd.read_sql_query(query, warehouse.conn)

    if df.empty:
        return {
            "summary": {"total_records": 0, "total_anomalies": 0, "anomaly_percentage": 0.0, "total_suspicious_amount_brl": 0.0, "risk_counts": {}},
            "anomalies": []
        }

    analyzed = ml_engine.detect_ensemble_anomalies(df)
    outliers = analyzed[analyzed["is_anomaly"] == True]

    if risk_level and risk_level.upper() != "ALL":
        target = risk_level.upper().replace("Í", "I").replace("É", "E")
        outliers = outliers[outliers["risk_level"].str.replace("Í", "I").str.replace("É", "E") == target]

    risk_counts = analyzed["risk_level"].value_counts().to_dict()
    total_suspicious = float(outliers["valor_liquido"].sum()) if not outliers.empty else 0.0
    pct = (len(outliers) / len(df) * 100) if len(df) > 0 else 0.0

    summary = {
        "total_records": len(df),
        "total_anomalies": len(outliers),
        "anomaly_percentage": round(pct, 2),
        "total_suspicious_amount_brl": round(total_suspicious, 2),
        "risk_counts": risk_counts,
    }

    # Slice up to requested limit
    outliers_page = outliers.head(limit)

    clean_records = []
    for r in outliers_page.to_dict(orient="records"):
        clean_records.append({
            "despesa_id": str(r.get("despesa_id", "")),
            "data_documento": str(r.get("data_documento", ""))[:10],
            "ano": int(r.get("ano") or 0),
            "mes": int(r.get("mes") or 0),
            "valor_liquido": float(r.get("valor_liquido") or 0.0),
            "nome_deputado": str(r.get("nome_deputado") or "PARLAMENTAR"),
            "partido": str(r.get("partido") or "OUTROS"),
            "uf": str(r.get("uf") or "BR"),
            "tipo_despesa": str(r.get("tipo_despesa") or ""),
            "macro_categoria": str(r.get("macro_categoria") or ""),
            "nome_fornecedor": str(r.get("nome_fornecedor") or ""),
            "cnpj_cpf": str(r.get("cnpj_cpf") or ""),
            "risk_score": round(float(r.get("risk_score") or 0.0), 1),
            "risk_level": str(r.get("risk_level") or "MÉDIO"),
            "models_flagged": list(r.get("models_flagged") or []),
            "anomaly_reasons_str": str(r.get("anomaly_reasons_str") or "ANOMALIA"),
        })

    return {"summary": summary, "anomalies": clean_records}


@app.post("/api/v1/etl/run")
def trigger_etl(deputados_limit: int = Query(default=25, ge=1, le=50)) -> Dict[str, Any]:
    writer_warehouse = SpendingWarehouse(read_only=False)
    deputados = pipeline.fetch_deputados(limit=deputados_limit)
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    raw = pipeline._generate_rich_multiyear_dataset(deputados, years, items_per_year=35)
    cleaned = pipeline.clean_expenses_dataframe(raw)
    metrics = writer_warehouse.load_dimensions_and_facts(cleaned, deputados_list=deputados)

    return {
        "status": "success",
        "years": years,
        "deputados_count": len(deputados),
        "total_receipts_loaded": len(cleaned),
        "warehouse_metrics": metrics,
    }


# ============================================================================
# MODERN ENTERPRISE DASHBOARD UI WITH INTERACTIVE YEAR FILTER (2019 - 2026)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    # Return raw modern single-page dashboard with interactive filter engine
    html = """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gastos Públicos Analytics | Câmara dos Deputados (CEAP 2019-2026)</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        darkBg: '#080c14',
                        darkCard: '#0f172a',
                        cardBorder: '#1e293b',
                        accentCyan: '#06b6d4',
                        accentBlue: '#38bdf8',
                        accentPurple: '#818cf8',
                        alertRed: '#f43f5e',
                        statusGreen: '#10b981'
                    }
                }
            }
        }
    </script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            background-color: #080c14;
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            background-image: radial-gradient(rgba(56, 189, 248, 0.04) 1px, transparent 1px);
            background-size: 28px 28px;
        }
        .font-mono { font-family: 'JetBrains Mono', monospace; }
        .glass {
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.07);
        }
        .glass:hover {
            border-color: rgba(6, 182, 212, 0.3);
        }
        .glow-cyan { box-shadow: 0 0 25px rgba(6, 182, 212, 0.18); }
        .glow-red { box-shadow: 0 0 25px rgba(244, 63, 94, 0.18); }
        .btn-year.active {
            background: linear-gradient(135deg, #06b6d4, #0284c7);
            color: #ffffff;
            border-color: #38bdf8;
            box-shadow: 0 0 16px rgba(6, 182, 212, 0.4);
            font-weight: 700;
        }
    </style>
</head>
<body class="min-h-screen flex flex-col justify-between selection:bg-accentCyan selection:text-black">

    <!-- Top Navigation Header -->
    <header class="glass sticky top-0 z-50 px-6 py-4 border-b border-cardBorder">
        <div class="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
            <div class="flex items-center gap-3.5">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white">
                    <i data-lucide="landmark" class="w-6 h-6"></i>
                </div>
                <div>
                    <h1 class="text-xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-sky-300 bg-clip-text text-transparent flex items-center gap-2">
                        GASTOS PÚBLICOS ANALYTICS <span class="text-xs px-2.5 py-0.5 rounded-full border border-cyan-500/30 text-cyan-400 font-mono">CEAP 2019-2026</span>
                    </h1>
                    <p class="text-xs text-slate-400 font-mono">Data Warehouse Dimensional (DuckDB & Parquet) • Machine Learning Anomaly Detection</p>
                </div>
            </div>

            <div class="flex items-center gap-3">
                <div class="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-xs font-mono" id="header-status-badge">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    <span id="badge-total-records">CARREGANDO...</span>
                </div>
                <button onclick="triggerETL()" id="btn-etl" class="flex items-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium text-xs px-4 py-2 rounded-lg shadow-lg shadow-cyan-600/20 transition">
                    <i data-lucide="refresh-cw" class="w-4 h-4" id="icon-etl"></i>
                    <span>Recarregar Base Multi-Anual</span>
                </button>
            </div>
        </div>
    </header>

    <!-- Interactive Year Filter Toolbar -->
    <div class="bg-darkBg/95 border-b border-cardBorder px-6 py-3 sticky top-[73px] z-40 backdrop-blur-md">
        <div class="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
            <div class="flex items-center gap-2 text-xs font-mono text-slate-400">
                <i data-lucide="filter" class="w-4 h-4 text-cyan-400"></i>
                <span>FILTRAR ANO DE ANÁLISE:</span>
            </div>
            
            <!-- Quick Year Filter Buttons -->
            <div class="flex flex-wrap items-center gap-1.5 font-mono text-xs" id="year-buttons-container">
                <button onclick="setYear(null)" class="btn-year active px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 transition" data-year="all">
                    🌐 Todos os Anos (2019-2026)
                </button>
                <button onclick="setYear(2026)" class="btn-year px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 transition" data-year="2026">
                    2026
                </button>
                <button onclick="setYear(2025)" class="btn-year px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 transition" data-year="2025">
                    2025
                </button>
                <button onclick="setYear(2024)" class="btn-year px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 transition" data-year="2024">
                    2024
                </button>
                <button onclick="setYear(2023)" class="btn-year px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 transition" data-year="2023">
                    2023
                </button>
                <button onclick="setYear(2022)" class="btn-year px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 transition" data-year="2022">
                    2022
                </button>
                <button onclick="setYear(2021)" class="btn-year px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 transition" data-year="2021">
                    2021
                </button>
                <button onclick="setYear(2020)" class="btn-year px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 transition" data-year="2020">
                    2020
                </button>
                <button onclick="setYear(2019)" class="btn-year px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 transition" data-year="2019">
                    2019
                </button>
            </div>
        </div>
    </div>

    <!-- Main Content Grid -->
    <main class="max-w-7xl mx-auto w-full px-6 py-8 space-y-8 flex-1">

        <!-- Active Filter Indicator Banner -->
        <div class="flex items-center justify-between text-xs font-mono text-cyan-300 bg-cyan-950/30 border border-cyan-500/20 px-4 py-2.5 rounded-xl">
            <span class="flex items-center gap-2">
                <i data-lucide="check-circle-2" class="w-4 h-4 text-cyan-400"></i>
                <span id="current-filter-label">Visualizando histórico consolidado completo (2019 a 2026)</span>
            </span>
            <span class="text-slate-400">Atualizado dinamicamente via DuckDB Engine</span>
        </div>

        <!-- Macro KPIs Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="glass p-5 rounded-2xl glow-cyan">
                <div class="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
                    <span>VOLUME TOTAL ANALISADO</span>
                    <i data-lucide="dollar-sign" class="w-4 h-4 text-cyan-400"></i>
                </div>
                <div class="text-2xl font-black text-white font-mono tracking-tight" id="kpi-total-gasto">R$ 0,00</div>
                <div class="text-xs text-emerald-400 mt-2 flex items-center gap-1 font-mono" id="kpi-sub-total">
                    <i data-lucide="calendar" class="w-3.5 h-3.5"></i> Período: 2019 – 2026
                </div>
            </div>

            <div class="glass p-5 rounded-2xl">
                <div class="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
                    <span>RECIBOS / NOTAS FISCAIS</span>
                    <i data-lucide="file-text" class="w-4 h-4 text-sky-400"></i>
                </div>
                <div class="text-2xl font-black text-white font-mono" id="kpi-total-tx">0</div>
                <div class="text-xs text-slate-400 mt-2 font-mono" id="kpi-sub-deputados">0 Parlamentares</div>
            </div>

            <div class="glass p-5 rounded-2xl">
                <div class="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
                    <span>TICKET MÉDIO POR RECIBO</span>
                    <i data-lucide="trending-up" class="w-4 h-4 text-purple-400"></i>
                </div>
                <div class="text-2xl font-black text-white font-mono" id="kpi-ticket-medio">R$ 0,00</div>
                <div class="text-xs text-purple-300 mt-2 font-mono" id="kpi-sub-fornecedores">0 Fornecedores</div>
            </div>

            <div class="glass p-5 rounded-2xl glow-red">
                <div class="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
                    <span>MAIOR DESPESA ÚNICA</span>
                    <i data-lucide="alert-triangle" class="w-4 h-4 text-rose-400"></i>
                </div>
                <div class="text-2xl font-black text-rose-400 font-mono" id="kpi-maior-despesa">R$ 0,00</div>
                <div class="text-xs text-rose-300 mt-2 font-mono">Sob Auditoria de Anomalia</div>
            </div>
        </div>

        <!-- Charts Grid 1: Evolution Timeline & Party Distribution -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="glass lg:col-span-2 p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h2 class="text-base font-bold text-slate-100 flex items-center gap-2">
                            <i data-lucide="line-chart" class="w-5 h-5 text-cyan-400"></i>
                            <span id="title-chart-timeline">Evolução Histórica de Gastos (2019 – 2026)</span>
                        </h2>
                        <p class="text-xs text-slate-400 font-mono" id="sub-chart-timeline">Distribuição anual consolidada da Cota Parlamentar</p>
                    </div>
                </div>
                <div class="h-72">
                    <canvas id="chartYears"></canvas>
                </div>
            </div>

            <div class="glass p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h2 class="text-base font-bold text-slate-100 flex items-center gap-2">
                            <i data-lucide="pie-chart" class="w-5 h-5 text-purple-400"></i>
                            Gastos por Partido
                        </h2>
                        <p class="text-xs text-slate-400 font-mono">Concentração financeira por sigla partidária</p>
                    </div>
                </div>
                <div class="h-72">
                    <canvas id="chartParties"></canvas>
                </div>
            </div>
        </div>

        <!-- Charts Grid 2: Spending by State (UF) & Top Categories -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="glass p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h2 class="text-base font-bold text-slate-100 flex items-center gap-2">
                            <i data-lucide="map-pin" class="w-5 h-5 text-emerald-400"></i>
                            Gastos por Unidade Federativa (UF)
                        </h2>
                        <p class="text-xs text-slate-400 font-mono">Concentração regional de despesas parlamentares</p>
                    </div>
                </div>
                <div class="h-72">
                    <canvas id="chartStates"></canvas>
                </div>
            </div>

            <div class="glass p-6 rounded-2xl">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h2 class="text-base font-bold text-slate-100 flex items-center gap-2">
                            <i data-lucide="layers" class="w-5 h-5 text-amber-400"></i>
                            Top Categorias de Gastos
                        </h2>
                        <p class="text-xs text-slate-400 font-mono">Divulgação, Passagens, Locação e Consultorias</p>
                    </div>
                </div>
                <div class="h-72">
                    <canvas id="chartCategories"></canvas>
                </div>
            </div>
        </div>

        <!-- ADVANCED MACHINE LEARNING & FORENSIC SUITE -->
        <div class="space-y-6">
            <!-- ML Suite Header Card -->
            <div class="glass p-6 rounded-2xl border-indigo-500/30">
                <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-4 border-b border-cardBorder">
                    <div>
                        <h2 class="text-lg font-bold text-slate-100 flex items-center gap-2">
                            <i data-lucide="brain-circuit" class="w-6 h-6 text-indigo-400"></i>
                            <span>Inteligência Artificial & Machine Learning Forense</span>
                            <span class="text-xs px-2.5 py-0.5 rounded-full border border-indigo-500/30 text-indigo-400 font-mono">Ensemble • Benford • K-Means • Forecast</span>
                        </h2>
                        <p class="text-xs text-slate-400 font-mono mt-1">
                            Comitê Não Supervisionado (<strong>Isolation Forest, LOF, Elliptic Envelope</strong>), Teste Forense de <strong>Newcomb-Benford</strong> e Clustering Comportamental de Parlamentares.
                        </p>
                    </div>
                    <!-- Quick Risk Summary Badges -->
                    <div class="flex flex-wrap items-center gap-2 font-mono text-xs" id="ml-summary-badges">
                        <span class="px-2.5 py-1 rounded-lg bg-rose-950/80 border border-rose-500/40 text-rose-300 font-bold" id="badge-critico">🚨 0 Críticos</span>
                        <span class="px-2.5 py-1 rounded-lg bg-amber-950/80 border border-amber-500/40 text-amber-300 font-bold" id="badge-alto">⚠️ 0 Altos</span>
                        <span class="px-2.5 py-1 rounded-lg bg-yellow-950/80 border border-yellow-500/40 text-yellow-300 font-bold" id="badge-medio">🟡 0 Médios</span>
                        <span class="px-2.5 py-1 rounded-lg bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 font-bold" id="badge-benford-status">BENFORD: CALCULANDO</span>
                    </div>
                </div>

                <!-- 2 ML Charts: Benford Law & Predictive Forecast -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                    <!-- Chart 1: Benford's Law Forensic Audit -->
                    <div class="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
                        <div class="flex items-center justify-between mb-3">
                            <div>
                                <h3 class="text-sm font-bold text-white flex items-center gap-2 font-mono">
                                    <i data-lucide="bar-chart-2" class="w-4 h-4 text-cyan-400"></i>
                                    Auditoria Forense: Lei de Newcomb-Benford (1º Dígito)
                                </h3>
                                <p class="text-[11px] text-slate-400 font-mono">Distribuição Observada vs Teórica P(d)=log10(1 + 1/d)</p>
                            </div>
                            <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300" id="benford-mad-badge">MAD: 0.000</span>
                        </div>
                        <div class="h-60">
                            <canvas id="chartBenford"></canvas>
                        </div>
                        <div class="mt-3 text-[10px] font-mono text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-2">
                            <span id="benford-conformity-text">Conformidade TCU: Avaliando...</span>
                            <span id="benford-chi2-text">χ² Stat: 0.00</span>
                        </div>
                    </div>

                    <!-- Chart 2: Predictive Spending Forecast -->
                    <div class="p-5 rounded-2xl bg-slate-900/60 border border-slate-800">
                        <div class="flex items-center justify-between mb-3">
                            <div>
                                <h3 class="text-sm font-bold text-white flex items-center gap-2 font-mono">
                                    <i data-lucide="trending-up" class="w-4 h-4 text-indigo-400"></i>
                                    Projeção Preditiva de Gastos (Próximos 6 Meses)
                                </h3>
                                <p class="text-[11px] text-slate-400 font-mono">Modelo Ridge com Sazonalidade & Intervalo de Confiança (95%)</p>
                            </div>
                            <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950/80 border border-indigo-500/40 text-indigo-300" id="forecast-trend-badge">TENDÊNCIA: ...</span>
                        </div>
                        <div class="h-60">
                            <canvas id="chartForecast"></canvas>
                        </div>
                        <div class="mt-3 text-[10px] font-mono text-slate-400 flex items-center justify-between border-t border-slate-800/80 pt-2">
                            <span>Histórico Recente + Projeção Futura</span>
                            <span id="forecast-slope-text">Variação Mensal: R$ 0,00</span>
                        </div>
                    </div>
                </div>

                <!-- Behavioral Clusters (K-Means) Cards -->
                <div class="mt-6">
                    <h3 class="text-xs font-bold text-slate-300 flex items-center gap-2 mb-3 font-mono">
                        <i data-lucide="network" class="w-4 h-4 text-purple-400"></i>
                        SEGMENTAÇÃO COMPORTAMENTAL DE PARLAMENTARES (K-MEANS CLUSTERING)
                    </h3>
                    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3" id="clusters-container">
                        <div class="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-slate-500 font-mono text-xs">Carregando clusters...</div>
                    </div>
                </div>

                <!-- Multi-Model Ensemble Anomaly Audit Table -->
                <div class="mt-8 pt-6 border-t border-cardBorder">
                    <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 mb-4">
                        <div>
                            <h3 class="text-sm font-bold text-rose-300 flex items-center gap-2 font-mono">
                                <i data-lucide="shield-alert" class="w-4 h-4 text-rose-500"></i>
                                DESPESAS IDENTIFICADAS PELO COMITÊ ENSEMBLE DE MACHINE LEARNING
                            </h3>
                            <p class="text-[11px] text-slate-400 font-mono">
                                Votação cruzada: <strong>Isolation Forest</strong> (raridade), <strong>LOF</strong> (densidade local), <strong>Elliptic Envelope</strong> (covariância robusta) e <strong>Z-Score (3σ)</strong>.
                            </p>
                        </div>
                        <!-- Risk Filter Pills -->
                        <div class="flex items-center gap-1.5 font-mono text-xs">
                            <span class="text-slate-400 text-[11px] mr-1">Filtrar Risco:</span>
                            <button onclick="filterAnomaliesByRisk('ALL')" class="btn-risk-filter active px-2.5 py-1 rounded-lg border border-slate-700 bg-slate-800 text-slate-200 text-xs" data-risk="ALL">
                                Todos
                            </button>
                            <button onclick="filterAnomaliesByRisk('CRÍTICO')" class="btn-risk-filter px-2.5 py-1 rounded-lg border border-rose-500/40 bg-rose-950/60 text-rose-300 text-xs" data-risk="CRÍTICO">
                                🚨 Crítico
                            </button>
                            <button onclick="filterAnomaliesByRisk('ALTO')" class="btn-risk-filter px-2.5 py-1 rounded-lg border border-amber-500/40 bg-amber-950/60 text-amber-300 text-xs" data-risk="ALTO">
                                ⚠️ Alto
                            </button>
                            <button onclick="filterAnomaliesByRisk('MÉDIO')" class="btn-risk-filter px-2.5 py-1 rounded-lg border border-yellow-500/40 bg-yellow-950/60 text-yellow-300 text-xs" data-risk="MÉDIO">
                                🟡 Médio
                            </button>
                        </div>
                    </div>

                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs font-mono">
                            <thead class="text-slate-400 border-b border-cardBorder pb-2">
                                <tr>
                                    <th class="py-2.5 px-3">NÍVEL / SCORE</th>
                                    <th class="py-2.5 px-3">DATA</th>
                                    <th class="py-2.5 px-3">PARLAMENTAR</th>
                                    <th class="py-2.5 px-3">FORNECEDOR</th>
                                    <th class="py-2.5 px-3">CATEGORIA</th>
                                    <th class="py-2.5 px-3 text-right">VALOR (R$)</th>
                                    <th class="py-2.5 px-3">MODELOS QUE VOTARAM / MOTIVOS</th>
                                </tr>
                            </thead>
                            <tbody id="anomalies-table-body" class="divide-y divide-slate-800/60">
                                <tr>
                                    <td colspan="7" class="py-6 text-center text-slate-500 font-mono">Carregando auditoria ensemble...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Top Deputies Ranking & Fornecedores -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="glass p-6 rounded-2xl">
                <h2 class="text-base font-bold text-slate-100 flex items-center gap-2 mb-4">
                    <i data-lucide="users" class="w-5 h-5 text-cyan-400"></i>
                    Ranking de Parlamentares (Top Gastos)
                </h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs font-mono">
                        <thead class="text-slate-400 border-b border-cardBorder pb-2">
                            <tr>
                                <th class="py-2 px-2">NOME</th>
                                <th class="py-2 px-2">PARTIDO/UF</th>
                                <th class="py-2 px-2 text-right">RECIBOS</th>
                                <th class="py-2 px-2 text-right">TOTAL (R$)</th>
                                <th class="py-2 px-2 text-center">DOSSIÊ</th>
                            </tr>
                        </thead>
                        <tbody id="deputies-table-body" class="divide-y divide-slate-800/60">
                            <tr><td colspan="5" class="py-4 text-center text-slate-500">Carregando ranking...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="glass p-6 rounded-2xl">
                <h2 class="text-base font-bold text-slate-100 flex items-center gap-2 mb-4">
                    <i data-lucide="building-2" class="w-5 h-5 text-indigo-400"></i>
                    Maiores Fornecedores & CNPJs Beneficiados
                </h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs font-mono">
                        <thead class="text-slate-400 border-b border-cardBorder pb-2">
                            <tr>
                                <th class="py-2 px-2">FORNECEDOR</th>
                                <th class="py-2 px-2">CNPJ / CPF</th>
                                <th class="py-2 px-2 text-right">TOTAL PAGO</th>
                            </tr>
                        </thead>
                        <tbody id="suppliers-table-body" class="divide-y divide-slate-800/60">
                            <tr><td colspan="3" class="py-4 text-center text-slate-500">Carregando fornecedores...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- DETAILED ITEM-BY-ITEM EXPENSES AUDIT: QUEM, O QUE E ONDE -->
        <div class="glass p-6 rounded-2xl border-cyan-500/20" id="section-detalhamento">
            <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-6 pb-4 border-b border-cardBorder">
                <div>
                    <h2 class="text-lg font-bold text-slate-100 flex items-center gap-2">
                        <i data-lucide="scan-search" class="w-6 h-6 text-cyan-400"></i>
                        <span>Raio-X das Despesas: Nome aos Bois</span>
                        <span class="text-xs px-2.5 py-0.5 rounded-full border border-cyan-500/30 text-cyan-400 font-mono">Quem Gastou, O Quê e Onde</span>
                    </h2>
                    <p class="text-xs text-slate-400 font-mono mt-1">
                        Auditoria item a item: identificação do parlamentar (QUEM), tipo de despesa (O QUÊ) e empresa beneficiária com CNPJ (ONDE).
                    </p>
                </div>
                <div id="matching-expenses-badge" class="px-3.5 py-1.5 rounded-full bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 font-mono text-xs font-semibold">
                    Carregando notas...
                </div>
            </div>

            <!-- Search and Filter Bar -->
            <div class="grid grid-cols-1 md:grid-cols-12 gap-3 mb-6">
                <!-- Text search -->
                <div class="md:col-span-5 relative">
                    <i data-lucide="search" class="w-4 h-4 text-slate-400 absolute left-3.5 top-3"></i>
                    <input type="text" id="expense-search-input" placeholder="Buscar parlamentar, fornecedor, CNPJ ou tipo de gasto..." 
                           class="w-full bg-slate-900/90 border border-slate-700 rounded-xl pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-400 font-mono transition"
                           onkeyup="if(event.key === 'Enter') triggerExpenseSearch()">
                </div>

                <!-- Deputy selector -->
                <div class="md:col-span-3">
                    <select id="expense-deputy-select" onchange="triggerExpenseSearch()"
                            class="w-full bg-slate-900/90 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-400 font-mono transition">
                        <option value="">👤 Todos os Parlamentares</option>
                    </select>
                </div>

                <!-- Sort selector -->
                <div class="md:col-span-2">
                    <select id="expense-sort-select" onchange="triggerExpenseSearch()"
                            class="w-full bg-slate-900/90 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-400 font-mono transition">
                        <option value="valor_desc">💰 Maior Valor (R$)</option>
                        <option value="data_desc">📅 Mais Recente</option>
                        <option value="valor_asc">📉 Menor Valor (R$)</option>
                    </select>
                </div>

                <!-- Action Buttons -->
                <div class="md:col-span-2 flex gap-2">
                    <button onclick="triggerExpenseSearch()" class="w-full flex items-center justify-center gap-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-mono font-medium py-2 transition shadow-lg shadow-cyan-600/20">
                        <i data-lucide="filter" class="w-3.5 h-3.5"></i>
                        Filtrar
                    </button>
                    <button onclick="resetExpenseFilters()" title="Limpar Filtros" class="px-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-mono transition flex items-center justify-center">
                        <i data-lucide="rotate-ccw" class="w-3.5 h-3.5"></i>
                    </button>
                </div>
            </div>

            <!-- Itemized Expenses Table -->
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="text-slate-400 border-b border-cardBorder pb-2">
                        <tr>
                            <th class="py-3 px-3">QUEM (PARLAMENTAR)</th>
                            <th class="py-3 px-3">O QUÊ (DESPESA)</th>
                            <th class="py-3 px-3">ONDE (FORNECEDOR / CNPJ)</th>
                            <th class="py-3 px-3">DATA</th>
                            <th class="py-3 px-3 text-right">VALOR (R$)</th>
                            <th class="py-3 px-3 text-center">DOSSIÊ</th>
                        </tr>
                    </thead>
                    <tbody id="detailed-expenses-body" class="divide-y divide-slate-800/60">
                        <tr><td colspan="6" class="py-6 text-center text-slate-500">Carregando despesas detalhadas...</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Pagination Bar -->
            <div class="flex items-center justify-between pt-4 mt-4 border-t border-cardBorder text-xs font-mono text-slate-400">
                <span id="pagination-info">Mostrando 0 de 0</span>
                <div class="flex items-center gap-2">
                    <button id="btn-prev-page" onclick="changeExpensePage(-1)" disabled class="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 disabled:opacity-40 hover:bg-slate-700 transition">
                        ◀ Anterior
                    </button>
                    <span id="current-page-num" class="px-2 font-bold text-slate-200">1</span>
                    <button id="btn-next-page" onclick="changeExpensePage(1)" class="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/80 disabled:opacity-40 hover:bg-slate-700 transition">
                        Próxima ▶
                    </button>
                </div>
            </div>
        </div>

    </main>

    <!-- Footer -->
    <footer class="glass mt-12 py-6 px-6 border-t border-cardBorder text-center text-xs text-slate-500 font-mono">
        Gastos Públicos Analytics — Arquitetura Kimball Star Schema em DuckDB & Parquet • FastAPI Analytical Server • Antigravity AI
    </footer>

    <!-- Deputy Dossier Modal -->
    <div id="deputy-modal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-md hidden flex items-center justify-center p-4">
        <div class="glass max-w-4xl w-full max-h-[90vh] overflow-y-auto rounded-3xl p-6 md:p-8 border border-cyan-500/30 relative">
            <!-- Close button -->
            <button onclick="closeDeputyModal()" class="absolute top-6 right-6 text-slate-400 hover:text-white bg-slate-800/80 hover:bg-slate-700 p-2 rounded-xl transition">
                <i data-lucide="x" class="w-5 h-5"></i>
            </button>

            <!-- Deputy Dossier Header -->
            <div class="flex items-center gap-4 mb-6 pb-6 border-b border-cardBorder">
                <div class="w-14 h-14 rounded-2xl bg-gradient-to-tr from-cyan-600 via-sky-500 to-indigo-600 flex items-center justify-center text-white text-xl font-black font-mono shadow-xl shadow-cyan-500/20" id="modal-deputy-avatar">
                    DP
                </div>
                <div>
                    <div class="flex items-center gap-2">
                        <h2 class="text-xl md:text-2xl font-black text-white" id="modal-deputy-name">NOME DO PARLAMENTAR</h2>
                        <span class="px-2.5 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/50 text-cyan-300 font-mono text-xs font-bold" id="modal-deputy-badge">
                            PARTIDO / UF
                        </span>
                    </div>
                    <p class="text-xs text-slate-400 font-mono mt-1">Dossiê de Transparência Parlamentar — Cota para Exercício da Atividade Parlamentar (CEAP)</p>
                </div>
            </div>

            <!-- Modal Deputy KPIs -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                <div class="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                    <span class="text-[10px] font-mono text-slate-400 block mb-1">TOTAL GASTO</span>
                    <span class="text-base md:text-lg font-bold text-cyan-400 font-mono" id="modal-total-gasto">R$ 0,00</span>
                </div>
                <div class="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                    <span class="text-[10px] font-mono text-slate-400 block mb-1">TOTAL DE NOTAS</span>
                    <span class="text-base md:text-lg font-bold text-white font-mono" id="modal-total-recibos">0</span>
                </div>
                <div class="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                    <span class="text-[10px] font-mono text-slate-400 block mb-1">TICKET MÉDIO</span>
                    <span class="text-base md:text-lg font-bold text-purple-400 font-mono" id="modal-ticket-medio">R$ 0,00</span>
                </div>
                <div class="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                    <span class="text-[10px] font-mono text-slate-400 block mb-1">MAIOR NOTA FISCAL</span>
                    <span class="text-base md:text-lg font-bold text-rose-400 font-mono" id="modal-maior-gasto">R$ 0,00</span>
                </div>
            </div>

            <!-- Modal Breakdown Grids: Onde Gastou vs O Que Comprou -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <!-- Onde Gastou (Top Fornecedores) -->
                <div class="p-5 rounded-2xl bg-slate-900/50 border border-slate-800">
                    <h3 class="text-xs font-bold text-slate-200 flex items-center gap-2 mb-3 font-mono">
                        <i data-lucide="building" class="w-4 h-4 text-emerald-400"></i>
                        ONDE GASTOU (TOP FORNECEDORES)
                    </h3>
                    <div id="modal-suppliers-list" class="space-y-2 text-xs font-mono max-h-56 overflow-y-auto pr-1">
                        <div class="text-slate-500">Carregando fornecedores...</div>
                    </div>
                </div>

                <!-- O Que Comprou (Top Categorias) -->
                <div class="p-5 rounded-2xl bg-slate-900/50 border border-slate-800">
                    <h3 class="text-xs font-bold text-slate-200 flex items-center gap-2 mb-3 font-mono">
                        <i data-lucide="tag" class="w-4 h-4 text-amber-400"></i>
                        O QUE COMPROU (TOP CATEGORIAS)
                    </h3>
                    <div id="modal-categories-list" class="space-y-2 text-xs font-mono max-h-56 overflow-y-auto pr-1">
                        <div class="text-slate-500">Carregando categorias...</div>
                    </div>
                </div>
            </div>

            <!-- Modal Extrato de Notas -->
            <div>
                <h3 class="text-xs font-bold text-slate-200 flex items-center gap-2 mb-3 font-mono">
                    <i data-lucide="receipt" class="w-4 h-4 text-cyan-400"></i>
                    EXTRATO DE NOTAS FISCAIS DO PARLAMENTAR (MAIORES VALORES)
                </h3>
                <div class="overflow-x-auto max-h-64 rounded-xl border border-slate-800">
                    <table class="w-full text-left text-xs font-mono">
                        <thead class="bg-slate-900/90 text-slate-400 sticky top-0 border-b border-slate-800">
                            <tr>
                                <th class="py-2.5 px-3">DATA</th>
                                <th class="py-2.5 px-3">CATEGORIA</th>
                                <th class="py-2.5 px-3">FORNECEDOR</th>
                                <th class="py-2.5 px-3">DOC / NOTA</th>
                                <th class="py-2.5 px-3 text-right">VALOR</th>
                            </tr>
                        </thead>
                        <tbody id="modal-receipts-body" class="divide-y divide-slate-800/60 bg-slate-950/40">
                            <tr><td colspan="5" class="py-4 text-center text-slate-500">Carregando extrato...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- Interactive Front-End Engine -->
    <script>
        lucide.createIcons();

        let chartYears = null;
        let chartParties = null;
        let chartStates = null;
        let chartCategories = null;
        let chartBenford = null;
        let chartForecast = null;
        let activeYear = null;
        let activeRiskFilter = 'ALL';

        function formatBRL(value) {
            return Number(value || 0).toLocaleString('pt-BR', { style: 'currency', currency: 'BRL' });
        }

        function formatChartAxis(v) {
            if (v >= 1e6) return 'R$ ' + (v / 1e6).toFixed(1) + 'M';
            if (v >= 1e3) return 'R$ ' + (v / 1e3).toFixed(0) + 'k';
            if (v > 0) return 'R$ ' + v.toFixed(0);
            return 'R$ 0';
        }

        function setYear(year) {
            activeYear = year;

            // Update UI buttons
            document.querySelectorAll('.btn-year').forEach(btn => {
                const bYear = btn.getAttribute('data-year');
                if ((year === null && bYear === 'all') || (String(year) === bYear)) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });

            // Update labels
            const label = document.getElementById('current-filter-label');
            const subTimeline = document.getElementById('sub-chart-timeline');
            const titleTimeline = document.getElementById('title-chart-timeline');

            if (year) {
                label.innerText = `Visualizando dados exclusivos do Ano ${year}`;
                titleTimeline.innerText = `Evolução Mensal (${year})`;
                subTimeline.innerText = `Distribuição mês a mês ao longo de ${year}`;
            } else {
                label.innerText = 'Visualizando histórico consolidado completo (2019 a 2026)';
                titleTimeline.innerText = 'Evolução Histórica de Gastos (2019 – 2026)';
                subTimeline.innerText = 'Distribuição anual consolidada da Cota Parlamentar';
            }

            // Reload all dashboard sections with the new year filter
            loadAllData(year);
        }

        function loadAllData(year = null) {
            const yearParam = year ? `?ano=${year}` : '';

            // 1. KPIs
            fetch(`/api/v1/kpis${yearParam}`)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}`);
                    return r.json();
                })
                .then(k => {
                    document.getElementById('kpi-total-gasto').innerText = formatBRL(k.total_gasto);
                    document.getElementById('kpi-total-tx').innerText = Number(k.total_transacoes || 0).toLocaleString('pt-BR');
                    document.getElementById('kpi-ticket-medio').innerText = formatBRL(k.ticket_medio);
                    document.getElementById('kpi-maior-despesa').innerText = formatBRL(k.maior_despesa);
                    document.getElementById('kpi-sub-deputados').innerText = `${k.total_deputados || 0} Parlamentares Ativos`;
                    document.getElementById('kpi-sub-fornecedores').innerText = `${k.total_fornecedores || 0} Fornecedores / CNPJs`;
                    document.getElementById('badge-total-records').innerText = `DUCKDB: ${Number(k.total_transacoes || 0).toLocaleString('pt-BR')} FATOS`;
                    document.getElementById('kpi-sub-total').innerHTML = year
                        ? `<i data-lucide="calendar" class="w-3.5 h-3.5 inline"></i> Ano Filtrado: ${year}`
                        : `<i data-lucide="calendar" class="w-3.5 h-3.5 inline"></i> Período: 2019 – 2026`;
                    lucide.createIcons();
                })
                .catch(err => console.error("Error loading KPIs:", err));

            // 2. Timeline Chart (Annual if year=null, Monthly if year is set)
            const timelineUrl = year ? `/api/v1/timeline?ano=${year}` : '/api/v1/years';
            fetch(timelineUrl)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}`);
                    return r.json();
                })
                .then(data => {
                    const ctx = document.getElementById('chartYears').getContext('2d');
                    if (chartYears) chartYears.destroy();

                    if (!data || data.length === 0) return;

                    const labels = year 
                        ? data.map(d => `Mês ${d.mes}`)
                        : data.map(d => String(d.ano));
                    const values = data.map(d => d.total_valor);

                    chartYears = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Gasto (R$)',
                                data: values,
                                borderColor: '#06b6d4',
                                backgroundColor: 'rgba(6, 182, 212, 0.12)',
                                fill: true,
                                tension: 0.35,
                                pointBackgroundColor: '#38bdf8',
                                pointRadius: 5,
                                pointHoverRadius: 7
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    callbacks: {
                                        label: ctx => ` Gasto: ${formatBRL(ctx.parsed.y)}`
                                    }
                                }
                            },
                            scales: {
                                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                                y: {
                                    grid: { color: 'rgba(255,255,255,0.05)' },
                                    ticks: {
                                        color: '#94a3b8',
                                        callback: v => formatChartAxis(v)
                                    }
                                }
                            }
                        }
                    });
                });

            // 3. Parties Chart
            fetch(`/api/v1/parties${yearParam}${year ? '&' : '?'}limit=8`)
                .then(r => r.json())
                .then(data => {
                    const ctx = document.getElementById('chartParties').getContext('2d');
                    if (chartParties) chartParties.destroy();

                    chartParties = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: data.map(d => d.partido),
                            datasets: [{
                                data: data.map(d => d.total_valor),
                                backgroundColor: ['#06b6d4', '#818cf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#f472b6', '#64748b'],
                                borderWidth: 0
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { position: 'bottom', labels: { color: '#cbd5e1', font: { size: 10 } } },
                                tooltip: {
                                    callbacks: {
                                        label: ctx => ` ${ctx.label}: ${formatBRL(ctx.parsed)}`
                                    }
                                }
                            }
                        }
                    });
                });

            // 4. States Chart
            fetch(`/api/v1/states${yearParam}${year ? '&' : '?'}limit=10`)
                .then(r => r.json())
                .then(data => {
                    const ctx = document.getElementById('chartStates').getContext('2d');
                    if (chartStates) chartStates.destroy();

                    chartStates = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.map(d => d.uf),
                            datasets: [{
                                label: 'Total (R$)',
                                data: data.map(d => d.total_valor),
                                backgroundColor: '#10b981',
                                borderRadius: 6
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    callbacks: {
                                        label: ctx => ` Total UF: ${formatBRL(ctx.parsed.y)}`
                                    }
                                }
                            },
                            scales: {
                                x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                                y: {
                                    grid: { color: 'rgba(255,255,255,0.05)' },
                                    ticks: {
                                        color: '#94a3b8',
                                        callback: v => formatChartAxis(v)
                                    }
                                }
                            }
                        }
                    });
                });

            // 5. Categories Chart
            fetch(`/api/v1/top-categories${yearParam}${year ? '&' : '?'}limit=6`)
                .then(r => r.json())
                .then(data => {
                    const ctx = document.getElementById('chartCategories').getContext('2d');
                    if (chartCategories) chartCategories.destroy();

                    chartCategories = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.map(d => d.tipo_despesa.length > 25 ? d.tipo_despesa.slice(0, 25) + '...' : d.tipo_despesa),
                            datasets: [{
                                label: 'Valor',
                                data: data.map(d => d.total_valor),
                                backgroundColor: '#f59e0b',
                                borderRadius: 6
                            }]
                        },
                        options: {
                            indexAxis: 'y',
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    callbacks: {
                                        label: ctx => ` Total Categoria: ${formatBRL(ctx.parsed.x)}`
                                    }
                                }
                            },
                            scales: {
                                x: {
                                    grid: { color: 'rgba(255,255,255,0.05)' },
                                    ticks: { color: '#94a3b8', callback: v => formatChartAxis(v) }
                                },
                                y: { grid: { display: false }, ticks: { color: '#cbd5e1', font: { size: 9 } } }
                            }
                        }
                    });
                });

            // 6. ML & AI Forensic Suite
            loadMLSuite(year);

            // 7. Deputies Ranking Table
            fetch(`/api/v1/deputies${yearParam}${year ? '&' : '?'}limit=8`)
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('deputies-table-body');
                    if (!data || data.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" class="py-4 text-center text-slate-500">Nenhum parlamentar registrado neste ano.</td></tr>';
                        return;
                    }
                    tbody.innerHTML = data.map(d => `
                        <tr class="hover:bg-slate-800/30 transition">
                            <td class="py-2 px-2 text-white font-medium hover:text-cyan-400 cursor-pointer" onclick="openDeputyModal('${d.nome}')">${d.nome}</td>
                            <td class="py-2 px-2 text-cyan-400 font-bold">${d.partido}/${d.uf}</td>
                            <td class="py-2 px-2 text-right text-slate-400">${Number(d.qtd_recibos).toLocaleString('pt-BR')}</td>
                            <td class="py-2 px-2 text-right font-bold text-slate-200">${formatBRL(d.total_gasto)}</td>
                            <td class="py-2 px-2 text-center">
                                <button onclick="openDeputyModal('${d.nome}')" class="px-2 py-0.5 rounded text-[10px] bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500 hover:text-white transition inline-flex items-center gap-1 font-mono">
                                    <i data-lucide="user" class="w-3 h-3"></i> Dossiê
                                </button>
                            </td>
                        </tr>
                    `).join('');
                    lucide.createIcons();
                });

            // 8. Top Suppliers Table
            fetch(`/api/v1/suppliers${yearParam}${year ? '&' : '?'}limit=8`)
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('suppliers-table-body');
                    if (!data || data.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="3" class="py-4 text-center text-slate-500">Nenhum fornecedor registrado neste ano.</td></tr>';
                        return;
                    }
                    tbody.innerHTML = data.map(s => `
                        <tr class="hover:bg-slate-800/30">
                            <td class="py-2 px-2 text-white font-medium max-w-xs truncate">${s.nome_fornecedor}</td>
                            <td class="py-2 px-2 text-slate-400 font-mono text-[11px]">${s.cnpj_cpf}</td>
                            <td class="py-2 px-2 text-right font-bold text-emerald-400">${formatBRL(s.total_valor)}</td>
                        </tr>
                    `).join('');
                });

            // 9. Load detailed expenses & dropdowns
            loadDeputiesDropdown();
            loadDetailedExpenses(1);
        }

        // ====================================================================
        // ADVANCED ML & FORENSIC SUITE ENGINE
        // ====================================================================
        function loadMLSuite(year = null) {
            const yearParam = year ? `?ano=${year}` : '';

            // 1. Benford's Law Forensic Audit
            fetch(`/api/v1/ml/benford${yearParam}`)
                .then(r => r.json())
                .then(data => {
                    const ctx = document.getElementById('chartBenford').getContext('2d');
                    if (chartBenford) chartBenford.destroy();

                    const digits = data.digits || [1,2,3,4,5,6,7,8,9];
                    const observed = data.observed_pct || [];
                    const expected = data.expected_pct || [];

                    document.getElementById('benford-mad-badge').innerText = `MAD: ${data.mad || '0.000'}`;
                    document.getElementById('benford-conformity-text').innerText = `Conformidade TCU: ${data.conformity_status || 'AVALIANDO'}`;
                    document.getElementById('benford-chi2-text').innerText = `χ² Stat: ${data.chi2_stat || '0.00'} (p=${data.p_value || '0.0'})`;

                    const statusBadge = document.getElementById('badge-benford-status');
                    if (data.forensic_risk === 'BAIXO' || data.forensic_risk === 'MODERADO') {
                        statusBadge.className = 'px-2.5 py-1 rounded-lg bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 font-bold';
                        statusBadge.innerText = 'BENFORD: PADRÃO NATURAL';
                    } else {
                        statusBadge.className = 'px-2.5 py-1 rounded-lg bg-rose-950/80 border border-rose-500/40 text-rose-300 font-bold';
                        statusBadge.innerText = 'BENFORD: DESVIO DETECTADO';
                    }

                    chartBenford = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: digits.map(d => `Dígito ${d}`),
                            datasets: [
                                {
                                    label: 'Frequência Real Observada (%)',
                                    data: observed,
                                    backgroundColor: '#06b6d4',
                                    borderRadius: 4
                                },
                                {
                                    label: 'Curva Teórica de Benford (%)',
                                    data: expected,
                                    backgroundColor: 'rgba(244, 63, 94, 0.65)',
                                    borderRadius: 4
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { position: 'top', labels: { color: '#cbd5e1', font: { size: 10 } } },
                                tooltip: {
                                    callbacks: {
                                        label: ctx => ` ${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(1)}%`
                                    }
                                }
                            },
                            scales: {
                                x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                                y: {
                                    grid: { color: 'rgba(255,255,255,0.05)' },
                                    ticks: { color: '#94a3b8', callback: v => v + '%' }
                                }
                            }
                        }
                    });
                });

            // 2. Predictive Spending Forecast
            fetch('/api/v1/ml/forecast?months=6')
                .then(r => r.json())
                .then(data => {
                    const ctx = document.getElementById('chartForecast').getContext('2d');
                    if (chartForecast) chartForecast.destroy();

                    const hist = (data.historical || []).slice(-12);
                    const fore = data.forecast || [];

                    document.getElementById('forecast-trend-badge').innerText = `TENDÊNCIA: ${data.trend || 'ESTÁVEL'}`;
                    document.getElementById('forecast-slope-text').innerText = `Variação Mensal: ${formatBRL(data.monthly_slope_brl)}`;

                    const labels = [...hist.map(h => h.label), ...fore.map(f => f.label)];
                    const histData = [...hist.map(h => h.valor), ...Array(fore.length).fill(null)];

                    const lastHistVal = hist.length > 0 ? hist[hist.length - 1].valor : null;
                    const foreData = [...Array(Math.max(0, hist.length - 1)).fill(null), lastHistVal, ...fore.map(f => f.predicted_valor)];
                    const upperBounds = [...Array(Math.max(0, hist.length - 1)).fill(null), lastHistVal, ...fore.map(f => f.upper_bound)];
                    const lowerBounds = [...Array(Math.max(0, hist.length - 1)).fill(null), lastHistVal, ...fore.map(f => f.lower_bound)];

                    chartForecast = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [
                                {
                                    label: 'Histórico Real',
                                    data: histData,
                                    borderColor: '#38bdf8',
                                    backgroundColor: '#38bdf8',
                                    tension: 0.3,
                                    pointRadius: 4
                                },
                                {
                                    label: 'Previsão ML (Ridge)',
                                    data: foreData,
                                    borderColor: '#818cf8',
                                    borderDash: [5, 5],
                                    backgroundColor: 'rgba(129, 140, 248, 0.15)',
                                    tension: 0.3,
                                    pointRadius: 5,
                                    pointBackgroundColor: '#818cf8'
                                },
                                {
                                    label: 'Limite Superior (95%)',
                                    data: upperBounds,
                                    borderColor: 'rgba(129, 140, 248, 0.3)',
                                    borderDash: [2, 2],
                                    pointRadius: 0,
                                    fill: false
                                },
                                {
                                    label: 'Limite Inferior (95%)',
                                    data: lowerBounds,
                                    borderColor: 'rgba(129, 140, 248, 0.3)',
                                    borderDash: [2, 2],
                                    pointRadius: 0,
                                    fill: '-1',
                                    backgroundColor: 'rgba(129, 140, 248, 0.08)'
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { position: 'top', labels: { color: '#cbd5e1', font: { size: 9 }, filter: item => !item.text.includes('Limite') } },
                                tooltip: {
                                    callbacks: {
                                        label: ctx => ` ${ctx.dataset.label}: ${formatBRL(ctx.parsed.y)}`
                                    }
                                }
                            },
                            scales: {
                                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', maxRotation: 45 } },
                                y: {
                                    grid: { color: 'rgba(255,255,255,0.05)' },
                                    ticks: { color: '#94a3b8', callback: v => formatChartAxis(v) }
                                }
                            }
                        }
                    });
                });

            // 3. Behavioral Clusters (K-Means)
            fetch(`/api/v1/ml/clusters${yearParam}`)
                .then(r => r.json())
                .then(data => {
                    const clusters = data.clusters || [];
                    const container = document.getElementById('clusters-container');
                    if (clusters.length === 0) {
                        container.innerHTML = '<div class="text-slate-500 font-mono text-xs col-span-4">Dados insuficientes para agrupamento K-Means.</div>';
                        return;
                    }
                    container.innerHTML = clusters.map(c => `
                        <div class="p-4 rounded-xl bg-slate-900/70 border border-slate-700/60 hover:border-indigo-500/50 transition">
                            <div class="flex items-center justify-between mb-1.5">
                                <span class="font-bold text-white text-xs truncate" title="${c.name}">${c.name}</span>
                                <span class="px-1.5 py-0.5 rounded text-[10px] bg-indigo-950 border border-indigo-500/40 text-indigo-300 font-mono font-bold">${c.total_deputies} Deps</span>
                            </div>
                            <p class="text-[10px] text-slate-400 font-mono mb-2 line-clamp-2">${c.description}</p>
                            <div class="flex items-center justify-between text-[11px] font-mono text-cyan-300 font-bold border-t border-slate-800 pt-2">
                                <span>Média: ${formatBRL(c.avg_spending)}</span>
                                <span class="text-slate-400 font-normal">Ticket: ${formatBRL(c.avg_ticket)}</span>
                            </div>
                        </div>
                    `).join('');
                });

            // 4. Load Multi-Model Ensemble Anomalies Table
            loadEnsembleAnomalies(year, activeRiskFilter);
        }

        function loadEnsembleAnomalies(year = null, risk = 'ALL') {
            const yearParam = year ? `&ano=${year}` : '';
            const riskParam = (risk && risk !== 'ALL') ? `&risk_level=${encodeURIComponent(risk)}` : '';
            const url = `/api/v1/anomalies?limit=250${yearParam}${riskParam}`;

            fetch(url)
                .then(r => r.json())
                .then(res => {
                    const summary = res.summary || {};
                    const riskCounts = summary.risk_counts || {};
                    const anomalies = res.anomalies || [];

                    document.getElementById('badge-critico').innerText = `🚨 ${riskCounts['CRÍTICO'] || 0} Críticos`;
                    document.getElementById('badge-alto').innerText = `⚠️ ${riskCounts['ALTO'] || 0} Altos`;
                    document.getElementById('badge-medio').innerText = `🟡 ${riskCounts['MÉDIO'] || 0} Médios`;

                    const tbody = document.getElementById('anomalies-table-body');
                    if (anomalies.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" class="py-6 text-center text-emerald-400 font-mono">Nenhuma anomalia identificada com o filtro selecionado.</td></tr>';
                        return;
                    }

                    tbody.innerHTML = anomalies.slice(0, 10).map(a => {
                        const riskBadge = a.risk_level === 'CRÍTICO' 
                            ? `<span class="px-2 py-0.5 rounded text-[10px] bg-rose-950 border border-rose-500/80 text-rose-300 font-bold">🚨 CRÍTICO (${a.risk_score} pts)</span>`
                            : a.risk_level === 'ALTO'
                            ? `<span class="px-2 py-0.5 rounded text-[10px] bg-amber-950 border border-amber-500/70 text-amber-300 font-bold">⚠️ ALTO (${a.risk_score} pts)</span>`
                            : `<span class="px-2 py-0.5 rounded text-[10px] bg-yellow-950 border border-yellow-500/50 text-yellow-300 font-bold">🟡 MÉDIO (${a.risk_score} pts)</span>`;

                        const modelsBadges = (a.models_flagged || []).map(m => 
                            `<span class="px-1.5 py-0.2 rounded text-[9px] bg-slate-800 text-slate-300 border border-slate-700">${m}</span>`
                        ).join(' ');

                        return `
                            <tr class="hover:bg-slate-800/40 transition">
                                <td class="py-2.5 px-3 whitespace-nowrap">${riskBadge}</td>
                                <td class="py-2.5 px-3 text-slate-300 whitespace-nowrap">${a.data_documento || ''}</td>
                                <td class="py-2.5 px-3 font-semibold text-white">
                                    <span class="hover:text-cyan-400 cursor-pointer" onclick="openDeputyModal('${a.nome_deputado}')">${a.nome_deputado || 'DEPUTADO'}</span>
                                    <span class="text-[10px] text-cyan-400 font-bold ml-1">(${a.partido || ''}/${a.uf || ''})</span>
                                </td>
                                <td class="py-2.5 px-3 text-slate-300 max-w-xs truncate" title="${a.nome_fornecedor}">${a.nome_fornecedor || ''}</td>
                                <td class="py-2.5 px-3 text-purple-300 max-w-xs truncate" title="${a.tipo_despesa}">${a.tipo_despesa || ''}</td>
                                <td class="py-2.5 px-3 text-right font-bold font-mono text-rose-400">${formatBRL(a.valor_liquido)}</td>
                                <td class="py-2.5 px-3 text-xs">
                                    <div class="flex flex-wrap gap-1">${modelsBadges}</div>
                                </td>
                            </tr>
                        `;
                    }).join('');
                });
        }

        function filterAnomaliesByRisk(level) {
            activeRiskFilter = level;
            document.querySelectorAll('.btn-risk-filter').forEach(btn => {
                if (btn.getAttribute('data-risk') === level) {
                    btn.classList.add('active', 'border-indigo-400', 'bg-indigo-900/60');
                } else {
                    btn.classList.remove('active', 'border-indigo-400', 'bg-indigo-900/60');
                }
            });
            loadEnsembleAnomalies(activeYear, level);
        }

        // ====================================================================
        // ITEM-BY-ITEM EXPENSES ENGINE & DEPUTY DOSSIER MODAL
        // ====================================================================
        let currentExpensePage = 1;
        const expensePageSize = 25;
        let totalExpenseRecords = 0;
        let deputiesLoaded = false;

        function loadDeputiesDropdown() {
            if (deputiesLoaded) return;
            fetch('/api/v1/deputies/list')
                .then(r => r.json())
                .then(list => {
                    const sel = document.getElementById('expense-deputy-select');
                    sel.innerHTML = '<option value="">👤 Todos os Parlamentares</option>';
                    list.forEach(d => {
                        const opt = document.createElement('option');
                        opt.value = d.nome;
                        opt.innerText = `${d.nome} (${d.partido}/${d.uf})`;
                        sel.appendChild(opt);
                    });
                    deputiesLoaded = true;
                })
                .catch(err => console.error('Error loading deputies list:', err));
        }

        function loadDetailedExpenses(page = 1) {
            currentExpensePage = page;
            const searchVal = document.getElementById('expense-search-input').value.trim();
            const depVal = document.getElementById('expense-deputy-select').value;
            const sortVal = document.getElementById('expense-sort-select').value;
            const offset = (page - 1) * expensePageSize;

            let url = `/api/v1/expenses?limit=${expensePageSize}&offset=${offset}&sort_by=${sortVal}`;
            if (activeYear) url += `&ano=${activeYear}`;
            if (searchVal) url += `&search=${encodeURIComponent(searchVal)}`;
            if (depVal) url += `&deputado=${encodeURIComponent(depVal)}`;

            const tbody = document.getElementById('detailed-expenses-body');
            tbody.innerHTML = '<tr><td colspan="6" class="py-6 text-center text-slate-400 font-mono">Buscando notas fiscais no DuckDB...</td></tr>';

            fetch(url)
                .then(r => r.json())
                .then(res => {
                    const total = res.total_matching || 0;
                    totalExpenseRecords = total;
                    const expenses = res.expenses || [];

                    document.getElementById('matching-expenses-badge').innerText = `📄 ${total.toLocaleString('pt-BR')} NOTAS ENCONTRADAS`;

                    const totalPages = Math.ceil(total / expensePageSize) || 1;
                    const startIdx = total === 0 ? 0 : offset + 1;
                    const endIdx = Math.min(offset + expenses.length, total);
                    document.getElementById('pagination-info').innerText = `Mostrando ${startIdx} a ${endIdx} de ${total.toLocaleString('pt-BR')} despesas auditadas`;
                    document.getElementById('current-page-num').innerText = `${page} / ${totalPages}`;

                    document.getElementById('btn-prev-page').disabled = (page <= 1);
                    document.getElementById('btn-next-page').disabled = (page >= totalPages);

                    if (expenses.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" class="py-8 text-center text-slate-500 font-mono">Nenhuma despesa localizada com os filtros selecionados.</td></tr>';
                        return;
                    }

                    tbody.innerHTML = expenses.map(e => `
                        <tr class="hover:bg-slate-800/40 transition">
                            <td class="py-3 px-3">
                                <div class="font-bold text-white hover:text-cyan-400 cursor-pointer" onclick="openDeputyModal('${e.deputado_nome}')">${e.deputado_nome}</div>
                                <div class="flex items-center gap-1.5 mt-0.5">
                                    <span class="px-1.5 py-0.2 rounded text-[10px] bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 font-mono font-bold">${e.deputado_partido}/${e.deputado_uf}</span>
                                    <button onclick="filterByDeputyFromRow('${e.deputado_nome}')" class="text-[10px] text-slate-400 hover:text-cyan-300 underline font-mono">filtrar este</button>
                                </div>
                            </td>
                            <td class="py-3 px-3 max-w-xs">
                                <div class="text-slate-200 font-medium truncate" title="${e.categoria}">${e.categoria}</div>
                                <div class="text-[10px] text-amber-400 font-mono mt-0.5 flex items-center gap-1">
                                    <span>${e.macro_categoria}</span>
                                    ${e.num_documento ? `<span class="text-slate-500">• NF ${e.num_documento}</span>` : ''}
                                </div>
                            </td>
                            <td class="py-3 px-3 max-w-xs">
                                <div class="text-slate-200 font-medium truncate" title="${e.fornecedor_nome}">${e.fornecedor_nome}</div>
                                <div class="text-[10px] text-slate-400 font-mono">${e.fornecedor_cnpj || 'CNPJ NÃO INFORMADO'}</div>
                            </td>
                            <td class="py-3 px-3 whitespace-nowrap text-slate-300 font-mono">
                                <div>${e.data_documento || ''}</div>
                                ${e.is_fim_de_semana ? '<span class="text-[9px] px-1 rounded bg-rose-950 border border-rose-500/50 text-rose-300">FIM DE SEMANA</span>' : ''}
                            </td>
                            <td class="py-3 px-3 text-right font-black font-mono ${e.valor_liquido > 10000 ? 'text-rose-400' : e.valor_liquido > 3000 ? 'text-amber-400' : 'text-emerald-400'}">
                                ${formatBRL(e.valor_liquido)}
                            </td>
                            <td class="py-3 px-3 text-center">
                                <button onclick="openDeputyModal('${e.deputado_nome}')" class="px-2.5 py-1 rounded-lg text-[10px] bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 hover:bg-cyan-500 hover:text-white transition inline-flex items-center gap-1 font-mono">
                                    <i data-lucide="user" class="w-3 h-3"></i> Dossiê
                                </button>
                            </td>
                        </tr>
                    `).join('');
                    lucide.createIcons();
                })
                .catch(err => {
                    tbody.innerHTML = `<tr><td colspan="6" class="py-6 text-center text-rose-400 font-mono">Erro ao carregar despesas: ${err}</td></tr>`;
                });
        }

        function triggerExpenseSearch() {
            loadDetailedExpenses(1);
        }

        function resetExpenseFilters() {
            document.getElementById('expense-search-input').value = '';
            document.getElementById('expense-deputy-select').value = '';
            document.getElementById('expense-sort-select').value = 'valor_desc';
            loadDetailedExpenses(1);
        }

        function changeExpensePage(delta) {
            const newPage = currentExpensePage + delta;
            if (newPage >= 1) {
                loadDetailedExpenses(newPage);
            }
        }

        function filterByDeputyFromRow(depName) {
            const sel = document.getElementById('expense-deputy-select');
            sel.value = depName;
            triggerExpenseSearch();
            document.getElementById('section-detalhamento').scrollIntoView({ behavior: 'smooth' });
        }

        function openDeputyModal(deputyName) {
            const modal = document.getElementById('deputy-modal');
            const cleanName = deputyName.trim();
            const yearParam = activeYear ? `?ano=${activeYear}` : '';

            // Set temporary loading state
            document.getElementById('modal-deputy-name').innerText = cleanName;
            document.getElementById('modal-deputy-badge').innerText = 'CARREGANDO...';
            document.getElementById('modal-total-gasto').innerText = '...';
            document.getElementById('modal-total-recibos').innerText = '...';
            document.getElementById('modal-ticket-medio').innerText = '...';
            document.getElementById('modal-maior-gasto').innerText = '...';
            document.getElementById('modal-suppliers-list').innerHTML = '<div class="text-slate-500 py-3">Carregando fornecedores...</div>';
            document.getElementById('modal-categories-list').innerHTML = '<div class="text-slate-500 py-3">Carregando categorias...</div>';
            document.getElementById('modal-receipts-body').innerHTML = '<tr><td colspan="5" class="py-4 text-center text-slate-500">Carregando extrato...</td></tr>';

            const initials = cleanName.split(' ').filter(Boolean).slice(0, 2).map(n => n[0]).join('').toUpperCase() || 'DP';
            document.getElementById('modal-deputy-avatar').innerText = initials;

            modal.classList.remove('hidden');

            fetch(`/api/v1/deputies/${encodeURIComponent(cleanName)}/profile${yearParam}`)
                .then(r => {
                    if (!r.ok) throw new Error('Não encontrado');
                    return r.json();
                })
                .then(profile => {
                    const dep = profile.deputado || {};
                    document.getElementById('modal-deputy-badge').innerText = `${dep.partido || 'OUTROS'} / ${dep.uf || 'BR'}`;
                    document.getElementById('modal-total-gasto').innerText = formatBRL(dep.total_gasto);
                    document.getElementById('modal-total-recibos').innerText = Number(dep.total_recibos).toLocaleString('pt-BR');
                    document.getElementById('modal-ticket-medio').innerText = formatBRL(dep.ticket_medio);
                    document.getElementById('modal-maior-gasto').innerText = formatBRL(dep.maior_gasto);

                    // Render Top Suppliers (Onde Gastou)
                    const sups = profile.top_fornecedores || [];
                    const maxSup = sups.length > 0 ? sups[0].total_gasto : 1;
                    const supsList = document.getElementById('modal-suppliers-list');
                    if (sups.length === 0) {
                        supsList.innerHTML = '<div class="text-slate-500">Nenhum fornecedor registrado.</div>';
                    } else {
                        supsList.innerHTML = sups.map(s => {
                            const pct = Math.min(100, Math.round((s.total_gasto / maxSup) * 100));
                            return `
                                <div class="p-2 rounded-lg bg-slate-800/40 border border-slate-700/50">
                                    <div class="flex items-center justify-between gap-2 mb-1">
                                        <span class="font-bold text-white truncate max-w-xs" title="${s.nome_fornecedor}">${s.nome_fornecedor}</span>
                                        <span class="font-bold text-emerald-400 whitespace-nowrap">${formatBRL(s.total_gasto)}</span>
                                    </div>
                                    <div class="flex items-center justify-between text-[10px] text-slate-400 mb-1">
                                        <span>CNPJ: ${s.cnpj_cpf || 'N/A'}</span>
                                        <span>${s.qtd_recibos} recibos</span>
                                    </div>
                                    <div class="w-full bg-slate-700/50 h-1 rounded-full overflow-hidden">
                                        <div class="bg-emerald-400 h-full rounded-full" style="width: ${pct}%"></div>
                                    </div>
                                </div>
                            `;
                        }).join('');
                    }

                    // Render Top Categories (O Que Comprou)
                    const cats = profile.top_categorias || [];
                    const maxCat = cats.length > 0 ? cats[0].total_gasto : 1;
                    const catsList = document.getElementById('modal-categories-list');
                    if (cats.length === 0) {
                        catsList.innerHTML = '<div class="text-slate-500">Nenhuma categoria registrada.</div>';
                    } else {
                        catsList.innerHTML = cats.map(c => {
                            const pct = Math.min(100, Math.round((c.total_gasto / maxCat) * 100));
                            return `
                                <div class="p-2 rounded-lg bg-slate-800/40 border border-slate-700/50">
                                    <div class="flex items-center justify-between gap-2 mb-1">
                                        <span class="font-bold text-white truncate max-w-xs" title="${c.tipo_despesa}">${c.tipo_despesa}</span>
                                        <span class="font-bold text-amber-400 whitespace-nowrap">${formatBRL(c.total_gasto)}</span>
                                    </div>
                                    <div class="flex items-center justify-between text-[10px] text-slate-400 mb-1">
                                        <span class="text-cyan-400">${c.macro_categoria}</span>
                                        <span>${c.qtd_recibos} recibos</span>
                                    </div>
                                    <div class="w-full bg-slate-700/50 h-1 rounded-full overflow-hidden">
                                        <div class="bg-amber-400 h-full rounded-full" style="width: ${pct}%"></div>
                                    </div>
                                </div>
                            `;
                        }).join('');
                    }

                    // Render Recent/Top Receipts
                    const recs = profile.despesas_recentes || [];
                    const recBody = document.getElementById('modal-receipts-body');
                    if (recs.length === 0) {
                        recBody.innerHTML = '<tr><td colspan="5" class="py-4 text-center text-slate-500">Nenhuma nota fiscal encontrada.</td></tr>';
                    } else {
                        recBody.innerHTML = recs.map(r => `
                            <tr class="hover:bg-slate-800/40 transition">
                                <td class="py-2 px-3 text-slate-300 whitespace-nowrap">${r.data_documento || ''}</td>
                                <td class="py-2 px-3 text-white max-w-xs truncate" title="${r.categoria}">${r.categoria}</td>
                                <td class="py-2 px-3 text-slate-300 max-w-xs truncate" title="${r.nome_fornecedor}">${r.nome_fornecedor}</td>
                                <td class="py-2 px-3 text-slate-400 font-mono text-[11px]">${r.num_documento || 'S/N'}</td>
                                <td class="py-2 px-3 text-right font-bold text-cyan-400">${formatBRL(r.valor_liquido)}</td>
                            </tr>
                        `).join('');
                    }
                    lucide.createIcons();
                })
                .catch(err => {
                    document.getElementById('modal-deputy-badge').innerText = 'ERRO';
                    document.getElementById('modal-receipts-body').innerHTML = `<tr><td colspan="5" class="py-4 text-center text-rose-400 font-mono">Falha ao obter dossiê: ${err}</td></tr>`;
                });
        }

        function closeDeputyModal() {
            document.getElementById('deputy-modal').classList.add('hidden');
        }

        // Close modal when clicking on overlay outside
        document.getElementById('deputy-modal').addEventListener('click', function(e) {
            if (e.target === this) closeDeputyModal();
        });

        // Close on Escape key
        window.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeDeputyModal();
        });

        function triggerETL() {
            const btn = document.getElementById('btn-etl');
            const icon = document.getElementById('icon-etl');
            btn.disabled = true;
            icon.classList.add('animate-spin');
            btn.querySelector('span').innerText = 'Executando Carga...';

            fetch('/api/v1/etl/run?deputados_limit=25', { method: 'POST' })
                .then(r => r.json())
                .then(res => {
                    alert(`[✓] Carga Multi-Anual Concluída com Sucesso! ${res.total_receipts_loaded} novos fatos inseridos.`);
                    window.location.reload();
                })
                .catch(err => {
                    alert('Erro ao atualizar: ' + err);
                    btn.disabled = false;
                    icon.classList.remove('animate-spin');
                });
        }

        // Initialize dashboard on page load
        loadAllData(null);
    </script>
</body>
</html>"""
    return html
