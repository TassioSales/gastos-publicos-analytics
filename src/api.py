"""FastAPI Analytics REST API & Modern Enterprise Dashboard for Brazilian Parliamentary Expenses (CEAP).

Exposes high-performance analytical endpoints and serves a responsive, Bloomberg/Stark-style Web UI.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from src.pipeline import CeapPipeline
from src.warehouse import SpendingWarehouse
from src.anomaly_detector import SpendingAnomalyDetector

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gastos Públicos Analytics API & Dashboard",
    description="Enterprise Analytics Engine for Brazilian Parliamentary Spending (CEAP) & Machine Learning Anomaly Detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

warehouse = SpendingWarehouse()
pipeline = CeapPipeline()
detector = SpendingAnomalyDetector()


@app.get("/api/v1/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "service": "Gastos Públicos Analytics Engine v2.0", "engine": "DuckDB + FastAPI"}


@app.get("/api/v1/kpis")
def get_kpis() -> Dict[str, Any]:
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


@app.get("/api/v1/years")
def get_years_summary() -> List[Dict[str, Any]]:
    df = warehouse.get_spending_by_year()
    return df.to_dict(orient="records") if not df.empty else []


@app.get("/api/v1/parties")
def get_parties_summary(limit: int = Query(default=15, ge=1, le=50)) -> List[Dict[str, Any]]:
    df = warehouse.get_spending_by_party(limit=limit)
    return df.to_dict(orient="records") if not df.empty else []


@app.get("/api/v1/states")
def get_states_summary(limit: int = Query(default=27, ge=1, le=27)) -> List[Dict[str, Any]]:
    df = warehouse.get_spending_by_uf(limit=limit)
    return df.to_dict(orient="records") if not df.empty else []


@app.get("/api/v1/top-categories")
def get_top_categories(limit: int = Query(default=10, ge=1, le=50)) -> List[Dict[str, Any]]:
    df = warehouse.get_top_spending_categories(limit=limit)
    return df.to_dict(orient="records") if not df.empty else []


@app.get("/api/v1/suppliers")
def get_top_suppliers(limit: int = Query(default=15, ge=1, le=50)) -> List[Dict[str, Any]]:
    df = warehouse.get_top_suppliers(limit=limit)
    return df.to_dict(orient="records") if not df.empty else []


@app.get("/api/v1/deputies")
def get_deputies_ranking(limit: int = Query(default=25, ge=1, le=100)) -> List[Dict[str, Any]]:
    df = warehouse.get_deputy_ranking(limit=limit)
    return df.to_dict(orient="records") if not df.empty else []


@app.get("/api/v1/timeline")
def get_timeline() -> List[Dict[str, Any]]:
    df = warehouse.get_monthly_timeline()
    return df.to_dict(orient="records") if not df.empty else []


@app.get("/api/v1/anomalies")
def get_anomalies(limit: int = Query(default=200, ge=10, le=1000)) -> Dict[str, Any]:
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
        ORDER BY f.valor_liquido DESC
        LIMIT {limit}
    """
    if warehouse.use_duckdb:
        df = warehouse.conn.execute(query).df()
    else:
        df = pd.read_sql_query(query, warehouse.conn)

    if df.empty:
        return {"summary": {"total_anomalies": 0}, "anomalies": []}

    analyzed = detector.analyze_dataframe(df)
    summary = detector.summarize_anomalies(analyzed)
    outliers = analyzed[analyzed["is_anomaly"] == True]

    return {
        "summary": summary,
        "anomalies": outliers.to_dict(orient="records"),
    }


@app.post("/api/v1/etl/run")
def trigger_etl(deputados_limit: int = Query(default=25, ge=1, le=50)) -> Dict[str, Any]:
    deputados = pipeline.fetch_deputados(limit=deputados_limit)
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    raw = pipeline.fetch_multiyear_expenses(deputados, years=years, limit_per_deputy=40)
    cleaned = pipeline.clean_expenses_dataframe(raw)
    metrics = warehouse.load_dimensions_and_facts(cleaned, deputados_list=deputados)

    return {
        "status": "success",
        "years": years,
        "deputados_count": len(deputados),
        "total_receipts_loaded": len(cleaned),
        "warehouse_metrics": metrics,
    }


# ============================================================================
# MODERN ENTERPRISE DASHBOARD UI (HTML5 + Tailwind CSS + Chart.js + Lucide)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    kpis = warehouse.get_summary_kpis()
    total_gasto = kpis.get("total_gasto", 0.0)
    total_tx = int(kpis.get("total_transacoes", 0))
    ticket_medio = kpis.get("ticket_medio", 0.0)
    maior_despesa = kpis.get("maior_despesa", 0.0)
    total_deps = int(kpis.get("total_deputados", 0))
    total_forns = int(kpis.get("total_fornecedores", 0))

    html = f"""<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gastos Públicos Analytics | Câmara dos Deputados (CEAP)</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    colors: {{
                        darkBg: '#080c14',
                        darkCard: '#0f172a',
                        cardBorder: '#1e293b',
                        accentBlue: '#38bdf8',
                        accentCyan: '#06b6d4',
                        accentPurple: '#818cf8',
                        alertRed: '#f43f5e',
                        statusGreen: '#10b981'
                    }}
                }}
            }}
        }}
    </script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {{
            background-color: #080c14;
            color: #f1f5f9;
            font-family: 'Inter', sans-serif;
            background-image: radial-gradient(rgba(56, 189, 248, 0.03) 1px, transparent 1px);
            background-size: 28px 28px;
        }}
        .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
        .glass {{
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.07);
        }}
        .glass:hover {{
            border-color: rgba(56, 189, 248, 0.25);
        }}
        .glow-cyan {{ box-shadow: 0 0 25px rgba(6, 182, 212, 0.15); }}
        .glow-red {{ box-shadow: 0 0 25px rgba(244, 63, 94, 0.15); }}
    </style>
</head>
<body class="min-h-screen flex flex-col justify-between selection:bg-accentCyan selection:text-black">

    <!-- Top Navigation -->
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
                    <p class="text-xs text-slate-400 font-mono">Data Warehouse Dimensional (DuckDB) & Detecção de Anomalias (Isolation Forest)</p>
                </div>
            </div>

            <div class="flex items-center gap-3">
                <div class="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-xs font-mono">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    <span>DUCKDB DW: {total_tx:,} FATOS</span>
                </div>
                <button onclick="triggerETL()" id="btn-etl" class="flex items-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium text-xs px-4 py-2 rounded-lg shadow-lg shadow-cyan-600/20 transition">
                    <i data-lucide="refresh-cw" class="w-4 h-4" id="icon-etl"></i>
                    <span>Atualizar DW Multi-Anual</span>
                </button>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto w-full px-6 py-8 space-y-8 flex-1">

        <!-- Macro KPIs -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div class="glass p-5 rounded-2xl glow-cyan">
                <div class="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
                    <span>VOLUME TOTAL ANALISADO</span>
                    <i data-lucide="dollar-sign" class="w-4 h-4 text-cyan-400"></i>
                </div>
                <div class="text-2xl font-black text-white font-mono tracking-tight">R$ {total_gasto:,.2f}</div>
                <div class="text-xs text-emerald-400 mt-2 flex items-center gap-1 font-mono">
                    <i data-lucide="calendar" class="w-3.5 h-3.5"></i> Período: 2019 a 2026 (8 Anos)
                </div>
            </div>

            <div class="glass p-5 rounded-2xl">
                <div class="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
                    <span>RECIBOS / NOTAS FISCAIS</span>
                    <i data-lucide="file-text" class="w-4 h-4 text-sky-400"></i>
                </div>
                <div class="text-2xl font-black text-white font-mono">{total_tx:,}</div>
                <div class="text-xs text-slate-400 mt-2 font-mono">{total_deps} Parlamentares Monitorados</div>
            </div>

            <div class="glass p-5 rounded-2xl">
                <div class="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
                    <span>TICKET MÉDIO POR RECIBO</span>
                    <i data-lucide="trending-up" class="w-4 h-4 text-purple-400"></i>
                </div>
                <div class="text-2xl font-black text-white font-mono">R$ {ticket_medio:,.2f}</div>
                <div class="text-xs text-purple-300 mt-2 font-mono">{total_forns} Fornecedores / CNPJs</div>
            </div>

            <div class="glass p-5 rounded-2xl glow-red">
                <div class="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
                    <span>MAIOR DESPESA DETECTADA</span>
                    <i data-lucide="alert-triangle" class="w-4 h-4 text-rose-400"></i>
                </div>
                <div class="text-2xl font-black text-rose-400 font-mono">R$ {maior_despesa:,.2f}</div>
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
                            Evolução Histórica de Gastos (2019 – 2026)
                        </h2>
                        <p class="text-xs text-slate-400 font-mono">Distribuição anual consolidada da Cota Parlamentar</p>
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
                        <p class="text-xs text-slate-400 font-mono">Volume financeiro por sigla partidária</p>
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

        <!-- Machine Learning Anomaly Detection Hub -->
        <div class="glass p-6 rounded-2xl border-rose-500/30">
            <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-6 pb-4 border-b border-cardBorder">
                <div>
                    <h2 class="text-lg font-bold text-rose-300 flex items-center gap-2">
                        <i data-lucide="shield-alert" class="w-6 h-6 text-rose-500"></i>
                        Auditoria de Conformidade & Anomalias de Machine Learning
                    </h2>
                    <p class="text-xs text-slate-400 font-mono mt-1">
                        Algoritmos aplicados: <strong>Isolation Forest Multivariado</strong> e <strong>Z-Score Estatístico (3σ)</strong> para identificação de outliers, notas duplicadas e despesas em fins de semana.
                    </p>
                </div>
                <div id="anomaly-badge" class="px-3.5 py-1.5 rounded-full bg-rose-950/50 border border-rose-500/40 text-rose-300 font-mono text-xs font-semibold">
                    Calculando anomalias...
                </div>
            </div>

            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="text-slate-400 border-b border-cardBorder pb-2">
                        <tr>
                            <th class="py-2.5 px-3">DATA</th>
                            <th class="py-2.5 px-3">PARLAMENTAR</th>
                            <th class="py-2.5 px-3">UF/PARTIDO</th>
                            <th class="py-2.5 px-3">FORNECEDOR</th>
                            <th class="py-2.5 px-3">CATEGORIA</th>
                            <th class="py-2.5 px-3 text-right">VALOR (R$)</th>
                            <th class="py-2.5 px-3 text-center">MOTIVO DO ALERTA</th>
                        </tr>
                    </thead>
                    <tbody id="anomalies-table-body" class="divide-y divide-slate-800/60">
                        <tr>
                            <td colspan="7" class="py-6 text-center text-slate-500 font-mono">Carregando auditoria de anomalias...</td>
                        </tr>
                    </tbody>
                </table>
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
                            </tr>
                        </thead>
                        <tbody id="deputies-table-body" class="divide-y divide-slate-800/60">
                            <tr><td colspan="4" class="py-4 text-center text-slate-500">Carregando ranking...</td></tr>
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

    </main>

    <!-- Footer -->
    <footer class="glass mt-12 py-6 px-6 border-t border-cardBorder text-center text-xs text-slate-500 font-mono">
        Gastos Públicos Analytics — Arquitetura Kimball Star Schema em DuckDB & Parquet • FastAPI Analytical Server • Antigravity AI
    </footer>

    <!-- Front-End Interactive Script -->
    <script>
        lucide.createIcons();

        // 1. Fetch Timeline Evolution
        fetch('/api/v1/years')
            .then(r => r.json())
            .then(data => {{
                const ctx = document.getElementById('chartYears').getContext('2d');
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: data.map(d => String(d.ano)),
                        datasets: [{{
                            label: 'Gasto Total (R$)',
                            data: data.map(d => d.total_valor),
                            borderColor: '#38bdf8',
                            backgroundColor: 'rgba(56, 189, 248, 0.1)',
                            fill: true,
                            tension: 0.35,
                            pointBackgroundColor: '#0284c7',
                            pointRadius: 5
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ display: false }} }},
                        scales: {{
                            x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                            y: {{
                                grid: {{ color: 'rgba(255,255,255,0.05)' }},
                                ticks: {{
                                    color: '#94a3b8',
                                    callback: v => 'R$ ' + (v/1e6).toFixed(1) + 'M'
                                }}
                            }}
                        }}
                    }}
                }});
            }});

        // 2. Fetch Party Distribution
        fetch('/api/v1/parties?limit=8')
            .then(r => r.json())
            .then(data => {{
                const ctx = document.getElementById('chartParties').getContext('2d');
                new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: data.map(d => d.partido),
                        datasets: [{{
                            data: data.map(d => d.total_valor),
                            backgroundColor: ['#38bdf8', '#818cf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#f472b6', '#64748b'],
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{ position: 'bottom', labels: {{ color: '#cbd5e1', font: {{ size: 10 }} }} }}
                        }}
                    }}
                }});
            }});

        // 3. Fetch States Distribution
        fetch('/api/v1/states?limit=10')
            .then(r => r.json())
            .then(data => {{
                const ctx = document.getElementById('chartStates').getContext('2d');
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: data.map(d => d.uf),
                        datasets: [{{
                            label: 'Total (R$)',
                            data: data.map(d => d.total_valor),
                            backgroundColor: '#10b981',
                            borderRadius: 6
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ display: false }} }},
                        scales: {{
                            x: {{ grid: {{ display: false }}, ticks: {{ color: '#94a3b8' }} }},
                            y: {{
                                grid: {{ color: 'rgba(255,255,255,0.05)' }},
                                ticks: {{ color: '#94a3b8', callback: v => 'R$ ' + (v/1e6).toFixed(1) + 'M' }}
                            }}
                        }}
                    }}
                }});
            }});

        // 4. Fetch Categories Distribution
        fetch('/api/v1/top-categories?limit=6')
            .then(r => r.json())
            .then(data => {{
                const ctx = document.getElementById('chartCategories').getContext('2d');
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: data.map(d => d.tipo_despesa.length > 25 ? d.tipo_despesa.slice(0, 25) + '...' : d.tipo_despesa),
                        datasets: [{{
                            label: 'Valor',
                            data: data.map(d => d.total_valor),
                            backgroundColor: '#f59e0b',
                            borderRadius: 6
                        }}]
                    }},
                    options: {{
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ display: false }} }},
                        scales: {{
                            x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8', callback: v => 'R$ ' + (v/1e6).toFixed(1) + 'M' }} }},
                            y: {{ grid: {{ display: false }}, ticks: {{ color: '#cbd5e1', font: {{ size: 9 }} }} }}
                        }}
                    }}
                }});
            }});

        // 5. Fetch Anomalies
        fetch('/api/v1/anomalies?limit=250')
            .then(r => r.json())
            .then(res => {{
                const summary = res.summary || {{}};
                const anomalies = res.anomalies || [];
                document.getElementById('anomaly-badge').innerText = `🚨 ${{anomalies.length}} ANOMALIAS IDENTIFICADAS (${{summary.anomaly_percentage || 0}}%)`;
                
                const tbody = document.getElementById('anomalies-table-body');
                if (anomalies.length === 0) {{
                    tbody.innerHTML = '<tr><td colspan="7" class="py-4 text-center text-emerald-400">Nenhuma anomalia crítica detectada nos registros avaliados.</td></tr>';
                    return;
                }}
                tbody.innerHTML = anomalies.slice(0, 8).map(a => `
                    <tr class="hover:bg-slate-800/40 transition">
                        <td class="py-2.5 px-3 text-slate-300">${{String(a.data_documento || '').slice(0, 10)}}</td>
                        <td class="py-2.5 px-3 font-semibold text-white">${{a.nome_deputado || 'DEPUTADO'}}</td>
                        <td class="py-2.5 px-3 text-cyan-400">${{a.partido || ''}}/${{a.uf || ''}}</td>
                        <td class="py-2.5 px-3 text-slate-300 max-w-xs truncate">${{a.nome_fornecedor || ''}}</td>
                        <td class="py-2.5 px-3 text-purple-300 max-w-xs truncate">${{a.tipo_despesa || ''}}</td>
                        <td class="py-2.5 px-3 text-right font-bold text-rose-400">R$ ${{Number(a.valor_liquido || 0).toLocaleString('pt-BR', {{minimumFractionDigits: 2}})}}</td>
                        <td class="py-2.5 px-3 text-center">
                            <span class="px-2 py-0.5 rounded text-[10px] bg-rose-950/80 border border-rose-500/50 text-rose-300">
                                ${{a.anomaly_reasons_str || 'OUTLIER'}}
                            </span>
                        </td>
                    </tr>
                `).join('');
            }});

        // 6. Fetch Deputies Ranking
        fetch('/api/v1/deputies?limit=8')
            .then(r => r.json())
            .then(data => {{
                const tbody = document.getElementById('deputies-table-body');
                tbody.innerHTML = data.map(d => `
                    <tr class="hover:bg-slate-800/30">
                        <td class="py-2 px-2 text-white font-medium">${{d.nome}}</td>
                        <td class="py-2 px-2 text-cyan-400">${{d.partido}}/${{d.uf}}</td>
                        <td class="py-2 px-2 text-right text-slate-400">${{d.qtd_recibos}}</td>
                        <td class="py-2 px-2 text-right font-bold text-slate-200">R$ ${{Number(d.total_gasto).toLocaleString('pt-BR', {{minimumFractionDigits: 2}})}}</td>
                    </tr>
                `).join('');
            }});

        // 7. Fetch Top Suppliers
        fetch('/api/v1/suppliers?limit=8')
            .then(r => r.json())
            .then(data => {{
                const tbody = document.getElementById('suppliers-table-body');
                tbody.innerHTML = data.map(s => `
                    <tr class="hover:bg-slate-800/30">
                        <td class="py-2 px-2 text-white font-medium max-w-xs truncate">${{s.nome_fornecedor}}</td>
                        <td class="py-2 px-2 text-slate-400 font-mono text-[11px]">${{s.cnpj_cpf}}</td>
                        <td class="py-2 px-2 text-right font-bold text-emerald-400">R$ ${{Number(s.total_valor).toLocaleString('pt-BR', {{minimumFractionDigits: 2}})}}</td>
                    </tr>
                `).join('');
            }});

        function triggerETL() {{
            const btn = document.getElementById('btn-etl');
            const icon = document.getElementById('icon-etl');
            btn.disabled = true;
            icon.classList.add('animate-spin');
            btn.querySelector('span').innerText = 'Executando Carga...';

            fetch('/api/v1/etl/run?deputados_limit=25', {{ method: 'POST' }})
                .then(r => r.json())
                .then(res => {{
                    alert(`[✓] Carga Multi-Anual Concluída com Sucesso! ${{res.total_receipts_loaded}} novos fatos inseridos.`);
                    window.location.reload();
                }})
                .catch(err => {{
                    alert('Erro ao atualizar: ' + err);
                    btn.disabled = false;
                    icon.classList.remove('animate-spin');
                }});
        }}
    </script>
</body>
</html>"""
    return html
