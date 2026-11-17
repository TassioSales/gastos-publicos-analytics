"""Command Line Interface (CLI) for Gastos Públicos Analytics.

Usage:
    python cli.py etl          # Run extraction and load into DuckDB Star Schema
    python cli.py kpis         # Display executive KPIs
    python cli.py audit        # Run ML Anomaly Detection report
    python cli.py dashboard    # Launch Streamlit interactive dashboard
    python cli.py api          # Launch FastAPI backend
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
import pandas as pd

from src.pipeline import CeapPipeline
from src.warehouse import SpendingWarehouse
from src.anomaly_detector import SpendingAnomalyDetector


def cmd_etl(args: argparse.Namespace) -> None:
    print(f"[*] Starting CEAP ETL Pipeline for year {args.ano} (Limit: {args.limit} deputados)...")
    pipeline = CeapPipeline()
    warehouse = SpendingWarehouse()

    deputados = pipeline.fetch_deputados(limit=args.limit)
    all_expenses = []
    for dep in deputados:
        exp = pipeline.fetch_despesas_deputado(deputado_id=dep["id"], ano=args.ano, limit=args.itens)
        all_expenses.extend(exp)

    cleaned = pipeline.clean_expenses_dataframe(all_expenses)
    metrics = warehouse.load_dimensions_and_facts(cleaned, deputados_list=deputados)
    print(f"[✓] ETL successfully finished! Loaded: {metrics['fact_rows']} fact rows, {metrics['dim_deputados']} deputies.")


def cmd_kpis(args: argparse.Namespace) -> None:
    print("[*] Retrieving Macro KPIs from Data Warehouse...")
    warehouse = SpendingWarehouse()
    kpis = warehouse.get_summary_kpis()
    if not kpis:
        print("[!] No records found in warehouse. Run 'python cli.py etl' first.")
        return

    print("\n================= EXECUTIVE KPIS =================")
    print(f" Total Despesas Analisadas:  R$ {kpis.get('total_gasto', 0.0):,.2f}")
    print(f" Total de Recibos/Fatos:    {kpis.get('total_transacoes', 0):,}")
    print(f" Ticket Médio por Recibo:   R$ {kpis.get('ticket_medio', 0.0):,.2f}")
    print(f" Maior Despesa Única:       R$ {kpis.get('maior_despesa', 0.0):,.2f}")
    print(f" Total de Deputados:        {kpis.get('total_deputados', 0)}")
    print(f" Total de Fornecedores:     {kpis.get('total_fornecedores', 0)}")
    print("==================================================\n")


def cmd_audit(args: argparse.Namespace) -> None:
    print("[*] Running Anomaly Detection & Compliance Engine (Z-Score + Isolation Forest)...")
    warehouse = SpendingWarehouse()
    detector = SpendingAnomalyDetector()

    query = """
        SELECT f.despesa_id, f.id_deputado, f.data_documento, f.valor_liquido,
               f.is_fim_de_semana, c.tipo_despesa, forn.nome_fornecedor
        FROM fact_despesas f
        LEFT JOIN dim_categoria c ON f.categoria_key = c.categoria_key
        LEFT JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
        LIMIT 1000
    """
    df = warehouse.conn.execute(query).df() if warehouse.use_duckdb else pd.read_sql_query(query, warehouse.conn)
    if df.empty:
        print("[!] No data available to audit. Run 'python cli.py etl' first.")
        return

    analyzed = detector.analyze_dataframe(df)
    summary = detector.summarize_anomalies(analyzed)

    print("\n================ AUDIT & ANOMALIES ================")
    print(f" Total de Registros Analisados: {summary['total_records']}")
    print(f" Total de Anomalias Detectadas: {summary['total_anomalies']} ({summary['anomaly_percentage']}%)")
    print(f" Volume Financeiro sob Alerta:  R$ {summary['total_suspicious_amount_brl']:,.2f}")
    print(" Principais Motivos Flagrados:")
    for reason, count in summary["most_common_reasons"].items():
        print(f"   - [{count}x] {reason}")
    print("==================================================\n")


def cmd_dashboard(args: argparse.Namespace) -> None:
    print("[*] Starting Streamlit Analytics Dashboard on port 8501...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard.py", "--server.port", "8501"])


def cmd_api(args: argparse.Namespace) -> None:
    print("[*] Starting FastAPI Analytical Server on port 8000...")
    subprocess.run([sys.executable, "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Gastos Públicos Analytics CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ETL
    p_etl = subparsers.add_parser("etl", help="Run ingestion and load into Star Schema")
    p_etl.add_argument("--ano", type=int, default=2024, help="Year of analysis")
    p_etl.add_argument("--limit", type=int, default=10, help="Number of deputies to ingest")
    p_etl.add_argument("--itens", type=int, default=50, help="Receipts per deputy")
    p_etl.set_defaults(func=cmd_etl)

    # KPIs
    p_kpis = subparsers.add_parser("kpis", help="Display macro KPIs from Data Warehouse")
    p_kpis.set_defaults(func=cmd_kpis)

    # Audit
    p_audit = subparsers.add_parser("audit", help="Run statistical and ML anomaly detection")
    p_audit.set_defaults(func=cmd_audit)

    # Dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    # API
    p_api = subparsers.add_parser("api", help="Launch FastAPI server")
    p_api.set_defaults(func=cmd_api)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
