"""Interactive Analytics Dashboard for Public Spending (CEAP).

Run with:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline import CeapPipeline
from src.warehouse import SpendingWarehouse
from src.anomaly_detector import SpendingAnomalyDetector


def main() -> None:
    st.set_page_config(
        page_title="Gastos Públicos Analytics | CEAP",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏛️ Gastos Públicos Analytics: Cota Parlamentar (CEAP)")
    st.caption("Pipeline de Dados, Data Warehouse Dimensional e Detecção Inteligente de Anomalias | Câmara dos Deputados")

    warehouse = SpendingWarehouse()
    pipeline = CeapPipeline()
    detector = SpendingAnomalyDetector()

    # Sidebar
    st.sidebar.header("⚙️ Configurações & Filtros")
    ano = st.sidebar.selectbox("Ano de Análise", [2024, 2023], index=0)
    top_limit = st.sidebar.slider("Top Categorias / Fornecedores", min_value=5, max_value=20, value=10)

    if st.sidebar.button("🔄 Executar Carga / Atualizar Dados"):
        with st.spinner("Extraindo e carregando dados da API Dados Abertos..."):
            deputados = pipeline.fetch_deputados(limit=10)
            all_expenses = []
            for dep in deputados:
                exp = pipeline.fetch_despesas_deputado(deputado_id=dep["id"], ano=ano, limit=50)
                all_expenses.extend(exp)
            cleaned = pipeline.clean_expenses_dataframe(all_expenses)
            metrics = warehouse.load_dimensions_and_facts(cleaned, deputados_list=deputados)
            st.sidebar.success(f"Carga concluída! {metrics['fact_rows']} fatos carregados.")

    # KPI Metrics
    kpis = warehouse.get_summary_kpis()
    col1, col2, col3, col4 = st.columns(4)

    total_gasto = kpis.get("total_gasto", 0.0)
    total_tx = kpis.get("total_transacoes", 0)
    ticket_medio = kpis.get("ticket_medio", 0.0)
    maior_despesa = kpis.get("maior_despesa", 0.0)

    col1.metric("💰 Total Gasto Analisado", f"R$ {total_gasto:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col2.metric("📑 Total de Recibos/Notas", f"{total_tx:,}")
    col3.metric("🎯 Ticket Médio por Recibo", f"R$ {ticket_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col4.metric("🚨 Maior Despesa Única", f"R$ {maior_despesa:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Análise de Categorias & Macro", "🔍 Detector de Anomalias", "🏛️ Detalhamento Dimensional"])

    with tab1:
        st.subheader("Distribuição de Gastos por Categoria")
        cat_df = warehouse.get_top_spending_categories(limit=top_limit)
        if not cat_df.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_bar = px.bar(
                    cat_df,
                    x="total_valor",
                    y="tipo_despesa",
                    orientation="h",
                    color="macro_categoria",
                    title="Volume Total por Tipo de Despesa (R$)",
                    labels={"total_valor": "Total (R$)", "tipo_despesa": "Tipo de Despesa"},
                )
                fig_bar.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig_bar, use_container_width=True)
            with c2:
                fig_pie = px.pie(
                    cat_df,
                    names="macro_categoria",
                    values="total_valor",
                    title="Distribuição Macro Categoria",
                    hole=0.4,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Execute a carga de dados pela barra lateral para visualizar os gráficos analíticos.")

    with tab2:
        st.subheader("🚨 Auditoria de Conformidade e Detecção de Padrões Suspeitos")
        st.markdown("""
        O motor aplica **Z-Score Estatístico (3σ)**, **Isolation Forest Multivariado** e regras heurísticas:
        - Fracionamento de despesas no mesmo dia para mesmo fornecedor
        - Gastos atípicos em finais de semana
        - Valores fora da distribuição padrão da categoria
        """)
        # Query fact and detect
        if total_tx > 0:
            query = """
                SELECT f.despesa_id, f.id_deputado, f.data_documento, f.valor_liquido,
                       f.is_fim_de_semana, c.tipo_despesa, forn.nome_fornecedor
                FROM fact_despesas f
                LEFT JOIN dim_categoria c ON f.categoria_key = c.categoria_key
                LEFT JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
                LIMIT 500
            """
            try:
                sample_df = warehouse.conn.execute(query).df() if warehouse.use_duckdb else pd.read_sql_query(query, warehouse.conn)
                analyzed_df = detector.analyze_dataframe(sample_df)
                anomalies_summary = detector.summarize_anomalies(analyzed_df)

                a1, a2, a3 = st.columns(3)
                a1.metric("Anomalias Detectadas", f"{anomalies_summary['total_anomalies']}")
                a2.metric("Percentual Suspeito", f"{anomalies_summary['anomaly_percentage']}%")
                a3.metric("Volume Financeiro Sob Alerta", f"R$ {anomalies_summary['total_suspicious_amount_brl']:,.2f}")

                outliers = analyzed_df[analyzed_df["is_anomaly"] == True]
                if not outliers.empty:
                    st.dataframe(
                        outliers[["data_documento", "nome_fornecedor", "tipo_despesa", "valor_liquido", "anomaly_reasons_str"]],
                        use_container_width=True,
                    )
                else:
                    st.success("Nenhuma anomalia crítica detectada no lote atual.")
            except Exception as e:
                st.warning(f"Aviso ao analisar anomalias: {e}")
        else:
            st.info("Carregue dados na barra lateral para executar os modelos de auditoria.")

    with tab3:
        st.subheader("Visualização da Tabela Fato")
        st.markdown("Exibindo registros normalizados na camada analítica do Data Warehouse.")
        if total_tx > 0:
            fact_sample = warehouse.conn.execute("SELECT * FROM fact_despesas LIMIT 100").df() if warehouse.use_duckdb else pd.read_sql_query("SELECT * FROM fact_despesas LIMIT 100", warehouse.conn)
            st.dataframe(fact_sample, use_container_width=True)
        else:
            st.info("Nenhum dado na tabela fato.")


if __name__ == "__main__":
    main()
