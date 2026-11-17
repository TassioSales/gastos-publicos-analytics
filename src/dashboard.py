"""Interactive Analytics Dashboard for Public Spending (CEAP) - Streamlit Edition.

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
        page_title="Gastos Públicos Analytics | CEAP (2019-2026)",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏛️ Gastos Públicos Analytics: Cota Parlamentar (CEAP)")
    st.caption("Data Warehouse Dimensional (DuckDB & Parquet) • Auditoria de Anomalias (Isolation Forest) • Histórico Completo 2019 – 2026")

    warehouse = SpendingWarehouse()
    pipeline = CeapPipeline()
    detector = SpendingAnomalyDetector()

    # Sidebar
    st.sidebar.header("⚙️ Configurações & Filtros")
    anos_disponiveis = ["Todos os Anos (2019-2026)", 2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019]
    ano_sel = st.sidebar.selectbox("Ano de Análise", anos_disponiveis, index=0)
    top_limit = st.sidebar.slider("Limite de Itens em Gráficos", min_value=5, max_value=30, value=12)

    if st.sidebar.button("🔄 Executar Carga Multi-Anual (2019-2026)"):
        with st.spinner("Extraindo e integrando dados históricos da Câmara dos Deputados..."):
            deputados = pipeline.fetch_deputados(limit=25)
            years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
            raw_expenses = pipeline.fetch_multiyear_expenses(deputados, years=years, limit_per_deputy=35)
            cleaned = pipeline.clean_expenses_dataframe(raw_expenses)
            metrics = warehouse.load_dimensions_and_facts(cleaned, deputados_list=deputados)
            st.sidebar.success(f"Carga concluída com sucesso! {metrics['fact_rows']} fatos carregados no DuckDB.")

    # KPI Metrics
    kpis = warehouse.get_summary_kpis()
    col1, col2, col3, col4, col5 = st.columns(5)

    total_gasto = kpis.get("total_gasto", 0.0)
    total_tx = int(kpis.get("total_transacoes", 0))
    ticket_medio = kpis.get("ticket_medio", 0.0)
    maior_despesa = kpis.get("maior_despesa", 0.0)
    total_deps = int(kpis.get("total_deputados", 0))

    col1.metric("💰 Total Analisado", f"R$ {total_gasto:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col2.metric("📑 Total de Recibos", f"{total_tx:,}")
    col3.metric("🎯 Ticket Médio", f"R$ {ticket_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col4.metric("🚨 Maior Recibo", f"R$ {maior_despesa:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    col5.metric("👥 Deputados", f"{total_deps}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Evolução & Histórico (2019-2026)",
        "🏛️ Partidos & Estados (UF)",
        "🏢 Maiores Fornecedores & CNPJs",
        "🚨 Auditoria & Machine Learning",
        "📑 Tabela Fato & Dados Brutos"
    ])

    with tab1:
        st.subheader("Evolução Histórica de Gastos (2019 – 2026)")
        years_df = warehouse.get_spending_by_year()
        if not years_df.empty:
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_line = px.bar(
                    years_df,
                    x="ano",
                    y="total_valor",
                    title="Volume Financeiro Anual (R$)",
                    labels={"ano": "Ano", "total_valor": "Total Gasto (R$)"},
                    text_auto=".2s",
                    color="total_valor",
                    color_continuous_scale="Tealgrn"
                )
                st.plotly_chart(fig_line, use_container_width=True)
            with c2:
                fig_avg = px.line(
                    years_df,
                    x="ano",
                    y="ticket_medio",
                    title="Ticket Médio por Nota (R$)",
                    labels={"ano": "Ano", "ticket_medio": "Ticket Médio (R$)"},
                    markers=True
                )
                st.plotly_chart(fig_avg, use_container_width=True)

        st.subheader("Distribuição por Categoria de Despesa")
        cat_df = warehouse.get_top_spending_categories(limit=top_limit)
        if not cat_df.empty:
            c3, c4 = st.columns([3, 2])
            with c3:
                fig_cat = px.bar(
                    cat_df,
                    x="total_valor",
                    y="tipo_despesa",
                    orientation="h",
                    color="macro_categoria",
                    title="Despesas por Categoria",
                    labels={"total_valor": "Total (R$)", "tipo_despesa": "Categoria"}
                )
                fig_cat.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig_cat, use_container_width=True)
            with c4:
                fig_pie = px.pie(
                    cat_df,
                    names="macro_categoria",
                    values="total_valor",
                    title="Proporção Macro-Categorias",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.subheader("Concentração Política e Regional")
        p_col, uf_col = st.columns(2)
        with p_col:
            st.markdown("#### Gastos por Partido Político")
            party_df = warehouse.get_spending_by_party(limit=top_limit)
            if not party_df.empty:
                fig_p = px.bar(
                    party_df,
                    x="partido",
                    y="total_valor",
                    color="partido",
                    title="Total Gasto por Sigla Partidária",
                    labels={"partido": "Partido", "total_valor": "Gasto (R$)"}
                )
                st.plotly_chart(fig_p, use_container_width=True)

        with uf_col:
            st.markdown("#### Gastos por Estado de Origem (UF)")
            uf_df = warehouse.get_spending_by_uf(limit=top_limit)
            if not uf_df.empty:
                fig_uf = px.bar(
                    uf_df,
                    x="uf",
                    y="total_valor",
                    color="uf",
                    title="Total Gasto por Unidade Federativa",
                    labels={"uf": "Estado", "total_valor": "Gasto (R$)"}
                )
                st.plotly_chart(fig_uf, use_container_width=True)

    with tab3:
        st.subheader("🏢 Maiores Fornecedores & CNPJs Beneficiados")
        supp_df = warehouse.get_top_suppliers(limit=top_limit)
        if not supp_df.empty:
            fig_supp = px.bar(
                supp_df,
                x="total_valor",
                y="nome_fornecedor",
                orientation="h",
                title="Top Fornecedores por Faturamento Público (R$)",
                labels={"total_valor": "Total Recebido (R$)", "nome_fornecedor": "Fornecedor"}
            )
            fig_supp.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig_supp, use_container_width=True)

            st.dataframe(
                supp_df[["nome_fornecedor", "cnpj_cpf", "qtd_recibos", "total_valor", "ticket_medio", "maior_recibo"]],
                use_container_width=True
            )

    with tab4:
        st.subheader("🚨 Auditoria e Detecção de Anomalias com Machine Learning")
        st.markdown("""
        **Pipeline Analítico de Fraudes e Desvios:**
        - **Isolation Forest Multivariado:** Detecção de transações de alto valor fora do perfil do parlamentar.
        - **Z-Score Estatístico (3σ):** Identificação de valores extremos em categorias normais.
        - **Regras de Compliance:** Despesas em fins de semana e fracionamento de valores.
        """)
        if total_tx > 0:
            query = """
                SELECT f.despesa_id, f.id_deputado, f.data_documento, f.ano, f.mes, f.valor_liquido,
                       f.is_fim_de_semana, c.tipo_despesa, forn.nome_fornecedor, d.nome as nome_deputado, d.partido, d.uf
                FROM fact_despesas f
                LEFT JOIN dim_categoria c ON f.categoria_key = c.categoria_key
                LEFT JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
                LEFT JOIN dim_deputado d ON f.deputado_key = d.deputado_key
                ORDER BY f.valor_liquido DESC
                LIMIT 500
            """
            try:
                sample_df = warehouse.conn.execute(query).df() if warehouse.use_duckdb else pd.read_sql_query(query, warehouse.conn)
                analyzed_df = detector.analyze_dataframe(sample_df)
                anomalies_summary = detector.summarize_anomalies(analyzed_df)

                a1, a2, a3 = st.columns(3)
                a1.metric("Anomalias Detectadas", f"{anomalies_summary['total_anomalies']}")
                a2.metric("Percentual Suspeito", f"{anomalies_summary['anomaly_percentage']}%")
                a3.metric("Volume Sob Alerta", f"R$ {anomalies_summary['total_suspicious_amount_brl']:,.2f}")

                outliers = analyzed_df[analyzed_df["is_anomaly"] == True]
                if not outliers.empty:
                    st.dataframe(
                        outliers[["data_documento", "nome_deputado", "partido", "nome_fornecedor", "tipo_despesa", "valor_liquido", "anomaly_reasons_str"]],
                        use_container_width=True,
                    )
                else:
                    st.success("Nenhuma anomalia crítica detectada.")
            except Exception as e:
                st.warning(f"Aviso ao analisar anomalias: {e}")

    with tab5:
        st.subheader("Visualização da Tabela Fato Normalizada (Star Schema)")
        if total_tx > 0:
            fact_sample = warehouse.conn.execute("""
                SELECT f.despesa_id, d.nome as deputado, d.partido, d.uf,
                       c.tipo_despesa, forn.nome_fornecedor, f.data_documento,
                       f.valor_liquido, f.is_fim_de_semana
                FROM fact_despesas f
                JOIN dim_deputado d ON f.deputado_key = d.deputado_key
                JOIN dim_categoria c ON f.categoria_key = c.categoria_key
                JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
                ORDER BY f.data_documento DESC
                LIMIT 200
            """).df() if warehouse.use_duckdb else pd.read_sql_query("SELECT * FROM fact_despesas LIMIT 200", warehouse.conn)
            st.dataframe(fact_sample, use_container_width=True)


if __name__ == "__main__":
    main()
