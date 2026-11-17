"""Dimensional Data Warehouse for CEAP (Câmara dos Deputados).

Implements Kimball Star Schema on DuckDB with Parquet persistence.
Supports full multi-year historical analysis (2019-2026) and granular year filtering.
"""

from __future__ import annotations

import os
import sqlite3
import logging
import threading
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class SpendingWarehouse:
    """Analytical Data Warehouse manager supporting DuckDB and SQLite."""

    def __init__(self, db_path: str = "data/warehouse/gastos_dw.duckdb", read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self.use_duckdb = HAS_DUCKDB

        if self.use_duckdb:
            try:
                self.conn = duckdb.connect(self.db_path, read_only=read_only)
                if not read_only:
                    self._init_duckdb_schema()
            except Exception as e:
                logger.warning("DuckDB connection failed (%s). Opening read-only...", e)
                try:
                    self.conn = duckdb.connect(self.db_path, read_only=True)
                except Exception as e2:
                    logger.error("DuckDB read-only also failed: %s. Falling back to SQLite.", e2)
                    self.use_duckdb = False
                    self.sqlite_path = self.db_path.replace(".duckdb", ".db")
                    self.conn = sqlite3.connect(self.sqlite_path)
                    if not read_only:
                        self._init_sqlite_schema()
        else:
            self.sqlite_path = self.db_path.replace(".duckdb", ".db")
            self.conn = sqlite3.connect(self.sqlite_path)
            if not read_only:
                self._init_sqlite_schema()

    def _init_duckdb_schema(self) -> None:
        """Create Star Schema tables in DuckDB."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dim_deputado (
                deputado_key INTEGER PRIMARY KEY,
                id_deputado INTEGER UNIQUE,
                nome VARCHAR,
                partido VARCHAR,
                uf VARCHAR
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dim_fornecedor (
                fornecedor_key INTEGER PRIMARY KEY,
                cnpj_cpf VARCHAR UNIQUE,
                nome_fornecedor VARCHAR
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dim_categoria (
                categoria_key INTEGER PRIMARY KEY,
                tipo_despesa VARCHAR UNIQUE,
                macro_categoria VARCHAR
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fact_despesas (
                despesa_id VARCHAR PRIMARY KEY,
                deputado_key INTEGER,
                fornecedor_key INTEGER,
                categoria_key INTEGER,
                data_documento DATE,
                ano INTEGER,
                mes INTEGER,
                valor_documento DOUBLE,
                valor_liquido DOUBLE,
                is_fim_de_semana BOOLEAN,
                num_documento VARCHAR
            );
        """)

    def _init_sqlite_schema(self) -> None:
        """Create Star Schema tables in SQLite (fallback mode)."""
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS dim_deputado (
                deputado_key INTEGER PRIMARY KEY AUTOINCREMENT,
                id_deputado INTEGER UNIQUE,
                nome TEXT,
                partido TEXT,
                uf TEXT
            );
            CREATE TABLE IF NOT EXISTS dim_fornecedor (
                fornecedor_key INTEGER PRIMARY KEY AUTOINCREMENT,
                cnpj_cpf TEXT UNIQUE,
                nome_fornecedor TEXT
            );
            CREATE TABLE IF NOT EXISTS dim_categoria (
                categoria_key INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo_despesa TEXT UNIQUE,
                macro_categoria TEXT
            );
            CREATE TABLE IF NOT EXISTS fact_despesas (
                despesa_id TEXT PRIMARY KEY,
                deputado_key INTEGER,
                fornecedor_key INTEGER,
                categoria_key INTEGER,
                data_documento TEXT,
                ano INTEGER,
                mes INTEGER,
                valor_documento REAL,
                valor_liquido REAL,
                is_fim_de_semana INTEGER,
                num_documento TEXT
            );
        """)
        self.conn.commit()

    def query_df(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query thread-safely with connection lock and cursor isolation."""
        with self._lock:
            if self.use_duckdb:
                cur = self.conn.cursor()
                try:
                    return cur.execute(sql).df()
                finally:
                    cur.close()
            return pd.read_sql_query(sql, self.conn)

    def query_scalar(self, sql: str) -> Any:
        """Execute a scalar SQL query thread-safely."""
        with self._lock:
            if self.use_duckdb:
                cur = self.conn.cursor()
                try:
                    res = cur.execute(sql).fetchone()
                    return res[0] if res else None
                finally:
                    cur.close()
            df = pd.read_sql_query(sql, self.conn)
            return df.iloc[0, 0] if not df.empty else None

    def reset_and_reload(self) -> None:
        """Wipe existing facts and dimensions to perform a clean multi-year reload."""
        with self._lock:
            if self.use_duckdb:
                self.conn.execute("DELETE FROM fact_despesas")
                self.conn.execute("DELETE FROM dim_deputado")
                self.conn.execute("DELETE FROM dim_fornecedor")
                self.conn.execute("DELETE FROM dim_categoria")
            else:
                cur = self.conn.cursor()
                cur.execute("DELETE FROM fact_despesas")
                cur.execute("DELETE FROM dim_deputado")
                cur.execute("DELETE FROM dim_fornecedor")
                cur.execute("DELETE FROM dim_categoria")
                self.conn.commit()

    def load_dimensions_and_facts(
        self,
        expenses_df: pd.DataFrame,
        deputados_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """Ingest cleaned DataFrame into Star Schema with Kimball dimensional conformance."""
        if expenses_df.empty:
            return {"fact_rows": 0, "dim_deputados": 0}

        # Build Dim Deputado lookup
        dep_lookup = {}
        if deputados_list:
            for idx, d in enumerate(deputados_list, start=1):
                dep_id = d.get("id")
                dep_lookup[dep_id] = {
                    "key": idx,
                    "id": dep_id,
                    "nome": str(d.get("nome", f"DEPUTADO_{dep_id}")).upper(),
                    "partido": str(d.get("siglaPartido") or d.get("partido") or "OUTROS").upper(),
                    "uf": str(d.get("siglaUf") or d.get("uf") or "DF").upper(),
                }

        unique_dep_ids = expenses_df["id_deputado"].drop_duplicates().tolist()
        for dep_id in unique_dep_ids:
            if dep_id not in dep_lookup:
                dep_lookup[dep_id] = {
                    "key": len(dep_lookup) + 1,
                    "id": dep_id,
                    "nome": f"DEPUTADO_{dep_id}",
                    "partido": "OUTROS",
                    "uf": "BR",
                }

        dep_df = pd.DataFrame(list(dep_lookup.values()))

        # Build Dim Fornecedor
        fornecedores = expenses_df[["cnpj_cpf_fornecedor", "nome_fornecedor"]].drop_duplicates().reset_index(drop=True)
        fornecedores["fornecedor_key"] = range(1, len(fornecedores) + 1)
        fornecedores = fornecedores.rename(columns={"cnpj_cpf_fornecedor": "cnpj_cpf"})

        # Build Dim Categoria
        categorias = expenses_df[["tipo_despesa"]].drop_duplicates().reset_index(drop=True)
        categorias["categoria_key"] = range(1, len(categorias) + 1)
        categorias["macro_categoria"] = categorias["tipo_despesa"].apply(self._classify_macro_category)

        # Merge surrogate keys onto fact
        fact = expenses_df.copy()
        fact = fact.merge(fornecedores, left_on=["cnpj_cpf_fornecedor", "nome_fornecedor"], right_on=["cnpj_cpf", "nome_fornecedor"], how="left")
        fact = fact.merge(categorias, on="tipo_despesa", how="left")

        dep_key_map = dict(zip(dep_df["id"], dep_df["key"]))
        fact["deputado_key"] = fact["id_deputado"].map(dep_key_map).fillna(1).astype(int)

        fact["despesa_id"] = (
            fact["id_deputado"].astype(str) + "_" +
            fact["ano"].astype(str) + "_" +
            fact["mes"].astype(str) + "_" +
            fact["num_documento"].astype(str) + "_" +
            fact.index.astype(str)
        )

        # Filter out negative or zero refunds for clean spending analytics
        fact = fact[fact["valor_liquido"] > 0].copy()

        # Store in Warehouse
        if self.use_duckdb:
            with self._lock:
                self.conn.register("tmp_dep", dep_df)
                self.conn.execute("""
                    INSERT OR IGNORE INTO dim_deputado (deputado_key, id_deputado, nome, partido, uf)
                    SELECT key, id, nome, partido, uf FROM tmp_dep
                """)

                self.conn.register("tmp_forn", fornecedores)
                self.conn.execute("""
                    INSERT OR IGNORE INTO dim_fornecedor (fornecedor_key, cnpj_cpf, nome_fornecedor)
                    SELECT fornecedor_key, cnpj_cpf, nome_fornecedor FROM tmp_forn
                """)

                self.conn.register("tmp_cat", categorias)
                self.conn.execute("""
                    INSERT OR IGNORE INTO dim_categoria (categoria_key, tipo_despesa, macro_categoria)
                    SELECT categoria_key, tipo_despesa, macro_categoria FROM tmp_cat
                """)

                fact_cols = fact[[
                    "despesa_id", "deputado_key", "fornecedor_key", "categoria_key",
                    "data_documento", "ano", "mes", "valor_documento", "valor_liquido",
                    "is_fim_de_semana", "num_documento"
                ]]
                self.conn.register("tmp_fact", fact_cols)
                self.conn.execute("""
                    INSERT OR IGNORE INTO fact_despesas (
                        despesa_id, deputado_key, fornecedor_key, categoria_key,
                        data_documento, ano, mes, valor_documento, valor_liquido,
                        is_fim_de_semana, num_documento
                    )
                    SELECT despesa_id, deputado_key, fornecedor_key, categoria_key,
                           data_documento, ano, mes, valor_documento, valor_liquido,
                           is_fim_de_semana, num_documento
                    FROM tmp_fact
                """)
        else:
            with self._lock:
                dep_df.to_sql("dim_deputado", self.conn, if_exists="replace", index=False)
                fornecedores.to_sql("dim_fornecedor", self.conn, if_exists="replace", index=False)
                categorias.to_sql("dim_categoria", self.conn, if_exists="replace", index=False)
                fact.to_sql("fact_despesas", self.conn, if_exists="replace", index=False)

        return {"fact_rows": len(fact), "dim_deputados": len(dep_df)}

    def get_summary_kpis(self, ano: Optional[int] = None) -> Dict[str, Any]:
        """Return high-level KPIs across the analytical warehouse, optionally filtered by year."""
        where_clause = f"WHERE ano = {int(ano)}" if ano else ""
        query = f"""
            SELECT
                COUNT(*) as total_transacoes,
                COALESCE(SUM(valor_liquido), 0.0) as total_gasto,
                COALESCE(AVG(valor_liquido), 0.0) as ticket_medio,
                COALESCE(MAX(valor_liquido), 0.0) as maior_despesa,
                COUNT(DISTINCT deputado_key) as total_deputados,
                COUNT(DISTINCT fornecedor_key) as total_fornecedores
            FROM fact_despesas
            {where_clause}
        """
        df = self.query_df(query)
        return df.iloc[0].to_dict() if not df.empty else {}

    def get_top_spending_categories(self, ano: Optional[int] = None, limit: int = 10) -> pd.DataFrame:
        """Return spending aggregated by category, optionally filtered by year."""
        where_clause = f"WHERE f.ano = {int(ano)}" if ano else ""
        query = f"""
            SELECT
                c.tipo_despesa,
                c.macro_categoria,
                COUNT(f.despesa_id) as qtd_recibos,
                SUM(f.valor_liquido) as total_valor
            FROM fact_despesas f
            JOIN dim_categoria c ON f.categoria_key = c.categoria_key
            {where_clause}
            GROUP BY c.tipo_despesa, c.macro_categoria
            ORDER BY total_valor DESC
            LIMIT {limit}
        """
        return self.query_df(query)

    def get_spending_by_year(self) -> pd.DataFrame:
        """Return historical spending evolution by year (2019-2026)."""
        query = """
            SELECT
                ano,
                COUNT(*) as qtd_recibos,
                SUM(valor_liquido) as total_valor,
                AVG(valor_liquido) as ticket_medio
            FROM fact_despesas
            WHERE ano IS NOT NULL AND valor_liquido > 0
            GROUP BY ano
            ORDER BY ano ASC
        """
        return self.query_df(query)

    def get_spending_by_party(self, ano: Optional[int] = None, limit: int = 15) -> pd.DataFrame:
        """Return spending grouped by political party, optionally filtered by year."""
        where_clause = f"WHERE f.ano = {int(ano)}" if ano else ""
        query = f"""
            SELECT
                COALESCE(d.partido, 'OUTROS') as partido,
                COUNT(DISTINCT d.id_deputado) as total_deputados,
                COUNT(f.despesa_id) as qtd_recibos,
                SUM(f.valor_liquido) as total_valor,
                AVG(f.valor_liquido) as ticket_medio
            FROM fact_despesas f
            JOIN dim_deputado d ON f.deputado_key = d.deputado_key
            {where_clause}
            GROUP BY d.partido
            ORDER BY total_valor DESC
            LIMIT {limit}
        """
        return self.query_df(query)

    def get_spending_by_uf(self, ano: Optional[int] = None, limit: int = 27) -> pd.DataFrame:
        """Return spending grouped by Brazilian State (UF), optionally filtered by year."""
        where_clause = f"WHERE f.ano = {int(ano)}" if ano else ""
        query = f"""
            SELECT
                COALESCE(d.uf, 'BR') as uf,
                COUNT(DISTINCT d.id_deputado) as total_deputados,
                COUNT(f.despesa_id) as qtd_recibos,
                SUM(f.valor_liquido) as total_valor
            FROM fact_despesas f
            JOIN dim_deputado d ON f.deputado_key = d.deputado_key
            {where_clause}
            GROUP BY d.uf
            ORDER BY total_valor DESC
            LIMIT {limit}
        """
        return self.query_df(query)

    def get_top_suppliers(self, ano: Optional[int] = None, limit: int = 15) -> pd.DataFrame:
        """Return top suppliers / CNPJs receiving public parliamentary funds."""
        where_clause = f"WHERE f.ano = {int(ano)}" if ano else ""
        query = f"""
            SELECT
                forn.nome_fornecedor,
                forn.cnpj_cpf,
                COUNT(f.despesa_id) as qtd_recibos,
                SUM(f.valor_liquido) as total_valor,
                AVG(f.valor_liquido) as ticket_medio,
                MAX(f.valor_liquido) as maior_recibo
            FROM fact_despesas f
            JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
            {where_clause}
            GROUP BY forn.nome_fornecedor, forn.cnpj_cpf
            ORDER BY total_valor DESC
            LIMIT {limit}
        """
        return self.query_df(query)

    def get_deputy_ranking(self, ano: Optional[int] = None, limit: int = 25) -> pd.DataFrame:
        """Return top spending deputies ranking, optionally filtered by year."""
        where_clause = f"WHERE f.ano = {int(ano)}" if ano else ""
        query = f"""
            SELECT
                d.nome,
                d.partido,
                d.uf,
                COUNT(f.despesa_id) as qtd_recibos,
                SUM(f.valor_liquido) as total_gasto,
                AVG(f.valor_liquido) as ticket_medio,
                MAX(f.valor_liquido) as maior_gasto
            FROM fact_despesas f
            JOIN dim_deputado d ON f.deputado_key = d.deputado_key
            {where_clause}
            GROUP BY d.nome, d.partido, d.uf
            ORDER BY total_gasto DESC
            LIMIT {limit}
        """
        return self.query_df(query)

    def get_available_years(self) -> List[int]:
        """Return all distinct years present in the warehouse sorted descending."""
        query = "SELECT DISTINCT ano FROM fact_despesas WHERE ano IS NOT NULL ORDER BY ano DESC"
        df = self.query_df(query)
        return [int(y) for y in df["ano"].tolist()] if not df.empty else []

    @staticmethod
    def _classify_macro_category(tipo: str) -> str:
        """Classify granular expense types into macro categories."""
        tipo_upper = str(tipo).upper()
        if any(w in tipo_upper for w in ["AÉREO", "PASSAGEM", "LOCOMOÇÃO", "COMBUSTÍVEL", "VEÍCULO", "FRETAMENTO"]):
            return "TRANSPORTE & LOGÍSTICA"
        if any(w in tipo_upper for w in ["DIVULGAÇÃO", "IMPRENSA", "PROPAGANDA", "PUBLICIDADE"]):
            return "COMUNICAÇÃO & MARKETING"
        if any(w in tipo_upper for w in ["CONSULTORIA", "PESQUISA", "ASSESSORIA", "JURÍDICO"]):
            return "CONSULTORIA & SERVIÇOS TÉCNICOS"
        if any(w in tipo_upper for w in ["ALIMENTAÇÃO", "REFEIÇÃO", "HOSPEDAGEM"]):
            return "DIÁRIAS & HOSPEDAGEM"
        return "OUTROS"

    def get_all_deputies_list(self) -> List[Dict[str, Any]]:
        """Return distinct list of deputies for search/filter dropdowns."""
        query = """
            SELECT DISTINCT
                d.id_deputado,
                d.nome,
                d.partido,
                d.uf
            FROM dim_deputado d
            JOIN fact_despesas f ON d.deputado_key = f.deputado_key
            ORDER BY d.nome ASC
        """
        df = self.query_df(query)
        return df.to_dict(orient="records") if not df.empty else []

    def get_detailed_expenses(
        self,
        ano: Optional[int] = None,
        deputado: Optional[str] = None,
        fornecedor: Optional[str] = None,
        categoria: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "valor_desc",
    ) -> Dict[str, Any]:
        """Return granular itemized expenses: who spent, what they bought, and where (supplier/CNPJ)."""
        conditions = []
        if ano:
            conditions.append(f"f.ano = {int(ano)}")
        if deputado:
            dep_clean = deputado.replace("'", "''")
            conditions.append(f"UPPER(d.nome) LIKE UPPER('%{dep_clean}%')")
        if fornecedor:
            forn_clean = fornecedor.replace("'", "''")
            conditions.append(f"(UPPER(forn.nome_fornecedor) LIKE UPPER('%{forn_clean}%') OR forn.cnpj_cpf LIKE '%{forn_clean}%')")
        if categoria:
            cat_clean = categoria.replace("'", "''")
            conditions.append(f"UPPER(c.tipo_despesa) LIKE UPPER('%{cat_clean}%')")
        if search:
            s_clean = search.replace("'", "''")
            conditions.append(
                f"(UPPER(d.nome) LIKE UPPER('%{s_clean}%') OR "
                f"UPPER(forn.nome_fornecedor) LIKE UPPER('%{s_clean}%') OR "
                f"forn.cnpj_cpf LIKE '%{s_clean}%' OR "
                f"UPPER(c.tipo_despesa) LIKE UPPER('%{s_clean}%') OR "
                f"f.num_documento LIKE '%{s_clean}%')"
            )

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        if sort_by == "data_desc":
            order_clause = "ORDER BY f.data_documento DESC, f.valor_liquido DESC"
        elif sort_by == "valor_asc":
            order_clause = "ORDER BY f.valor_liquido ASC"
        else:
            order_clause = "ORDER BY f.valor_liquido DESC, f.data_documento DESC"

        count_query = f"""
            SELECT COUNT(*) as total_matching
            FROM fact_despesas f
            JOIN dim_deputado d ON f.deputado_key = d.deputado_key
            JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
            JOIN dim_categoria c ON f.categoria_key = c.categoria_key
            {where_clause}
        """

        data_query = f"""
            SELECT
                f.despesa_id,
                d.id_deputado,
                d.nome AS deputado_nome,
                d.partido AS deputado_partido,
                d.uf AS deputado_uf,
                c.tipo_despesa AS categoria,
                c.macro_categoria,
                forn.nome_fornecedor AS fornecedor_nome,
                forn.cnpj_cpf AS fornecedor_cnpj,
                f.data_documento,
                f.ano,
                f.mes,
                f.valor_liquido,
                f.is_fim_de_semana,
                f.num_documento
            FROM fact_despesas f
            JOIN dim_deputado d ON f.deputado_key = d.deputado_key
            JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
            JOIN dim_categoria c ON f.categoria_key = c.categoria_key
            {where_clause}
            {order_clause}
            LIMIT {limit} OFFSET {offset}
        """

        total_matching = int(self.query_scalar(count_query) or 0)
        df = self.query_df(data_query)

        records = df.to_dict(orient="records") if not df.empty else []
        return {
            "total_matching": total_matching,
            "limit": limit,
            "offset": offset,
            "expenses": records,
        }

    def get_deputy_profile(self, deputado_nome: str, ano: Optional[int] = None) -> Dict[str, Any]:
        """Return comprehensive breakdown/dossier for a specific deputy (who, what, where)."""
        dep_clean = deputado_nome.replace("'", "''")
        where_ano = f"AND f.ano = {int(ano)}" if ano else ""

        kpi_query = f"""
            SELECT
                d.nome,
                d.partido,
                d.uf,
                COUNT(f.despesa_id) as total_recibos,
                COALESCE(SUM(f.valor_liquido), 0.0) as total_gasto,
                COALESCE(AVG(f.valor_liquido), 0.0) as ticket_medio,
                COALESCE(MAX(f.valor_liquido), 0.0) as maior_gasto
            FROM fact_despesas f
            JOIN dim_deputado d ON f.deputado_key = d.deputado_key
            WHERE UPPER(d.nome) = UPPER('{dep_clean}') {where_ano}
            GROUP BY d.nome, d.partido, d.uf
        """

        suppliers_query = f"""
            SELECT
                forn.nome_fornecedor,
                forn.cnpj_cpf,
                COUNT(f.despesa_id) as qtd_recibos,
                SUM(f.valor_liquido) as total_gasto
            FROM fact_despesas f
            JOIN dim_deputado d ON f.deputado_key = d.deputado_key
            JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
            WHERE UPPER(d.nome) = UPPER('{dep_clean}') {where_ano}
            GROUP BY forn.nome_fornecedor, forn.cnpj_cpf
            ORDER BY total_gasto DESC
            LIMIT 10
        """

        categories_query = f"""
            SELECT
                c.tipo_despesa,
                c.macro_categoria,
                COUNT(f.despesa_id) as qtd_recibos,
                SUM(f.valor_liquido) as total_gasto
            FROM fact_despesas f
            JOIN dim_deputado d ON f.deputado_key = d.deputado_key
            JOIN dim_categoria c ON f.categoria_key = c.categoria_key
            WHERE UPPER(d.nome) = UPPER('{dep_clean}') {where_ano}
            GROUP BY c.tipo_despesa, c.macro_categoria
            ORDER BY total_gasto DESC
            LIMIT 10
        """

        recent_expenses_query = f"""
            SELECT
                f.despesa_id,
                c.tipo_despesa AS categoria,
                forn.nome_fornecedor,
                forn.cnpj_cpf,
                f.data_documento,
                f.valor_liquido,
                f.num_documento,
                f.is_fim_de_semana
            FROM fact_despesas f
            JOIN dim_deputado d ON f.deputado_key = d.deputado_key
            JOIN dim_fornecedor forn ON f.fornecedor_key = forn.fornecedor_key
            JOIN dim_categoria c ON f.categoria_key = c.categoria_key
            WHERE UPPER(d.nome) = UPPER('{dep_clean}') {where_ano}
            ORDER BY f.valor_liquido DESC, f.data_documento DESC
            LIMIT 25
        """

        kpi_df = self.query_df(kpi_query)
        sup_df = self.query_df(suppliers_query)
        cat_df = self.query_df(categories_query)
        rec_df = self.query_df(recent_expenses_query)

        if kpi_df.empty:
            return {}

        kpi = kpi_df.iloc[0].to_dict()
        return {
            "deputado": {
                "nome": str(kpi.get("nome", "")),
                "partido": str(kpi.get("partido", "")),
                "uf": str(kpi.get("uf", "")),
                "total_gasto": float(kpi.get("total_gasto", 0.0)),
                "total_recibos": int(kpi.get("total_recibos", 0)),
                "ticket_medio": float(kpi.get("ticket_medio", 0.0)),
                "maior_gasto": float(kpi.get("maior_gasto", 0.0)),
            },
            "top_fornecedores": sup_df.to_dict(orient="records") if not sup_df.empty else [],
            "top_categorias": cat_df.to_dict(orient="records") if not cat_df.empty else [],
            "despesas_recentes": rec_df.to_dict(orient="records") if not rec_df.empty else [],
        }
