# 🏛️ Gastos Públicos Analytics (CEAP)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-Fast%20Analytical%20DW-FFF000.svg?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CI Pipeline](https://img.shields.io/badge/CI-GitHub_Actions-2088FF.svg?logo=github-actions&logoColor=white)](https://github.com/TassioSales/gastos-publicos-analytics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **End-to-End Analytics Pipeline, Star Schema Data Warehouse (DuckDB/Parquet), and Machine Learning Anomaly Detection on Brazilian Parliamentary Expenses (CEAP / Dados Abertos).**

---

## 📌 Visão Geral do Projeto

O **Gastos Públicos Analytics** é uma plataforma analítica robusta desenvolvida para monitorar, auditar e extrair inteligência a partir dos dados abertos da **Câmara dos Deputados** do Brasil.

O sistema processa a **Cota para o Exercício da Atividade Parlamentar (CEAP)**, estruturando milhões em gastos públicos em um modelo dimensional analítico (Star Schema) otimizado com **DuckDB** e **Parquet**, aplicando algoritmos estatísticos e **Isolation Forest (Machine Learning)** para detectar automaticamente comportamentos atípicos, fracionamentos e potenciais irregularidades.

---

## 🏗️ Arquitetura do Sistema

```mermaid
flowchart TD
    subgraph Ingestao ["1. Ingestão & ETL"]
        API["Câmara dos Deputados REST API"] --> Pipeline["CeapPipeline (Python / Polars / Pandas)"]
        Pipeline --> Clean["Data Cleansing & Normalization"]
    end

    subgraph DW ["2. Data Warehouse Dimensional (Star Schema)"]
        Clean --> DimDep["dim_deputado"]
        Clean --> DimForn["dim_fornecedor"]
        Clean --> DimCat["dim_categoria"]
        Clean --> Fact["fact_despesas (DuckDB / Parquet)"]
    end

    subgraph ML ["3. Inteligência & Auditoria"]
        Fact --> ZScore["Z-Score Estatístico (3σ)"]
        Fact --> IsoForest["Machine Learning (Isolation Forest)"]
        Fact --> Rules["Regras Heurísticas (Fim de semana / Fracionamento)"]
    end

    subgraph Delivery ["4. Camada de Entrega & Visualização"]
        DW --> FastAPI["FastAPI REST Endpoints"]
        DW --> Streamlit["Streamlit Interactive Dashboard"]
        ML --> Alerts["Painel de Auditoria & Alertas"]
    end
```

---

## 🌟 Principais Recursos

1. **Pipeline ETL Resiliente:**
   - Extração automática de deputados, partidos e recibos fiscais com paginação e rate-limiting.
   - Normalização de tipos de dados, deduplicação por hash de documento e tratamento de nulos.
   - Enriquecimento temporal com flags de fim de semana e dias úteis.

2. **Data Warehouse Dimensional (Kimball Star Schema):**
   - **`fact_despesas`**: Fato com métricas quantitativas (`valor_documento`, `valor_liquido`), granularidade por recibo fiscal.
   - **`dim_deputado`**: Dimensão com ID parlamentar, nome político, partido e UF de representação.
   - **`dim_fornecedor`**: Dimensão de emitentes com CPF/CNPJ limpo e razão social.
   - **`dim_categoria`**: Classificação granular e agrupamento macro (Transporte, Consultoria, Comunicação, etc.).
   - Processamento colunar ultra-rápido com **DuckDB** nativo e fallback automático em **SQLite**.

3. **Motor Híbrido de Detecção de Anomalias:**
   - **Z-Score por Categoria:** Identifica gastos que excedem 3 desvios padrão da média histórica de despesas similares.
   - **Machine Learning (Isolation Forest):** Detecção não-supervisionada de transações multivariadas anômalas.
   - **Detecção de Fracionamento:** Alerta sobre múltiplas notas consecutivas de mesmo valor e fornecedor no mesmo dia para desvio de limites licitatórios.
   - **Gastos de Fim de Semana:** Auditoria de despesas substanciais fora dos dias de sessão parlamentar.

4. **Dashboard Streamlit & FastAPI:**
   - KPIs executivos em tempo real (Total Gasto, Ticket Médio, Maior Despesa Única).
   - Gráficos interativos com **Plotly** (Treemaps, barras horizontais, distribuição setorial).
   - API REST documentada via OpenAPI/Swagger (`/docs`).

---

## 📊 Modelo Dimensional (Star Schema)

```
       +--------------------+
       |    dim_deputado    |
       +--------------------+
       | PK deputado_key    |
       |    id_deputado     |
       |    nome            |
       |    partido         |
       |    uf              |
       +---------+----------+
                 |
                 | 1:N
                 v
       +--------------------+          +--------------------+
       |    fact_despesas   | N:1      |   dim_fornecedor   |
       +--------------------+--------->+--------------------+
       | PK despesa_id      |          | PK fornecedor_key  |
       | FK deputado_key    |          |    cnpj_cpf        |
       | FK fornecedor_key  |          |    nome_fornecedor |
       | FK categoria_key   |          +--------------------+
       |    data_documento  |
       |    ano / mes       |          +--------------------+
       |    valor_documento | N:1      |    dim_categoria   |
       |    valor_liquido   +--------->+--------------------+
       |    is_fim_de_semana|          | PK categoria_key   |
       +--------------------+          |    tipo_despesa    |
                                       |    macro_categoria |
                                       +--------------------+
```

---

## 🚀 Como Executar

### 1. Clonar o Repositório e Instalar Dependências

```bash
git clone https://github.com/TassioSales/gastos-publicos-analytics.git
cd gastos-publicos-analytics

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar pacotes
pip install -r requirements.txt
```

### 2. Iniciar o Dashboard Interativo

```bash
streamlit run src/dashboard.py
```
Acesse no navegador: `http://localhost:8501`

### 3. Iniciar a API REST (FastAPI)

```bash
uvicorn src.api:app --reload --port 8000
```
Swagger UI disponível em: `http://localhost:8000/docs`

### 4. Executar os Testes Unitários

```bash
pytest tests/ -v
```

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Processamento de Dados:** Pandas, Polars, DuckDB, NumPy, PyArrow
- **Machine Learning & Estatística:** Scikit-Learn (Isolation Forest), SciPy
- **APIs & Web:** FastAPI, Uvicorn, Requests, Pydantic
- **Visualização de Dados:** Streamlit, Plotly Express
- **Qualidade & CI/CD:** Pytest, Ruff, GitHub Actions

---

## 👤 Autor

**Tássio Lucian de Jesus Sales**  
*Cientista & Analista de Dados | Analytics Engineer*  
📍 Brasília - DF, Brasil  
🔗 [LinkedIn](https://www.linkedin.com/in/tassio-sales/) | 🐙 [GitHub](https://github.com/TassioSales)

---

## 📄 Licença

Este projeto é distribuído sob a licença **MIT**. Consulte o arquivo `LICENSE` para mais detalhes.
