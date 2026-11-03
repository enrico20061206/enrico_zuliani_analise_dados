
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURAÇÕES E CARREGAMENTO DOS DADOS
# ============================================================

BANCO = "Banco de Dados Final.db"
DATA_REF = pd.Timestamp("2024-03-28")
DATA_INICIAL = "2024-03-29"
DATA_FINAL = "2026-07-01"

N_ACOES = 10
LIQUIDEZ_MOMENTUM_VALUE = 1_000_000
LIQUIDEZ_SIZE = 5_000_000

with sqlite3.connect(BANCO) as con:

    precos = pd.read_sql("""
        SELECT ticker, data, adjustedClose
        FROM precos_historicos
    """, con)

    empresas = pd.read_sql("""
        SELECT ticker, cd_cvm, empresa, volume_medio_60d
        FROM base_consolidada_factor_investing_mar2024
    """, con)

    base_value = pd.read_sql("""
        SELECT *
        FROM base_value_size_mar2024
    """, con)

precos["data"] = pd.to_datetime(precos["data"])

for base in [precos, empresas, base_value]:
    base["ticker"] = (
        base["ticker"].astype(str).str.strip().str.upper()
    )

for base in [empresas, base_value]:
    base["cd_cvm"] = pd.to_numeric(
        base["cd_cvm"], errors="coerce"
    ).astype("Int64")

empresas["volume_medio_60d"] = pd.to_numeric(
    empresas["volume_medio_60d"], errors="coerce"
)

base_value["valor_mercado"] = pd.to_numeric(
    base_value["valor_mercado"], errors="coerce"
)

base_value["earnings_to_price"] = pd.to_numeric(
    base_value["earnings_to_price"], errors="coerce"
)

# Universos de liquidez
empresas_mv = empresas[
    empresas["volume_medio_60d"] >= LIQUIDEZ_MOMENTUM_VALUE
].copy()

empresas_size = empresas[
    empresas["volume_medio_60d"] >= LIQUIDEZ_SIZE
].copy()


# ============================================================
# 2. FUNÇÃO PARA ESCOLHER A CLASSE MAIS LÍQUIDA
# ============================================================

def selecionar_classe_liquida(dados):
    return (
        dados
        .sort_values(
            ["volume_medio_60d", "ticker"],
            ascending=[False, True]
        )
        .drop_duplicates("cd_cvm")
    )


# ============================================================
# 3. FATOR MOMENTUM (12-2 MESES)
# ============================================================

def calcular_momentum(dados):

    dados = dados.sort_values("data")

    inicio = dados[
        dados["data"] <= DATA_REF - pd.DateOffset(months=12)
    ].tail(1)

    fim = dados[
        dados["data"] <= DATA_REF - pd.DateOffset(months=1)
    ].tail(1)

    if inicio.empty or fim.empty:
        return None

    p0 = inicio["adjustedClose"].iloc[0]
    p1 = fim["adjustedClose"].iloc[0]

    if pd.isna(p0) or pd.isna(p1) or p0 <= 0 or p1 <= 0:
        return None

    return p1 / p0 - 1


resultados = []

for ticker, dados in precos.groupby("ticker"):

    momentum = calcular_momentum(dados)

    if momentum is not None:
        resultados.append({
            "ticker": ticker,
            "momentum": momentum
        })

carteira_momentum = (
    pd.DataFrame(resultados)
    .merge(empresas_mv, on="ticker")
    .pipe(selecionar_classe_liquida)
    .sort_values("momentum", ascending=False)
    .head(N_ACOES)
    .reset_index(drop=True)
)

print("\n" + "=" * 65)
print("CARTEIRA MOMENTUM — 28/03/2024")
print("=" * 65)

exibicao = carteira_momentum[
    ["ticker", "empresa", "momentum"]
].copy()

exibicao["momentum"] *= 100
exibicao.index += 1

print(
    exibicao.rename(columns={"momentum": "Momentum (%)"})
    .to_string(float_format=lambda x: f"{x:.2f}")
)


# ============================================================
# 4. FATOR VALUE (EARNINGS-TO-PRICE)
# ============================================================

value = base_value[
    (base_value["earnings_to_price"] > 0) &
    (base_value["valor_mercado"] > 0) &
    (base_value["cd_cvm"].notna())
].copy()

value = (
    value
    .merge(
        empresas_mv[["ticker", "volume_medio_60d"]],
        on="ticker"
    )
    .pipe(selecionar_classe_liquida)
)

# Excluir empresas já selecionadas em Momentum
value = value[
    ~value["cd_cvm"].isin(carteira_momentum["cd_cvm"]) &
    ~value["ticker"].isin(carteira_momentum["ticker"])
].copy()

carteira_value = (
    value
    .sort_values("earnings_to_price", ascending=False)
    .head(N_ACOES)
    .reset_index(drop=True)
)

carteira_value["P/L"] = (
    1 / carteira_value["earnings_to_price"]
)

carteira_value["E/P (%)"] = (
    carteira_value["earnings_to_price"] * 100
)

print("\n" + "=" * 65)
print("CARTEIRA VALUE — 10 EMPRESAS EXCLUSIVAS")
print("=" * 65)

exibicao = carteira_value[
    ["ticker", "empresa", "E/P (%)", "P/L"]
].copy()

exibicao.index += 1

print(
    exibicao.to_string(float_format=lambda x: f"{x:.2f}")
)


# ============================================================
# 5. FATOR SIZE — MID CAPS
# ============================================================

size = base_value[
    (base_value["valor_mercado"] > 0) &
    (base_value["cd_cvm"].notna())
].copy()

size = (
    size
    .merge(
        empresas_size[["ticker", "volume_medio_60d"]],
        on="ticker"
    )
    .pipe(selecionar_classe_liquida)
)

# Calcular a mediana ANTES das exclusões
mediana_mercado = size["valor_mercado"].median()

size["distancia_mediana"] = np.abs(
    np.log(size["valor_mercado"] / mediana_mercado)
)

# Excluir empresas já selecionadas em Momentum e Value
codigos_excluidos = pd.concat([
    carteira_momentum["cd_cvm"],
    carteira_value["cd_cvm"]
])

tickers_excluidos = pd.concat([
    carteira_momentum["ticker"],
    carteira_value["ticker"]
])

size_disponivel = size[
    ~size["cd_cvm"].isin(codigos_excluidos) &
    ~size["ticker"].isin(tickers_excluidos)
].copy()

carteira_size = (
    size_disponivel
    .sort_values(
        ["distancia_mediana", "ticker"],
        ascending=[True, True]
    )
    .head(N_ACOES)
    .reset_index(drop=True)
)

print("\n" + "=" * 65)
print("CARTEIRA SIZE — MID CAPS — SEM REPETIÇÕES")
print("=" * 65)

print(f"Liquidez mínima: R$ {LIQUIDEZ_SIZE / 1e6:.0f} milhões/dia")
print(f"Mediana: R$ {mediana_mercado / 1e9:.2f} bilhões")

exibicao = carteira_size[
    ["ticker", "empresa", "valor_mercado", "volume_medio_60d"]
].copy()

exibicao["valor_mercado"] /= 1_000_000
exibicao["volume_medio_60d"] /= 1_000_000

exibicao = exibicao.rename(columns={
    "valor_mercado": "Valor mercado (R$ mi)",
    "volume_medio_60d": "Liquidez diária (R$ mi)"
})

exibicao.index += 1

print(
    exibicao.to_string(float_format=lambda x: f"{x:,.2f}")
)


# ============================================================
# 6. CONSTRUIR CARTEIRA MULTIFATORIAL
# ============================================================

carteira_momentum["fator"] = "Momentum"
carteira_value["fator"] = "Value"
carteira_size["fator"] = "Size"

acoes = pd.concat([
    carteira_momentum[["ticker", "cd_cvm", "empresa", "fator"]],
    carteira_value[["ticker", "cd_cvm", "empresa", "fator"]],
    carteira_size[["ticker", "cd_cvm", "empresa", "fator"]]
], ignore_index=True)

if (
    len(acoes) != 30 or
    acoes["ticker"].duplicated().any() or
    acoes["cd_cvm"].isna().any() or
    acoes["cd_cvm"].duplicated().any()
):
    raise ValueError(
        "A carteira não possui 30 empresas distintas."
    )

print("\n" + "=" * 65)
print("CARTEIRA MULTIFATORIAL — 30 EMPRESAS")
print("=" * 65)

print(
    acoes[["ticker", "empresa", "fator"]]
    .to_string(index=False)
)


# ============================================================
# 7. BAIXAR COTAÇÕES HISTÓRICAS E  CALCULAR RENTABILIDADES
# ============================================================


tickers_acoes = acoes["ticker"].add(".SA").tolist()
tickers = tickers_acoes + ["^BVSP"]

cotacoes = yf.download(
    tickers,
    start=DATA_INICIAL,
    end=DATA_FINAL,
    auto_adjust=True,
    progress=False,
    threads=False
)

fechamentos = (
    cotacoes["Close"]
    .reindex(columns=tickers)
    .sort_index()
    .dropna(subset=["^BVSP"])
    .ffill()
)

if fechamentos.empty:
    raise ValueError("Não foram encontradas cotações.")

sem_dados = fechamentos.columns[
    fechamentos.iloc[0].isna() |
    fechamentos.iloc[-1].isna()
].tolist()

if sem_dados:
    raise ValueError(
        f"Ativos sem cotações completas: {sem_dados}"
    )



# ============================================================
# 8. ANÁLISE DE DESEMPENHO DA CARTEIRA MULTIFATORIAL
# ============================================================

# ============================================================
# 8.1 RENTABILIDADE ACUMULADA
# ============================================================

retornos_acumulados = (
    fechamentos.div(fechamentos.iloc[0]) - 1
)

rentabilidades = retornos_acumulados.iloc[-1]

retorno_acoes = rentabilidades[tickers_acoes]

retorno_carteira = retorno_acoes.mean()
retorno_ibovespa = rentabilidades["^BVSP"]

diferenca = retorno_carteira - retorno_ibovespa

# Rentabilidade separada por fator
acoes["retorno"] = (
    acoes["ticker"].add(".SA").map(retorno_acoes)
)

retorno_fatores = (
    acoes.groupby("fator")["retorno"].mean()
)

# ============================================================
# 8.2 VOLATILIDADE ANUALIZADA
# ============================================================

# Carteira com pesos iguais no investimento inicial
# e sem rebalanceamento.

valor_carteira = (
    fechamentos[tickers_acoes]
    .div(fechamentos[tickers_acoes].iloc[0])
    .mean(axis=1)
)

# Retornos diários
retornos_diarios_carteira = (
    valor_carteira.pct_change().dropna()
)

retornos_diarios_ibovespa = (
    fechamentos["^BVSP"].pct_change().dropna()
)

# Volatilidade anualizada: desvio padrão diário * raiz(252)

volatilidade_carteira = (
    retornos_diarios_carteira.std(ddof=1)
    * np.sqrt(252)
)

volatilidade_ibovespa = (
    retornos_diarios_ibovespa.std(ddof=1)
    * np.sqrt(252)
)

# ============================================================
# 8.3 RENTABILIDADE ANUALIZADA
# ============================================================

anos = (
    fechamentos.index[-1] - fechamentos.index[0]
).days / 365.25

rentabilidade_anual_carteira = (
    (1 + retorno_carteira) ** (1 / anos) - 1
)

rentabilidade_anual_ibovespa = (
    (1 + retorno_ibovespa) ** (1 / anos) - 1
)

# ============================================================
# 8.4 ÍNDICE - RENTABILIDADE FINAL / VOLATILIDADE ANUALIZADA
# ============================================================

# Fórmula combinada:
# Rentabilidade acumulada / Volatilidade anualizada

sharpe_carteira = (
    retorno_carteira / volatilidade_carteira
)

sharpe_ibovespa = (
    retorno_ibovespa / volatilidade_ibovespa
)

diferenca_sharpe = (
    sharpe_carteira - sharpe_ibovespa
)

# ============================================================
# 8.5 RESULTADOS FINAIS
# ============================================================

print("\n" + "=" * 75)
print("8. RESULTADO FINAL - CARTEIRA MULTIFATORIAL X IBOVESPA")
print("=" * 75)

print(f"Data inicial: {fechamentos.index[0]:%d/%m/%Y}")
print(f"Data final:   {fechamentos.index[-1]:%d/%m/%Y}")
print(f"Quantidade de ações: {len(acoes)}")

print("\n" + "-" * 75)
print(f"{'INDICADOR':<32}{'CARTEIRA':>20}{'IBOVESPA':>20}")
print("-" * 75)

print(
    f"{'Rentabilidade acumulada':<32}"
    f"{retorno_carteira:>20.2%}"
    f"{retorno_ibovespa:>20.2%}"
)

print(
    f"{'Rentabilidade anualizada':<32}"
    f"{rentabilidade_anual_carteira:>20.2%}"
    f"{rentabilidade_anual_ibovespa:>20.2%}"
)

print(
    f"{'Volatilidade anualizada':<32}"
    f"{volatilidade_carteira:>20.2%}"
    f"{volatilidade_ibovespa:>20.2%}"
)

print(
    f"{'Indice de Sharpe (formula combinada)':<40}"
    f"{sharpe_carteira:>16.3f}"
    f"{sharpe_ibovespa:>19.3f}"
)

print("-" * 75)

print(
    f"Diferença de rentabilidade: "
    f"{diferenca * 100:+.2f} p.p."
)

print(
    f"Diferença entre os índices: "
    f"{diferenca_sharpe:+.3f}"
)

# ============================================================
# 8.6 RENTABILIDADE POR FATOR
# ============================================================

print("\nRENTABILIDADE POR FATOR")
print("-" * 75)

for fator in ["Momentum", "Value", "Size"]:
    print(
        f"{fator:<12}"
        f"{retorno_fatores[fator]:>10.2%}"
    )

print("=" * 75)




# ============================================================
# 9. ANÁLISE GRÁFICA - CARTEIRA MULTIFATORIAL X IBOVESPA
# ============================================================

# ============================================================
# 9.1 EVOLUÇÃO DA RENTABILIDADE ACUMULADA
# ============================================================

evolucao_carteira = (
    retornos_acumulados[tickers_acoes].mean(axis=1)
)

evolucao_ibovespa = retornos_acumulados["^BVSP"]

plt.figure(figsize=(12, 6))

plt.plot(
    evolucao_carteira.index,
    evolucao_carteira * 100,
    label="Multifatorial (Momentum + Value + Size)",
    linewidth=2
)

plt.plot(
    evolucao_ibovespa.index,
    evolucao_ibovespa * 100,
    label="Ibovespa",
    linewidth=2
)

plt.title(
    "Carteira Multifatorial x Ibovespa (2024-2026)"
)

plt.xlabel("Data")
plt.ylabel("Rentabilidade acumulada (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 9.2 COMPARAÇÃO DA RENTABILIDADE ACUMULADA
# ============================================================

nomes = ["Carteira Multifatorial", "Ibovespa"]

rentabilidades_grafico = [
    retorno_carteira * 100,
    retorno_ibovespa * 100
]

plt.figure(figsize=(9, 5))

barras = plt.bar(
    nomes,
    rentabilidades_grafico
)

plt.title("Rentabilidade Acumulada - 2024 a 2026")
plt.ylabel("Rentabilidade acumulada (%)")

for barra, valor in zip(barras, rentabilidades_grafico):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height(),
        f"{valor:.2f}%",
        ha="center",
        va="bottom"
    )

plt.ylim(0, max(rentabilidades_grafico) * 1.15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 9.3 COMPARAÇÃO DA VOLATILIDADE ANUALIZADA
# ============================================================

volatilidades_grafico = [
    volatilidade_carteira * 100,
    volatilidade_ibovespa * 100
]

plt.figure(figsize=(9, 5))

barras = plt.bar(
    nomes,
    volatilidades_grafico
)

plt.title("Volatilidade Anualizada - 2024 a 2026")
plt.ylabel("Volatilidade anualizada (%)")

for barra, valor in zip(barras, volatilidades_grafico):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height(),
        f"{valor:.2f}%",
        ha="center",
        va="bottom"
    )

plt.ylim(0, max(volatilidades_grafico) * 1.15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 9.4 COMPARAÇÃO DO ÍNDICE CALCULADO
# ============================================================

# Utiliza os índices calculados no Tópico 8:
# Rentabilidade acumulada / Volatilidade anualizada

indices_grafico = [
    sharpe_carteira,
    sharpe_ibovespa
]

plt.figure(figsize=(9, 5))

barras = plt.bar(
    nomes,
    indices_grafico
)

plt.title("Índice de Retorno / Volatilidade - 2024 a 2026")
plt.ylabel("Rentabilidade acumulada / Volatilidade anualizada")

for barra, valor in zip(barras, indices_grafico):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height(),
        f"{valor:.3f}",
        ha="center",
        va="bottom"
    )

plt.ylim(0, max(indices_grafico) * 1.15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()



