"""
pricing_dados.py — dados e contas do Dashboard Pricing da NOSSA Energia
======================================================================
Lê o conjunto que o `pricing_dados.py` do projeto «Compra OMIE» gera (a carteira
com o tarifário e os preços de cada CPE, os pesos dos períodos e o OMIE pesado
pelos perfis da ERSE, os pressupostos de omissão do simulador) e faz as contas
que a tab mostra, incluindo o simulador de tarifários (cálculo MAX e FLEX).

Duas origens, pela ordem (como no Dashboard Trading, e com a mesma palavra-passe):
  pricing/dados/*.csv   em claro — só existe no computador de quem gera os dados
  pricing/pub/*.bin     cifrado — é o que vai para o git e para o servidor

Convenções:
  - preços de energia em €/kWh (como nos tarifários); pressupostos em €/MWh;
  - os períodos são P (ponta), C (cheias), VN (vazio normal) e SV (super vazio);
    o tri-horário tem o vazio em VN e SV, o bi-horário o fora de vazio em P e C;
  - o preço médio de um CPE é a média dos preços dele pesada pela parte do
    consumo do perfil da ERSE em cada período;
  - médias da carteira pesadas pelo consumo anual (MWh).
Sem dependências do Streamlit: testa-se sozinho (scripts/teste_pricing.py).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

import trading_dados as TD

RAIZ = Path(__file__).resolve().parent.parent / "pricing"
PERIODOS = ["P", "C", "VN", "SV"]
NOME_PERIODO = {"P": "Ponta", "C": "Cheias", "VN": "Vazio normal", "SV": "Super vazio"}
# Como cada horário arruma os quatro períodos nos preços que o cliente vê (o que está na fatura)
PRECOS_DO_HORARIO = {
    "SIMPLES": [("Simples", ["P", "C", "VN", "SV"])],
    "BI": [("Fora de vazio", ["P", "C"]), ("Vazio", ["VN", "SV"])],
    "TRI": [("Ponta", ["P"]), ("Cheias", ["C"]), ("Vazio", ["VN", "SV"])],
    "TETRA": [("Ponta", ["P"]), ("Cheias", ["C"]), ("Vazio normal", ["VN"]), ("Super vazio", ["SV"])],
}
NIVEIS = ["BTN", "BTE", "MT", "AT"]
_DATAS = ["ativacao", "desativacao", "desativacao_prevista", "fim_contrato", "renovacao_contrato"]


class PalavraErrada(TD.PalavraErrada):
    """A palavra-passe não abre os dados cifrados."""


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def origem(raiz: Path = RAIZ) -> str | None:
    """'claro', 'cifrado' ou None (ainda não há dados)."""
    if (raiz / "dados" / "cpes.csv").exists():
        return "claro"
    if any((raiz / "pub").glob("*.bin")):
        return "cifrado"
    return None


def assinatura(raiz: Path = RAIZ) -> str:
    pasta = raiz / ("dados" if origem(raiz) == "claro" else "pub")
    return "|".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in sorted(pasta.glob("*")))


def ler_claro(raiz: Path = RAIZ) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in (raiz / "dados").glob("*") if p.is_file()}


def ler_cifrado(palavra: str, raiz: Path = RAIZ) -> dict[str, str]:
    chaves: dict = {}
    saida: dict[str, str] = {}
    try:
        for caminho in sorted((raiz / "pub").glob("*.bin")):
            saida.update(TD._decifrar(caminho.read_bytes(), palavra, chaves))
    except TD.PalavraErrada:
        raise PalavraErrada() from None
    return saida


def quadros(textos: dict[str, str]) -> dict:
    """Os textos -> DataFrames."""
    def csv(nome, **kw):
        t = textos.get(nome, "")
        return pd.read_csv(io.StringIO(t), **kw) if t.strip() else pd.DataFrame()

    cpes = csv("cpes.csv", dtype={"cae": "string", "rating": "string", "versao_tarifa": "string",
                                  "cliente": "string", "tarifa": "string", "em_mudanca_para": "string"})
    if not cpes.empty:
        for c in _DATAS:
            if c in cpes:
                cpes[c] = pd.to_datetime(cpes[c], errors="coerce")
        for c in ("agente", "familia", "calculo", "inclui_redes", "estado", "nivel", "horario", "tipo", "segmento",
                  "preco_especial", "modalidade", "perfil", "tipo_ciclo", "k_fonte", "concelho"):
            if c in cpes:
                cpes[c] = cpes[c].fillna("").astype(str)
    hist = csv("historico.csv", parse_dates=["dia", "lista"])
    return {
        "cpes": cpes,
        "pesos": csv("pesos.csv"),
        "omie": csv("omie_perfis.csv", parse_dates=["de", "ate"]),
        "historico": hist,
        "parametros": json.loads(textos.get("parametros.json", "{}") or "{}"),
        "meta": json.loads(textos.get("meta.json", "{}") or "{}"),
    }


# ---------------------------------------------------------------------------
# Contas da carteira
# ---------------------------------------------------------------------------
def media_ponderada(v: pd.Series, w: pd.Series) -> float:
    """Média de v pesada por w, só onde os dois existem e w > 0; NaN sem nenhum."""
    m = v.notna() & w.notna() & (w > 0)
    return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")


def _com_pesos(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas auxiliares para as médias pesadas pelo consumo se fazerem com somas (sem um ciclo por grupo)."""
    x = df.copy()
    w = x["consumo_mwh"]
    for v in ("preco_medio", "k"):
        m = x[v].notna() & w.notna() & (w > 0)
        x[f"_{v}_n"] = (x[v] * w).where(m, 0.0)
        x[f"_{v}_d"] = w.where(m, 0.0)
    x["_redes"] = (x["inclui_redes"] == "Sim").astype(float)
    x["_agente"] = x["agente"].replace("", np.nan)
    x["_indexado"] = w.where(x["tipo"] == "Indexada", 0.0).fillna(0.0)
    x["_especial"] = (x["preco_especial"] == "Sim").astype(float)
    return x


def _razao(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den > 0)


def _agrupar(df: pd.DataFrame, chave) -> pd.DataFrame:
    """CPEs, clientes, consumo e preços (médias pesadas pelo consumo) por grupo."""
    if df.empty:
        return pd.DataFrame()
    chaves = chave if isinstance(chave, list) else [chave]
    x = _com_pesos(df)
    out = x.groupby(chaves, dropna=False, sort=False).agg(
        n_cpes=("cpe", "size"), n_clientes=("cliente_id", "nunique"), consumo_mwh=("consumo_mwh", "sum"),
        preco_min=("preco_medio", "min"), preco_max=("preco_medio", "max"), n_agentes=("_agente", "nunique"),
        pct_redes=("_redes", "mean"), _pn=("_preco_medio_n", "sum"), _pd=("_preco_medio_d", "sum"),
        _kn=("_k_n", "sum"), _kd=("_k_d", "sum"), _ind=("_indexado", "sum")).reset_index()
    out["preco_medio"] = _razao(out["_pn"], out["_pd"])
    out["k_medio"] = _razao(out["_kn"], out["_kd"])
    out["pct_redes"] *= 100
    out["pct_indexado"] = 100 * _razao(out["_ind"], out["consumo_mwh"])
    out = out.drop(columns=["_pn", "_pd", "_kn", "_kd", "_ind"])
    tot = out["consumo_mwh"].sum()
    out["quota_consumo"] = 100 * out["consumo_mwh"] / tot if tot else np.nan
    return out.sort_values(["consumo_mwh", "n_cpes"], ascending=False, na_position="last").reset_index(drop=True)


def por_tarifario(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por tarifário (com a família, o tipo, o cálculo e se inclui redes)."""
    return _agrupar(df, ["tarifa", "familia", "tipo", "calculo", "inclui_redes"])


def por_familia(df: pd.DataFrame) -> pd.DataFrame:
    return _agrupar(df, "familia")


def por_agente(df: pd.DataFrame) -> pd.DataFrame:
    return _agrupar(df, "agente")


def _juntar(df: pd.DataFrame, col: str) -> pd.Series:
    """Os valores diferentes de `col` de cada cliente, por ordem, num texto («a · b»)."""
    x = df[["cliente_id", col]].dropna()
    x = x.assign(**{col: x[col].astype(str)})
    x = x[x[col] != ""].drop_duplicates().sort_values(col)
    return x.groupby("cliente_id")[col].agg(" · ".join)


def clientes(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por cliente: CPEs, consumo, tarifários, agentes, preço e K médios, o fim de contrato mais próximo."""
    if df.empty:
        return pd.DataFrame()
    x = _com_pesos(df)
    g = x.groupby("cliente_id")
    out = g.agg(cliente=("cliente", "first"), n_cpes=("cpe", "size"), consumo_mwh=("consumo_mwh", "sum"),
                fim_contrato=("fim_contrato", "min"), _pn=("_preco_medio_n", "sum"), _pd=("_preco_medio_d", "sum"),
                _kn=("_k_n", "sum"), _kd=("_k_d", "sum"), _esp=("_especial", "max"))
    for col, nome in (("segmento", "segmento"), ("nivel", "niveis"), ("tarifa", "tarifarios"),
                      ("familia", "familias"), ("agente", "agentes"), ("risco", "risco")):
        out[nome] = _juntar(x, col) if col in x else ""
        out[nome] = out[nome].fillna("")
    out["preco_medio"] = _razao(out["_pn"], out["_pd"])
    out["k_medio"] = _razao(out["_kn"], out["_kd"])
    out["especial"] = np.where(out["_esp"] > 0, "Sim", "")
    out = out.drop(columns=["_pn", "_pd", "_kn", "_kd", "_esp"])
    out = out.reset_index().sort_values("consumo_mwh", ascending=False, na_position="last").reset_index(drop=True)
    tot = out["consumo_mwh"].sum()
    out["quota_consumo"] = 100 * out["consumo_mwh"] / tot if tot else np.nan
    out["quota_acumulada"] = out["quota_consumo"].cumsum()
    return out


def extremos(df: pd.DataFrame, n: int = 15, coluna: str = "preco_medio") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Os n CPEs com o preço mais alto e os n com o mais baixo (só os que o têm)."""
    x = df[df[coluna].notna()]
    return (x.sort_values(coluna, ascending=False).head(n).reset_index(drop=True),
            x.sort_values(coluna, ascending=True).head(n).reset_index(drop=True))


def a_terminar(df: pd.DataFrame, hoje: pd.Timestamp, dias: int) -> pd.DataFrame:
    """Os CPEs cujo contrato acaba nos próximos `dias` dias."""
    fim = df["fim_contrato"]
    m = fim.notna() & (fim >= hoje) & (fim <= hoje + pd.Timedelta(days=dias))
    return df[m].sort_values("fim_contrato").reset_index(drop=True)


def equivalente_indexado(df: pd.DataFrame, omie: pd.DataFrame, parametros: dict) -> pd.Series:
    """Preço equivalente (€/kWh) das tarifas indexadas, com a fórmula delas — (OMIE + CGS + K) × perdas — e o OMIE
    do perfil de cada CPE (o médio, pesado pelo perfil, nos dias que os dados têm). Uma estimativa: o que cada uma
    teria custado com o OMIE desses dias."""
    if df.empty or omie.empty:
        return pd.Series(np.nan, index=df.index)
    medio = omie.set_index(["perfil", "tipo"])["medio"]
    o = pd.Series([medio.get((p, t), np.nan) for p, t in zip(df["perfil"], df["tipo_ciclo"])], index=df.index)
    cgs = df["nivel"].map(parametros.get("cgs", {})).astype(float)
    perdas = df["nivel"].map(parametros.get("perdas", {})).astype(float)
    eq = (o + cgs + df["k"]) * perdas / 1000
    return eq.where(df["tipo"] == "Indexada")


def conclusoes(df: pd.DataFrame, todos: pd.DataFrame, meta: dict, hoje: pd.Timestamp) -> list[tuple[str, str]]:
    """Frases tiradas dos dados: (nível, texto), com nível em 'alerta', 'atencao', 'bom' ou 'info'."""
    out: list[tuple[str, str]] = []
    if df.empty:
        return [("info", "Nenhum CPE com estes filtros.")]
    tot = df["consumo_mwh"].sum()
    t = por_tarifario(df)
    if not t.empty:
        mais_cpes = t.sort_values("n_cpes", ascending=False).iloc[0]
        mais_mwh = t.iloc[0]
        out.append(("info", f"O tarifário com mais CPEs é «{mais_cpes['tarifa']}» ({n(mais_cpes['n_cpes'])} CPEs); "
                            f"o de mais consumo, «{mais_mwh['tarifa']}» ({n(mais_mwh['quota_consumo'], 1)} % do "
                            f"consumo, {n(mais_mwh['n_cpes'])} CPEs)."))
    c = clientes(df)
    if len(c) >= 10:
        q10 = c["quota_consumo"].head(10).sum()
        out.append(("atencao" if q10 >= 50 else "info",
                    f"Os 10 maiores clientes têm {n(q10, 1)} % do consumo da seleção "
                    f"({n(c['consumo_mwh'].head(10).sum())} de {n(tot)} MWh/ano)."))
    ind = df.loc[df["tipo"] == "Indexada", "consumo_mwh"].sum()
    if tot:
        out.append(("info", f"{n(100 * ind / tot, 1)} % do consumo está em tarifas indexadas (o risco de preço é do "
                            f"cliente); {n(100 * (1 - ind / tot), 1)} % em fixas (o risco é nosso)."))
    f90 = a_terminar(df, hoje, 90)
    if len(f90):
        out.append(("atencao", f"{cpes(len(f90))} ({n(f90['consumo_mwh'].sum())} MWh/ano) "
                               f"{'acaba' if len(f90) == 1 else 'acabam'} o contrato nos próximos 90 dias."))
    mud = df[df["em_mudanca_para"].fillna("").str.strip() != ""]
    if len(mud):
        out.append(("alerta", f"{cpes(len(mud))} ({n(mud['consumo_mwh'].sum(), 1)} MWh/ano) "
                              f"{'está' if len(mud) == 1 else 'estão'} em mudança para outro comercializador."))
    sem_k = df[(df["tipo"] == "Fixa") & df["k"].isna()]
    if len(sem_k):
        out.append(("info", f"{cpes(len(sem_k))} de tarifas fixas sem K na Lista de Preços (nem no «Escalão», nem «K<n>» "
                            f"no K fixo)."))
    lista = meta.get("lista", {}).get("data")
    if lista:
        idade = (hoje - pd.Timestamp(lista)).days
        if idade > 21:
            out.append(("atencao", f"Os preços são os da Lista de Preços de {pd.Timestamp(lista):%d/%m/%Y} "
                                   f"({idade} dias): o que mudou desde então não está cá."))
    return out


# ---------------------------------------------------------------------------
# Simulador de tarifários
# ---------------------------------------------------------------------------
def omie_do_perfil(omie: pd.DataFrame, perfil: str, tipo: str) -> dict | None:
    """O OMIE histórico pesado pelo perfil em cada período, a média do perfil e a do baseload."""
    if omie.empty:
        return None
    x = omie[(omie["perfil"] == perfil) & (omie["tipo"] == tipo)]
    if x.empty:
        return None
    r = x.iloc[0]
    return {**{p: float(r[p]) for p in PERIODOS}, "medio": float(r["medio"]), "base": float(r["base"]),
            "de": r["de"], "ate": r["ate"]}


def pesos_do_perfil(pesos: pd.DataFrame, perfil: str, tipo: str) -> dict | None:
    if pesos.empty:
        return None
    x = pesos[(pesos["perfil"] == perfil) & (pesos["tipo"] == tipo)]
    return {p: float(x.iloc[0][p]) for p in PERIODOS} if not x.empty else None


def omie_por_referencia(historico: dict, referencia: float) -> dict:
    """O OMIE de cada período para uma referência de baseload (um contrato OMIP, ou um valor à mão): a forma do perfil
    (o período histórico ÷ o baseload histórico) vezes a referência."""
    return {p: referencia * historico[p] / historico["base"] for p in PERIODOS}


def simular(calculo: str, omie: dict, pesos: dict, cgs: float, custos: float, k_interno: float, k_agente: float,
            perdas: float, tar: dict | None = None) -> pd.DataFrame:
    """O preço de energia de cada período (€/kWh) e as parcelas dele (€/MWh), com a fórmula do cálculo:

        MAX:  (OMIE + custos indiretos + K interno + K do agente) / 1000
        FLEX: (OMIE + CGS + custos indiretos + K interno + K do agente) × perdas / 1000 + TAR

    omie e pesos: {período: valor}; tar: {período: €/kWh} (só no FLEX). As parcelas somam o preço: no FLEX, o efeito
    das perdas é uma parcela à parte, (OMIE + CGS + custos + K) × (perdas − 1)."""
    if calculo not in ("MAX", "FLEX"):
        raise ValueError(f"cálculo desconhecido: {calculo!r}")
    flex = calculo == "FLEX"
    tar = tar or {}
    linhas = []
    for p in PERIODOS:
        base = omie[p] + (cgs if flex else 0.0) + custos + k_interno + k_agente
        f = perdas if flex else 1.0
        t = float(tar.get(p) or 0.0) if flex else 0.0
        linhas.append({
            "periodo": p, "nome": NOME_PERIODO[p], "peso": pesos[p],
            "omie": omie[p], "cgs": cgs if flex else 0.0, "custos": custos, "k_interno": k_interno,
            "k_agente": k_agente, "perdas": base * (f - 1), "tar": t * 1000,
            "preco": base * f / 1000 + t,
        })
    return pd.DataFrame(linhas)


def precos_do_horario(sim: pd.DataFrame, horario: str) -> pd.DataFrame:
    """Os preços que o cliente vê, pelo horário: cada um é a média dos períodos que junta, pesada pelo consumo do
    perfil (assim a fatura do ano é a mesma que com os quatro preços)."""
    s = sim.set_index("periodo")
    out = []
    for nome, pers in PRECOS_DO_HORARIO[horario]:
        w = s.loc[pers, "peso"]
        preco = float((s.loc[pers, "preco"] * w).sum() / w.sum()) if w.sum() else float(s.loc[pers, "preco"].mean())
        out.append({"preco_nome": nome, "periodos": " + ".join(NOME_PERIODO[p] for p in pers), "peso": float(w.sum()),
                    "preco": preco})
    return pd.DataFrame(out)


def resumo_simulacao(sim: pd.DataFrame, calculo: str, consumo_mwh: float, perdas: float,
                     potencia_dia: float | None = None) -> dict:
    """O preço médio (€/kWh), a fatura de energia do ano (€) e o que fica de cada K (€/ano)."""
    w = sim["peso"]
    medio = float((sim["preco"] * w).sum() / w.sum()) if w.sum() else float("nan")
    f = perdas if calculo == "FLEX" else 1.0
    energia = medio * consumo_mwh * 1000
    pot = (potencia_dia or 0.0) * 365
    parcelas = {c: float((sim[c] * w).sum() / w.sum()) for c in
                ("omie", "cgs", "custos", "k_interno", "k_agente", "perdas", "tar")} if w.sum() else {}
    return {"preco_medio": medio, "energia_ano": energia, "potencia_ano": pot, "total_ano": energia + pot,
            "margem_k_interno": float(sim["k_interno"].iloc[0]) * f * consumo_mwh,
            "comissao_agente": float(sim["k_agente"].iloc[0]) * f * consumo_mwh,
            "parcelas": parcelas}


def percentil(valor: float, referencia: pd.Series) -> float:
    """Que parte (%) da referência tem um preço abaixo deste."""
    r = referencia.dropna()
    return float(100 * (r < valor).mean()) if len(r) else float("nan")


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
def n(v, casas: int = 0, sinal: bool = False) -> str:
    return TD.n(v, casas, sinal)


def cpes(k: int) -> str:
    return f"{n(k)} CPE" if k == 1 else f"{n(k)} CPEs"
