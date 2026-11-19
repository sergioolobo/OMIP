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

import calendar
import datetime as dt
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
        "forma": csv("forma_omie.csv"),
        "sazonal": csv("sazonal.csv"),
        "sazonal_perfis": csv("sazonal_perfis.csv"),
        "forma_perfis": csv("forma_perfis.csv"),
        "perfis_cpe": csv("perfis_cpe.csv"),
        "olmc_cpes": csv("olmc_cpes.csv", dtype={"cae": "string", "tarifa_acesso": "string", "ciclo_prog": "string",
                                                 "aviso": "string", "concelho": "string"}),
        "olmc_leituras": csv("olmc_leituras.csv", parse_dates=["de", "ate"]),
        "olmc_perfis": csv("olmc_perfis.csv"),
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
            perdas: float, tar: dict | None = None, margem: float = 1.0) -> pd.DataFrame:
    """O preço de energia de cada período (€/kWh) e as parcelas dele (€/MWh), com a fórmula do cálculo:

        MAX:  (OMIE × margem + custos indiretos + K interno + K do agente) / 1000
        FLEX: (OMIE × margem + CGS + custos indiretos + K interno + K do agente) × perdas / 1000 + TAR

    omie e pesos: {período: valor}; tar: {período: €/kWh} (só no FLEX); margem: a margem sobre o OMIE das ofertas
    personalizadas (1,1 = 10 %). As parcelas somam o preço: a margem é uma parcela à parte, OMIE × (margem − 1), e no
    FLEX também o efeito das perdas, (OMIE × margem + CGS + custos + K) × (perdas − 1)."""
    if calculo not in ("MAX", "FLEX"):
        raise ValueError(f"cálculo desconhecido: {calculo!r}")
    flex = calculo == "FLEX"
    tar = tar or {}
    linhas = []
    for p in PERIODOS:
        base = omie[p] * margem + (cgs if flex else 0.0) + custos + k_interno + k_agente
        f = perdas if flex else 1.0
        t = float(tar.get(p) or 0.0) if flex else 0.0
        linhas.append({
            "periodo": p, "nome": NOME_PERIODO[p], "peso": pesos[p],
            "omie": omie[p], "margem": omie[p] * (margem - 1), "cgs": cgs if flex else 0.0, "custos": custos,
            "k_interno": k_interno, "k_agente": k_agente, "perdas": base * (f - 1), "tar": t * 1000,
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
    cols = [c for c in ("omie", "margem", "cgs", "custos", "k_interno", "k_agente", "perdas", "tar") if c in sim]
    parcelas = {c: float((sim[c] * w).sum() / w.sum()) for c in cols} if w.sum() else {}
    margem_omie = parcelas.get("margem", 0.0) * f * consumo_mwh
    k_int = float(sim["k_interno"].iloc[0]) * f * consumo_mwh
    return {"preco_medio": medio, "energia_ano": energia, "potencia_ano": pot, "total_ano": energia + pot,
            "margem_k_interno": k_int, "margem_omie": margem_omie, "margem_total": k_int + margem_omie,
            "comissao_agente": float(sim["k_agente"].iloc[0]) * f * consumo_mwh,
            "parcelas": parcelas}


def percentil(valor: float, referencia: pd.Series) -> float:
    """Que parte (%) da referência tem um preço abaixo deste."""
    r = referencia.dropna()
    return float(100 * (r < valor).mean()) if len(r) else float("nan")


def tar_omissao(tar: dict, nivel: str, horario: str, potencia_kva: float | None = None) -> dict:
    """A TAR de energia ativa (€/kWh) de cada período para o cliente: em MAT, AT, MT e BTE pelo nível; em BTN pela opção
    horária (o tetra-horário não existe em BTN: vale o tri; acima de 20,7 kVA só há tri-horário, com outra tabela).
    {} se o config não a tiver."""
    t = (tar or {}).get(nivel) or {}
    if nivel != "BTN":
        return {p: float(t.get(p, 0.0)) for p in PERIODOS} if t else {}
    if potencia_kva and potencia_kva > 20.7:
        op = "TRI>20.7"
    else:
        op = {"SIMPLES": "SIMPLES", "BI": "BI"}.get(horario, "TRI")
    x = t.get(op) or {}
    return {p: float(x.get(p, 0.0)) for p in PERIODOS} if x else {}


def tar_potencia_omissao(tar_pot: dict, nivel: str, potencia_kva: float | None) -> str:
    """A TAR de potência em texto, para referência: €/(kW.dia) de ponta e contratada; em BTN, o €/dia do escalão de
    potência contratada (o primeiro igual ou acima)."""
    t = (tar_pot or {}).get(nivel)
    if not t:
        return ""
    if nivel != "BTN":
        return f"potência em horas de ponta {t['ponta']:.4f} €/(kW.dia), contratada {t['contratada']:.4f} €/(kW.dia)"
    escaloes = sorted((float(k), v) for k, v in t.items())
    if not potencia_kva:
        return ""
    for kva, v in escaloes:
        if kva >= potencia_kva - 1e-9:
            return f"potência contratada de {kva:g} kVA: {v:.4f} €/dia"
    return ""


def modalidades(preco_dd_fe: float) -> dict:
    """Os preços das quatro modalidades a partir do de DD e FE, como nas ofertas e na Ficha de Produtos:
    Base = DD e FE ÷ 0,75; DD ou FE = Base × 0,775; MB e FP = Base × 0,8."""
    base = preco_dd_fe / 0.75
    return {"DD e FE": preco_dd_fe, "DD ou FE": base * 0.775, "MB e FP": base * 0.8, "Base": base}


# ---------------------------------------------------------------------------
# Oferta personalizada: previsão a prazo, períodos de cada dia e curva do cliente
# ---------------------------------------------------------------------------
OMIP_CSV = Path(__file__).resolve().parent / "data" / "raw" / "omip_futures_historical.csv"
TIPOS_DIA = ["util", "sab", "dom"]
NOME_TIPO = {"util": "Dias úteis", "sab": "Sábados", "dom": "Domingos e feriados"}
_MESES_EN = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10,
             "Nov": 11, "Dec": 12}


def _m(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _tabela(*trocos) -> list[str]:
    """[(início, fim, período)] -> o período de cada um dos 96 quartos de hora (sem buracos nem sobreposições)."""
    out = [None] * 96
    for a, b, p in trocos:
        for q in range(_m(a) // 15, _m(b) // 15):
            assert out[q] is None, (a, b)
            out[q] = p
    assert all(out), trocos
    return out


# Os períodos em vigor em Portugal continental (as grelhas «Vigente» do documento de enquadramento da consulta pública
# 137 da ERSE, de novembro de 2025, lidas ao meio quarto de hora; a ponta de verão do ciclo semanal nos dias úteis é a das
# 09:15 às 12:15, que a grelha desenha como 09:30–12:30 por simplificação). As grelhas não falam de feriados, mas os
# contadores de MT contam-nos como domingo nos ciclos semanais (o ciclo programado «com feriados» do OLMC; conferido a
# 18/11/2026 com as curvas e as leituras de seis MT: só assim batem) — `periodos_do_dia` fá-lo em MAT, AT e MT.
_DOM_V = _tabela(("00:00", "02:00", "VN"), ("02:00", "06:00", "SV"), ("06:00", "24:00", "VN"))
_DOM_OPC = _tabela(("00:00", "04:00", "VN"), ("04:00", "08:00", "SV"), ("08:00", "24:00", "VN"))
VIGENTE = {
    "diario": {
        "inverno": _tabela(("00:00", "02:00", "VN"), ("02:00", "06:00", "SV"), ("06:00", "08:00", "VN"),
                           ("08:00", "09:00", "C"), ("09:00", "10:30", "P"), ("10:30", "18:00", "C"),
                           ("18:00", "20:30", "P"), ("20:30", "22:00", "C"), ("22:00", "24:00", "VN")),
        "verao": _tabela(("00:00", "02:00", "VN"), ("02:00", "06:00", "SV"), ("06:00", "08:00", "VN"),
                         ("08:00", "10:30", "C"), ("10:30", "13:00", "P"), ("13:00", "19:30", "C"),
                         ("19:30", "21:00", "P"), ("21:00", "22:00", "C"), ("22:00", "24:00", "VN")),
    },
    "semanal": {
        ("util", "inverno"): _tabela(("00:00", "02:00", "VN"), ("02:00", "06:00", "SV"), ("06:00", "07:00", "VN"),
                                     ("07:00", "09:30", "C"), ("09:30", "12:00", "P"), ("12:00", "18:30", "C"),
                                     ("18:30", "21:00", "P"), ("21:00", "24:00", "C")),
        ("util", "verao"): _tabela(("00:00", "02:00", "VN"), ("02:00", "06:00", "SV"), ("06:00", "07:00", "VN"),
                                   ("07:00", "09:15", "C"), ("09:15", "12:15", "P"), ("12:15", "24:00", "C")),
        ("sab", "inverno"): _tabela(("00:00", "02:00", "VN"), ("02:00", "06:00", "SV"), ("06:00", "09:30", "VN"),
                                    ("09:30", "13:00", "C"), ("13:00", "18:30", "VN"), ("18:30", "22:00", "C"),
                                    ("22:00", "24:00", "VN")),
        ("sab", "verao"): _tabela(("00:00", "02:00", "VN"), ("02:00", "06:00", "SV"), ("06:00", "09:00", "VN"),
                                  ("09:00", "14:00", "C"), ("14:00", "20:00", "VN"), ("20:00", "22:00", "C"),
                                  ("22:00", "24:00", "VN")),
        ("dom", "inverno"): _DOM_V, ("dom", "verao"): _DOM_V,
    },
    "opcional": {
        ("util", "inverno"): _tabela(("00:00", "00:30", "C"), ("00:30", "02:00", "VN"), ("02:00", "06:00", "SV"),
                                     ("06:00", "07:30", "VN"), ("07:30", "17:00", "C"), ("17:00", "22:00", "P"),
                                     ("22:00", "24:00", "C")),
        ("util", "verao"): _tabela(("00:00", "00:30", "C"), ("00:30", "02:00", "VN"), ("02:00", "06:00", "SV"),
                                   ("06:00", "07:30", "VN"), ("07:30", "14:00", "C"), ("14:00", "17:00", "P"),
                                   ("17:00", "24:00", "C")),
        ("sab", "inverno"): _tabela(("00:00", "03:00", "VN"), ("03:00", "07:00", "SV"), ("07:00", "10:30", "VN"),
                                    ("10:30", "12:30", "C"), ("12:30", "17:30", "VN"), ("17:30", "22:30", "C"),
                                    ("22:30", "24:00", "VN")),
        ("sab", "verao"): _tabela(("00:00", "03:30", "VN"), ("03:30", "07:30", "SV"), ("07:30", "10:00", "VN"),
                                  ("10:00", "13:30", "C"), ("13:30", "19:30", "VN"), ("19:30", "23:00", "C"),
                                  ("23:00", "24:00", "VN")),
        ("dom", "inverno"): _DOM_OPC, ("dom", "verao"): _DOM_OPC,
    },
}
# Os da Diretiva ERSE n.º 3/2026 (o periodos.py do projeto Compra OMIE): MAT, AT, MT e BTE a 1/04/2027, BTN bi e
# tri-horário no 2.º semestre de 2027. O ciclo semanal opcional acaba (MAT, AT e MT passam ao semanal) e, nesses
# níveis, os feriados nacionais contam como domingo.
_DOM_N = _tabela(("00:00", "02:30", "VN"), ("02:30", "06:30", "SV"), ("06:30", "24:00", "VN"))
NOVO = {
    "diario": _tabela(("00:00", "02:30", "VN"), ("02:30", "06:30", "SV"), ("06:30", "08:30", "VN"),
                      ("08:30", "17:30", "C"), ("17:30", "21:30", "P"), ("21:30", "22:30", "C"),
                      ("22:30", "24:00", "VN")),
    "semanal": {
        ("util", "inverno"): _tabela(("00:00", "00:30", "C"), ("00:30", "02:30", "VN"), ("02:30", "06:30", "SV"),
                                     ("06:30", "07:30", "VN"), ("07:30", "17:00", "C"), ("17:00", "22:00", "P"),
                                     ("22:00", "24:00", "C")),
        ("util", "verao"): _tabela(("00:00", "00:30", "C"), ("00:30", "02:30", "VN"), ("02:30", "06:30", "SV"),
                                   ("06:30", "07:30", "VN"), ("07:30", "19:30", "C"), ("19:30", "22:30", "P"),
                                   ("22:30", "24:00", "C")),
        ("sab", "inverno"): _tabela(("00:00", "02:30", "VN"), ("02:30", "06:30", "SV"), ("06:30", "17:00", "VN"),
                                    ("17:00", "24:00", "C")),
        ("dom", "inverno"): _DOM_N, ("dom", "verao"): _DOM_N,
    },
}
NOVO["semanal"][("sab", "verao")] = NOVO["semanal"][("sab", "inverno")]
NIVEIS_SO_SEMANAL = ("MAT", "AT", "MT")


def _ultimo_domingo(ano: int, mes: int) -> dt.date:
    d = dt.date(ano, mes, 31)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)


def hora_de_verao(d: dt.date) -> bool:
    return _ultimo_domingo(d.year, 3) <= d < _ultimo_domingo(d.year, 10)


def feriados_nacionais(ano: int) -> set:
    """Os feriados obrigatórios do Código do Trabalho (art. 234.º), com a Páscoa calculada."""
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    mes, dia = divmod(h + l_ - 7 * m + 114, 31)
    p = dt.date(ano, mes, dia + 1)
    D = lambda mm, dd: dt.date(ano, mm, dd)
    return {D(1, 1), p - dt.timedelta(days=2), p, D(4, 25), D(5, 1), p + dt.timedelta(days=60), D(6, 10), D(8, 15),
            D(10, 5), D(11, 1), D(12, 1), D(12, 8), D(12, 25)}


def tipo_de_dia(d: dt.date, feriados=()) -> str:
    if d.weekday() == 6 or d in feriados:
        return "dom"
    return "sab" if d.weekday() == 5 else "util"


def periodos_do_dia(d: dt.date, nivel: str, ciclo: str, novos_desde: dt.date | None, feriados=(),
                    feriados_vigentes: bool | None = None) -> list[str]:
    """O período de cada quarto de hora de um dia, com os períodos que valem nesse dia. ciclo: 'diario', 'semanal' ou
    'opcional' (o semanal opcional de MAT, AT e MT, que acaba com os períodos novos). feriados_vigentes: se os feriados
    contam como domingo nos ciclos semanais de hoje (por omissão, em MAT, AT e MT, como os contadores fazem)."""
    est = "verao" if hora_de_verao(d) else "inverno"
    if novos_desde is not None and d >= novos_desde:
        if ciclo == "diario" and nivel not in NIVEIS_SO_SEMANAL:
            return NOVO["diario"]
        so = nivel in NIVEIS_SO_SEMANAL
        return NOVO["semanal"][(tipo_de_dia(d, feriados if so else ()), est)]
    if ciclo == "diario":
        return VIGENTE["diario"][est]
    fv = nivel in NIVEIS_SO_SEMANAL if feriados_vigentes is None else feriados_vigentes
    return VIGENTE["opcional" if ciclo == "opcional" else "semanal"][(tipo_de_dia(d, feriados if fv else ()), est)]


def cotacoes_omip(caminho: Path = OMIP_CSV, ate: dt.date | None = None) -> dict:
    """As cotações OMIP (fecho) mais recentes de cada contrato do ficheiro do projeto da previsão (atualizado todos os
    dias). -> {'data', 'mes': {1.º dia: €/MWh}, 'tri': {(ano, trimestre): €/MWh}, 'ano': {ano: €/MWh}}"""
    df = pd.read_csv(caminho, comment="#", index_col=0, parse_dates=True)
    if ate is not None:
        df = df[df.index <= pd.Timestamp(ate)]
    df = df.dropna(how="all")
    data = df.index.max()
    ult = df.loc[data].dropna()
    out = {"data": data.date(), "mes": {}, "tri": {}, "ano": {}}
    for nome, v in ult.items():
        s = str(nome).replace("omip_", "")
        if s.startswith("M "):
            mes, ano = s[2:].split("_")
            out["mes"][dt.date(2000 + int(ano), _MESES_EN[mes], 1)] = float(v)
        elif s.startswith("Q"):
            q, ano = s[1:].split("_")
            out["tri"][(2000 + int(ano), int(q))] = float(v)
        elif s.startswith("YR"):
            out["ano"][2000 + int(s.split("_")[1])] = float(v)
    return out


def _horas(d: dt.date) -> int:
    return calendar.monthrange(d.year, d.month)[1] * 24


def previsao_mensal(cot: dict, sazonal: pd.DataFrame, meses: list[dt.date]) -> tuple[pd.DataFrame, list[str]]:
    """O preço de cada mês (baseload, €/MWh), como a folha «Previsao» — o OMIP do mês; senão o do trimestre; senão o do
    ano — mas conciliado: os meses sem cotação levam o que falta para o trimestre (ou o ano) dar a cotação dele,
    repartido pelos pesos históricos (O no trimestre, K no ano), com as médias pesadas pelas horas. Assim cada trimestre
    e cada ano cotados dão exatamente a cotação, e um mês cotado nunca é contrariado.
    -> (DataFrame mes, preco, fonte, cotacao; avisos)"""
    K = dict(zip(sazonal["mes"], sazonal["k_ano"]))
    O = dict(zip(sazonal["mes"], sazonal["o_trimestre"]))
    avisos: list[str] = []
    preco, fonte = {}, {}
    for d, v in cot["mes"].items():
        preco[d], fonte[d] = v, "OMIP mês"
    for (a, q), v in sorted(cot["tri"].items()):
        ms = [dt.date(a, 3 * (q - 1) + k, 1) for k in (1, 2, 3)]
        falta = [d for d in ms if d not in preco]
        if not falta:
            continue
        resto = v * sum(_horas(d) for d in ms) - sum(preco[d] * _horas(d) for d in ms if d in preco)
        esc = resto / sum(O[d.month] * _horas(d) for d in falta)
        if esc <= 0:
            avisos.append(f"Q{q}/{a}: os meses cotados já passam a cotação do trimestre; os outros ficam com o "
                          f"trimestre × peso, sem conciliar")
            esc = v
        for d in falta:
            preco[d], fonte[d] = esc * O[d.month], f"OMIP Q{q}/{a % 100:02d}"
    for a, v in sorted(cot["ano"].items()):
        ms = [dt.date(a, k, 1) for k in range(1, 13)]
        falta = [d for d in ms if d not in preco]
        if not falta:
            continue
        resto = v * sum(_horas(d) for d in ms) - sum(preco[d] * _horas(d) for d in ms if d in preco)
        esc = resto / sum(K[d.month] * _horas(d) for d in falta)
        if esc <= 0:
            avisos.append(f"{a}: os meses e trimestres cotados já passam a cotação do ano; os outros meses ficam com o "
                          f"ano × peso, sem conciliar")
            esc = v
        for d in falta:
            preco[d], fonte[d] = esc * K[d.month], f"OMIP YR{a % 100:02d}"
    linhas = []
    for d in meses:
        if d not in preco:
            avisos.append(f"{d:%m/%Y}: sem cotação OMIP do mês, do trimestre nem do ano")
        linhas.append({"mes": d, "preco": preco.get(d, np.nan), "fonte": fonte.get(d, "—")})
    return pd.DataFrame(linhas), avisos


def conferencia_previsao(prev: pd.DataFrame, cot: dict) -> pd.DataFrame:
    """Cada trimestre e cada ano cotados, com a média da previsão (pesada pelas horas) ao lado: têm de ser iguais."""
    p = dict(zip(prev["mes"], prev["preco"]))
    out = []
    for (a, q), v in sorted(cot["tri"].items()):
        ms = [dt.date(a, 3 * (q - 1) + k, 1) for k in (1, 2, 3)]
        if all(d in p and pd.notna(p[d]) for d in ms):
            out.append({"contrato": f"Q{q}/{a}", "cotacao": v,
                        "previsao": sum(p[d] * _horas(d) for d in ms) / sum(_horas(d) for d in ms)})
    for a, v in sorted(cot["ano"].items()):
        ms = [dt.date(a, k, 1) for k in range(1, 13)]
        if all(d in p and pd.notna(p[d]) for d in ms):
            out.append({"contrato": f"YR{a}", "cotacao": v,
                        "previsao": sum(p[d] * _horas(d) for d in ms) / sum(_horas(d) for d in ms)})
    return pd.DataFrame(out)


def meses_do_contrato(inicio: dt.date, n: int) -> list[dt.date]:
    a, m = inicio.year, inicio.month
    out = []
    for _ in range(n):
        out.append(dt.date(a, m, 1))
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)
    return out


def _matriz_forma(forma: pd.DataFrame) -> dict:
    """forma (mes, tipo, q, coef) -> {(mês, tipo): array de 96}."""
    out = {}
    for (m, t), g in forma.groupby(["mes", "tipo"]):
        a = np.zeros(96)
        a[g["q"].astype(int).values] = g["coef"].values
        out[(int(m), str(t))] = a
    return out


def oferta_contrato(prev: pd.DataFrame, forma: pd.DataFrame, perfil_tipo: dict, partes_mes: dict,
                    consumo_anual_mwh: float, nivel: str, ciclo: str, novos_desde: dt.date | None,
                    perfil_mes: dict | None = None) -> dict:
    """A curva de preços e de consumos de cada quarto de hora do contrato e o que dá em cada período.

    prev: previsao_mensal dos meses do contrato; forma: o preço de cada quarto de hora ÷ a média do mês, por mês e tipo
    de dia; perfil_tipo: {tipo: 96 kW} (a curva média do cliente em cada tipo de dia); partes_mes: {mês: parte da
    energia do ano}; novos_desde: desde quando valem os períodos de 2027; perfil_mes: {mês: {tipo: 96 kW}}, a curva de
    cada mês do ano (a das leituras do OLMC), que vale sobre perfil_tipo nos meses que tiver.
    Em cada mês, o preço de cada quarto de hora é a forma × o preço do mês, escalada para a média do mês dar o preço do
    mês; o consumo do mês (consumo anual × parte do mês) reparte-se pelos dias e quartos de hora pela curva do cliente.
    -> {'periodos': DataFrame (periodo, mwh, custo, omie, parte), 'meses': DataFrame, 'omie': {P..SV}, 'pesos': {P..SV},
        'base', 'pesado'}"""
    fm = _matriz_forma(forma)
    soma_partes = sum(partes_mes.values()) or 1.0
    per_e = dict.fromkeys(PERIODOS, 0.0)
    per_c = dict.fromkeys(PERIODOS, 0.0)
    linhas_mes = []
    fer = set()
    for d in prev["mes"]:
        fer |= feriados_nacionais(d.year)
    for d, ag, fonte in zip(prev["mes"], prev["preco"], prev["fonte"]):
        dias = [d + dt.timedelta(days=k) for k in range(calendar.monthrange(d.year, d.month)[1])]
        tipos = [tipo_de_dia(x, fer) for x in dias]
        precos = np.array([fm[(d.month, t)] for t in tipos])                  # dias × 96
        precos = precos * (ag / precos.mean())
        pm = (perfil_mes or {}).get(d.month) or perfil_tipo
        w = np.array([pm[t] for t in tipos], dtype=float)
        e_mes = consumo_anual_mwh * partes_mes.get(d.month, 0.0) / soma_partes
        cons = w / w.sum() * e_mes if w.sum() > 0 else np.zeros_like(w)
        per = np.array([periodos_do_dia(x, nivel, ciclo, novos_desde, fer) for x in dias])
        custo = cons * precos
        for p in PERIODOS:
            msk = per == p
            per_e[p] += float(cons[msk].sum())
            per_c[p] += float(custo[msk].sum())
        linhas_mes.append({"mes": d, "fonte": fonte, "preco_base": ag, "mwh": float(cons.sum()),
                           "custo": float(custo.sum()),
                           "preco_pesado": float(custo.sum() / cons.sum()) if cons.sum() else np.nan,
                           "periodos": "2027" if novos_desde is not None and dias[-1] >= novos_desde else "atuais"})
    tot = sum(per_e.values())
    tabela = pd.DataFrame([{"periodo": p, "nome": NOME_PERIODO[p], "mwh": per_e[p], "custo": per_c[p],
                            "omie": per_c[p] / per_e[p] if per_e[p] else np.nan,
                            "parte": per_e[p] / tot if tot else np.nan} for p in PERIODOS])
    meses = pd.DataFrame(linhas_mes)
    omie = {p: (per_c[p] / per_e[p] if per_e[p] else float(meses["preco_base"].mean())) for p in PERIODOS}
    return {"periodos": tabela, "meses": meses, "omie": omie,
            "pesos": {p: (per_e[p] / tot if tot else 0.25) for p in PERIODOS},
            "base": float((meses["preco_base"] * meses["mes"].map(_horas)).sum() / meses["mes"].map(_horas).sum()),
            "pesado": float(sum(per_c.values()) / tot) if tot else np.nan}


def perfil_de_tabela(tab: pd.DataFrame, chave: str, valor: str, coluna: str = "kw") -> dict | None:
    """{tipo: 96 valores} de uma tabela (perfis_cpe ou forma_perfis) para um CPE ou um perfil; None se faltar algum
    tipo de dia."""
    if tab.empty:
        return None
    x = tab[tab[chave] == valor]
    out = {}
    for t in TIPOS_DIA:
        g = x[x["tipo"] == t]
        if len(g) != 96:
            return None
        a = np.zeros(96)
        a[g["q"].astype(int).values] = g[coluna].astype(float).values
        out[t] = a
    return out


def partes_do_perfil(sazonal_perfis: pd.DataFrame, perfil: str) -> dict:
    x = sazonal_perfis[sazonal_perfis["perfil"] == perfil]
    return dict(zip(x["mes"].astype(int), x["parte"].astype(float))) if len(x) == 12 else {m: 1 / 12 for m in range(1, 13)}


# ---------------------------------------------------------------------------
# Clientes novos: as leituras do OLMC
# ---------------------------------------------------------------------------
REGISTOS_OLMC = {"P": ("P",), "C": ("C",), "VN": ("VN",), "SV": ("SV",), "V": ("VN", "SV"), "FV": ("P", "C"),
                 "S": ("P", "C", "VN", "SV")}


def _grupos(linha) -> list[tuple[tuple, float]]:
    """Os registadores de um intervalo -> [(períodos, kWh)]. Um tetra dá os quatro; o tri, P, C e o vazio (VN e SV
    juntos); o bi, fora de vazio (P e C) e vazio; o simples, tudo junto. Registadores que não cubram os quatro períodos
    sem se sobreporem valem como um total só."""
    regs = {r: float(linha[r]) for r in REGISTOS_OLMC if r in linha and pd.notna(linha[r])}
    for escolha in (("P", "C", "VN", "SV"), ("P", "C", "V"), ("FV", "V"), ("S",)):
        if all(r in regs for r in escolha):
            return [(REGISTOS_OLMC[r], regs[r]) for r in escolha]
    return [(tuple(PERIODOS), sum(regs.values()))] if regs else []


def perfil_das_leituras(base: dict, leituras: pd.DataFrame, nivel: str, ciclo: str, feriados: bool,
                        dias_ano: int = 365) -> dict:
    """A curva de cada mês de um cliente a partir das leituras do OLMC (energia de cada período entre duas leituras) e
    de uma forma base ({tipo: 96}: a curva do CPE, se houver, ou a forma do perfil da ERSE).

    Em cada intervalo dos últimos `dias_ano` dias com leituras, a energia de cada período reparte-se pelos quartos de
    hora desse período com a forma base — com os períodos que valiam nesses dias e o ciclo do contador (feriados: se os
    conta como domingo) —, e cada dia fica com a sua curva: a soma dela em cada período e intervalo é a das leituras. A
    curva de cada mês do ano é a média dos dias desse mês por tipo de dia; a parte de cada mês, a energia média por dia
    × os dias do mês.
    -> {'perfil_mes': {mês: {tipo: 96 kW}}, 'partes': {mês: parte} (None sem os 12 meses), 'kwh_ano' (None com menos
    de 330 dias), 'de', 'ate', 'dias', 'meses': DataFrame (mês, kWh de cada período nas leituras), 'avisos'}"""
    lt = leituras.sort_values("ate")
    if lt.empty:
        raise ValueError("sem leituras")
    fim = lt["ate"].max().date()
    desde = fim - dt.timedelta(days=dias_ano)
    lt = lt[lt["ate"].dt.date > desde]
    fer = set()
    for a in range(desde.year, fim.year + 1):
        fer |= feriados_nacionais(a)
    avisos: list[str] = []
    soma = {(m, t): np.zeros(96) for m in range(1, 13) for t in TIPOS_DIA}
    n = dict.fromkeys(soma, 0)
    e_mes = dict.fromkeys(range(1, 13), 0.0)
    d_mes = dict.fromkeys(range(1, 13), 0)
    linhas_mes: dict = {}
    total, dias_cob = 0.0, 0
    for _, r in lt.iterrows():
        d0, d1 = r["de"].date(), r["ate"].date()
        dias = [d0 + dt.timedelta(days=k) for k in range(1, (d1 - d0).days + 1)]
        dias = [x for x in dias if x > desde]
        if not dias:
            continue
        grupos = _grupos(r)
        if not grupos:
            continue
        frac = len(dias) / (d1 - d0).days                     # o intervalo que corta o início da janela
        tipos = [tipo_de_dia(x, fer) for x in dias]
        forma = np.array([base[t] for t in tipos], dtype=float)                      # dias × 96
        per = np.array([periodos_do_dia(x, nivel, ciclo, None, fer, feriados) for x in dias])
        fator = np.zeros_like(forma)
        for ps, kwh in grupos:
            msk = np.isin(per, ps)
            b = float(forma[msk].sum())
            if b <= 0 < kwh:                                 # a forma não tem consumo nessas horas: por igual
                forma[msk] = 1.0
                b = float(msk.sum())
                avisos.append(f"{d0:%d/%m/%Y}–{d1:%d/%m/%Y}: a forma base não tem consumo em "
                              f"{'/'.join(ps)}; a energia das leituras repartiu-se por igual nessas horas")
            if b > 0:
                fator[msk] = kwh * frac / b
        kwh_q = forma * fator                                                         # kWh de cada quarto de hora
        for x, t, a, pp in zip(dias, tipos, kwh_q, per):
            soma[(x.month, t)] += a * 4                                               # kW
            n[(x.month, t)] += 1
            e_mes[x.month] += a.sum()
            d_mes[x.month] += 1
            lm = linhas_mes.setdefault(dt.date(x.year, x.month, 1), dict.fromkeys(PERIODOS, 0.0))
            for p in PERIODOS:
                lm[p] += float(a[pp == p].sum())
        total += kwh_q.sum()
        dias_cob += len(dias)
    perfil_mes = {}
    for m in range(1, 13):
        if all(n[(m, t)] for t in TIPOS_DIA):
            perfil_mes[m] = {t: soma[(m, t)] / n[(m, t)] for t in TIPOS_DIA}
        elif any(n[(m, t)] for t in TIPOS_DIA):
            avisos.append(f"o mês {m} não tem todos os tipos de dia nas leituras: vale a forma base nesse mês")
    partes = None
    if all(d_mes[m] for m in range(1, 13)):
        media = {m: e_mes[m] / d_mes[m] * calendar.monthrange(2027, m)[1] for m in range(1, 13)}
        tot = sum(media.values())
        partes = {m: media[m] / tot for m in range(1, 13)} if tot > 0 else None
    meses = pd.DataFrame([{"mes": k, **v, "total": sum(v.values())} for k, v in sorted(linhas_mes.items())])
    return {"perfil_mes": perfil_mes, "partes": partes,
            "kwh_ano": total * 365 / dias_cob if dias_cob >= 330 else None,
            "de": lt["de"].min().date(), "ate": fim, "dias": dias_cob, "meses": meses, "avisos": list(dict.fromkeys(avisos))}


def ler_curva(dados: bytes, nome: str) -> dict:
    """Uma exportação de curvas de carga (a «Curvas de carga» da EGASI — CPE, FUNCAO, UNIDADE, VALOR, DATA, HORA —, a
    «Lista Agregadas…» da compra — Data, Hora, Minuto, Potência —, ou uma folha com data, hora e kW). Da EGASI só a
    energia ativa (kW). -> {'perfil': {tipo: 96 kW}, 'partes': {mês: parte} ou None (menos de 330 dias), 'dias', 'de',
    'ate', 'kwh_ano', 'cpes'}"""
    import unicodedata

    def simples(s):
        return " ".join(unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().split())

    if nome.lower().endswith((".xlsx", ".xlsm", ".xls")):
        cru = pd.read_excel(io.BytesIO(dados), header=None, dtype=object)
    else:
        texto = dados.decode("utf-8-sig", errors="replace")
        primeira = next((l for l in texto.splitlines() if l.strip()), "")
        sep = ";" if primeira.count(";") >= primeira.count(",") else ","
        cru = pd.read_csv(io.StringIO(texto), sep=sep, header=None, dtype=str, keep_default_na=False,
                          engine="python", on_bad_lines="skip")
    i_cab = None
    for i in range(min(40, len(cru))):
        nomes = [simples(v) for v in cru.iloc[i].tolist()]
        if "data" in nomes and "hora" in nomes:
            i_cab = i
            break
    if i_cab is None:
        raise ValueError("não encontro o cabeçalho (as colunas Data e Hora)")
    cab = [simples(v) for v in cru.iloc[i_cab].tolist()]
    df = cru.iloc[i_cab + 1:].copy()
    df.columns = [c or f"col{k}" for k, c in enumerate(cab)]
    if "unidade" in df.columns:
        df = df[df["unidade"].astype(str).str.strip().str.lower() == "kw"]
    elif "funcao" in df.columns:
        df = df[df["funcao"].astype(str).str.strip().str.upper() == "A+"]
    col_v = next((c for c in ("valor", "potencia", "kw", "potencia (kw)") if c in df.columns), None)
    if col_v is None:
        raise ValueError("não encontro a coluna da potência (VALOR, Potência ou kW)")

    def numero(s):
        if isinstance(s, (int, float)):
            return float(s)
        s = str(s).strip().replace(" ", "")
        if s.count(",") == 1 and s.count(".") <= 1 and s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        return float(s) if s else np.nan

    def data(s):
        if isinstance(s, (dt.datetime, pd.Timestamp)):
            return s.date()
        s = str(s).strip()[:10]
        if len(s) == 8 and s.isdigit():
            return dt.date(int(s[:4]), int(s[4:6]), int(s[6:]))
        if "/" in s:
            d_, m_, a_ = s.split("/")[:3]
            return dt.date(int(a_), int(m_), int(d_))
        return dt.date.fromisoformat(s)

    def quarto(h, mi=None):
        """O quarto de hora (0..95) pelo fim dele: «0015»…«2400», ou Hora + Minuto (15…60), ou «HH:MM»."""
        if mi is not None:
            fim = int(float(h)) * 60 + int(float(mi))
        elif isinstance(h, (dt.time, dt.datetime)):
            fim = h.hour * 60 + h.minute + 15          # uma hora em formato de tempo é o início do quarto
        else:
            s = str(h).strip().replace(":", "")
            s = s.zfill(4)[-4:]
            fim = int(s[:2]) * 60 + int(s[2:])
        return (fim - 15) // 15

    tem_minuto = "minuto" in df.columns
    regs = {}
    for _, r in df.iterrows():
        try:
            d = data(r["data"])
            q = quarto(r["hora"], r["minuto"] if tem_minuto else None)
            v = numero(r[col_v])
        except (ValueError, TypeError, KeyError):
            continue
        if 0 <= q < 96 and pd.notna(v):
            regs[(d, q)] = v
    por_dia: dict = {}
    for (d, q), v in regs.items():
        por_dia.setdefault(d, {})[q] = v
    completos = {d: v for d, v in por_dia.items() if len(v) == 96}
    if len(completos) < 7:
        raise ValueError(f"a curva só tem {len(completos)} dias completos (96 quartos de hora): são precisos 7")
    fer = set()
    for a in {d.year for d in completos}:
        fer |= feriados_nacionais(a)
    soma = {t: np.zeros(96) for t in TIPOS_DIA}
    n = dict.fromkeys(TIPOS_DIA, 0)
    em = dict.fromkeys(range(1, 13), 0.0)
    nm = dict.fromkeys(range(1, 13), 0)
    for d, v in completos.items():
        t = tipo_de_dia(d, fer)
        a = np.array([v[q] for q in range(96)])
        soma[t] += a
        n[t] += 1
        em[d.month] += a.sum() / 4
        nm[d.month] += 1
    falta = [NOME_TIPO[t] for t in TIPOS_DIA if not n[t]]
    if falta:
        raise ValueError("a curva não tem nenhum dia de " + " nem de ".join(falta))
    perfil = {t: soma[t] / n[t] for t in TIPOS_DIA}
    partes = None
    if len(completos) >= 330 and all(nm[m] for m in range(1, 13)):
        media = {m: em[m] / nm[m] * calendar.monthrange(2027, m)[1] for m in range(1, 13)}
        tot = sum(media.values())
        partes = {m: media[m] / tot for m in range(1, 13)}
    dias_ano = sum(perfil[t].sum() / 4 * k for t, k in (("util", 250), ("sab", 52), ("dom", 63)))
    return {"perfil": perfil, "partes": partes, "dias": len(completos), "de": min(completos),
            "ate": max(completos), "kwh_ano": dias_ano,
            "cpes": sorted(set(df["cpe"].astype(str))) if "cpe" in df.columns else []}


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
def n(v, casas: int = 0, sinal: bool = False) -> str:
    return TD.n(v, casas, sinal)


def cpes(k: int) -> str:
    return f"{n(k)} CPE" if k == 1 else f"{n(k)} CPEs"
