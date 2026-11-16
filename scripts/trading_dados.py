"""
trading_dados.py — dados e contas do Dashboard Trading da NOSSA Energia
======================================================================
Lê o conjunto de dados que o `trading_dados.py` do projeto «Compra OMIE» gera
(liquidação da REN por quarto de hora, ofertas enviadas, preços públicos do
OMIE e da REN, desvios dos outros agentes) e faz as contas que a tab mostra.

Duas origens, pela ordem:
  trading/dados/*.csv   em claro — só existe no computador de quem gera os dados
  trading/pub/*.bin     cifrado, um ficheiro por mês — é o que vai para o git e
                        para o servidor; abre-se com a palavra-passe da equipa

Convenções (as da REN e do livro «Cálculo do CGS»):
  - o dia e os períodos são os do OMIE, em hora de Espanha (Lisboa + 1 h);
  - posição final e quantidade verificada em MWh por quarto de hora;
  - desvio = posição − verificada: positivo é EXCESSO (comprou-se a mais),
    negativo é DEFEITO (comprou-se a menos);
  - custo do desvio (o «CGS (€)» do livro) = o que o desvio custou a mais do que
    ter comprado no diário a energia certa:
        defeito × (preço de defeito − OMIE)  ou  excesso × (OMIE − preço de excesso)
Sem dependências do Streamlit: testa-se sozinho (scripts/teste_trading.py).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent / "trading"
MARCA = b"NETRD1"
PBKDF2 = 600_000
HORAS_PERIODO = 0.25
PERIODOS_HORARIOS = 25        # um dia cujo último período não passa disto foi liquidado à hora
NOVAE = "NOVAE"
DIAS_SEMANA = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

_POR_MES = ("periodos.csv", "mercado.csv", "agentes_dia.csv")
# colunas que se somam ao agregar (dia -> semana -> mês)
ADITIVAS = ["pos", "qv", "exc", "dfc", "cgs_exc", "cgs_def", "cgs", "enc", "custo_omie", "custo_ref",
            "val_def", "val_exc", "nao_casado", "cgs_nao_casado", "oferta_mwh", "n", "n_sem_pos",
            "soma_p_omie", "soma_p_exc", "soma_p_def", "n_preco", "dias"]


class PalavraErrada(Exception):
    """A palavra-passe não abre os dados cifrados."""


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def origem(raiz: Path = RAIZ) -> str | None:
    """'claro', 'cifrado' ou None (ainda não há dados)."""
    if (raiz / "dados" / "periodos.csv").exists():
        return "claro"
    if any((raiz / "pub").glob("*.bin")):
        return "cifrado"
    return None


def assinatura(raiz: Path = RAIZ) -> str:
    """Muda quando os ficheiros mudam: serve de chave à cache do Streamlit."""
    pasta = raiz / ("dados" if origem(raiz) == "claro" else "pub")
    return "|".join(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in sorted(pasta.glob("*")))


def ler_claro(raiz: Path = RAIZ) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in (raiz / "dados").glob("*") if p.is_file()}


def _decifrar(dados: bytes, palavra: str, chaves: dict) -> dict[str, str]:
    from cryptography.fernet import Fernet, InvalidToken

    n = len(MARCA)
    if dados[:n] != MARCA:
        raise ValueError("não é um ficheiro de dados do trading")
    sal = dados[n:n + 16]
    if sal not in chaves:        # a derivação é lenta de propósito; o sal é o mesmo em todos os ficheiros
        chaves[sal] = base64.urlsafe_b64encode(
            hashlib.pbkdf2_hmac("sha256", palavra.encode("utf-8"), sal, PBKDF2))
    try:
        claro = Fernet(chaves[sal]).decrypt(dados[n + 16:])
    except InvalidToken:
        raise PalavraErrada() from None
    with zipfile.ZipFile(io.BytesIO(claro)) as z:
        return {i.filename: z.read(i).decode("utf-8") for i in z.infolist()}


def ler_cifrado(palavra: str, raiz: Path = RAIZ) -> dict[str, str]:
    """Abre trading/pub/*.bin e junta os meses outra vez num ficheiro de cada tipo."""
    chaves: dict = {}
    partes: dict[str, list[str]] = {}
    saida: dict[str, str] = {}
    for caminho in sorted((raiz / "pub").glob("*.bin")):
        for nome, texto in _decifrar(caminho.read_bytes(), palavra, chaves).items():
            if nome in _POR_MES:
                cab, _, corpo = texto.partition("\n")
                partes.setdefault(nome, [cab + "\n"]).append(corpo)
            else:
                saida[nome] = texto
    saida.update({nome: "".join(p) for nome, p in partes.items()})
    return saida


def quadros(textos: dict[str, str]) -> dict:
    """Os textos CSV -> DataFrames já com as contas feitas por período."""
    def csv(nome, **kw):
        t = textos.get(nome, "")
        return pd.read_csv(io.StringIO(t), **kw) if t.strip() else pd.DataFrame()

    per = csv("periodos.csv", parse_dates=["dia"], dtype={"estado": "string"})
    mer = csv("mercado.csv", parse_dates=["dia"])
    agentes_dia = csv("agentes_dia.csv", parse_dates=["dia"])
    return {
        "periodos": enriquecer(per) if not per.empty else per,
        "mercado": _horas(mer) if not mer.empty else mer,
        "agentes_dia": agentes_dia,
        "agentes": csv("agentes.csv"),
        "ofertas": csv("ofertas.csv", parse_dates=["dia"]),
        "meta": json.loads(textos.get("meta.json", "{}") or "{}"),
    }


# ---------------------------------------------------------------------------
# Contas por período
# ---------------------------------------------------------------------------
def _horas(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a hora de mercado (Espanha), 1 a 24. Até março de 2025 a REN liquidava à hora:
    nesses dias o período já é a hora (reconhecem-se por não passarem do período 25)."""
    df = df.copy()
    a_hora = df.groupby("dia")["periodo"].transform("max") <= PERIODOS_HORARIOS
    df["hora"] = np.where(a_hora, df["periodo"], (df["periodo"] - 1) // 4 + 1).astype(int)
    df["horas_periodo"] = np.where(a_hora, 1.0, HORAS_PERIODO)
    return df


def enriquecer(per: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a cada quarto de hora o desvio, o seu custo e o que não casou."""
    p = _horas(per)
    for c in ("pos", "qv", "p_omie", "p_exc", "p_def", "enc", "val_exc", "val_def", "oferta_mw", "preco_lim"):
        if c not in p.columns:
            p[c] = np.nan
        p[c] = pd.to_numeric(p[c], errors="coerce")
    liq = p["qv"].notna()                                  # o período já foi liquidado pela REN
    pos0 = p["pos"].fillna(0.0)                            # sem posição = não se comprou (como no livro)
    p["desvio"] = np.where(liq, pos0 - p["qv"], np.nan)
    p["exc"] = np.where(liq, np.clip(p["desvio"], 0, None), np.nan)
    p["dfc"] = np.where(liq, np.clip(-p["desvio"], 0, None), np.nan)
    p["cgs_exc"] = p["exc"] * (p["p_omie"] - p["p_exc"])
    p["cgs_def"] = p["dfc"] * (p["p_def"] - p["p_omie"])
    p["cgs"] = p["cgs_exc"] + p["cgs_def"]
    p["custo_omie"] = np.where(liq, pos0 * p["p_omie"], np.nan)      # o que se pagou no diário
    p["custo_ref"] = p["qv"] * p["p_omie"]                 # o que custaria comprar o consumo exato no diário
    p["oferta_mwh"] = p["oferta_mw"] * p["horas_periodo"]
    p["sem_pos"] = liq & p["pos"].isna()
    falhou = p["sem_pos"] & (p["oferta_mwh"] > 0)          # ofereceu-se e a posição ficou a zero
    p["nao_casado"] = np.where(falhou, p["oferta_mwh"], 0.0)
    p["cgs_nao_casado"] = np.where(falhou, p["cgs_def"], 0.0)
    p["dia_semana"] = p["dia"].dt.dayofweek
    return p


def diario(per: pd.DataFrame) -> pd.DataFrame:
    """Um registo por dia liquidado, só com colunas que se podem somar; `racios` faz o resto."""
    liq = per[per["qv"].notna()]
    if liq.empty:
        return pd.DataFrame(columns=["dia"] + ADITIVAS)
    g = liq.groupby("dia")
    d = g[["pos", "qv", "exc", "dfc", "cgs_exc", "cgs_def", "cgs", "enc", "custo_omie", "custo_ref",
           "val_def", "val_exc", "nao_casado", "cgs_nao_casado", "oferta_mwh"]].sum(min_count=1)
    d["n"] = g["qv"].size()
    d["n_sem_pos"] = g["sem_pos"].sum()
    d["soma_p_omie"] = g["p_omie"].sum()
    d["soma_p_exc"] = g["p_exc"].sum()
    d["soma_p_def"] = g["p_def"].sum()
    d["n_preco"] = g["p_omie"].count()
    d["dias"] = 1
    d["pos"] = d["pos"].fillna(0.0)
    d["estado"] = g["estado"].first()
    d["versao"] = g["versao"].max()
    return d.reset_index()


def racios(d: pd.DataFrame) -> pd.DataFrame:
    """Sobre somas (de um dia, de uma semana, de um mês, do período todo): percentagens e €/MWh."""
    d = d.copy()
    qv = d["qv"].where(d["qv"] > 0)
    d["desvio"] = d["pos"] - d["qv"]
    d["desvio_pct"] = 100 * d["desvio"] / qv
    d["abs_pct"] = 100 * (d["exc"] + d["dfc"]) / qv
    d["exc_pct"] = 100 * d["exc"] / qv
    d["dfc_pct"] = 100 * d["dfc"] / qv
    d["cgs_mwh"] = d["cgs"] / qv
    d["enc_mwh"] = d["enc"] / qv
    d["cgs_total"] = d["cgs"] + d["enc"]
    d["cgs_total_mwh"] = d["cgs_total"] / qv
    d["p_base"] = d["soma_p_omie"] / d["n_preco"].where(d["n_preco"] > 0)      # média simples do OMIE
    d["p_exc"] = d["soma_p_exc"] / d["n_preco"].where(d["n_preco"] > 0)
    d["p_def"] = d["soma_p_def"] / d["n_preco"].where(d["n_preco"] > 0)
    d["p_compra"] = d["custo_omie"] / d["pos"].where(d["pos"] > 0)             # preço médio do que se comprou
    d["p_consumo"] = d["custo_ref"] / qv                                       # OMIE pesado pelo consumo
    d["custo_total"] = d["custo_ref"] + d["cgs"] + d["enc"]
    d["custo_total_mwh"] = d["custo_total"] / qv
    d["cgs_def_mwh"] = d["cgs_def"] / d["dfc"].where(d["dfc"] > 0)             # € por MWh em defeito
    d["cgs_exc_mwh"] = d["cgs_exc"] / d["exc"].where(d["exc"] > 0)             # € por MWh em excesso
    return d


def agregar(d: pd.DataFrame, grao: str) -> pd.DataFrame:
    """O diário somado por 'Dia', 'Semana' (de segunda a domingo) ou 'Mês', já com os rácios."""
    if d.empty:
        return racios(d)
    if grao == "Dia":
        chave = d["dia"]
    elif grao == "Semana":
        chave = d["dia"] - pd.to_timedelta(d["dia"].dt.dayofweek, unit="D")
    else:
        chave = d["dia"].dt.to_period("M").dt.start_time
    cols = [c for c in ADITIVAS if c in d.columns]
    return racios(d.groupby(chave.rename("dia"))[cols].sum().reset_index())


def total(d: pd.DataFrame) -> pd.Series:
    """O período todo numa linha."""
    cols = [c for c in ADITIVAS if c in d.columns]
    return racios(pd.DataFrame([d[cols].sum()])).iloc[0]


def por_hora(per: pd.DataFrame) -> pd.DataFrame:
    """Perfil médio por hora de mercado: compra, consumo, desvio e o seu custo."""
    liq = per[per["qv"].notna()]
    g = liq.groupby("hora")
    h = g[["pos", "qv", "exc", "dfc", "cgs", "cgs_exc", "cgs_def", "enc"]].sum()
    h["pos"] = h["pos"].fillna(0.0)
    h["dias"] = g["dia"].nunique()
    h["desvio_pct"] = 100 * (h["pos"] - h["qv"]) / h["qv"].where(h["qv"] > 0)
    h["abs_pct"] = 100 * (h["exc"] + h["dfc"]) / h["qv"].where(h["qv"] > 0)
    h["cgs_mwh"] = h["cgs"] / h["qv"].where(h["qv"] > 0)
    h["p_omie"] = g["p_omie"].mean()
    h["p_exc"] = g["p_exc"].mean()
    h["p_def"] = g["p_def"].mean()
    # em quantos dias a hora fechou em defeito (para distinguir viés de ruído)
    hd = liq.groupby(["dia", "hora"])["desvio"].sum().reset_index()
    h["dias_defeito_pct"] = 100 * hd.assign(x=hd["desvio"] < 0).groupby("hora")["x"].mean()
    return h.reset_index()


def por_dia_semana(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    cols = [c for c in ADITIVAS if c in d.columns]
    s = racios(d.groupby(d["dia"].dt.dayofweek.rename("dia_semana"))[cols].sum().reset_index())
    s["nome"] = s["dia_semana"].map(lambda i: DIAS_SEMANA[int(i)])
    return s


def mapa(per: pd.DataFrame, valor: str, modo: str = "soma") -> pd.DataFrame:
    """Tabela dia × hora de mercado para os mapas de calor (`desvio_pct` é a razão das somas)."""
    liq = per[per["qv"].notna()] if valor not in ("p_omie", "p_exc", "p_def") else per
    if valor == "desvio_pct":
        g = liq.groupby(["dia", "hora"])[["desvio", "qv"]].sum()
        s = 100 * g["desvio"] / g["qv"].where(g["qv"] > 0)
    else:
        s = liq.groupby(["dia", "hora"])[valor].agg("sum" if modo == "soma" else "mean")
    return s.unstack("hora").sort_index()


# ---------------------------------------------------------------------------
# Mercado
# ---------------------------------------------------------------------------
def mercado_diario(mer: pd.DataFrame) -> pd.DataFrame:
    if mer.empty:
        return mer
    g = mer.groupby("dia")
    m = pd.DataFrame({
        "p_omie": g["p_omie"].mean(), "p_min": g["p_omie"].min(), "p_max": g["p_omie"].max(),
        "p_exc": g["p_exc"].mean(), "p_def": g["p_def"].mean(), "n": g["p_omie"].count(),
    })
    m["spread_def"] = m["p_def"] - m["p_omie"]          # quanto custa a mais um MWh em defeito
    m["spread_exc"] = m["p_omie"] - m["p_exc"]          # quanto se perde num MWh em excesso
    return m.reset_index()


def ranking(agentes_dia: pd.DataFrame, agentes: pd.DataFrame) -> pd.DataFrame:
    """Custo dos desvios de cada agente em €/MWh, do mais barato ao mais caro.

    Os dados são os do livro «Análise Desvios do Mercado - Balanço Anual» do utilizador, e a média é a
    dele: a **média simples** dos €/MWh dos dias do período (a «MÉDIA ANUAL» do livro, no ano todo). Só
    vêm os dias que o livro conta: os dias em que o agente não teve consumo e, na NOVAE, os dias por
    preencher já ficaram de fora no exportador. Entram todos os agentes que o livro tem.
    """
    if agentes_dia.empty:
        return pd.DataFrame()
    g = agentes_dia.groupby("unidade")
    r = pd.DataFrame({
        "eur_mwh": g["eur_mwh"].mean(), "dias": g["dia"].nunique(),
        "consumo": g["consumo"].sum(min_count=1), "desvios": g["desvios"].sum(min_count=1),
        "encargos": g["encargos"].sum(min_count=1),
    })
    r["mwh_dia"] = r["consumo"] / r["dias"].where(r["dias"] > 0)
    r = r.reset_index()
    if not agentes.empty:
        r = r.merge(agentes[["unidade", "nome"]], on="unidade", how="left")
    else:
        r["nome"] = r["unidade"]
    r["nome"] = r["nome"].fillna(r["unidade"])
    r = r.dropna(subset=["eur_mwh"]).sort_values("eur_mwh").reset_index(drop=True)
    r["lugar"] = r.index + 1
    return r


def media_do_mercado(r: pd.DataFrame) -> float:
    """A linha «MÉDIA» do livro: a média simples das médias dos agentes."""
    return float(r["eur_mwh"].mean()) if not r.empty else float("nan")


# ---------------------------------------------------------------------------
# Conclusões automáticas
# ---------------------------------------------------------------------------
def n(v, casas: int = 0, sinal: bool = False) -> str:
    """Número à portuguesa: 12 345,6 (espaço fino nos milhares, vírgula decimal)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "—"
    t = f"{v:+,.{casas}f}" if sinal else f"{v:,.{casas}f}"
    return t.replace(",", " ").replace(".", ",").replace("-", "−")


def conclusoes(per: pd.DataFrame, d: pd.DataFrame, cgs_ref: float = 0.0) -> list[tuple[str, str]]:
    """[(tipo, frase)] sobre o período escolhido; tipo: 'alerta', 'atencao', 'bom' ou 'info'."""
    saida: list[tuple[str, str]] = []
    if d.empty:
        return saida
    t = total(d)
    # 1 — balanço
    lado = "a menos" if t["desvio"] < 0 else "a mais"
    saida.append(("info",
                  f"Comprou-se **{n(t['pos'])} MWh** para um consumo verificado de **{n(t['qv'])} MWh**: "
                  f"{n(abs(t['desvio']))} MWh {lado} ({n(t['desvio_pct'], 1, True)} %). Somando excessos e defeitos "
                  f"quarto de hora a quarto de hora, o desvio foi de **{n(t['abs_pct'], 1)} %** do consumo "
                  f"({n(t['exc_pct'], 1)} % em excesso, {n(t['dfc_pct'], 1)} % em defeito)."))
    # 2 — custo
    frase = (f"Os desvios custaram **{n(t['cgs'])} €** ({n(t['cgs_mwh'], 2)} €/MWh) e os encargos de regulação "
             f"**{n(t['enc'])} €** ({n(t['enc_mwh'], 2)} €/MWh): CGS de **{n(t['cgs_total_mwh'], 2)} €/MWh**.")
    if cgs_ref > 0 and pd.notna(t["cgs_total_mwh"]):
        dif = t["cgs_total_mwh"] - cgs_ref
        frase += (f" Fica {n(abs(dif), 2)} €/MWh {'acima' if dif > 0 else 'abaixo'} da referência de "
                  f"{n(cgs_ref, 2)} €/MWh ({n(dif * t['qv'], 0, True)} € no período).")
        saida.append(("alerta" if dif > 0 else "bom", frase))
    else:
        saida.append(("info", frase))
    # 3 — assimetria: de que lado sai mais barato errar
    if pd.notna(t["cgs_def_mwh"]) and pd.notna(t["cgs_exc_mwh"]):
        caro, barato = ("defeito", "excesso") if t["cgs_def_mwh"] > t["cgs_exc_mwh"] else ("excesso", "defeito")
        saida.append(("atencao",
                      f"Cada MWh em **defeito** custou {n(t['cgs_def_mwh'], 2)} € acima do OMIE e cada MWh em "
                      f"**excesso** perdeu {n(t['cgs_exc_mwh'], 2)} €. Neste período saiu mais caro errar por "
                      f"{caro}: na dúvida, a margem do arredondamento rende mais do lado do {barato}."))
    # 4 — dias sem posição e períodos que não casaram
    sem = d[(d["pos"] <= 0) & (d["qv"] > 0)]
    if not sem.empty:
        dias = ", ".join(x.strftime("%d/%m") for x in sem["dia"])
        saida.append(("alerta",
                      f"**{len(sem)} dia(s) sem posição no mercado** ({dias}): o consumo inteiro "
                      f"({n(sem['qv'].sum())} MWh) foi liquidado como defeito e custou {n(sem['cgs'].sum())} € "
                      f"acima do OMIE — {n(100 * sem['cgs'].sum() / t['cgs'], 0) if t['cgs'] else '—'} % do custo "
                      f"dos desvios do período."))
    nc = d[(d["nao_casado"] > 0) & (d["pos"] > 0)]
    if not nc.empty:
        saida.append(("alerta",
                      f"Em **{len(nc)} dia(s)** houve períodos da oferta que **não casaram** (preço do OMIE acima "
                      f"do nosso preço máximo): {n(nc['nao_casado'].sum(), 1)} MWh ficaram por comprar e foram a "
                      f"defeito, com um custo de {n(nc['cgs_nao_casado'].sum())} € acima do OMIE."))
    # 5 — horas com viés
    h = por_hora(per)
    if not h.empty and h["dias"].max() >= 7:
        vies = h[(h["desvio_pct"].abs() >= 3) & ((h["dias_defeito_pct"] >= 70) | (h["dias_defeito_pct"] <= 30))]
        dfc = vies[vies["desvio_pct"] < 0].sort_values("desvio_pct").head(4)
        exc = vies[vies["desvio_pct"] > 0].sort_values("desvio_pct", ascending=False).head(4)
        if not dfc.empty:
            saida.append(("atencao", "Horas em que se compra **sistematicamente a menos**: "
                          + ", ".join(f"H{int(r.hora)} ({n(r.desvio_pct, 1, True)} %)" for r in dfc.itertuples())
                          + " — hora de mercado; em Lisboa é uma hora menos."))
        if not exc.empty:
            saida.append(("atencao", "Horas em que se compra **sistematicamente a mais**: "
                          + ", ".join(f"H{int(r.hora)} ({n(r.desvio_pct, 1, True)} %)" for r in exc.itertuples())
                          + "."))
        cara = h.sort_values("cgs", ascending=False).head(3)
        if t["cgs"] > 0:
            saida.append(("info", "As horas que mais pesam no custo dos desvios: "
                          + ", ".join(f"H{int(r.hora)} ({n(100 * r.cgs / t['cgs'], 0)} %)" for r in cara.itertuples())
                          + "."))
    # 6 — dia da semana
    s = por_dia_semana(d)
    if len(s) >= 5 and len(d) >= 14:
        pior, melhor = s.sort_values("abs_pct").iloc[-1], s.sort_values("abs_pct").iloc[0]
        saida.append(("info",
                      f"Por dia da semana, o desvio é maior à **{pior['nome']}** ({n(pior['abs_pct'], 1)} % do "
                      f"consumo, {n(pior['cgs_mwh'], 2)} €/MWh) e menor à {melhor['nome']} "
                      f"({n(melhor['abs_pct'], 1)} %)."))
    # 7 — piores dias
    dr = racios(d).sort_values("cgs", ascending=False).head(3)
    if t["cgs"] > 0 and len(d) >= 5:
        saida.append(("info", "Os dias mais caros em desvios: "
                      + "; ".join(f"{r.dia.strftime('%d/%m')} ({n(r.cgs)} €, desvio {n(r.desvio_pct, 1, True)} %)"
                                  for r in dr.itertuples()) + "."))
    # 8 — preço
    if pd.notna(t["p_base"]) and pd.notna(t["p_consumo"]):
        dif = t["p_consumo"] - t["p_base"]
        saida.append(("info",
                      f"O OMIE fez em média **{n(t['p_base'], 2)} €/MWh** (base). Pesado pelo nosso consumo dá "
                      f"{n(t['p_consumo'], 2)} €/MWh: o perfil da carteira custa {n(dif, 2, True)} €/MWh face à base. "
                      f"Com desvios e encargos, a energia saiu a **{n(t['custo_total_mwh'], 2)} €/MWh**."))
    return saida
