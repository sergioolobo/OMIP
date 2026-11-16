"""
pricing_tab.py — a tab «Dashboard Pricing»
==========================================
Tarifários, preços e clientes da carteira, a partir dos dados do
`pricing_dados.py`, e o simulador de tarifários (cálculo MAX e FLEX) para criar
tarifários especiais para clientes estratégicos.

Chamado pelo 07_dashboard.py dentro de `with tab_pricing:` — nunca se chama
`st.stop()` aqui (pararia também as outras tabs): sai-se com `return`.

Os dados abrem-se com a mesma palavra-passe do Dashboard Trading: quem já a
escreveu numa das tabs não a escreve outra vez.
"""
from __future__ import annotations

import datetime as dt
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pricing_dados as PD
import trading_dados as TD
import trading_tab as TT
from pricing_dados import n

COR = TT.COR
COR_CALCULO = {"MAX": "#2a78d6", "FLEX": "#eb6834", "Indexado": "#1baf7a"}
COR_REDES = {"Não": "#2a78d6", "Sim": "#6250d6"}
COR_PERIODO = {"P": "#e34948", "C": "#eda100", "VN": "#2a78d6", "SV": "#0d366b"}
PARCELAS = [("omie", "OMIE", "#1baf7a"), ("cgs", "CGS", "#eda100"), ("custos", "Custos indiretos", "#898781"),
            ("k_interno", "K interno", "#6250d6"), ("k_agente", "K do agente", "#eb6834"),
            ("perdas", "Perdas", "#c3c2b7"), ("tar", "TAR", "#0d366b")]
_LARGO = TT._LARGO
_fig, _mostra, _tabela, _metrica = TT._fig, TT._mostra, TT._tabela, TT._metrica
_PRECO = ("preco_medio", "Preço médio")
_COLUNAS_PRECO = {"preco_medio": "Preço médio", "p_P": "Ponta", "p_C": "Cheias", "p_VN": "Vazio normal",
                  "p_SV": "Super vazio"}


# ---------------------------------------------------------------------------
# Dados (com cache) e entrada
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="A carregar os dados de pricing…")
def _quadros_claro(assinatura: str) -> dict:
    return PD.quadros(PD.ler_claro())


@st.cache_data(show_spinner="A abrir os dados de pricing…")
def _quadros_cifrado(assinatura: str, palavra: str) -> dict:
    return PD.quadros(PD.ler_cifrado(palavra))


def _obter() -> dict | None:
    origem = PD.origem()
    if origem is None:
        st.info("Ainda não há dados de pricing neste servidor. Geram-se no computador da empresa com "
                "`python pricing_dados.py --publicar` (projeto «Compra OMIE»), e todos os dias às 13:30 com os do "
                "Dashboard Trading.")
        return None
    assinatura = PD.assinatura()
    if origem == "claro":
        return _quadros_claro(assinatura)
    for chave in ("pricing_palavra", "trading_palavra"):          # a da equipa é a mesma das duas tabs
        palavra = st.session_state.get(chave)
        if palavra:
            try:
                return _quadros_cifrado(assinatura, palavra)
            except PD.PalavraErrada:
                if chave == "pricing_palavra":
                    st.session_state.pop(chave, None)
    st.markdown("#### 🔒 Área reservada à equipa da NOSSA Energia")
    st.caption("Os tarifários, os preços e os clientes só se abrem com a palavra-passe da equipa (a mesma do "
               "Dashboard Trading). As previsões OMIP e OMIE continuam abertas.")
    with st.form("pricing_entrada"):
        tentativa = st.text_input("Palavra-passe", type="password")
        if st.form_submit_button("Entrar") and tentativa:
            try:
                _quadros_cifrado(assinatura, tentativa)
            except PD.PalavraErrada:
                st.error("Palavra-passe errada.")
            else:
                st.session_state["pricing_palavra"] = tentativa
                st.session_state.setdefault("trading_palavra", tentativa)
                st.rerun()
    return None


def _cgs_real() -> tuple[float, pd.Timestamp, pd.Timestamp] | None:
    """A CGS real da NOVAE (desvios + encargos por MWh consumido) nos últimos 365 dias liquidados, dos dados do
    Dashboard Trading, se estiverem abertos; None se não."""
    try:
        origem = TD.origem()
        if origem == "claro":
            q = TT._quadros_claro(TD.assinatura())
        elif origem == "cifrado" and st.session_state.get("trading_palavra"):
            q = TT._quadros_cifrado(TD.assinatura(), st.session_state["trading_palavra"])
        else:
            return None
        d = q["diario"]
        if d.empty:
            return None
        fim = d["dia"].max()
        x = d[d["dia"] > fim - pd.Timedelta(days=365)]
        return float(TD.total(x)["cgs_total_mwh"]), x["dia"].min(), fim
    except Exception:            # o pricing não depende do trading
        return None


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
def _opcoes(s: pd.Series) -> list[str]:
    return sorted(x for x in s.dropna().astype(str).unique() if x)


def _filtros(c: pd.DataFrame) -> pd.DataFrame:
    with st.container(border=True):
        l1 = st.columns([1.2, 1, 1, 1, 1.1])
        estados = l1[0].multiselect("Estado", _opcoes(c["estado"]), default=["Ativo"], key="prc_estado",
                                    placeholder="Todos")
        niveis = l1[1].multiselect("Nível de tensão", _opcoes(c["nivel"]), key="prc_nivel", placeholder="Todos")
        tipos = l1[2].multiselect("Tipo", _opcoes(c["tipo"]), key="prc_tipo", placeholder="Fixa e indexada")
        redes = l1[3].multiselect("Inclui redes", ["Sim", "Não"], key="prc_redes", placeholder="Com e sem")
        segm = l1[4].multiselect("Segmento", _opcoes(c["segmento"]), key="prc_segmento", placeholder="Todos")
        l2 = st.columns([1, 1, 1])
        calc = l2[0].multiselect("Cálculo", _opcoes(c["calculo"]), key="prc_calculo", placeholder="MAX, FLEX e indexado")
        fam = l2[1].multiselect("Família", _opcoes(c["familia"]), key="prc_familia", placeholder="Todas")
        ag = l2[2].multiselect("Agente", _opcoes(c["agente"]), key="prc_agente", placeholder="Todos")
    m = pd.Series(True, index=c.index)
    for col, sel in (("estado", estados), ("nivel", niveis), ("tipo", tipos), ("inclui_redes", redes),
                     ("segmento", segm), ("calculo", calc), ("familia", fam), ("agente", ag)):
        if sel:
            m &= c[col].isin(sel)
    return c[m]


# ---------------------------------------------------------------------------
# Peças
# ---------------------------------------------------------------------------
def _eur_kwh(v, casas: int = 4) -> str:
    return "—" if v is None or pd.isna(v) else f"{n(v, casas)} €/kWh"


def _barras_h(nomes, valores, cor, unidade, casas=0, hover_extra=None, altura=None) -> go.Figure:
    """Barras horizontais, a maior em cima."""
    fig = go.Figure(go.Bar(y=nomes, x=valores, orientation="h", marker=dict(color=cor),
                           customdata=hover_extra, text=[n(v, casas) for v in valores], textposition="outside",
                           cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:,." + str(casas) + "f} " + unidade +
                                         ("<br>%{customdata}" if hover_extra is not None else "") + "<extra></extra>"))
    fig = _fig(fig, altura or max(220, 26 * len(nomes) + 60), legenda=False, unificado=False)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridcolor=COR["grelha"], title_text=unidade)
    return fig


_COL_CPE = {"cliente": ("Cliente", "texto"), "cpe": ("CPE", "texto"), "estado": ("Estado", "texto"),
            "nivel": ("Nível", "texto"), "horario": ("Horário", "texto"), "tarifa": ("Tarifário", "texto"),
            "familia": ("Família", "texto"), "calculo": ("Cálculo", "texto"), "inclui_redes": ("Redes", "texto"),
            "agente": ("Agente", "texto"), "consumo_mwh": ("Consumo (MWh/ano)", 1),
            "preco_medio": ("Preço médio (€/kWh)", 4), "p_P": ("Ponta", 4), "p_C": ("Cheias", 4),
            "p_VN": ("Vazio normal", 4), "p_SV": ("Super vazio", 4), "k": ("K (€/MWh)", 1),
            "preco_potencia": ("Potência (€/dia)", 4), "potencia_kva": ("kVA", 2),
            "fim_contrato": ("Fim de contrato", "data"), "renovacao_automatica": ("Renovação", "texto"),
            "em_mudanca_para": ("Em mudança para", "texto"), "preco_especial": ("Preço especial", "texto"),
            "concelho": ("Concelho", "texto"), "risco": ("Risco", "texto")}


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
def _resumo(df: pd.DataFrame, todos: pd.DataFrame, q: dict, hoje: pd.Timestamp):
    w = df["consumo_mwh"]
    fixas = df[df["tipo"] == "Fixa"]
    k1 = st.columns(4)
    _metrica(k1[0], "CPEs", n(len(df)))
    _metrica(k1[1], "Clientes", n(df["cliente_id"].nunique()))
    _metrica(k1[2], "Consumo anual", f"{n(w.sum() / 1000, 2)} GWh",
             ajuda="Consumo ORD de cada CPE (senão o comercial), da Lista de Pontos de Entrega de hoje.")
    _metrica(k1[3], "Tarifários em uso", n(df["tarifa"].nunique()), ajuda=f"{n(df['familia'].nunique())} famílias")
    k2 = st.columns(4)
    for col, redes, rot in ((k2[0], "Não", "Preço médio · fixas sem redes"), (k2[1], "Sim", "Preço médio · fixas com redes")):
        x = fixas[fixas["inclui_redes"] == redes]
        _metrica(col, rot, _eur_kwh(PD.media_ponderada(x["preco_medio"], x["consumo_mwh"])),
                 ajuda="Média dos preços de energia de cada CPE (pesados pelo perfil da ERSE em cada período), "
                       "pesada pelo consumo. Com redes, inclui as tarifas de acesso: compara-se só com com redes.")
    _metrica(k2[2], "K médio", f"{n(PD.media_ponderada(df['k'], w), 1)} €/MWh",
             ajuda="K do agente de cada CPE (o «Escalão» da Lista de Preços; senão o K indexado ou o K fixo), "
                   "pesado pelo consumo.")
    ind = df.loc[df["tipo"] == "Indexada", "consumo_mwh"].sum()
    _metrica(k2[3], "Consumo indexado", f"{n(100 * ind / w.sum(), 1) if w.sum() else '—'} %")

    st.markdown("##### Conclusões")
    for nivel, texto in PD.conclusoes(df, todos, q["meta"], hoje):
        st.markdown(f"{TT.ICONE[nivel]} {texto}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Consumo por família")
        f = PD.por_familia(df).head(14)
        if not f.empty:
            extra = [f"{n(a)} CPEs · {n(b)} clientes" for a, b in zip(f["n_cpes"], f["n_clientes"])]
            _mostra(_barras_h(f["familia"], f["consumo_mwh"], COR["exc"], "MWh/ano", hover_extra=extra), "prc_res_fam")
    with c2:
        st.markdown("##### Como se calcula o preço, por nível de tensão")
        st.caption("Parte do consumo de cada nível em tarifas MAX, FLEX e indexadas.")
        x = df[df["calculo"] != ""].groupby(["nivel", "calculo"])["consumo_mwh"].sum().unstack(fill_value=0)
        if not x.empty:
            x = x.loc[[v for v in PD.NIVEIS if v in x.index] + [v for v in x.index if v not in PD.NIVEIS]]
            pct = 100 * x.div(x.sum(axis=1), axis=0)
            fig = go.Figure()
            for calc in [c for c in ("MAX", "FLEX", "Indexado") if c in pct.columns]:
                fig.add_trace(go.Bar(y=pct.index, x=pct[calc], name=calc, orientation="h",
                                     marker=dict(color=COR_CALCULO[calc], line=dict(color=COR["fundo"], width=1)),
                                     customdata=x[calc], hovertemplate="%{x:.1f} % · %{customdata:,.0f} MWh"))
            fig = _fig(fig, 60 + 46 * len(pct), unificado=False)
            fig.update_layout(barmode="stack", legend_traceorder="normal")
            fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(range=[0, 100], ticksuffix=" %", showgrid=True, gridcolor=COR["grelha"])
            _mostra(fig, "prc_res_calc")

    h = q["historico"]
    st.markdown("##### Evolução da carteira ativa")
    if h.empty or h["dia"].nunique() < 2:
        st.caption("O histórico guarda-se uma vez por dia desde 16/11/2026: a evolução aparece a partir do segundo dia.")
    else:
        g = h.groupby("dia").agg(n=("n_cpes", "sum"), mwh=("consumo_mwh", "sum"), sp=("soma_preco_consumo", "sum"),
                                 cp=("consumo_com_preco", "sum")).reset_index()
        g["preco"] = g["sp"] / g["cp"].where(g["cp"] > 0)
        c1, c2 = st.columns(2)
        with c1:
            fig = _fig(go.Figure(TT._linha(g["dia"], g["n"], "CPEs ativos", COR["compra"], "CPEs", 0)), 280, legenda=False)
            _mostra(fig, "prc_res_hist_n")
        with c2:
            fig = _fig(go.Figure(TT._linha(g["dia"], g["preco"], "Preço médio (todas as fixas)", COR["consumo"],
                                           "€/kWh", 4)), 280, legenda=False)
            _mostra(fig, "prc_res_hist_p")


# ---------------------------------------------------------------------------
# Tarifários
# ---------------------------------------------------------------------------
_COL_TARIFARIO = {"tarifa": ("Tarifário", "texto"), "familia": ("Família", "texto"), "tipo": ("Tipo", "texto"),
                  "calculo": ("Cálculo", "texto"), "inclui_redes": ("Redes", "texto"), "n_cpes": ("CPEs", 0),
                  "n_clientes": ("Clientes", 0), "consumo_mwh": ("Consumo (MWh/ano)", 0),
                  "quota_consumo": ("% do consumo", 1), "preco_medio": ("Preço médio (€/kWh)", 4),
                  "preco_min": ("Mais baixo", 4), "preco_max": ("Mais alto", 4), "k_medio": ("K médio (€/MWh)", 1),
                  "n_agentes": ("Agentes", 0)}


def _tarifarios(df: pd.DataFrame):
    t = PD.por_tarifario(df)
    if t.empty:
        st.info("Nenhum CPE com estes filtros.")
        return
    c1, c2 = st.columns([3, 1])
    medida = c2.radio("Mais usados por", ["Consumo", "Nº de CPEs"], key="prc_tar_medida")
    top = c2.slider("Quantos", 5, 40, 15, key="prc_tar_top")
    col, unidade = ("consumo_mwh", "MWh/ano") if medida == "Consumo" else ("n_cpes", "CPEs")
    x = t.sort_values(col, ascending=False).head(top)
    extra = [f"{fam} · {calc} · redes: {red}<br>{n(nc)} CPEs · {n(mwh)} MWh/ano · preço médio {_eur_kwh(p)}"
             for fam, calc, red, nc, mwh, p in zip(x["familia"], x["calculo"], x["inclui_redes"], x["n_cpes"],
                                                    x["consumo_mwh"], x["preco_medio"])]
    with c1:
        st.markdown(f"##### Os {top} tarifários mais usados ({medida.lower()})")
        cores = [COR_CALCULO.get(c, COR["neutro"]) for c in x["calculo"]]
        _mostra(_barras_h(x["tarifa"].str.slice(0, 60), x[col], cores, unidade, hover_extra=extra), "prc_tar_top_g")
        st.caption("Cor: " + " · ".join(f"<span style='color:{v}'>■</span> {k}" for k, v in COR_CALCULO.items()),
                   unsafe_allow_html=True)
    st.markdown("##### Todos os tarifários")
    _tabela(t, _COL_TARIFARIO, "prc_tar_tab", altura=420, baixar="tarifarios.csv")
    st.markdown("##### Por família")
    f = PD.por_familia(df)
    _tabela(f, {"familia": ("Família", "texto"), **{k: v for k, v in _COL_TARIFARIO.items()
                                                   if k not in ("tarifa", "familia", "tipo", "calculo", "inclui_redes")},
                "pct_redes": ("% com redes", 0)}, "prc_tar_fam", baixar="familias.csv")
    st.caption("Preço médio: a média dos preços de energia de cada CPE pesados pelo perfil da ERSE em cada período "
               "(períodos de 2027), pesada pelo consumo; mais baixo e mais alto são os de um CPE. As tarifas "
               "indexadas não têm preço fixo: contam pelo K.")


# ---------------------------------------------------------------------------
# Preços
# ---------------------------------------------------------------------------
def _precos(df: pd.DataFrame, q: dict):
    c = st.columns([1, 1, 2])
    redes = c[0].radio("Inclui redes", ["Não", "Sim"], horizontal=True, key="prc_pr_redes",
                       help="Os preços com redes têm as tarifas de acesso lá dentro: comparam-se só entre si.")
    qual = c[1].selectbox("Preço", list(_COLUNAS_PRECO), format_func=_COLUNAS_PRECO.get, key="prc_pr_qual")
    c[2].caption("Só tarifas fixas, que têm preço de energia. O preço médio de cada CPE é a média dos preços dele "
                 "pesada pela parte do consumo do perfil da ERSE em cada período.")
    x = df[(df["tipo"] == "Fixa") & (df["inclui_redes"] == redes) & df[qual].notna()]
    if x.empty:
        st.info("Nenhuma tarifa fixa com estes filtros.")
    else:
        st.markdown(f"##### {_COLUNAS_PRECO[qual]} por nível de tensão e horário ({'com' if redes == 'Sim' else 'sem'} redes)")
        fig = go.Figure()
        for h, cor in zip(["SIMPLES", "BI", "TRI", "TETRA"], ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]):
            y = x[x["horario"] == h]
            if y.empty:
                continue
            fig.add_trace(go.Box(x=y["nivel"], y=y[qual], name=h.capitalize(), marker=dict(color=cor, size=4),
                                 boxpoints="outliers", customdata=np.stack([y["cliente"], y["tarifa"]], axis=-1),
                                 hovertemplate="%{y:.4f} €/kWh<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
        fig = _fig(fig, 380, unificado=False)
        fig.update_layout(boxmode="group")
        fig.update_xaxes(categoryorder="array", categoryarray=PD.NIVEIS)
        fig.update_yaxes(title_text="€/kWh", tickformat=".3f")
        _mostra(fig, "prc_pr_box")

        st.markdown("##### Preço e consumo de cada CPE")
        fig = go.Figure()
        for calc in ("MAX", "FLEX"):
            y = x[(x["calculo"] == calc) & (x["consumo_mwh"] > 0)]
            if y.empty:
                continue
            fig.add_trace(go.Scatter(x=y["consumo_mwh"], y=y[qual], mode="markers", name=calc,
                                     marker=dict(color=COR_CALCULO[calc], size=7, opacity=0.7,
                                                 line=dict(color=COR["fundo"], width=0.5)),
                                     customdata=np.stack([y["cliente"], y["tarifa"], y["nivel"]], axis=-1),
                                     hovertemplate="%{customdata[0]}<br>%{customdata[1]} (%{customdata[2]})<br>"
                                                   "%{x:,.1f} MWh/ano · %{y:.4f} €/kWh<extra></extra>"))
        fig = _fig(fig, 380, unificado=False)
        fig.update_xaxes(type="log", title_text="Consumo (MWh/ano, escala logarítmica)")
        fig.update_yaxes(title_text="€/kWh", tickformat=".3f")
        _mostra(fig, "prc_pr_disp")

        alto, baixo = PD.extremos(x, 15, qual)
        cols = {k: _COL_CPE[k] for k in ("cliente", "cpe", "nivel", "horario", "tarifa", "agente", "consumo_mwh",
                                         qual, "k", "fim_contrato")}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Os 15 mais caros")
            _tabela(alto, cols, "prc_pr_alto")
        with c2:
            st.markdown("##### Os 15 mais baratos")
            _tabela(baixo, cols, "prc_pr_baixo")

    st.markdown("##### K do agente")
    k = df[df["k"].notna()]
    if not k.empty:
        fig = go.Figure()
        for calc in ("MAX", "FLEX", "Indexado"):
            y = k[k["calculo"] == calc]
            if not y.empty:
                fig.add_trace(go.Histogram(x=y["k"], name=calc, marker=dict(color=COR_CALCULO[calc]), xbins=dict(size=1),
                                           hovertemplate="K %{x} €/MWh: %{y} CPEs<extra></extra>"))
        fig = _fig(fig, 300, unificado=False)
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title_text="K (€/MWh)")
        fig.update_yaxes(title_text="CPEs")
        _mostra(fig, "prc_pr_k")
    sem = df[(df["calculo"] != "") & df["k"].isna()]
    if len(sem):
        st.caption(f"{n(len(sem))} CPEs sem K na Lista de Preços (nem no «Escalão», nem no K indexado, nem «K<n>» no "
                   f"K fixo).")

    st.markdown("##### Tarifas indexadas")
    ix = df[df["tipo"] == "Indexada"].copy()
    if ix.empty:
        st.caption("Nenhuma com estes filtros.")
    else:
        ix["equivalente"] = PD.equivalente_indexado(ix, q["omie"], q["parametros"])
        om = q["meta"].get("omie", {})
        st.caption(f"O preço destas segue o OMIE de cada hora: (OMIE + CGS + K) × perdas. O equivalente é o que cada uma "
                   f"teria dado com o OMIE de {om.get('de', '—')} a {om.get('ate', '—')} pesado pelo perfil do CPE, a "
                   f"CGS e as perdas de omissão do nível — uma estimativa para comparar com as fixas, não um preço.")
        _tabela(ix.sort_values("consumo_mwh", ascending=False),
                {**{c: _COL_CPE[c] for c in ("cliente", "cpe", "nivel", "tarifa", "agente", "consumo_mwh", "k")},
                 "equivalente": ("Equivalente (€/kWh)", 4), "inclui_redes": ("Redes", "texto")},
                "prc_pr_ind", altura=320)

    st.markdown("##### Preço da potência")
    p = df[df["preco_potencia"].notna() & df["potencia_kva"].notna() & (df["potencia_kva"] > 0)]
    if not p.empty:
        fig = go.Figure()
        for r in ("Não", "Sim"):
            y = p[p["inclui_redes"] == r]
            col = "preco_potencia_redes" if r == "Sim" and y["preco_potencia_redes"].notna().any() else "preco_potencia"
            if not y.empty:
                fig.add_trace(go.Scatter(x=y["potencia_kva"], y=y[col], mode="markers",
                                         name=f"{'Com' if r == 'Sim' else 'Sem'} redes",
                                         marker=dict(color=COR_REDES[r], size=6, opacity=0.6),
                                         customdata=np.stack([y["cliente"], y["tarifa"]], axis=-1),
                                         hovertemplate="%{x} kVA · %{y:.4f} €/dia<br>%{customdata[0]}<br>"
                                                       "%{customdata[1]}<extra></extra>"))
        fig = _fig(fig, 320, unificado=False)
        fig.update_xaxes(type="log", title_text="Potência contratada (kVA, escala logarítmica)")
        fig.update_yaxes(title_text="€/dia")
        _mostra(fig, "prc_pr_pot")


# ---------------------------------------------------------------------------
# Clientes
# ---------------------------------------------------------------------------
_COL_CLIENTE = {"cliente": ("Cliente", "texto"), "segmento": ("Segmento", "texto"), "n_cpes": ("CPEs", 0),
                "consumo_mwh": ("Consumo (MWh/ano)", 1), "quota_consumo": ("% do consumo", 2),
                "preco_medio": ("Preço médio (€/kWh)", 4), "k_medio": ("K médio (€/MWh)", 1),
                "niveis": ("Níveis", "texto"), "familias": ("Famílias", "texto"), "tarifarios": ("Tarifários", "texto"),
                "agentes": ("Agentes", "texto"), "fim_contrato": ("Fim de contrato (1.º)", "data"),
                "especial": ("Preço especial", "texto"), "risco": ("Risco", "texto")}


def _clientes(df: pd.DataFrame, hoje: pd.Timestamp):
    cl = PD.clientes(df)
    if cl.empty:
        st.info("Nenhum CPE com estes filtros.")
        return
    procura = st.text_input("Procurar cliente, tarifário ou agente", key="prc_cl_procura",
                            placeholder="parte do nome…").strip().lower()
    vis = cl
    if procura:
        texto = (cl["cliente"].fillna("") + " " + cl["tarifarios"] + " " + cl["agentes"]).str.lower()
        vis = cl[texto.str.contains(procura, regex=False)]
    st.markdown(f"##### Clientes ({n(len(vis))})")
    _tabela(vis, _COL_CLIENTE, "prc_cl_tab", altura=420, baixar="clientes.csv")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("##### Concentração do consumo")
        top = cl.head(50)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1, len(top) + 1)), y=top["quota_consumo"], name="% do consumo",
                             marker=dict(color=COR["exc"]), customdata=top["cliente"],
                             hovertemplate="%{customdata}: %{y:.2f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=list(range(1, len(top) + 1)), y=top["quota_acumulada"], name="Acumulado",
                                 mode="lines", line=dict(color=COR["consumo"], width=2), yaxis="y2",
                                 hovertemplate="%{y:.1f} % acumulado<extra></extra>"))
        fig = _fig(fig, 340, unificado=False)
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], ticksuffix=" %", showgrid=False,
                                      automargin=True, tickcolor=COR["eixo"]))
        fig.update_xaxes(title_text="Os 50 maiores clientes")
        fig.update_yaxes(ticksuffix=" %")
        _mostra(fig, "prc_cl_pareto")
    with c2:
        st.markdown("##### Um cliente")
        nomes = cl.set_index("cliente_id")["cliente"].fillna("").to_dict()
        escolha = st.selectbox("Cliente", list(vis["cliente_id"]) or list(cl["cliente_id"]),
                               format_func=lambda i: f"{nomes.get(i, i)} ({i})", key="prc_cl_um")
        x = df[df["cliente_id"] == escolha]
        if not x.empty:
            r = cl[cl["cliente_id"] == escolha].iloc[0]
            k = st.columns(3)
            _metrica(k[0], "CPEs", n(len(x)))
            _metrica(k[1], "Consumo", f"{n(x['consumo_mwh'].sum(), 1)} MWh/ano")
            _metrica(k[2], "Preço médio", _eur_kwh(r["preco_medio"]))
            _tabela(x, {c: _COL_CPE[c] for c in ("cpe", "estado", "nivel", "horario", "tarifa", "inclui_redes",
                                                  "consumo_mwh", "preco_medio", "p_P", "p_C", "p_VN", "p_SV", "k",
                                                  "agente", "fim_contrato")}, "prc_cl_cpes")

    st.markdown("##### Contratos a terminar")
    dias = st.radio("Nos próximos", [30, 60, 90, 180, 365], index=2, horizontal=True, key="prc_cl_dias",
                    format_func=lambda d: f"{d} dias")
    t = PD.a_terminar(df, hoje, dias)
    st.caption(f"{n(len(t))} CPEs, {n(t['consumo_mwh'].sum())} MWh/ano.")
    _tabela(t, {c: _COL_CPE[c] for c in ("fim_contrato", "cliente", "cpe", "nivel", "tarifa", "consumo_mwh",
                                         "preco_medio", "k", "agente", "renovacao_automatica")},
            "prc_cl_fim", altura=320, baixar="contratos_a_terminar.csv")

    mud = df[df["em_mudanca_para"].fillna("").str.strip() != ""]
    st.markdown(f"##### Em mudança para outro comercializador ({n(len(mud))})")
    if len(mud):
        _tabela(mud, {c: _COL_CPE[c] for c in ("cliente", "cpe", "em_mudanca_para", "nivel", "tarifa", "consumo_mwh",
                                               "preco_medio", "agente")}, "prc_cl_mud")
    esp = df[df["preco_especial"] == "Sim"]
    st.markdown(f"##### Com preço especial ({n(len(esp))})")
    if len(esp):
        _tabela(esp, {c: _COL_CPE[c] for c in ("cliente", "cpe", "nivel", "horario", "tarifa", "consumo_mwh",
                                               "preco_medio", "k", "agente", "fim_contrato")}, "prc_cl_esp")


# ---------------------------------------------------------------------------
# Agentes
# ---------------------------------------------------------------------------
def _agentes(df: pd.DataFrame):
    a = PD.por_agente(df)
    if a.empty:
        st.info("Nenhum CPE com estes filtros.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Consumo por agente")
        extra = [f"{n(x)} CPEs · {n(y)} clientes" for x, y in zip(a["n_cpes"], a["n_clientes"])]
        _mostra(_barras_h(a["agente"], a["consumo_mwh"], COR["exc"], "MWh/ano", hover_extra=extra), "prc_ag_mwh")
    with c2:
        st.markdown("##### K médio por agente")
        b = a.dropna(subset=["k_medio"]).sort_values("k_medio", ascending=False)
        _mostra(_barras_h(b["agente"], b["k_medio"], COR["consumo"], "€/MWh", 1), "prc_ag_k")
    _tabela(a, {"agente": ("Agente", "texto"), "n_cpes": ("CPEs", 0), "n_clientes": ("Clientes", 0),
                "consumo_mwh": ("Consumo (MWh/ano)", 0), "quota_consumo": ("% do consumo", 1),
                "preco_medio": ("Preço médio (€/kWh)", 4), "k_medio": ("K médio (€/MWh)", 1),
                "pct_indexado": ("% indexado", 0), "pct_redes": ("% CPEs com redes", 0)},
            "prc_ag_tab", baixar="agentes.csv")
    st.caption("O preço médio mistura tarifas com e sem redes: para comparar agentes, filtra «Inclui redes».")


# ---------------------------------------------------------------------------
# Simulador
# ---------------------------------------------------------------------------
_CAMPOS_MODELO = ("nome", "calculo", "nivel", "horario", "ciclo", "perfil", "consumo", "potencia", "preco_potencia",
                  "cgs", "custos", "k_interno", "k_agente", "perdas", "omie", "tar")


def _contratos_omip(omip: pd.DataFrame | None) -> dict[str, float]:
    if omip is None or omip.empty or "current_price" not in omip:
        return {}
    x = omip.dropna(subset=["current_price"]).drop_duplicates("contract")
    return {str(c).replace("_", "/"): float(p) for c, p in zip(x["contract"], x["current_price"])}


def _excel(nome: str, entradas: dict, sim: pd.DataFrame, hv: pd.DataFrame, res: dict) -> bytes:
    """A proposta num livro: os pressupostos, os preços do tarifário e as parcelas de cada período."""
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        pd.DataFrame([{"Campo": k, "Valor": v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False)}
                      for k, v in entradas.items()]).to_excel(xl, sheet_name="Pressupostos", index=False)
        hv.rename(columns={"preco_nome": "Preço", "periodos": "Períodos", "peso": "Parte do consumo",
                           "preco": "€/kWh"}).to_excel(xl, sheet_name="Tarifário", index=False)
        sim.rename(columns={"periodo": "Período", "nome": "Nome", "peso": "Parte do consumo", "omie": "OMIE (€/MWh)",
                            "cgs": "CGS (€/MWh)", "custos": "Custos indiretos (€/MWh)", "k_interno": "K interno (€/MWh)",
                            "k_agente": "K do agente (€/MWh)", "perdas": "Perdas (€/MWh)", "tar": "TAR (€/MWh)",
                            "preco": "Preço (€/kWh)"}).to_excel(xl, sheet_name="Parcelas", index=False)
        pd.DataFrame([{"Preço médio (€/kWh)": res["preco_medio"], "Energia (€/ano)": res["energia_ano"],
                       "Potência (€/ano)": res["potencia_ano"], "Total (€/ano)": res["total_ano"],
                       "Margem K interno (€/ano)": res["margem_k_interno"],
                       "Comissão do agente (€/ano)": res["comissao_agente"]}]).to_excel(xl, sheet_name="Resumo",
                                                                                          index=False)
    return out.getvalue()


def _simulador(q: dict, todos: pd.DataFrame, omip: pd.DataFrame | None):
    par = q["parametros"]
    st.markdown("Cria um tarifário com os pressupostos que quiseres — para um cliente da carteira ou um novo — e vê o "
                "preço de cada período, a fatura do ano, a margem e onde fica face à carteira.")
    with st.expander("As duas fórmulas", expanded=False):
        st.markdown(
            "- **Cálculo MAX** — só OMIE, custos indiretos, K interno e K do agente:  \n"
            "  `preço (€/kWh) = (OMIE + custos indiretos + K interno + K do agente) / 1000`\n"
            "- **Cálculo FLEX** — todos os pressupostos:  \n"
            "  `preço (€/kWh) = (OMIE + CGS + custos indiretos + K interno + K do agente) × perdas / 1000 + TAR`\n\n"
            "OMIE, CGS, custos e K em €/MWh; perdas é o fator (1,16 = 16 %); TAR em €/kWh. Cada período (ponta, "
            "cheias, vazio normal, super vazio) tem o seu OMIE e a sua TAR; os preços do horário do cliente (simples, "
            "bi, tri) são a média dos períodos que juntam, pesada pelo consumo do perfil.")
    # ---- de onde se parte (um modelo carregado vale sobre tudo)
    cand = todos[todos["estado"].isin(["Ativo", "Contrato"]) & (todos["calculo"] != "")].sort_values(
        "consumo_mwh", ascending=False)
    pendente = st.session_state.pop("prc_sim_pendente", None)     # um modelo aberto: antes de desenhar os widgets
    if pendente:
        st.session_state["prc_sim_modelo"] = pendente["modelo"]
        cpe_m = pendente.get("cpe")
        st.session_state["prc_sim_origem"] = "Um CPE da carteira" if cpe_m in set(cand["cpe"]) else "Um cliente novo"
        if cpe_m in set(cand["cpe"]):
            st.session_state["prc_sim_cpe"] = cpe_m
    modelo = st.session_state.get("prc_sim_modelo") or {}
    c = st.columns([1.3, 3])
    origem = c[0].radio("Partir de", ["Um CPE da carteira", "Um cliente novo"], key="prc_sim_origem")
    base = None
    if origem == "Um CPE da carteira":
        rot = {r.cpe: f"{r.cliente} — {r.cpe} — {r.tarifa} ({n(r.consumo_mwh or 0, 1)} MWh)" for r in cand.itertuples()}
        cpe = c[1].selectbox("CPE (escreve para procurar pelo cliente, CPE ou tarifário)", list(rot), format_func=rot.get,
                             key="prc_sim_cpe")
        base = cand[cand["cpe"] == cpe].iloc[0] if cpe else None
    sufixo = f"{base['cpe'] if base is not None else 'novo'}_{modelo.get('_id', '')}"

    def dv(campo, omissao):
        return modelo.get(campo, omissao) if modelo else omissao

    perfis = sorted(q["pesos"]["perfil"].unique()) if not q["pesos"].empty else ["A", "B", "C"]
    nivel0 = base["nivel"] if base is not None and base["nivel"] in PD.NIVEIS else "BTE"
    hor0 = base["horario"] if base is not None and base["horario"] in PD.PRECOS_DO_HORARIO else "TETRA"
    st.markdown("##### O cliente")
    l = st.columns(6)
    nivel = l[0].selectbox("Nível de tensão", PD.NIVEIS, index=PD.NIVEIS.index(dv("nivel", nivel0)), key=f"prc_sim_nivel_{sufixo}")
    horarios = list(PD.PRECOS_DO_HORARIO)
    horario = l[1].selectbox("Horário", horarios, index=horarios.index(dv("horario", hor0)), key=f"prc_sim_hor_{sufixo}",
                             format_func=lambda h: {"SIMPLES": "Simples", "BI": "Bi-horário", "TRI": "Tri-horário",
                                                    "TETRA": "Tetra-horário"}[h])
    if nivel in ("MT", "AT"):
        l[2].selectbox("Ciclo", ["Semanal"], key=f"prc_sim_ciclo_mt_{sufixo}", disabled=True,
                       help="MT e AT só têm o ciclo semanal (períodos de 2027), com os feriados como domingos.")
        tipo, ciclo = "semanal_feriados", "Semanal"
    else:
        ciclo0 = "Semanal" if base is not None and "semanal" in str(base["ciclo"]).lower() else "Diário"
        ciclo = l[2].selectbox("Ciclo", ["Diário", "Semanal"], index=["Diário", "Semanal"].index(dv("ciclo", ciclo0)),
                               key=f"prc_sim_ciclo_{sufixo}")
        tipo = "semanal" if ciclo == "Semanal" else "diario"
    perfil0 = base["perfil"] if base is not None and base["perfil"] in perfis else ("A" if nivel != "BTN" else "C")
    perfil = l[3].selectbox("Perfil (ERSE)", perfis, index=perfis.index(dv("perfil", perfil0)) if dv("perfil", perfil0)
                            in perfis else 0, key=f"prc_sim_perfil_{sufixo}",
                            help="O perfil de consumo da ERSE: dá a parte do consumo em cada período e o OMIE de cada "
                                 "período pesado por ele.")
    consumo = l[4].number_input("Consumo (MWh/ano)", min_value=0.0, step=10.0, format="%.1f",
                                value=float(dv("consumo", base["consumo_mwh"] if base is not None and
                                               pd.notna(base["consumo_mwh"]) else 100.0)), key=f"prc_sim_cons_{sufixo}")
    potencia = l[5].number_input("Potência (kVA)", min_value=0.0, step=1.0, format="%.2f",
                                 value=float(dv("potencia", base["potencia_kva"] if base is not None and
                                                pd.notna(base["potencia_kva"]) else 0.0)), key=f"prc_sim_pot_{sufixo}")
    pesos = PD.pesos_do_perfil(q["pesos"], perfil, tipo)
    hist = PD.omie_do_perfil(q["omie"], perfil, tipo)
    if not pesos or not hist:
        st.warning("Não há pesos dos períodos ou OMIE para este perfil e ciclo nos dados: escolhe outro perfil.")
        return

    st.markdown("##### Os pressupostos")
    l = st.columns([1, 2, 1])
    calc0 = base["calculo"] if base is not None and base["calculo"] in ("MAX", "FLEX") else "MAX"
    calculo = l[0].radio("Cálculo", ["MAX", "FLEX"], index=["MAX", "FLEX"].index(dv("calculo", calc0)), horizontal=True,
                         key=f"prc_sim_calc_{sufixo}",
                         help="MAX: OMIE + custos indiretos + K interno + K do agente. FLEX: todos os pressupostos "
                              "(com CGS, perdas e TAR).")
    flex = calculo == "FLEX"
    contratos = _contratos_omip(omip)
    fontes = ["OMIE histórico, pesado pelo perfil"] + [f"OMIP {k} — {n(v, 2)} €/MWh" for k, v in contratos.items()] + \
             ["Valor à mão (baseload)"]
    fonte = l[1].selectbox("OMIE de referência", fontes, key=f"prc_sim_fonte_{sufixo}",
                           help="O OMIE de cada período: o histórico pesado pelo perfil; ou um preço de futuros (OMIP) "
                                "ou um valor à mão, ambos de baseload, com a forma do perfil (cada período vale o mesmo "
                                "em proporção do baseload que valeu no histórico). Todos editáveis na tabela abaixo.")
    if fonte.startswith("OMIP"):
        ref = contratos[fonte.split(" — ")[0][5:]]
        omie0 = PD.omie_por_referencia(hist, ref)
    elif fonte.startswith("Valor"):
        ref = l[2].number_input("Baseload (€/MWh)", value=round(hist["base"], 2), step=1.0, key=f"prc_sim_ref_{sufixo}")
        omie0 = PD.omie_por_referencia(hist, ref)
    else:
        omie0 = {p: hist[p] for p in PD.PERIODOS}
        l[2].caption(f"Histórico de {pd.Timestamp(hist['de']):%d/%m/%Y} a {pd.Timestamp(hist['ate']):%d/%m/%Y}: "
                     f"baseload {n(hist['base'], 2)} €/MWh; pesado pelo perfil {n(hist['medio'], 2)} €/MWh.")
    if modelo.get("omie"):
        omie0 = {p: float(modelo["omie"].get(p, omie0[p])) for p in PD.PERIODOS}

    cg = _cgs_real()
    l = st.columns(5)
    cgs = l[0].number_input("CGS (€/MWh)", value=float(dv("cgs", par.get("cgs", {}).get(nivel, 20.0))), step=0.5,
                            key=f"prc_sim_cgs_{sufixo}_{nivel}", disabled=not flex,
                            help="Custo de gestão do sistema (desvios e encargos de regulação). Só entra no FLEX." +
                                 (f" A real da NOVAE de {cg[1]:%d/%m/%Y} a {cg[2]:%d/%m/%Y}: {n(cg[0], 2)} €/MWh "
                                  f"(Dashboard Trading)." if cg else ""))
    custos = l[1].number_input("Custos indiretos (€/MWh)", value=float(dv("custos", par.get("custos", 2.4))), step=0.1,
                               key=f"prc_sim_custos_{sufixo}")
    k_int = l[2].number_input("K interno (€/MWh)", value=float(dv("k_interno", par.get("k_interno", 15.0))), step=0.5,
                              key=f"prc_sim_kint_{sufixo}")
    k0 = base["k"] if base is not None and pd.notna(base["k"]) else par.get("k_agente", 4.0)
    k_ag = l[3].number_input("K do agente (€/MWh)", value=float(dv("k_agente", k0)), step=0.5, key=f"prc_sim_kag_{sufixo}")
    perdas = l[4].number_input("Perdas (fator)", value=float(dv("perdas", par.get("perdas", {}).get(nivel, 1.16))),
                               step=0.01, format="%.3f", key=f"prc_sim_perdas_{sufixo}_{nivel}", disabled=not flex,
                               help="1,16 = 16 % de perdas na rede. Só entra no FLEX.")
    if cg:
        st.caption(f"CGS real da NOVAE nos últimos 365 dias liquidados ({cg[1]:%d/%m/%Y} a {cg[2]:%d/%m/%Y}): "
                   f"**{n(cg[0], 2)} €/MWh** (desvios + encargos por MWh consumido, do Dashboard Trading).")

    tar_omissao = (modelo.get("tar") or par.get("tar", {}).get(nivel) or {})
    tabela = pd.DataFrame({"Período": [PD.NOME_PERIODO[p] for p in PD.PERIODOS],
                           "Parte do consumo (%)": [100 * pesos[p] for p in PD.PERIODOS],
                           "OMIE (€/MWh)": [round(omie0[p], 2) for p in PD.PERIODOS],
                           "TAR (€/kWh)": [float(tar_omissao.get(p, 0.0)) for p in PD.PERIODOS]})
    st.markdown("**OMIE e TAR de cada período** — editáveis")
    chave_tab = f"prc_sim_tab_{sufixo}_{nivel}_{perfil}_{tipo}_{fonte}_{round(sum(omie0.values()), 4)}"
    ed = st.data_editor(tabela, hide_index=True, key=chave_tab, disabled=["Período", "Parte do consumo (%)"] +
                        ([] if flex else ["TAR (€/kWh)"]),
                        column_config={"Parte do consumo (%)": st.column_config.NumberColumn(format="%.1f"),
                                       "OMIE (€/MWh)": st.column_config.NumberColumn(format="%.2f", step=0.01),
                                       "TAR (€/kWh)": st.column_config.NumberColumn(format="%.4f", step=0.0001)},
                        **_LARGO)
    omie = {p: float(v) for p, v in zip(PD.PERIODOS, ed["OMIE (€/MWh)"])}
    tar = {p: float(v or 0) for p, v in zip(PD.PERIODOS, ed["TAR (€/kWh)"])}
    if flex and not any(tar.values()):
        st.caption("⚠️ TAR a 0: escreve as tarifas de acesso às redes da ERSE em vigor para este nível e ciclo (ou "
                   "deixa a 0 para um preço sem redes). Ficam por omissão quando estiverem no `config.PRICING_TAR` do "
                   "projeto «Compra OMIE»." + (f" As do config são de {par['tar_ano']}." if par.get("tar_ano") else ""))
    pot_dia = st.number_input("Preço da potência (€/dia), opcional — para a fatura do ano",
                              value=float(dv("preco_potencia", base["preco_potencia"] if base is not None and
                                             pd.notna(base["preco_potencia"]) else 0.0)),
                              step=0.01, format="%.4f", key=f"prc_sim_ppot_{sufixo}")

    sim = PD.simular(calculo, omie, pesos, cgs, custos, k_int, k_ag, perdas, tar)
    hv = PD.precos_do_horario(sim, horario)
    res = PD.resumo_simulacao(sim, calculo, consumo, perdas, pot_dia)

    # ---- resultado
    st.markdown("##### O tarifário")
    k = st.columns(4)
    _metrica(k[0], "Preço médio de energia", _eur_kwh(res["preco_medio"], 5))
    _metrica(k[1], "Energia no ano", f"{n(res['energia_ano'])} €", ajuda="Preço médio × consumo anual.")
    _metrica(k[2], "Margem K interno", f"{n(res['margem_k_interno'])} €/ano",
             ajuda="K interno × consumo" + (" × perdas (no FLEX o K também passa pelas perdas)." if flex else "."))
    _metrica(k[3], "Comissão do agente", f"{n(res['comissao_agente'])} €/ano")
    c1, c2 = st.columns([2, 3])
    with c1:
        _tabela(hv, {"preco_nome": ("Preço", "texto"), "periodos": ("Períodos", "texto"),
                     "peso": ("Parte do consumo", 3), "preco": ("€/kWh", 5)}, f"prc_sim_hv_{sufixo}")
        if res["potencia_ano"]:
            st.caption(f"Potência: {n(res['potencia_ano'])} €/ano · total {n(res['total_ano'])} €/ano (sem IVA nem "
                       f"impostos).")
    with c2:
        fig = go.Figure()
        xs = [PD.NOME_PERIODO[p] for p in PD.PERIODOS] + ["Média"]
        for col, nome, cor in PARCELAS:
            if col in ("cgs", "perdas", "tar") and not flex:
                continue
            ys = list(sim[col]) + [res["parcelas"].get(col, 0.0)]
            fig.add_trace(go.Bar(x=xs, y=ys, name=nome, marker=dict(color=cor, line=dict(color=COR["fundo"], width=1)),
                                 hovertemplate=f"{nome}: %{{y:,.2f}} €/MWh<extra></extra>"))
        fig = _fig(fig, 320, unificado=False)
        fig.update_layout(barmode="relative")
        fig.update_yaxes(title_text="€/MWh")
        _mostra(fig, f"prc_sim_parcelas_{sufixo}")

    # ---- comparação
    st.markdown("##### Comparação")
    com_redes = flex and any(tar.values())
    if base is not None:
        atual = base["preco_medio"]
        if pd.notna(atual):
            dif = res["preco_medio"] - atual
            st.markdown(f"Hoje **{base['cliente']}** ({base['cpe']}) está no «{base['tarifa']}» a **{_eur_kwh(atual, 5)}** "
                        f"(média dos preços dele pesada pelo mesmo perfil). Este tarifário dá "
                        f"**{n(dif * 1000, 2, True)} €/MWh**, ou seja **{n(dif * consumo * 1000, 0, True)} €/ano** "
                        f"para o cliente.")
            if (base["inclui_redes"] == "Sim") != com_redes:
                st.caption("⚠️ O tarifário atual " + ("inclui" if base["inclui_redes"] == "Sim" else "não inclui") +
                           " as redes e este " + ("inclui" if com_redes else "não") + ": a diferença não é só energia.")
        else:
            st.caption(f"O tarifário atual dele («{base['tarifa']}», {base['tipo'].lower()}) não tem preço de energia "
                       f"fixo para comparar" + (f" (K {n(base['k'], 1)} €/MWh)." if pd.notna(base["k"]) else "."))
    comp = todos[(todos["estado"] == "Ativo") & (todos["tipo"] == "Fixa") & (todos["nivel"] == nivel) &
                 (todos["horario"] == horario) & (todos["inclui_redes"] == ("Sim" if com_redes else "Não"))]
    if len(comp) >= 5:
        pc = PD.percentil(res["preco_medio"], comp["preco_medio"])
        st.markdown(f"Entre os **{n(len(comp))}** CPEs ativos comparáveis ({nivel}, {horario.lower()}, tarifa fixa "
                    f"{'com' if com_redes else 'sem'} redes), este preço é mais alto do que **{n(pc)} %** deles "
                    f"(mediana {_eur_kwh(comp['preco_medio'].median(), 5)}, de {_eur_kwh(comp['preco_medio'].min())} "
                    f"a {_eur_kwh(comp['preco_medio'].max())}).")
        fig = go.Figure(go.Histogram(x=comp["preco_medio"], marker=dict(color=COR["apagado"]), nbinsx=30,
                                     hovertemplate="%{x} €/kWh: %{y} CPEs<extra></extra>"))
        fig.add_vline(x=res["preco_medio"], line=dict(color=COR_CALCULO[calculo], width=3),
                      annotation_text="este tarifário", annotation_position="top")
        fig = _fig(fig, 260, legenda=False, unificado=False)
        fig.update_layout(margin=dict(t=36))
        fig.update_xaxes(title_text="Preço médio dos CPEs comparáveis (€/kWh)", tickformat=".3f")
        fig.update_yaxes(title_text="CPEs")
        _mostra(fig, f"prc_sim_hist_{sufixo}")
    else:
        st.caption(f"Só há {len(comp)} CPEs ativos comparáveis ({nivel}, {horario.lower()}, fixa "
                   f"{'com' if com_redes else 'sem'} redes): poucos para os comparar.")

    # ---- guardar e exportar
    st.markdown("##### Guardar")
    nome = st.text_input("Nome do tarifário", value=str(dv("nome", f"Especial {base['cliente']}" if base is not None
                                                           else "Tarifário especial")), key=f"prc_sim_nome_{sufixo}")
    entradas = {"nome": nome, "calculo": calculo, "nivel": nivel, "horario": horario, "ciclo": ciclo, "perfil": perfil,
                "consumo": consumo, "potencia": potencia, "preco_potencia": pot_dia, "cgs": cgs, "custos": custos,
                "k_interno": k_int, "k_agente": k_ag, "perdas": perdas, "omie": omie, "tar": tar,
                "cliente": base["cliente"] if base is not None else "", "cpe": base["cpe"] if base is not None else "",
                "fonte_omie": fonte, "gerado": dt.datetime.now().isoformat(" ", "minutes")}
    c = st.columns(3)
    seguro = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in nome).strip() or "tarifario"
    c[0].download_button("Descarregar a proposta (Excel)", _excel(nome, entradas, sim, hv, res),
                         file_name=f"{seguro}.xlsx", key=f"prc_sim_xlsx_{sufixo}",
                         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c[1].download_button("Descarregar o modelo (para o abrir mais tarde)",
                         json.dumps(entradas, ensure_ascii=False, indent=1).encode("utf-8"),
                         file_name=f"{seguro}.json", mime="application/json", key=f"prc_sim_json_{sufixo}")
    carregado = c[2].file_uploader("Abrir um modelo guardado", type=["json"], key="prc_sim_abrir")
    if carregado is not None:
        try:
            m = json.loads(carregado.getvalue().decode("utf-8"))
            if not isinstance(m, dict) or not {"calculo", "nivel", "horario"} <= set(m):
                raise ValueError("não é um modelo do simulador")
        except (ValueError, UnicodeDecodeError) as e:
            st.error(f"Esse ficheiro não se abre: {e}")
        else:
            ident = f"{carregado.name}:{carregado.size}"
            if modelo.get("_ficheiro") != ident and st.session_state.get("prc_sim_ignorar") != ident:
                st.session_state["prc_sim_pendente"] = {
                    "modelo": {**{k: m[k] for k in _CAMPOS_MODELO if k in m},
                               "_id": dt.datetime.now().strftime("%H%M%S%f"), "_ficheiro": ident},
                    "cpe": m.get("cpe") or None}
                st.rerun()
    if modelo:
        st.caption(f"Aberto o modelo «{modelo.get('nome', '—')}»: os valores dele valem sobre os de omissão.")
        if st.button("Esquecer o modelo aberto", key="prc_sim_esquecer"):
            st.session_state["prc_sim_ignorar"] = modelo.get("_ficheiro")
            st.session_state.pop("prc_sim_modelo", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def _dados(q: dict):
    meta, par = q["meta"], q["parametros"]
    st.subheader("De onde vêm os dados")
    li, pt, pf, om = meta.get("lista", {}), meta.get("pontos", {}), meta.get("perfis", {}), meta.get("omie", {})
    st.markdown(
        f"- **Lista de Preços** da EGASI (tarifário e preços de cada CPE): «{li.get('ficheiro', '—')}», de "
        f"**{li.get('data', '—')}** ({li.get('cpes', '—')} CPEs).\n"
        f"- **Lista de Pontos de Entrega** (o estado de hoje, o fim de contrato, o concelho, o perfil, o risco): "
        f"{pt.get('data', '—')} ({pt.get('cpes', '—')} CPEs) — a que a compra do diário descarrega todos os dias.\n"
        f"- **Perfis de consumo da ERSE** ({pf.get('ano', '—')}, do ficheiro da compra «{pf.get('ficheiro', '—')}»), "
        f"com os períodos horários de {par.get('ano_periodos', '—')} (Diretiva ERSE n.º 3/2026).\n"
        f"- **OMIE** (Portugal) pesado pelo perfil: {om.get('de', '—')} a {om.get('ate', '—')} ({om.get('dias', '—')} dias).\n"
        f"- Pressupostos de omissão: {par.get('fonte', '—')}.\n"
        f"- Gerado a **{meta.get('gerado', '—')}**.")
    st.caption("Os dados têm nomes de clientes: no servidor estão cifrados, como os do trading. O NIF, o IBAN, o e-mail "
               "e o telefone dos clientes não vêm para o dashboard; cada cliente tem um número (C0001…) que junta os "
               "CPEs dele.")
    avisos = meta.get("avisos", [])
    st.subheader(f"Avisos da leitura ({len(avisos)})")
    if avisos:
        st.dataframe(pd.DataFrame({"Aviso": avisos}), hide_index=True, key="prc_avisos", **_LARGO)
    else:
        st.markdown("🟢 Nenhum.")
    st.subheader("Pressupostos de omissão do simulador")
    linhas = [{"Nível": nv, "Perdas (fator)": par.get("perdas", {}).get(nv), "CGS (€/MWh)": par.get("cgs", {}).get(nv),
               **{f"TAR {PD.NOME_PERIODO[p]} (€/kWh)": (par.get("tar", {}).get(nv) or {}).get(p) for p in PD.PERIODOS}}
              for nv in PD.NIVEIS]
    st.dataframe(pd.DataFrame(linhas), hide_index=True, key="prc_par", **_LARGO)
    st.caption(f"Custos indiretos {n(par.get('custos'), 2)} €/MWh · K interno {n(par.get('k_interno'), 1)} €/MWh · K do "
               f"agente {n(par.get('k_agente'), 1)} €/MWh (quando o CPE não tem um).")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Parte do consumo em cada período, por perfil e ciclo**")
        st.dataframe(q["pesos"], hide_index=True, key="prc_pesos", **_LARGO)
    with c2:
        st.markdown("**OMIE de cada período pesado pelo perfil (€/MWh)**")
        st.dataframe(q["omie"].drop(columns=["de", "ate", "dias"], errors="ignore"), hide_index=True, key="prc_omie",
                     **_LARGO)


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
def render(omip: pd.DataFrame | None = None) -> None:
    """omip: a última previsão OMIP (contract, current_price), para a referência do OMIE no simulador."""
    q = _obter()
    if q is None:
        return
    todos = q["cpes"]
    if todos.empty:
        st.warning("Os dados de pricing estão vazios: corre outra vez o `pricing_dados.py`.")
        return
    hoje = pd.Timestamp(dt.date.today())
    li = q["meta"].get("lista", {})
    st.caption(f"Lista de Preços de {li.get('data', '—')} · estado dos CPEs de {q['meta'].get('pontos', {}).get('data', '—')}"
               f" · dados gerados a {q['meta'].get('gerado', '—')}")
    df = _filtros(todos)
    st.caption(f"{n(len(df))} de {n(len(todos))} CPEs com estes filtros (valem para Resumo, Tarifários, Preços, "
               f"Clientes e Agentes; o Simulador compara com a carteira ativa inteira).")
    abas = st.tabs(["Resumo", "Tarifários", "Preços", "Clientes", "Agentes", "Simulador", "Dados"])
    with abas[5]:
        _simulador(q, todos, omip)
    with abas[6]:
        _dados(q)
    with abas[0]:
        _resumo(df, todos, q, hoje)
    with abas[1]:
        _tarifarios(df)
    with abas[2]:
        _precos(df, q)
    with abas[3]:
        _clientes(df, hoje)
    with abas[4]:
        _agentes(df)
