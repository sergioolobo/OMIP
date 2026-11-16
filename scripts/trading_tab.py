"""
trading_tab.py — a tab «Dashboard Trading NOSSA Energia»
=======================================================
Desenha a tab principal do dashboard a partir dos dados do `trading_dados.py`:
resumo e conclusões, histórico de compras, desvios (excesso e defeito), preços
(OMIE e desvios), CGS, comparação com o mercado e estado dos dados.

Chamado pelo 07_dashboard.py dentro de `with tab_trading:` — por isso nunca se
chama `st.stop()` aqui (pararia também as outras tabs): sai-se com `return`.

Cores: cada entidade tem sempre a mesma cor (compra roxo, consumo laranja,
excesso azul, defeito vermelho, OMIE verde-água, desvios amarelo, encargos
verde); os pares que aparecem juntos foram validados para daltonismo.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import trading_dados as TD
from trading_dados import n

COR = dict(compra="#6250d6", consumo="#eb6834", exc="#2a78d6", dfc="#e34948", omie="#1baf7a",
           desvios="#eda100", encargos="#008300", neutro="#898781", tinta="#0b0b0b", tinta2="#52514e",
           grelha="#e1e0d9", eixo="#c3c2b7", fundo="#ffffff", apagado="#c3c2b7")
DIVERGENTE = [[0.0, COR["dfc"]], [0.5, "#f0efec"], [1.0, COR["exc"]]]       # defeito ← 0 → excesso
SEQUENCIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ICONE = {"alerta": "🔴", "atencao": "🟠", "bom": "🟢", "info": "⚪"}
_LARGO = ({"width": "stretch"} if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 49)
          else {"use_container_width": True})
_FMT_DATA = {"Dia": "%d/%m/%Y", "Semana": "semana de %d/%m/%Y", "Mês": "%m/%Y"}


# ---------------------------------------------------------------------------
# Dados (com cache) e entrada
# ---------------------------------------------------------------------------
def _preparar(textos: dict) -> dict:
    q = TD.quadros(textos)
    q["diario"] = TD.diario(q["periodos"]) if not q["periodos"].empty else pd.DataFrame()
    q["mercado_dia"] = TD.mercado_diario(q["mercado"])
    return q


@st.cache_data(show_spinner="A carregar os dados de trading…")
def _quadros_claro(assinatura: str) -> dict:
    return _preparar(TD.ler_claro())


@st.cache_data(show_spinner="A abrir os dados de trading…")
def _quadros_cifrado(assinatura: str, palavra: str) -> dict:
    return _preparar(TD.ler_cifrado(palavra))


def _obter() -> dict | None:
    origem = TD.origem()
    if origem is None:
        st.info("Ainda não há dados de trading neste servidor. Geram-se no computador da empresa com "
                "`python trading_dados.py --publicar` (projeto «Compra OMIE»).")
        return None
    assinatura = TD.assinatura()
    if origem == "claro":
        return _quadros_claro(assinatura)
    palavra = st.session_state.get("trading_palavra")
    if palavra:
        try:
            return _quadros_cifrado(assinatura, palavra)
        except TD.PalavraErrada:
            st.session_state.pop("trading_palavra", None)
    st.markdown("#### 🔒 Área reservada à equipa da NOSSA Energia")
    st.caption("As compras, os desvios e os custos da empresa só se abrem com a palavra-passe da equipa. "
               "As previsões OMIP e OMIE, nas outras tabs, continuam abertas.")
    with st.form("trading_entrada"):
        tentativa = st.text_input("Palavra-passe", type="password")
        if st.form_submit_button("Entrar") and tentativa:
            try:
                _quadros_cifrado(assinatura, tentativa)
            except TD.PalavraErrada:
                st.error("Palavra-passe errada.")
            else:
                st.session_state["trading_palavra"] = tentativa
                st.rerun()
    return None


# ---------------------------------------------------------------------------
# Peças
# ---------------------------------------------------------------------------
def _fig(fig: go.Figure, altura: int = 340, legenda: bool = True, unificado: bool = True) -> go.Figure:
    fig.update_layout(
        height=altura, title_text="", paper_bgcolor=COR["fundo"], plot_bgcolor=COR["fundo"],
        font=dict(color=COR["tinta2"], size=12, family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        margin=dict(l=8, r=8, t=34 if legenda else 12, b=8), separators=", ",
        hovermode="x unified" if unificado else "closest", showlegend=legenda,
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left", title_text=""),
        bargap=0.18, hoverlabel=dict(bgcolor="white", font_size=12),
    )
    # automargin: sem ele, com margens tão curtas, os números e os títulos dos eixos ficam cortados
    fig.update_xaxes(showgrid=False, linecolor=COR["eixo"], tickcolor=COR["eixo"], ticks="outside",
                     automargin=True, title_standoff=10)
    fig.update_yaxes(gridcolor=COR["grelha"], zerolinecolor=COR["eixo"], zerolinewidth=1, automargin=True,
                     title_standoff=10)
    return fig


def _mostra(fig: go.Figure, chave: str) -> None:
    st.plotly_chart(fig, key=chave, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                    **_LARGO)


def _linha(x, y, nome, cor, unidade, casas=1, tracejado=None, largura=2, **kw) -> go.Scatter:
    kw.setdefault("mode", "lines")
    if "markers" in kw["mode"]:
        kw.setdefault("marker", dict(size=8, color=cor, line=dict(color=COR["fundo"], width=2)))
    return go.Scatter(x=x, y=y, name=nome, line=dict(color=cor, width=largura, dash=tracejado),
                      hovertemplate=f"%{{y:,.{casas}f}} {unidade}", **kw)


def _barra(x, y, nome, cor, unidade, casas=1, **kw) -> go.Bar:
    return go.Bar(x=x, y=y, name=nome, marker=dict(color=cor, line=dict(color=COR["fundo"], width=1)),
                  hovertemplate=f"%{{y:,.{casas}f}} {unidade}", **kw)


def _eixo_tempo(fig: go.Figure, grao: str) -> None:
    fig.update_xaxes(tickformat="%m/%Y" if grao == "Mês" else "%d/%m", hoverformat=_FMT_DATA[grao])


def _tabela(df: pd.DataFrame, colunas: dict, chave: str, altura: int | None = None, baixar: str | None = None):
    """colunas: {coluna: (título, casas | 'data' | 'texto')}."""
    cfg, ordem = {}, []
    for c, (titulo, tipo) in colunas.items():
        if c not in df.columns:
            continue
        ordem.append(c)
        if tipo == "data":
            cfg[c] = st.column_config.DateColumn(titulo, format="DD/MM/YYYY")
        elif tipo == "texto":
            cfg[c] = st.column_config.TextColumn(titulo)
        else:
            cfg[c] = st.column_config.NumberColumn(titulo, format=f"%.{tipo}f")
    kw = dict(hide_index=True, column_config=cfg, column_order=ordem, key=chave, **_LARGO)
    if altura:
        kw["height"] = altura
    st.dataframe(df, **kw)
    if baixar:
        st.download_button("Descarregar CSV", df[ordem].to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=baixar, mime="text/csv", key=chave + "_csv")


def _metrica(col, rotulo: str, valor: str, delta: str | None = None, inverso: bool = False, ajuda: str | None = None):
    col.metric(rotulo, valor, delta, delta_color="inverse" if inverso else "normal", help=ajuda)


def _delta(agora, antes, casas=1, unidade="", pct=False) -> str | None:
    if antes is None or pd.isna(antes) or pd.isna(agora):
        return None
    if pct:
        return None if not antes else f"{n(100 * (agora - antes) / abs(antes), 1, True)} %"
    return f"{n(agora - antes, casas, True)}{unidade}"


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
_PRESETS = ["Últimos 30 dias", "Últimos 90 dias", "Mês corrente", "Mês anterior", "Ano corrente", "Tudo",
            "Personalizado"]


def _filtros(d_all: pd.DataFrame):
    primeiro, ultimo = d_all["dia"].min().date(), d_all["dia"].max().date()
    c1, c2, c3, c4, c5 = st.columns([1.5, 2.1, 1.6, 1.3, 1.9])
    preset = c1.selectbox("Período", _PRESETS, key="trd_preset",
                          help=f"Conta a partir do último dia liquidado pela REN ({ultimo:%d/%m/%Y}).")
    if preset == "Últimos 30 dias":
        ini, fim = ultimo - dt.timedelta(days=29), ultimo
    elif preset == "Últimos 90 dias":
        ini, fim = ultimo - dt.timedelta(days=89), ultimo
    elif preset == "Mês corrente":
        ini, fim = ultimo.replace(day=1), ultimo
    elif preset == "Mês anterior":
        fim = ultimo.replace(day=1) - dt.timedelta(days=1)
        ini = fim.replace(day=1)
    elif preset == "Ano corrente":
        ini, fim = ultimo.replace(month=1, day=1), ultimo
    else:
        ini, fim = primeiro, ultimo
    ini = max(ini, primeiro)
    if preset == "Personalizado":
        escolha = c2.date_input("Intervalo", (max(primeiro, ultimo - dt.timedelta(days=29)), ultimo),
                                min_value=primeiro, max_value=ultimo, format="DD/MM/YYYY", key="trd_datas")
        if isinstance(escolha, (tuple, list)) and len(escolha) == 2:
            ini, fim = escolha
    else:
        c2.text_input("Intervalo", f"{ini:%d/%m/%Y} a {fim:%d/%m/%Y}", disabled=True, key=f"trd_int_{preset}")
    dias = (fim - ini).days + 1
    # a agregação por omissão acompanha o tamanho do intervalo (a chave muda com ele, senão o
    # Streamlit ficava com a escolha anterior e «Tudo» desenhava 700 barras de um dia)
    sugerida = 0 if dias <= 92 else 1 if dias <= 400 else 2
    grao = c3.radio("Agregação", ["Dia", "Semana", "Mês"], horizontal=True, key=f"trd_grao_{sugerida}",
                    index=sugerida)
    ref = c4.number_input("CGS de referência (€/MWh)", 0.0, 200.0, 20.0, 0.5, key="trd_ref",
                          help="Linha de comparação nos gráficos do CGS. 20 €/MWh é o CGS da tabela dos "
                               "tarifários «flex»; põe 0 para a esconder.")
    sem = c5.toggle("Excluir dias sem posição no mercado", False, key="trd_sem",
                    help="Dias em que a REN não tem posição final nossa (a oferta não chegou ao mercado): o "
                         "consumo inteiro vai a defeito e domina as médias.")
    return pd.Timestamp(ini), pd.Timestamp(fim), grao, ref, sem


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
def _resumo(per, d, d_ant, ag, grao, ref):
    t = TD.total(d)
    a = TD.total(d_ant) if not d_ant.empty else None
    g = (lambda c: a[c]) if a is not None else (lambda c: None)
    st.caption("Variações face ao período anterior com a mesma duração." if a is not None
               else "Sem período anterior com a mesma duração para comparar.")
    k = st.columns(5)
    _metrica(k[0], "Compra (posição final)", f"{n(t['pos'])} MWh", _delta(t["pos"], g("pos"), pct=True),
             ajuda="O que ficou comprado no mercado (diário + intradiário), segundo a REN.")
    _metrica(k[1], "Consumo verificado", f"{n(t['qv'])} MWh", _delta(t["qv"], g("qv"), pct=True),
             ajuda="A «Quantidade Verificada» da REN: o que os clientes consumiram.")
    _metrica(k[2], "Desvio líquido", f"{n(t['desvio'], 0, True)} MWh", f"{n(t['desvio_pct'], 1, True)} % do consumo",
             ajuda="Compra − consumo. Positivo: comprou-se a mais (excesso). Negativo: a menos (defeito).")
    k[2].caption("")
    _metrica(k[3], "Desvio absoluto", f"{n(t['abs_pct'], 1)} %", _delta(t["abs_pct"], g("abs_pct"), 1, " p.p."),
             inverso=True, ajuda="(Excesso + defeito) ÷ consumo, somados quarto de hora a quarto de hora.")
    _metrica(k[4], "Preço médio OMIE", f"{n(t['p_base'], 2)} €/MWh", _delta(t["p_base"], g("p_base"), 2, " €/MWh"),
             inverso=True, ajuda="Média simples dos preços de 15 minutos de Portugal nos dias do período.")
    k = st.columns(5)
    _metrica(k[0], "Custo dos desvios", f"{n(t['cgs'])} €", f"{n(t['cgs_mwh'], 2)} €/MWh consumido", inverso=True,
             ajuda="O que os desvios custaram a mais do que comprar no diário a energia certa (o «CGS (€)» do livro).")
    _metrica(k[1], "Encargos de regulação", f"{n(t['enc'])} €", f"{n(t['enc_mwh'], 2)} €/MWh consumido", inverso=True,
             ajuda="Os encargos que a REN imputa ao nosso consumo, período a período.")
    _metrica(k[2], "CGS (desvios + encargos)", f"{n(t['cgs_total_mwh'], 2)} €/MWh",
             _delta(t["cgs_total_mwh"], g("cgs_total_mwh"), 2, " €/MWh"), inverso=True)
    _metrica(k[3], "Custo total da energia", f"{n(t['custo_total_mwh'], 2)} €/MWh",
             _delta(t["custo_total_mwh"], g("custo_total_mwh"), 2, " €/MWh"), inverso=True,
             ajuda="Consumo ao preço do OMIE + custo dos desvios + encargos, por MWh consumido.")
    _metrica(k[4], "Compra no OMIE", f"{n(t['custo_omie'])} €", f"{n(t['p_compra'], 2)} €/MWh comprado",
             ajuda="Posição final × preço do mercado diário.")
    k[4].caption("")

    st.subheader("O que os dados dizem")
    for tipo, frase in TD.conclusoes(per, d, ref):
        st.markdown(f"{ICONE[tipo]} {frase}")

    st.subheader("Compra face ao consumo")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06)
    fig.add_trace(_linha(ag["dia"], ag["pos"], "Compra (posição final)", COR["compra"], "MWh"), 1, 1)
    fig.add_trace(_linha(ag["dia"], ag["qv"], "Consumo verificado", COR["consumo"], "MWh"), 1, 1)
    fig.add_trace(_barra(ag["dia"], ag["exc"], "Excesso", COR["exc"], "MWh"), 2, 1)
    fig.add_trace(_barra(ag["dia"], -ag["dfc"], "Defeito", COR["dfc"], "MWh"), 2, 1)
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="MWh", row=1, col=1)
    fig.update_yaxes(title_text="Desvio (MWh)", row=2, col=1)
    _eixo_tempo(fig, grao)
    _mostra(_fig(fig, 470), "res_compra")
    st.caption("Em cima, o que se comprou e o que se consumiu. Em baixo, a soma dos excessos (para cima) e dos "
               "defeitos (para baixo) de cada quarto de hora — um dia pode ter os dois.")


# ---------------------------------------------------------------------------
# Compras
# ---------------------------------------------------------------------------
_COL_HIST = {
    "dia": ("Dia", "data"), "sem": ("", "texto"), "pos": ("Compra (MWh)", 1), "qv": ("Consumo (MWh)", 1),
    "desvio": ("Desvio (MWh)", 1), "desvio_pct": ("Desvio (%)", 1), "exc": ("Excesso (MWh)", 1),
    "dfc": ("Defeito (MWh)", 1), "p_base": ("OMIE médio (€/MWh)", 2), "p_compra": ("Preço da compra (€/MWh)", 2),
    "custo_omie": ("Compra no OMIE (€)", 0), "cgs": ("Custo dos desvios (€)", 0), "enc": ("Encargos (€)", 0),
    "cgs_total_mwh": ("CGS (€/MWh)", 2), "custo_total_mwh": ("Custo total (€/MWh)", 2),
    "nao_casado": ("Não casou (MWh)", 1), "estado": ("Liquidação", "texto"),
}


def _compras(per, d):
    st.subheader("Histórico das compras")
    h = TD.racios(d).sort_values("dia", ascending=False)
    h["sem"] = h["dia"].dt.dayofweek.map(lambda i: TD.DIAS_SEMANA[i])
    h["estado"] = h["estado"].map({"D": "Definitiva", "P": "Provisória"}).fillna("—")
    _tabela(h, _COL_HIST, "tab_hist", altura=390, baixar="compras_nossa_energia.csv")
    st.caption("Um dia por linha, do mais recente para o mais antigo. Compra = posição final na REN; consumo = "
               "quantidade verificada. O dia é o do mercado (hora de Espanha).")

    st.subheader("Um dia ao detalhe")
    dias = [x.date() for x in sorted(per["dia"].unique(), reverse=True)]
    liquidados = set(d["dia"].dt.date)
    c1, c2 = st.columns([1.2, 4])
    dia = c1.selectbox("Dia", dias, index=next((i for i, x in enumerate(dias) if x in liquidados), 0),
                       format_func=lambda x: f"{x:%d/%m/%Y} ({TD.DIAS_SEMANA[x.weekday()]})"
                                             + ("" if x in liquidados else " · por liquidar"), key="trd_dia")
    p = per[per["dia"] == pd.Timestamp(dia)].sort_values("periodo")
    x = p["periodo"]
    dd = TD.racios(d[d["dia"] == pd.Timestamp(dia)])
    if not dd.empty:
        r = dd.iloc[0]
        c2.markdown(f"**Compra** {n(r['pos'], 1)} MWh · **consumo** {n(r['qv'], 1)} MWh · **desvio** "
                    f"{n(r['desvio'], 1, True)} MWh ({n(r['desvio_pct'], 1, True)} %) · **custo dos desvios** "
                    f"{n(r['cgs'])} € · **encargos** {n(r['enc'])} € · **OMIE médio** {n(r['p_base'], 2)} €/MWh"
                    + (f" · **não casou** {n(r['nao_casado'], 1)} MWh" if r["nao_casado"] > 0 else ""))
    else:
        c2.markdown("A REN ainda não liquidou este dia: só há a oferta enviada.")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.42, 0.22, 0.36], vertical_spacing=0.05)
    if p["oferta_mwh"].notna().any():
        fig.add_trace(_linha(x, p["oferta_mwh"], "Oferta enviada", COR["neutro"], "MWh", 3, "dot", 1.5,
                             line_shape="hv"), 1, 1)
    fig.add_trace(_linha(x, p["pos"], "Compra (posição final)", COR["compra"], "MWh", 3, line_shape="hv",
                         connectgaps=False), 1, 1)
    fig.add_trace(_linha(x, p["qv"], "Consumo verificado", COR["consumo"], "MWh", 3), 1, 1)
    fig.add_trace(_barra(x, p["exc"], "Excesso", COR["exc"], "MWh", 3), 2, 1)
    fig.add_trace(_barra(x, -p["dfc"], "Defeito", COR["dfc"], "MWh", 3), 2, 1)
    fig.add_trace(_linha(x, p["p_omie"], "Preço OMIE", COR["omie"], "€/MWh", 2), 3, 1)
    fig.add_trace(_linha(x, p["p_def"], "Preço de defeito", COR["dfc"], "€/MWh", 2, largura=1.5), 3, 1)
    fig.add_trace(_linha(x, p["p_exc"], "Preço de excesso", COR["exc"], "€/MWh", 2, largura=1.5), 3, 1)
    if p["preco_lim"].notna().any():
        fig.add_trace(_linha(x, p["preco_lim"], "Nosso preço máximo", COR["neutro"], "€/MWh", 2, "dot", 1.5,
                             line_shape="hv"), 3, 1)
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="MWh / 15 min", row=1, col=1)
    fig.update_yaxes(title_text="Desvio", row=2, col=1)
    fig.update_yaxes(title_text="€/MWh", row=3, col=1)
    ultimo = int(p["periodo"].max())
    if ultimo <= TD.PERIODOS_HORARIOS:                     # dia liquidado à hora (até março de 2025)
        marcas = list(range(1, 25, 2))
        fig.update_xaxes(title_text="Hora de Espanha (Lisboa = uma hora menos)", row=3, col=1, tickmode="array",
                         tickvals=marcas, ticktext=[f"{v - 1:02d}:00" for v in marcas])
    elif ultimo == 96:
        marcas = list(range(1, 97, 8))
        fig.update_xaxes(title_text="Período de 15 minutos (hora de Espanha; Lisboa = uma hora menos)", row=3,
                         col=1, tickmode="array", tickvals=marcas,
                         ticktext=[f"{(v - 1) // 4:02d}:00" for v in marcas])
    else:                                                  # 92 ou 100 períodos: o relógio não é contínuo
        fig.update_xaxes(title_text=f"Período de 15 minutos (dia de mudança da hora: {ultimo} períodos)", row=3,
                         col=1, dtick=8)
    _mostra(_fig(fig, 620), "cmp_dia")
    st.caption("A compra interrompe-se onde a REN não tem posição nossa: são os períodos que não casaram (o preço "
               "do OMIE passou o nosso preço máximo) e o consumo dessas horas foi inteiro a defeito.")

    nc = TD.racios(d[d["nao_casado"] > 0]).sort_values("dia", ascending=False)
    if not nc.empty:
        st.subheader("Ofertas que não casaram")
        _tabela(nc, {"dia": ("Dia", "data"), "n_sem_pos": ("Períodos sem posição", 0),
                     "nao_casado": ("Oferta que não casou (MWh)", 1), "cgs_nao_casado": ("Custo acima do OMIE (€)", 0),
                     "dfc": ("Defeito do dia (MWh)", 1), "cgs": ("Custo dos desvios do dia (€)", 0)}, "tab_nc")
        st.caption("Só desde 16/09/2026, quando as ofertas passaram a ficar arquivadas. O custo é o que esses "
                   "períodos pagaram de defeito acima do preço do OMIE.")


# ---------------------------------------------------------------------------
# Desvios
# ---------------------------------------------------------------------------
def _desvios(per, d, ag, grao):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Excesso e defeito")
        fig = go.Figure([_barra(ag["dia"], ag["exc_pct"], "Excesso", COR["exc"], "% do consumo"),
                         _barra(ag["dia"], -ag["dfc_pct"], "Defeito", COR["dfc"], "% do consumo")])
        fig.update_layout(barmode="relative")
        fig.update_yaxes(title_text="% do consumo")
        _eixo_tempo(fig, grao)
        _mostra(_fig(fig), "dsv_pct")
    with c2:
        st.subheader("Quanto custou cada lado")
        fig = go.Figure([_barra(ag["dia"], ag["cgs_exc"], "Por excesso", COR["exc"], "€", 0),
                         _barra(ag["dia"], ag["cgs_def"], "Por defeito", COR["dfc"], "€", 0)])
        fig.update_layout(barmode="relative")
        fig.update_yaxes(title_text="€ acima do OMIE")
        _eixo_tempo(fig, grao)
        _mostra(_fig(fig), "dsv_custo")
    st.caption("À esquerda, o tamanho do erro; à direita, o que custou. Um valor negativo no custo é ganho: o preço "
               "do desvio foi melhor do que o do OMIE nesse sentido.")

    st.subheader("A que horas se erra")
    h = TD.por_hora(per)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=h["hora"], y=h["desvio_pct"], name="Desvio líquido",
            marker=dict(color=np.where(h["desvio_pct"] >= 0, COR["exc"], COR["dfc"]),
                        line=dict(color=COR["fundo"], width=1)),
            customdata=h["dias_defeito_pct"],
            hovertemplate="%{y:+,.1f} % do consumo · em defeito em %{customdata:.0f} % dos dias"))
        fig.update_yaxes(title_text="Desvio líquido (% do consumo)")
        fig.update_xaxes(title_text="Hora de mercado (Espanha)", dtick=2)
        _mostra(_fig(fig, 320, legenda=False), "dsv_hora")
        st.caption("Azul: nessa hora compra-se a mais. Vermelho: a menos. É o viés médio do período.")
    with c2:
        fig = go.Figure([_barra(h["hora"], h["cgs_exc"], "Por excesso", COR["exc"], "€", 0),
                         _barra(h["hora"], h["cgs_def"], "Por defeito", COR["dfc"], "€", 0)])
        fig.update_layout(barmode="relative")
        fig.update_yaxes(title_text="Custo dos desvios (€)")
        fig.update_xaxes(title_text="Hora de mercado (Espanha)", dtick=2)
        _mostra(_fig(fig, 320), "dsv_hora_custo")
        st.caption("Onde o erro custa: horas caras pesam mais do que horas com muito desvio e preço baixo.")

    st.subheader("Mapa do desvio: dia × hora")
    m = TD.mapa(per, "desvio_pct")
    if not m.empty:
        lim = float(np.nanpercentile(np.abs(m.to_numpy(dtype=float)), 95)) or 1.0
        fig = go.Figure(go.Heatmap(
            z=m.to_numpy(dtype=float), x=[f"H{c}" for c in m.columns], y=m.index, colorscale=DIVERGENTE, zmid=0,
            zmin=-lim, zmax=lim, xgap=1, ygap=1, colorbar=dict(title="% do consumo", thickness=12),
            hovertemplate="%{y|%d/%m/%Y} · %{x}<br>%{z:+,.1f} % do consumo<extra></extra>"))
        fig.update_yaxes(autorange="reversed", tickformat="%d/%m")
        _mostra(_fig(fig, max(320, min(900, 14 * len(m) + 80)), legenda=False, unificado=False), "dsv_mapa")
        st.caption("Cada célula é uma hora de um dia. Vermelho: comprou-se a menos; azul: a mais; cinzento: certo. "
                   "Colunas da mesma cor são horas com viés; linhas inteiras são dias em que a previsão falhou.")

    c1, c2 = st.columns(2)
    s = TD.por_dia_semana(d)
    with c1:
        st.subheader("Por dia da semana")
        if not s.empty:
            fig = go.Figure([_barra(s["nome"], s["exc_pct"], "Excesso", COR["exc"], "% do consumo"),
                             _barra(s["nome"], -s["dfc_pct"], "Defeito", COR["dfc"], "% do consumo")])
            fig.update_layout(barmode="relative")
            fig.update_yaxes(title_text="% do consumo")
            _mostra(_fig(fig, 300), "dsv_semana")
    with c2:
        st.subheader("Como se distribuem os dias")
        r = TD.racios(d)
        fig = go.Figure(go.Histogram(x=r["desvio_pct"], xbins=dict(size=2), name="Dias",
                                     marker=dict(color=COR["compra"], line=dict(color=COR["fundo"], width=1)),
                                     hovertemplate="%{y} dias entre %{x} %<extra></extra>"))
        fig.add_vline(x=0, line_color=COR["eixo"], line_width=1)
        fig.update_xaxes(title_text="Desvio líquido do dia (% do consumo)")
        fig.update_yaxes(title_text="N.º de dias")
        _mostra(_fig(fig, 300, legenda=False, unificado=False), "dsv_hist")
    with st.expander("Ver a tabela por hora"):
        _tabela(h, {"hora": ("Hora", 0), "pos": ("Compra (MWh)", 1), "qv": ("Consumo (MWh)", 1),
                    "desvio_pct": ("Desvio líquido (%)", 1), "abs_pct": ("Desvio absoluto (%)", 1),
                    "dias_defeito_pct": ("Dias em defeito (%)", 0), "cgs": ("Custo dos desvios (€)", 0),
                    "cgs_mwh": ("Custo (€/MWh)", 2), "p_omie": ("OMIE médio (€/MWh)", 2)}, "tab_hora")


# ---------------------------------------------------------------------------
# Preços
# ---------------------------------------------------------------------------
def _agregar_mercado(m: pd.DataFrame, grao: str) -> pd.DataFrame:
    if grao == "Dia" or m.empty:
        return m
    chave = (m["dia"] - pd.to_timedelta(m["dia"].dt.dayofweek, unit="D") if grao == "Semana"
             else m["dia"].dt.to_period("M").dt.start_time)
    return m.groupby(chave.rename("dia")).agg(
        p_omie=("p_omie", "mean"), p_min=("p_min", "min"), p_max=("p_max", "max"), p_exc=("p_exc", "mean"),
        p_def=("p_def", "mean"), spread_def=("spread_def", "mean"), spread_exc=("spread_exc", "mean")).reset_index()


def _precos(mer, mer_dia, ag, ini, fim, grao):
    md = mer_dia[(mer_dia["dia"] >= ini) & (mer_dia["dia"] <= fim)]
    if md.empty:
        st.info("Sem preços de mercado neste período.")
        return
    mg = _agregar_mercado(md, grao)
    st.subheader("Preço do mercado diário (OMIE, Portugal)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mg["dia"], y=mg["p_max"], mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mg["dia"], y=mg["p_min"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(27,175,122,0.14)", name="Mínimo–máximo", hoverinfo="skip"))
    fig.add_trace(_linha(mg["dia"], mg["p_omie"], "Média", COR["omie"], "€/MWh", 2,
                         customdata=np.stack([mg["p_min"], mg["p_max"]], axis=-1),
                         ))
    fig.data[-1].hovertemplate = "%{y:,.2f} €/MWh (mín. %{customdata[0]:,.2f} · máx. %{customdata[1]:,.2f})"
    fig.update_yaxes(title_text="€/MWh")
    _eixo_tempo(fig, grao)
    _mostra(_fig(fig, 330), "prc_omie")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Preços de desvio face ao OMIE")
        fig = go.Figure([_linha(mg["dia"], mg["p_omie"], "OMIE", COR["omie"], "€/MWh", 2),
                         _linha(mg["dia"], mg["p_def"], "Defeito", COR["dfc"], "€/MWh", 2),
                         _linha(mg["dia"], mg["p_exc"], "Excesso", COR["exc"], "€/MWh", 2)])
        fig.update_yaxes(title_text="€/MWh")
        _eixo_tempo(fig, grao)
        _mostra(_fig(fig, 320), "prc_desvio")
        st.caption("Defeito: o que se paga pela energia que faltou. Excesso: o que se recebe pela que sobrou.")
    with c2:
        st.subheader("Quanto custa errar 1 MWh")
        fig = go.Figure([_linha(mg["dia"], mg["spread_def"], "Em defeito (defeito − OMIE)", COR["dfc"], "€/MWh", 2),
                         _linha(mg["dia"], mg["spread_exc"], "Em excesso (OMIE − excesso)", COR["exc"], "€/MWh", 2)])
        fig.add_hline(y=0, line_color=COR["eixo"], line_width=1)
        fig.update_yaxes(title_text="€/MWh acima do OMIE")
        _eixo_tempo(fig, grao)
        _mostra(_fig(fig, 320), "prc_spread")
        st.caption("Média simples dos quartos de hora. Quando uma linha passa abaixo de zero, errar para esse lado "
                   "deu ganho.")

    mp = mer[(mer["dia"] >= ini) & (mer["dia"] <= fim)]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Perfil do preço ao longo do dia")
        hp = mp.groupby("hora")[["p_omie", "p_def", "p_exc"]].mean().reset_index()
        fig = go.Figure([_linha(hp["hora"], hp["p_omie"], "OMIE", COR["omie"], "€/MWh", 2),
                         _linha(hp["hora"], hp["p_def"], "Defeito", COR["dfc"], "€/MWh", 2),
                         _linha(hp["hora"], hp["p_exc"], "Excesso", COR["exc"], "€/MWh", 2)])
        fig.update_xaxes(title_text="Hora de mercado (Espanha)", dtick=2)
        fig.update_yaxes(title_text="€/MWh")
        _mostra(_fig(fig, 320), "prc_perfil")
    with c2:
        st.subheader("O preço a que comprámos")
        if ag.empty:
            st.info("Sem dias liquidados neste período.")
        else:
            fig = go.Figure([_linha(ag["dia"], ag["p_base"], "OMIE base (média simples)", COR["omie"], "€/MWh", 2),
                             _linha(ag["dia"], ag["p_compra"], "Preço médio da compra", COR["compra"], "€/MWh", 2),
                             _linha(ag["dia"], ag["custo_total_mwh"], "Custo total com desvios e encargos",
                                    COR["consumo"], "€/MWh", 2)])
            fig.update_yaxes(title_text="€/MWh")
            _eixo_tempo(fig, grao)
            _mostra(_fig(fig, 320), "prc_nosso")
            st.caption("A distância entre a base e o preço da compra é o custo do perfil da carteira; a distância "
                       "até ao custo total é o CGS.")

    st.subheader("Mapa do preço OMIE: dia × hora")
    m = TD.mapa(mp, "p_omie", "media")
    fig = go.Figure(go.Heatmap(z=m.to_numpy(dtype=float), x=[f"H{c}" for c in m.columns], y=m.index,
                               colorscale=SEQUENCIAL, xgap=1, ygap=1, colorbar=dict(title="€/MWh", thickness=12),
                               hovertemplate="%{y|%d/%m/%Y} · %{x}<br>%{z:,.2f} €/MWh<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickformat="%d/%m")
    _mostra(_fig(fig, max(320, min(900, 14 * len(m) + 80)), legenda=False, unificado=False), "prc_mapa")

    with st.expander("Ver a tabela mensal dos preços", expanded=False):
        mm = _agregar_mercado(md, "Mês")
        _tabela(mm, {"dia": ("Mês", "data"), "p_omie": ("OMIE médio", 2), "p_min": ("Mínimo", 2),
                     "p_max": ("Máximo", 2), "p_def": ("Defeito médio", 2), "p_exc": ("Excesso médio", 2),
                     "spread_def": ("Defeito − OMIE", 2), "spread_exc": ("OMIE − excesso", 2)}, "tab_precos",
                baixar="precos_mercado.csv")


# ---------------------------------------------------------------------------
# CGS
# ---------------------------------------------------------------------------
def _cgs(d, ag, grao, ref):
    st.subheader("CGS por MWh consumido")
    fig = go.Figure([_barra(ag["dia"], ag["enc_mwh"], "Encargos de regulação", COR["encargos"], "€/MWh", 2),
                     _barra(ag["dia"], ag["cgs_mwh"], "Desvios", COR["desvios"], "€/MWh", 2)])
    fig.update_layout(barmode="relative")
    if ref > 0:
        fig.add_hline(y=ref, line_color=COR["tinta2"], line_dash="dot", line_width=1.5,
                      annotation_text=f"referência {n(ref, 2)} €/MWh", annotation_position="top left",
                      annotation_font_color=COR["tinta2"])
    fig.update_yaxes(title_text="€/MWh")
    _eixo_tempo(fig, grao)
    _mostra(_fig(fig, 360), "cgs_mwh")
    st.caption("CGS = custo dos desvios + encargos de regulação, a dividir pelo consumo verificado — as contas do "
               "livro «Cálculo do CGS», refeitas a partir dos mesmos dados da REN.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Em euros")
        fig = go.Figure([_barra(ag["dia"], ag["enc"], "Encargos de regulação", COR["encargos"], "€", 0),
                         _barra(ag["dia"], ag["cgs"], "Desvios", COR["desvios"], "€", 0)])
        fig.update_layout(barmode="relative")
        fig.update_yaxes(title_text="€")
        _eixo_tempo(fig, grao)
        _mostra(_fig(fig, 320), "cgs_eur")
    with c2:
        if ref > 0:
            st.subheader("Acumulado face à referência")
            r = TD.racios(d).sort_values("dia")
            acum = (r["cgs_total"] - ref * r["qv"]).cumsum()
            fig = go.Figure(_linha(r["dia"], acum, "CGS real − referência", COR["tinta2"], "€", 0, fill="tozeroy",
                                   fillcolor="rgba(82,81,78,0.10)"))
            fig.add_hline(y=0, line_color=COR["eixo"], line_width=1)
            fig.update_yaxes(title_text="€ acumulados")
            fig.update_xaxes(tickformat="%d/%m", hoverformat="%d/%m/%Y")
            _mostra(_fig(fig, 320, legenda=False), "cgs_acum")
            st.caption(f"Acima de zero: o CGS real custou mais do que {n(ref, 2)} €/MWh × consumo; abaixo: menos.")
        else:
            st.info("Põe um CGS de referência nos filtros para ver o acumulado face a ele.")

    st.subheader(f"Tabela por {grao.lower()}")
    t = ag.sort_values("dia", ascending=False)
    _tabela(t, {"dia": (grao, "data"), "qv": ("Consumo (MWh)", 1), "cgs_mwh": ("Desvios (€/MWh)", 2),
                "enc_mwh": ("Encargos (€/MWh)", 2), "cgs_total_mwh": ("CGS (€/MWh)", 2),
                "cgs": ("Desvios (€)", 0), "enc": ("Encargos (€)", 0), "cgs_total": ("CGS (€)", 0),
                "abs_pct": ("Desvio absoluto (%)", 1), "dias": ("Dias", 0)}, "tab_cgs", altura=330,
            baixar="cgs_nossa_energia.csv")


# ---------------------------------------------------------------------------
# Mercado
# ---------------------------------------------------------------------------
def _mercado(agentes_dia, agentes, ini, fim, meta):
    fonte = meta.get("agentes", {}) if isinstance(meta.get("agentes"), dict) else {}
    livro = fonte.get("livro", "Análise Desvios do Mercado - Balanço Anual")
    if agentes_dia.empty:
        st.info(f"Sem dados do livro «{livro}».")
        return
    b = agentes_dia[(agentes_dia["dia"] >= ini) & (agentes_dia["dia"] <= fim)]
    if b.empty:
        if fim < agentes_dia["dia"].min():
            st.info(f"O livro dos desvios do mercado começa a {agentes_dia['dia'].min():%d/%m/%Y}: "
                    f"escolhe um período a partir daí.")
        else:
            st.info(f"O livro dos desvios do mercado ainda não tem este período (vai até "
                    f"{agentes_dia['dia'].max():%d/%m/%Y}).")
        return
    r = TD.ranking(b, agentes)
    if r.empty:
        st.info("Sem agentes neste período.")
        return
    nos = r[r["unidade"] == TD.NOVAE]
    media_mercado = TD.media_do_mercado(r)
    st.caption(f"Do livro «{livro}» (atualizado a {fonte.get('modificado', '—')}), de "
               f"{b['dia'].min():%d/%m/%Y} a {b['dia'].max():%d/%m/%Y}: custo dos desvios em € por MWh, "
               f"**média simples dos dias** do período, como a «MÉDIA ANUAL» do livro. Os dias sem consumo de um "
               f"agente não contam, e a NOSSA Energia é a linha escrita à mão no livro (os dias por preencher não "
               f"contam).")
    k = st.columns(4)
    if not nos.empty:
        x = nos.iloc[0]
        _metrica(k[0], "Lugar da NOSSA Energia", f"{int(x['lugar'])}.º de {len(r)}",
                 ajuda="1.º é o agente com o custo de desvios mais baixo por MWh.")
        _metrica(k[1], "NOSSA Energia", f"{n(x['eur_mwh'], 2)} €/MWh",
                 f"{n(x['eur_mwh'] - media_mercado, 2, True)} €/MWh face à média", inverso=True,
                 ajuda=f"Média simples de {int(x['dias'])} dias.")
    _metrica(k[2], "Média do mercado", f"{n(media_mercado, 2)} €/MWh",
             ajuda="A linha «MÉDIA» do livro: a média simples das médias dos agentes.")
    _metrica(k[3], "Melhor agente", f"{n(r['eur_mwh'].min(), 2)} €/MWh", r.iloc[0]["nome"][:28], inverso=True)
    k[3].caption("")

    st.subheader("Custo dos desvios por agente")
    rr = r.sort_values("eur_mwh", ascending=False)
    fig = go.Figure(go.Bar(
        x=rr["eur_mwh"], y=rr["nome"].str.slice(0, 38), orientation="h",
        marker=dict(color=np.where(rr["unidade"] == TD.NOVAE, COR["consumo"], COR["apagado"])),
        text=[n(v, 2) for v in rr["eur_mwh"]], textposition="outside", cliponaxis=False,
        customdata=np.stack([rr["dias"], rr["mwh_dia"].fillna(0)], axis=-1),
        hovertemplate="%{y}<br>%{x:,.2f} €/MWh (média simples de %{customdata[0]:.0f} dias) · "
                      "%{customdata[1]:,.0f} MWh/dia<extra></extra>"))
    fig.add_vline(x=media_mercado, line_color=COR["tinta2"], line_dash="dot", line_width=1.5,
                  annotation_text=f"média do mercado {n(media_mercado, 2)}", annotation_position="top",
                  annotation_font_color=COR["tinta2"])
    fig.update_xaxes(title_text="€/MWh — média simples dos dias", showgrid=True, gridcolor=COR["grelha"])
    fig.update_yaxes(showgrid=False)
    fig = _fig(fig, 26 * len(rr) + 90, legenda=False, unificado=False)
    fig.update_layout(margin=dict(t=36))                       # espaço para o rótulo da média do mercado
    _mostra(fig, "mrc_rank")

    st.subheader("NOSSA Energia face ao mercado, mês a mês")
    bm = b.assign(mes=b["dia"].dt.to_period("M").dt.start_time)
    linhas = []
    for mes, g in bm.groupby("mes"):
        rk = TD.ranking(g, agentes)
        eu = rk[rk["unidade"] == TD.NOVAE]
        linhas.append({"mes": mes, "nossa": eu["eur_mwh"].iloc[0] if not eu.empty else np.nan,
                       "media": TD.media_do_mercado(rk), "melhor": rk["eur_mwh"].min(),
                       "lugar": int(eu["lugar"].iloc[0]) if not eu.empty else None, "agentes": len(rk),
                       "dias": int(eu["dias"].iloc[0]) if not eu.empty else 0})
    ev = pd.DataFrame(linhas)
    fig = go.Figure([_linha(ev["mes"], ev["nossa"], "NOSSA Energia", COR["consumo"], "€/MWh", 2, mode="lines+markers"),
                     _linha(ev["mes"], ev["media"], "Média do mercado", COR["neutro"], "€/MWh", 2,
                            mode="lines+markers"),
                     _linha(ev["mes"], ev["melhor"], "Melhor agente", COR["neutro"], "€/MWh", 2, "dot", 1.5)])
    fig.update_yaxes(title_text="€/MWh — média simples dos dias")
    _eixo_tempo(fig, "Mês")
    _mostra(_fig(fig, 320), "mrc_evol")
    st.caption("Cada ponto é a média simples dos dias desse mês (dentro do período escolhido); a média do mercado é "
               "a média simples das médias dos agentes.")
    with st.expander("Ver as tabelas"):
        _tabela(ev, {"mes": ("Mês", "data"), "nossa": ("NOSSA (€/MWh)", 2), "media": ("Média do mercado (€/MWh)", 2),
                     "melhor": ("Melhor (€/MWh)", 2), "lugar": ("Lugar", 0), "agentes": ("Agentes", 0),
                     "dias": ("Dias NOSSA", 0)}, "tab_evol")
        _tabela(r, {"lugar": ("Lugar", 0), "nome": ("Agente", "texto"), "unidade": ("Código", "texto"),
                    "eur_mwh": ("€/MWh (média simples)", 2), "dias": ("Dias", 0),
                    "mwh_dia": ("Consumo (MWh/dia)", 0), "desvios": ("Desvios (€)", 0),
                    "encargos": ("Encargos (€)", 0)},
                "tab_rank", baixar="ranking_desvios.csv")


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def _dados(q):
    meta = q["meta"]
    st.subheader("De onde vêm os dados")
    st.markdown(
        f"- **Liquidação da REN** (posição final, quantidade verificada, preços, encargos): "
        f"{meta.get('liquidacao', {}).get('de', '—')} a {meta.get('liquidacao', {}).get('ate', '—')} "
        f"({meta.get('liquidacao', {}).get('dias', '—')} dias) — dos livros «Cálculo do CGS».\n"
        f"- **Ofertas enviadas** (XML do diário): {meta.get('ofertas', {}).get('de', '—')} a "
        f"{meta.get('ofertas', {}).get('ate', '—')}.\n"
        f"- **Preços de mercado** (OMIE Portugal e preços de desvio da REN, públicos): "
        f"{meta.get('mercado', {}).get('de', '—')} a {meta.get('mercado', {}).get('ate', '—')}.\n"
        f"- **Desvios dos agentes de mercado** (tab Mercado): livro «{meta.get('agentes', {}).get('livro', '—')}», "
        f"{meta.get('agentes', {}).get('de', '—')} a {meta.get('agentes', {}).get('ate', '—')} "
        f"({meta.get('agentes', {}).get('agentes', '—')} agentes; atualizado a "
        f"{meta.get('agentes', {}).get('modificado', '—')}).\n"
        f"- Gerado a **{meta.get('gerado', '—')}**."
    )
    st.caption("O dia e as horas são os do mercado (hora de Espanha): o período 1 vai das 23:00 às 23:15 de Lisboa "
               "da véspera. A REN revê os dias já liquidados; de cada um fica a versão mais recente que estiver nos "
               "livros (provisória ou definitiva).")
    if meta.get("livros"):
        _tabela(pd.DataFrame(meta["livros"]), {"livro": ("Livro", "texto"), "de": ("De", "texto"),
                                               "ate": ("Até", "texto"), "modificado": ("Modificado", "texto")},
                "tab_livros")
    avisos = meta.get("avisos", [])
    st.subheader(f"Avisos da leitura ({len(avisos)})")
    if avisos:
        st.caption("Dias em que os dados colados nos livros não estão completos ou não batem uns com os outros. "
                   "Não param o dashboard, mas esses dias merecem ser vistos no livro.")
        st.dataframe(pd.DataFrame({"Aviso": avisos}), hide_index=True, height=260, key="tab_avisos", **_LARGO)
    else:
        st.markdown("🟢 Nenhum.")
    if st.session_state.get("trading_palavra") and st.button("Sair", key="trd_sair"):
        st.session_state.pop("trading_palavra", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
def render() -> None:
    q = _obter()
    if q is None:
        return
    per, d_all = q["periodos"], q["diario"]
    if per.empty or d_all.empty:
        st.warning("Os dados de trading estão vazios: confirma os livros do CGS e corre outra vez o "
                   "`trading_dados.py`.")
        return
    ini, fim, grao, ref, sem_posicao = _filtros(d_all)
    d = d_all[(d_all["dia"] >= ini) & (d_all["dia"] <= fim)]
    p = per[(per["dia"] >= ini) & (per["dia"] <= fim)]
    if sem_posicao:
        fora = d[(d["pos"] <= 0) & (d["qv"] > 0)]["dia"]
        d, p = d[~d["dia"].isin(fora)], p[~p["dia"].isin(fora)]
    if d.empty:
        st.warning("Não há dias liquidados neste intervalo.")
        return
    dias = (fim - ini).days + 1
    d_ant = d_all[(d_all["dia"] >= ini - pd.Timedelta(days=dias)) & (d_all["dia"] < ini)]
    if len(d_ant) < 0.8 * len(d):
        d_ant = d_ant.iloc[0:0]
    ag = TD.agregar(d, grao)
    ultimo = d_all["dia"].max()
    st.caption(f"{len(d)} dias liquidados no intervalo · último dia liquidado pela REN: {ultimo:%d/%m/%Y} · "
               f"dados gerados a {q['meta'].get('gerado', '—')}")
    abas = st.tabs(["Resumo", "Compras", "Desvios", "Preços", "CGS", "Mercado", "Dados"])
    with abas[0]:
        _resumo(p, d, d_ant, ag, grao, ref)
    with abas[1]:
        _compras(p, d)
    with abas[2]:
        _desvios(p, d, ag, grao)
    with abas[3]:
        _precos(q["mercado"], q["mercado_dia"], ag, ini, fim, grao)
    with abas[4]:
        _cgs(d, ag, grao, ref)
    with abas[5]:
        _mercado(q["agentes_dia"], q["agentes"], ini, fim, q["meta"])
    with abas[6]:
        _dados(q)
