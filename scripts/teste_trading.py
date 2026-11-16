"""
teste_trading.py — as contas do Dashboard Trading, com números feitos à mão
===========================================================================
    python scripts/teste_trading.py        # tem de dar 0 falhas

Sem rede, sem Streamlit e sem os dados verdadeiros. Confere:
  - desvio, excesso, defeito e o seu custo por período (as fórmulas do livro «Cálculo do CGS»);
  - períodos sem posição final (oferta que não casou) e dias sem posição;
  - somas e rácios por dia, por semana e no período; perfil por hora; ranking dos agentes;
  - a leitura dos dados cifrados (o formato que o trading_dados.py da «Compra OMIE» escreve):
    abre com a palavra-passe, não abre com outra, e junta os meses outra vez.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trading_dados as TD  # noqa: E402

falhas = 0


def confere(nome: str, condicao: bool, detalhe="") -> None:
    global falhas
    print(("ok    " if condicao else "FALHA ") + nome + (f"  {detalhe}" if detalhe != "" else ""))
    falhas += not condicao


def perto(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


CAB = "dia,periodo,versao,estado,pos,qv,p_omie,p_exc,p_def,enc,val_exc,val_def,oferta_mw,preco_lim\n"
# dia 1 (segunda): P1 defeito 0,5; P2 excesso 0,5; P3 certo; P4 não casou (sem posição, oferta 8 MW)
# dia 2 (terça):  sem posição nenhuma e sem oferta: dia inteiro a defeito
PERIODOS = CAB + "".join([
    "2026-10-05,1,1,P,2.0,2.5,100,80,130,10,,,8,315\n",
    "2026-10-05,2,1,P,2.0,1.5,100,80,130,10,,,8,315\n",
    "2026-10-05,3,1,P,2.0,2.0,120,90,150,10,,,8,315\n",
    "2026-10-05,5,1,P,,2.0,400,300,430,10,,,8,315\n",
    "2026-10-05,96,,,,,,,,,,,,\n",                         # só para o dia ser de 15 minutos (96 períodos)
    "2026-10-06,1,2,D,,1.0,100,80,130,5,,,,\n",
    "2026-10-06,2,2,D,,1.0,100,80,130,5,,,,\n",
    "2026-10-06,96,,,,,,,,,,,,\n",
    "2026-10-07,1,,,,,,,,,,,8,315\n",                       # só a oferta: dia por liquidar
])
MERCADO = ("dia,periodo,p_omie,p_exc,p_def\n"
           "2026-10-05,1,100,80,130\n2026-10-05,2,120,90,150\n2026-10-06,1,50,40,70\n")
# o livro dos desvios do mercado, já com os dias que contam: €/MWh de cada dia, consumo, desvios, encargos.
# A NOVAE tem um dia de consumo enorme e outro pequeno: a média simples ignora o peso, a ponderada não.
AGENTES_DIA = ("dia,unidade,eur_mwh,consumo,desvios,encargos\n"
               "2026-10-05,NOVAE,1.0,900,900,900\n2026-10-06,NOVAE,3.0,100,300,100\n"
               "2026-10-05,AAA,4.0,1000,4000,4000\n2026-10-06,AAA,0.5,1000,500,4000\n"
               "2026-10-06,PEQ,0.1,5,0.5,10\n")
AGENTES = "unidade,nome,ranking\nNOVAE,Nossa Energia,1\nAAA,Agente A,1\nPEQ,Pequeno,1\n"
# dia 5: V1 e V2 (a V2 subiu o consumo para 11 e trouxe 1 MWh de defeito); dia 6: só a V1
VERSOES = ("dia,versao,ultima,estado,emitida,n,pos,qv,exc,dfc,cgs_exc,cgs_def,enc,custo_ref\n"
           "2026-10-05,1,0,Provisório,2026-10-08,96,10,10,0,0,0,0,10,1000\n"
           "2026-10-05,2,1,Definitivo,2026-12-20,96,10,11,0,1,0,20,12,1100\n"
           "2026-10-06,1,1,Provisório,2026-10-09,96,10,9,1,0,5,0,9,900\n")
TEXTOS = {"periodos.csv": PERIODOS, "mercado.csv": MERCADO, "agentes_dia.csv": AGENTES_DIA, "agentes.csv": AGENTES,
          "versoes.csv": VERSOES,
          "ofertas.csv": "dia,ficheiro,candidatas,antecipada\n2026-10-05,x.xml,1,0\n",
          "meta.json": '{"gerado": "teste", "avisos": []}'}


def cifrar(ficheiros: dict[str, str], palavra: str, sal: bytes) -> bytes:
    """O formato do exportador: marca + sal + Fernet(zip)."""
    from cryptography.fernet import Fernet
    chave = base64.urlsafe_b64encode(hashlib.pbkdf2_hmac("sha256", palavra.encode(), sal, TD.PBKDF2))
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zz:
        for nome, texto in ficheiros.items():
            zz.writestr(nome, texto.encode("utf-8"))
    return TD.MARCA + sal + Fernet(chave).encrypt(z.getvalue())


def main() -> int:
    q = TD.quadros(TEXTOS)
    p = q["periodos"].set_index(["dia", "periodo"])
    d1, d2 = pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06")

    # ---------------------------------------------------------------- por período
    confere("defeito: desvio −0,5 e custo 0,5 × (130 − 100)",
            perto(p.loc[(d1, 1), "desvio"], -0.5) and perto(p.loc[(d1, 1), "cgs"], 15.0), p.loc[(d1, 1), "cgs"])
    confere("excesso: desvio +0,5 e custo 0,5 × (100 − 80)",
            perto(p.loc[(d1, 2), "desvio"], 0.5) and perto(p.loc[(d1, 2), "cgs"], 10.0), p.loc[(d1, 2), "cgs"])
    confere("sem desvio não há custo", perto(p.loc[(d1, 3), "cgs"], 0.0))
    x = p.loc[(d1, 5)]
    confere("sem posição: o consumo inteiro é defeito", perto(x["dfc"], 2.0) and perto(x["cgs"], 2.0 * 30))
    confere("…e conta como oferta que não casou (8 MW × 0,25 h)",
            perto(x["nao_casado"], 2.0) and perto(x["cgs_nao_casado"], 60.0) and bool(x["sem_pos"]))
    confere("sem oferta arquivada não se diz que não casou", perto(p.loc[(d2, 1), "nao_casado"], 0.0))
    confere("período por liquidar não tem desvio", pd.isna(p.loc[(pd.Timestamp("2026-10-07"), 1), "desvio"]))
    confere("hora de mercado: período 5 é a hora 2", int(x["hora"]) == 2)
    antigo = TD._horas(pd.DataFrame({"dia": [d1] * 3 + [d2] * 2, "periodo": [1, 5, 24, 5, 96]}))
    confere("dia liquidado à hora (até março de 2025): o período já é a hora",
            list(antigo["hora"]) == [1, 5, 24, 2, 24] and list(antigo["horas_periodo"]) == [1, 1, 1, 0.25, 0.25],
            list(antigo["hora"]))

    # ---------------------------------------------------------------- por dia e no período
    d = TD.diario(q["periodos"])
    confere("só os dias liquidados", list(d["dia"]) == [d1, d2], list(d["dia"]))
    r = TD.racios(d).set_index("dia")
    confere("dia 1: compra 6, consumo 8", perto(r.loc[d1, "pos"], 6.0) and perto(r.loc[d1, "qv"], 8.0))
    confere("dia 1: excesso 0,5, defeito 2,5, desvio líquido −2",
            perto(r.loc[d1, "exc"], 0.5) and perto(r.loc[d1, "dfc"], 2.5) and perto(r.loc[d1, "desvio"], -2.0))
    confere("dia 1: custo dos desvios 85 €, encargos 40 €", perto(r.loc[d1, "cgs"], 85.0) and perto(r.loc[d1, "enc"], 40.0))
    confere("dia 1: CGS = 125 € ÷ 8 MWh", perto(r.loc[d1, "cgs_total_mwh"], 125.0 / 8))
    confere("dia 1: desvio absoluto 3 ÷ 8", perto(r.loc[d1, "abs_pct"], 37.5))
    confere("dia 1: OMIE base é a média simples", perto(r.loc[d1, "p_base"], (100 + 100 + 120 + 400) / 4))
    confere("dia 1: preço da compra é pesado pela posição", perto(r.loc[d1, "p_compra"], (200 + 200 + 240) / 6))
    confere("dia 1: custo total = consumo ao OMIE + desvios + encargos",
            perto(r.loc[d1, "custo_total"], (250 + 150 + 240 + 800) + 85 + 40))
    confere("dia 2: sem posição, compra 0 e tudo a defeito",
            perto(r.loc[d2, "pos"], 0.0) and perto(r.loc[d2, "dfc"], 2.0) and perto(r.loc[d2, "cgs"], 60.0))
    confere("o estado e a versão do dia", r.loc[d2, "estado"] == "D" and int(r.loc[d2, "versao"]) == 2)
    t = TD.total(d)
    confere("período: os rácios fazem-se sobre as somas, não se fazem médias de rácios",
            perto(t["cgs_total_mwh"], (85 + 60 + 40 + 10) / 10) and perto(t["desvio_pct"], -40.0))
    confere("período: custo por MWh em defeito e em excesso",
            perto(t["cgs_def_mwh"], (15 + 60 + 60) / 4.5) and perto(t["cgs_exc_mwh"], 20.0))
    sem = TD.agregar(d, "Semana")
    confere("semana: uma linha, na segunda-feira, com a soma", len(sem) == 1 and sem["dia"][0] == d1 and perto(sem["qv"][0], 10))
    confere("mês: uma linha no dia 1", TD.agregar(d, "Mês")["dia"][0] == pd.Timestamp("2026-10-01"))
    confere("dia: o agregado diário é o diário", len(TD.agregar(d, "Dia")) == 2)

    # ---------------------------------------------------------------- perfis
    h = TD.por_hora(q["periodos"]).set_index("hora")
    confere("por hora: a hora 1 junta os períodos 1 a 3 dos dois dias",
            perto(h.loc[1, "qv"], 8.0) and perto(h.loc[1, "pos"], 6.0) and perto(h.loc[1, "desvio_pct"], -25.0))
    confere("por hora: em quantos dias fechou em defeito", perto(h.loc[1, "dias_defeito_pct"], 50.0), h.loc[1, "dias_defeito_pct"])
    s = TD.por_dia_semana(d)
    confere("por dia da semana", list(s["nome"]) == ["Seg", "Ter"], list(s["nome"]))
    m = TD.mapa(q["periodos"], "desvio_pct")
    confere("mapa dia × hora do desvio", perto(m.loc[d1, 1], 100 * (6 - 6) / 6) and perto(m.loc[d1, 2], -100.0))
    md = TD.mercado_diario(q["mercado"]).set_index("dia")
    confere("mercado diário: média, mínimo, máximo e os dois spreads",
            perto(md.loc[d1, "p_omie"], 110) and perto(md.loc[d1, "p_max"], 120)
            and perto(md.loc[d1, "spread_def"], 30) and perto(md.loc[d1, "spread_exc"], 25))

    # ---------------------------------------------------------------- ranking
    rk = TD.ranking(q["agentes_dia"], q["agentes"])
    ri = rk.set_index("unidade")
    confere("ranking: média simples dos dias, não ponderada pelo consumo (NOVAE: (1 + 3) / 2, não 1,2)",
            perto(ri.loc["NOVAE", "eur_mwh"], 2.0) and perto(ri.loc["AAA", "eur_mwh"], 2.25))
    confere("ranking: entram todos os agentes do livro, pequenos incluídos", list(rk["unidade"]) == ["PEQ", "NOVAE", "AAA"],
            list(rk["unidade"]))
    confere("o lugar conta do mais barato", int(ri.loc["NOVAE", "lugar"]) == 2)
    confere("um agente só conta os dias que tem", int(ri.loc["PEQ", "dias"]) == 1 and perto(ri.loc["NOVAE", "mwh_dia"], 500))
    confere("a média do mercado é a média simples das médias dos agentes (a linha «MÉDIA» do livro)",
            perto(TD.media_do_mercado(rk), (0.1 + 2.0 + 2.25) / 3))
    so5 = TD.ranking(q["agentes_dia"][q["agentes_dia"]["dia"] == d1], q["agentes"]).set_index("unidade")
    confere("num período mais curto, a média é só desses dias", perto(so5.loc["AAA", "eur_mwh"], 4.0) and "PEQ" not in so5.index)

    # ---------------------------------------------------------------- versões da REN
    v = q["versoes"]
    a = TD.versoes_agregadas(v, "Semana")
    w = a.set_index("versao")
    confere("versões: na semana, a V1 tem os dois dias e a V2 só um (incompleta)",
            bool(w.loc[1, "completo"]) and not bool(w.loc[2, "completo"]) and int(w.loc[2, "dias"]) == 1)
    confere("versões: V1 = (10 + 5 + 9) € ÷ 19 MWh", perto(w.loc[1, "cgs_total_mwh"], 24 / 19))
    confere("versões: a última é a V2 do dia 5 com a V1 do dia 6: (20 + 12 + 5 + 9) € ÷ 20 MWh",
            perto(w.loc[TD.ULTIMA, "cgs_total_mwh"], 46 / 20) and perto(w.loc[TD.ULTIMA, "qv"], 20))
    c = TD.versoes_comparacao(a).iloc[0]
    confere("versões: a V2 incompleta não tem coluna na semana", "cgs_V2" not in c.index or pd.isna(c["cgs_V2"]))
    confere("versões: diferença da V1 para a última, em € e no consumo",
            perto(c["dif_eur"], 22) and perto(c["qv_var_pct"], 100 / 19) and perto(c["dif_mwh"], 46 / 20 - 24 / 19))
    dia5 = TD.versoes_comparacao(TD.versoes_agregadas(v, "Dia")).set_index("periodo").loc[pd.Timestamp("2026-10-05")]
    confere("versões: por dia, a V2 do dia 5 já é completa", perto(dia5["cgs_V2"], 32 / 11) and perto(dia5["cgs_V1"], 1.0))
    qd = TD.versoes_quando(v).set_index("versao")
    confere("versões: a V1 sai 3 dias depois e a V2 76; quantas são definitivas",
            perto(qd.loc[1, "mediana"], 3) and perto(qd.loc[2, "mediana"], 76) and perto(qd.loc[2, "definitiva_pct"], 100)
            and perto(qd.loc[1, "definitiva_pct"], 0))
    confere("versões: o rótulo da última", TD.rotulo_versao(TD.ULTIMA) == "Última" and TD.rotulo_versao(3) == "V3")
    confere("versões: sem dados, nada", TD.versoes_agregadas(v.iloc[0:0], "Mês").empty)

    # ---------------------------------------------------------------- conclusões e números
    frases = TD.conclusoes(q["periodos"], d, 20.0)
    texto = " ".join(f for _, f in frases)
    confere("conclusões: o balanço e o custo", "10 MWh" in texto and "CGS de" in texto, len(frases))
    confere("conclusões: o dia sem posição é um alerta", any(t == "alerta" and "sem posição" in f for t, f in frases))
    confere("conclusões: o que não casou é um alerta", any(t == "alerta" and "não casaram" in f for t, f in frases))
    confere("conclusões: sem dias, nada", TD.conclusoes(q["periodos"], d.iloc[0:0]) == [])
    confere("números à portuguesa", TD.n(12345.678, 1) == "12 345,7" and TD.n(-2.5, 1, True) == "−2,5" and TD.n(None) == "—")

    # ---------------------------------------------------------------- leitura: em claro e cifrada
    raiz = Path(tempfile.mkdtemp(prefix="teste_trading_"))
    try:
        confere("sem dados, não há origem", TD.origem(raiz) is None)
        (raiz / "pub").mkdir()
        sal = os.urandom(16)
        meses = {"2026-10": {"periodos.csv": PERIODOS, "mercado.csv": MERCADO, "agentes_dia.csv": AGENTES_DIA},
                 "2026-11": {"periodos.csv": CAB + "2026-11-01,1,1,P,1.0,1.0,90,70,110,3,,,,\n"},
                 "base": {k: v for k, v in TEXTOS.items() if k not in ("periodos.csv", "mercado.csv", "agentes_dia.csv")}}
        for nome, ficheiros in meses.items():
            (raiz / "pub" / f"{nome}.bin").write_bytes(cifrar(ficheiros, "palavra de teste", sal))
        confere("com pub/, a origem é a cifrada", TD.origem(raiz) == "cifrado")
        lido = TD.ler_cifrado("palavra de teste", raiz)
        confere("abre com a palavra-passe e junta os meses",
                lido["periodos.csv"] == PERIODOS + "2026-11-01,1,1,P,1.0,1.0,90,70,110,3,,,,\n"
                and lido["agentes.csv"] == AGENTES and lido["agentes_dia.csv"] == AGENTES_DIA)
        confere("os quadros do cifrado são os do claro mais o mês novo",
                len(TD.quadros(lido)["periodos"]) == len(q["periodos"]) + 1)
        try:
            TD.ler_cifrado("outra palavra", raiz)
            confere("não abre com outra palavra-passe", False, "abriu")
        except TD.PalavraErrada:
            confere("não abre com outra palavra-passe", True)
        (raiz / "dados").mkdir()
        for nome, texto in TEXTOS.items():
            (raiz / "dados" / nome).write_text(texto, encoding="utf-8")
        confere("havendo dados em claro, usam-se esses", TD.origem(raiz) == "claro" and TD.ler_claro(raiz) == TEXTOS)
        confere("a assinatura muda com os ficheiros", TD.assinatura(raiz) != "")
    finally:
        shutil.rmtree(raiz, ignore_errors=True)
    print(f"\n{falhas} falha(s)")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
