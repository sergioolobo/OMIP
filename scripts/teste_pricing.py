"""
teste_pricing.py — as contas do Dashboard Pricing com números feitos à mão
=========================================================================
Corre sem rede nem Streamlit:  python scripts/teste_pricing.py
Tem de acabar com «0 falha(s)».
"""
from __future__ import annotations

import base64
import calendar
import datetime as dt
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pricing_dados as PD  # noqa: E402

FALHAS: list[str] = []


def confere(nome: str, ok: bool, detalhe: str = "") -> None:
    print(("  ok    " if ok else "  FALHA ") + nome + ("" if ok else f" — {detalhe}"))
    if not ok:
        FALHAS.append(nome)


def perto(a, b, tol=1e-9) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and
                                         (math.isnan(a) and math.isnan(b) or abs(a - b) <= tol))


# ---------------------------------------------------------------------------
print("Simulador: as duas fórmulas")
omie = {"P": 100.0, "C": 80.0, "VN": 60.0, "SV": 50.0}
pesos = {"P": 0.2, "C": 0.5, "VN": 0.2, "SV": 0.1}
tar = {"P": 0.04, "C": 0.03, "VN": 0.02, "SV": 0.01}
mx = PD.simular("MAX", omie, pesos, cgs=20, custos=2.4, k_interno=15, k_agente=5, perdas=1.16, tar=tar)
confere("MAX ponta = (100 + 2,4 + 15 + 5) / 1000", perto(mx.loc[0, "preco"], 0.1224))
confere("MAX super vazio = (50 + 22,4) / 1000", perto(mx.loc[3, "preco"], 0.0724))
confere("MAX ignora CGS, perdas e TAR", (mx["cgs"] == 0).all() and (mx["perdas"] == 0).all() and (mx["tar"] == 0).all())
fx = PD.simular("FLEX", omie, pesos, cgs=20, custos=2.4, k_interno=15, k_agente=5, perdas=1.16, tar=tar)
confere("FLEX ponta = (100 + 20 + 2,4 + 15 + 5) × 1,16 / 1000 + 0,04", perto(fx.loc[0, "preco"], 142.4 * 1.16 / 1000 + 0.04))
confere("FLEX vazio normal = (60 + 42,4) × 1,16 / 1000 + 0,02", perto(fx.loc[2, "preco"], 102.4 * 1.16 / 1000 + 0.02))
soma = fx[["omie", "cgs", "custos", "k_interno", "k_agente", "perdas", "tar"]].sum(axis=1) / 1000
confere("FLEX: as parcelas somam o preço", np.allclose(soma, fx["preco"]))
confere("FLEX: a parcela das perdas é (OMIE + CGS + custos + K) × 0,16", perto(fx.loc[0, "perdas"], 142.4 * 0.16))
try:
    PD.simular("OUTRO", omie, pesos, 0, 0, 0, 0, 1)
    confere("cálculo desconhecido pára", False, "não parou")
except ValueError:
    confere("cálculo desconhecido pára", True)

print("Simulador: preços do horário e resumo")
p = list(fx["preco"])
bi = PD.precos_do_horario(fx, "BI")
confere("BI: fora de vazio = média de ponta e cheias pesada (0,2 e 0,5)",
        perto(bi.loc[0, "preco"], (0.2 * p[0] + 0.5 * p[1]) / 0.7))
confere("BI: vazio = média de VN e SV pesada (0,2 e 0,1)", perto(bi.loc[1, "preco"], (0.2 * p[2] + 0.1 * p[3]) / 0.3))
tri = PD.precos_do_horario(fx, "TRI")
confere("TRI: três preços, ponta e cheias como os períodos", len(tri) == 3 and perto(tri.loc[0, "preco"], p[0])
        and perto(tri.loc[1, "preco"], p[1]))
sim_ = PD.precos_do_horario(fx, "SIMPLES")
medio = sum(w * x for w, x in zip(pesos.values(), p))
confere("SIMPLES: um preço, a média pesada", len(sim_) == 1 and perto(sim_.loc[0, "preco"], medio))
confere("TETRA: os quatro períodos", len(PD.precos_do_horario(fx, "TETRA")) == 4)
r = PD.resumo_simulacao(fx, "FLEX", consumo_mwh=250, perdas=1.16, potencia_dia=2.0)
confere("preço médio pesado", perto(r["preco_medio"], medio))
confere("energia do ano = preço médio × 250 MWh × 1000", perto(r["energia_ano"], medio * 250_000, 1e-6))
confere("potência do ano = 2 €/dia × 365", perto(r["potencia_ano"], 730.0))
confere("FLEX: margem do K interno = 15 × 1,16 × 250", perto(r["margem_k_interno"], 15 * 1.16 * 250, 1e-9))
confere("FLEX: comissão do agente = 5 × 1,16 × 250", perto(r["comissao_agente"], 5 * 1.16 * 250, 1e-9))
r2 = PD.resumo_simulacao(mx, "MAX", consumo_mwh=250, perdas=1.16)
confere("MAX: margem do K interno = 15 × 250 (sem perdas)", perto(r2["margem_k_interno"], 3750.0))
confere("as parcelas médias somam o preço médio", perto(sum(r["parcelas"].values()) / 1000, r["preco_medio"]))

print("OMIE de referência")
hist = {"P": 130.0, "C": 50.0, "VN": 114.0, "SV": 111.0, "base": 88.0, "medio": 83.0}
o = PD.omie_por_referencia(hist, 122.0)
confere("OMIP 122 com a forma do perfil: ponta = 122 × 130 / 88", perto(o["P"], 122 * 130 / 88))
confere("com a referência igual ao baseload histórico, volta o histórico",
        all(perto(PD.omie_por_referencia(hist, 88.0)[k], hist[k]) for k in PD.PERIODOS))

print("Carteira")
cp = pd.DataFrame([
    dict(cpe="PT1", cliente_id="C0001", cliente="Alfa", tarifa="T1", familia="PowerMAX", tipo="Fixa", calculo="MAX",
         inclui_redes="Não", consumo_mwh=100.0, preco_medio=0.10, k=10.0, agente="Ag1", nivel="BTE", segmento="Empresarial",
         preco_especial="Não", fim_contrato=pd.Timestamp("2026-12-01"), risco="Baixo", estado="Ativo", em_mudanca_para=""),
    dict(cpe="PT2", cliente_id="C0001", cliente="Alfa", tarifa="T2", familia="PowerGO Fixo", tipo="Fixa", calculo="MAX",
         inclui_redes="Sim", consumo_mwh=300.0, preco_medio=0.20, k=20.0, agente="Ag2", nivel="BTN", segmento="Empresarial",
         preco_especial="Sim", fim_contrato=pd.Timestamp("2027-06-01"), risco="", estado="Ativo", em_mudanca_para=""),
    dict(cpe="PT3", cliente_id="C0002", cliente="Beta", tarifa="T1", familia="PowerMAX", tipo="Fixa", calculo="MAX",
         inclui_redes="Não", consumo_mwh=50.0, preco_medio=np.nan, k=np.nan, agente="", nivel="BTE", segmento="Doméstico",
         preco_especial="Não", fim_contrato=pd.NaT, risco="", estado="Ativo", em_mudanca_para="EDP"),
    dict(cpe="PT4", cliente_id="C0003", cliente="Gama", tarifa="T3", familia="PowerFLEX Indexado", tipo="Indexada",
         calculo="Indexado", inclui_redes="Não", consumo_mwh=0.0, preco_medio=np.nan, k=12.0, agente="Ag1", nivel="MT",
         segmento="Empresarial", preco_especial="Não", fim_contrato=pd.Timestamp("2026-11-20"), risco="", estado="Ativo",
         em_mudanca_para=""),
])
confere("média pesada ignora NaN e pesos nulos", perto(PD.media_ponderada(cp["preco_medio"], cp["consumo_mwh"]),
                                                       (0.1 * 100 + 0.2 * 300) / 400))
confere("média pesada sem nada é NaN", math.isnan(PD.media_ponderada(pd.Series([np.nan]), pd.Series([1.0]))))
t = PD.por_tarifario(cp).set_index("tarifa")
confere("T1: 2 CPEs, 2 clientes, 150 MWh", t.loc["T1", "n_cpes"] == 2 and t.loc["T1", "n_clientes"] == 2
        and perto(t.loc["T1", "consumo_mwh"], 150.0))
confere("T1: preço médio só dos que têm preço (0,10)", perto(t.loc["T1", "preco_medio"], 0.10))
confere("quota do consumo de T2 = 300 / 450", perto(t.loc["T2", "quota_consumo"], 100 * 300 / 450))
confere("o mais usado vem primeiro", list(PD.por_tarifario(cp)["tarifa"])[0] == "T2")
a = PD.por_agente(cp).set_index("agente")
confere("Ag1: K médio pesado (10 × 100 + 12 × 0) / 100", perto(a.loc["Ag1", "k_medio"], 10.0))
confere("Ag1: % indexado do consumo (0 de 100)", perto(a.loc["Ag1", "pct_indexado"], 0.0))
confere("CPE sem agente fica no grupo vazio, sem contar como agente", a.loc["", "n_agentes"] == 0)
c = PD.clientes(cp).set_index("cliente_id")
confere("Alfa: 2 CPEs, 400 MWh, preço médio 0,175", c.loc["C0001", "n_cpes"] == 2 and
        perto(c.loc["C0001", "consumo_mwh"], 400.0) and perto(c.loc["C0001", "preco_medio"], 0.175))
confere("Alfa: tarifários e agentes juntos por ordem", c.loc["C0001", "tarifarios"] == "T1 · T2"
        and c.loc["C0001", "agentes"] == "Ag1 · Ag2")
confere("Alfa: o fim de contrato mais próximo", c.loc["C0001", "fim_contrato"] == pd.Timestamp("2026-12-01"))
confere("Alfa: tem preço especial", c.loc["C0001", "especial"] == "Sim")
confere("Beta: sem preço fica NaN", math.isnan(c.loc["C0002", "preco_medio"]))
cl = PD.clientes(cp)
confere("clientes por consumo, quota acumulada a 100 %", list(cl["cliente_id"]) == ["C0001", "C0002", "C0003"]
        and perto(cl["quota_acumulada"].iloc[-1], 100.0))
hoje = pd.Timestamp("2026-11-16")
confere("a terminar em 30 dias: PT4 (20/11) e PT1 (01/12)", list(PD.a_terminar(cp, hoje, 30)["cpe"]) == ["PT4", "PT1"])
confere("a terminar em 3 dias: nenhum", PD.a_terminar(cp, hoje, 3).empty)
alto, baixo = PD.extremos(cp, 1)
confere("extremos: o mais caro e o mais barato", alto["cpe"].iloc[0] == "PT2" and baixo["cpe"].iloc[0] == "PT1")
confere("percentil: 0,15 é mais caro do que 50 % de [0,1; 0,2]", perto(PD.percentil(0.15, pd.Series([0.1, 0.2])), 50.0))
cp["perfil"], cp["tipo_ciclo"] = "A", ["diario", "diario", "diario", "semanal_feriados"]
om = pd.DataFrame([dict(perfil="A", tipo="semanal_feriados", medio=80.0, base=85.0, P=1, C=1, VN=1, SV=1)])
eq = PD.equivalente_indexado(cp, om, {"cgs": {"MT": 22.5}, "perdas": {"MT": 1.08}})
confere("indexada: (80 + 22,5 + 12) × 1,08 / 1000", perto(eq.iloc[3], (80 + 22.5 + 12) * 1.08 / 1000))
confere("as fixas não têm equivalente", eq.iloc[:3].isna().all())
frases = " ".join(t for _, t in PD.conclusoes(cp, cp, {"lista": {"data": "2026-10-12"}}, hoje))
confere("conclusões: em mudança (singular), indexado, lista antiga",
        "1 CPE (50,0 MWh/ano) está em mudança" in frases and "% do consumo está em tarifas indexadas" in frases
        and "12/10/2026" in frases, frases)

print("Leitura cifrada (a mesma cifra do Dashboard Trading)")
def cifrar(ficheiros: dict, palavra: str, sal: bytes) -> bytes:
    from cryptography.fernet import Fernet
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        for nome, texto in ficheiros.items():
            zf.writestr(nome, texto.encode("utf-8"))
    chave = base64.urlsafe_b64encode(hashlib.pbkdf2_hmac("sha256", palavra.encode(), sal, PD.TD.PBKDF2))
    return PD.TD.MARCA + sal + Fernet(chave).encrypt(z.getvalue())

with tempfile.TemporaryDirectory() as d:
    raiz = Path(d) / "pricing"
    confere("sem dados: origem None", PD.origem(raiz) is None)
    (raiz / "pub").mkdir(parents=True)
    textos = {"cpes.csv": cp.drop(columns=["fim_contrato"]).to_csv(index=False),
              "meta.json": json.dumps({"lista": {"data": "2026-10-12"}}), "parametros.json": "{}"}
    (raiz / "pub" / "base.bin").write_bytes(cifrar(textos, "palavra-certa-12", b"0123456789abcdef"))
    confere("só o cifrado: origem 'cifrado'", PD.origem(raiz) == "cifrado")
    q = PD.quadros(PD.ler_cifrado("palavra-certa-12", raiz))
    confere("abre com a palavra certa", len(q["cpes"]) == 4 and q["meta"]["lista"]["data"] == "2026-10-12")
    try:
        PD.ler_cifrado("palavra-errada-12", raiz)
        confere("a palavra errada pára", False, "abriu")
    except PD.PalavraErrada:
        confere("a palavra errada pára (PalavraErrada)", True)
    (raiz / "dados").mkdir()
    (raiz / "dados" / "cpes.csv").write_text(textos["cpes.csv"], encoding="utf-8")
    confere("com o claro: origem 'claro'", PD.origem(raiz) == "claro")

print("Simulador com margem sobre o OMIE e modalidades")
mg = PD.simular("MAX", omie, pesos, cgs=20, custos=2.4, k_interno=6, k_agente=0, perdas=1.08, margem=1.1)
confere("MAX com margem 1,1: ponta = (100 × 1,1 + 2,4 + 6) / 1000", perto(mg.loc[0, "preco"], 0.1184))
confere("a margem é uma parcela à parte (10 €/MWh na ponta) e as parcelas somam o preço",
        perto(mg.loc[0, "margem"], 10.0) and np.allclose(mg[["omie", "margem", "custos", "k_interno", "k_agente"]]
                                                     .sum(axis=1) / 1000, mg["preco"]))
fm = PD.simular("FLEX", omie, pesos, cgs=22, custos=2.4, k_interno=6, k_agente=0, perdas=1.08, margem=1.1)
confere("FLEX com margem: ponta = (110 + 22 + 2,4 + 6) × 1,08 / 1000", perto(fm.loc[0, "preco"], 140.4 * 1.08 / 1000))
rm = PD.resumo_simulacao(fm, "FLEX", consumo_mwh=100, perdas=1.08)
confere("margem do ano = (OMIE × 0,1 + K interno) × perdas × consumo",
        perto(rm["margem_total"], (sum(pesos[p] * omie[p] for p in PD.PERIODOS) * 0.1 + 6) * 1.08 * 100, 1e-6))
md = PD.modalidades(0.1005)
confere("modalidades da oferta da Araújo e Mesquita: 0,1005 -> Base 0,1340, DD ou FE 0,10385, MB e FP 0,1072",
        perto(md["Base"], 0.134) and perto(md["DD ou FE"], 0.103850) and perto(md["MB e FP"], 0.1072))

print("Períodos horários (em vigor e de 2027)")
D = dt.date
q = lambda hhmm: PD._m(hhmm) // 15
seg_ver, seg_inv, sab_ver, dom = D(2026, 7, 6), D(2026, 1, 12), D(2026, 7, 11), D(2026, 7, 12)
pv = PD.periodos_do_dia(seg_ver, "MT", "semanal", None)
confere("semanal em vigor, dia útil de verão: ponta das 09:15 às 12:15", pv[q("09:00")] == "C" and pv[q("09:15")] == "P"
        and pv[q("12:00")] == "P" and pv[q("12:15")] == "C")
pi = PD.periodos_do_dia(seg_inv, "MT", "semanal", None)
confere("semanal em vigor, dia útil de inverno: 5 h de ponta (09:30–12:00 e 18:30–21:00)", pi.count("P") == 20)
po = PD.periodos_do_dia(seg_ver, "MT", "opcional", None)
confere("opcional em vigor, verão: ponta das 14:00 às 17:00, cheias das 00:00 às 00:30", po[q("14:00")] == "P"
        and po[q("16:45")] == "P" and po[q("17:00")] == "C" and po[0] == "C" and po.count("P") == 12)
confere("opcional em vigor, sábado de verão: super vazio das 03:30 às 07:30, sem ponta",
        PD.periodos_do_dia(sab_ver, "MT", "opcional", None)[q("03:30")] == "SV" and "P" not in
        PD.periodos_do_dia(sab_ver, "MT", "opcional", None))
confere("domingo: sem ponta nem cheias", set(PD.periodos_do_dia(dom, "MT", "semanal", None)) == {"VN", "SV"})
pd_ = PD.periodos_do_dia(seg_ver, "BTE", "diario", None)
confere("diário em vigor, verão: ponta 10:30–13:00 e 19:30–21:00 (4 h)", pd_.count("P") == 16 and pd_[q("10:30")] == "P")
nov = D(2027, 4, 1)
d27 = D(2027, 7, 5)                                                     # segunda-feira de verão
confere("2027, MT: o opcional passa a semanal, ponta das 19:30 às 22:30 no verão",
        PD.periodos_do_dia(d27, "MT", "opcional", nov)[q("19:30")] == "P" and
        PD.periodos_do_dia(d27, "MT", "opcional", nov)[q("14:00")] == "C")
confere("antes da data dos períodos novos valem os de hoje", PD.periodos_do_dia(D(2027, 3, 29), "MT", "opcional", nov)
        == PD.VIGENTE["opcional"][("util", "verao")])
fer27 = PD.feriados_nacionais(2027)
confere("feriados de 2027: a Sexta-feira Santa é 26/03", D(2027, 3, 26) in fer27 and D(2027, 6, 10) in fer27)
cp = D(2027, 5, 27)                                                     # Corpo de Deus, quinta-feira
confere("2027, MT: um feriado conta como domingo; BTE semanal não",
        "P" not in PD.periodos_do_dia(cp, "MT", "semanal", nov, fer27) and
        "P" in PD.periodos_do_dia(cp, "BTE", "semanal", nov, fer27))
confere("2027, diário: igual todo o ano, ponta das 17:30 às 21:30",
        PD.periodos_do_dia(d27, "BTN", "diario", nov) == PD.periodos_do_dia(D(2027, 12, 6), "BTN", "diario", nov)
        and PD.periodos_do_dia(d27, "BTN", "diario", nov).count("P") == 16)
confere("hora de verão: do último domingo de março ao último de outubro", PD.hora_de_verao(D(2027, 3, 28)) and
        not PD.hora_de_verao(D(2027, 3, 27)) and not PD.hora_de_verao(D(2027, 10, 31)))

print("Cotações OMIP e previsão conciliada")
with tempfile.TemporaryDirectory() as d:
    f = Path(d) / "omip.csv"
    f.write_text("# fonte\ndate,omip_M Jan_27,omip_M Feb_27,omip_Q1_27,omip_Q2_27,omip_YR_27,omip_YR_28\n"
                 "2026-11-13,180,160,150,,,\n2026-11-16,184,167,152.5,86.5,118.9,69.25\n", encoding="utf-8")
    cot = PD.cotacoes_omip(f)
confere("lê a última data e os meses, trimestres e anos", cot["data"] == D(2026, 11, 16) and cot["mes"][D(2027, 1, 1)] == 184
        and cot["tri"][(2027, 2)] == 86.5 and cot["ano"][2028] == 69.25)
saz = pd.DataFrame({"mes": range(1, 13), "k_ano": [1.0] * 12, "o_trimestre": [1.0] * 12})
cot_t = {"data": D(2026, 11, 16), "mes": {D(2027, 1, 1): 184.0, D(2027, 2, 1): 167.0},
         "tri": {(2027, 1): 152.5, (2027, 2): 86.5}, "ano": {2027: 118.9}}
prev, av = PD.previsao_mensal(cot_t, saz, PD.meses_do_contrato(D(2027, 1, 1), 12))
p = dict(zip(prev["mes"], prev["preco"]))
mar = (152.5 * (31 + 28 + 31) * 24 - 184 * 31 * 24 - 167 * 28 * 24) / (31 * 24)
confere("março = o que falta para o Q1 dar 152,5 (pesado pelas horas)", perto(p[D(2027, 3, 1)], mar, 1e-9), p[D(2027, 3, 1)])
confere("abril a junho = o Q2 (pesos 1)", all(perto(p[D(2027, m, 1)], 86.5) for m in (4, 5, 6)))
h = {m: calendar.monthrange(2027, m)[1] * 24 for m in range(1, 13)}
resto = (118.9 * sum(h.values()) - sum(p[D(2027, m, 1)] * h[m] for m in range(1, 7))) / sum(h[m] for m in range(7, 13))
confere("julho a dezembro = o que falta para o ano dar 118,9", all(perto(p[D(2027, m, 1)], resto, 1e-9) for m in range(7, 13)))
cf = PD.conferencia_previsao(prev, cot_t)
confere("a conferência: cada trimestre e o ano dão a cotação", np.allclose(cf["cotacao"], cf["previsao"]))
confere("a fonte de cada mês", p and list(prev["fonte"])[:4] == ["OMIP mês", "OMIP mês", "OMIP Q1/27", "OMIP Q2/27"])
saz2 = saz.assign(o_trimestre=[1.5, 1.0, 0.5] * 4)
p2 = dict(zip(*PD.previsao_mensal({**cot_t, "mes": {}}, saz2, PD.meses_do_contrato(D(2027, 1, 1), 3))[0][["mes", "preco"]].values.T))
esc = 152.5 * sum(h[m] for m in (1, 2, 3)) / (1.5 * h[1] + 1.0 * h[2] + 0.5 * h[3])
confere("sem meses cotados: o trimestre repartido pelos pesos O", perto(p2[D(2027, 1, 1)], 1.5 * esc, 1e-9)
        and perto(p2[D(2027, 3, 1)], 0.5 * esc, 1e-9))
_, av3 = PD.previsao_mensal({"data": D(2026, 1, 1), "mes": {D(2027, 1, 1): 500.0, D(2027, 2, 1): 500.0},
                             "tri": {(2027, 1): 100.0}, "ano": {}}, saz, [D(2027, 3, 1)])
confere("meses cotados que já passam o trimestre: aviso, sem conciliar", any("Q1/2027" in a for a in av3), av3)

print("Oferta de um contrato")
forma = pd.DataFrame([{"mes": m, "tipo": t, "q": k, "coef": 1.0} for m in range(1, 13) for t in PD.TIPOS_DIA
                      for k in range(96)])
plano = {t: np.ones(96) for t in PD.TIPOS_DIA}
partes = {m: 1 / 12 for m in range(1, 13)}
prev1 = pd.DataFrame({"mes": [D(2026, 7, 1)], "preco": [100.0], "fonte": ["OMIP mês"]})
r = PD.oferta_contrato(prev1, forma, plano, partes, 120.0, "MT", "opcional", None)
dias = [D(2026, 7, 1) + dt.timedelta(days=k) for k in range(31)]
cont = {p_: sum(PD.periodos_do_dia(x, "MT", "opcional", None).count(p_) for x in dias) for p_ in PD.PERIODOS}
confere("preço constante: o OMIE de cada período é o do mês", all(perto(r["omie"][p_], 100.0) for p_ in PD.PERIODOS))
confere("consumo plano: a parte de cada período é a parte dos quartos de hora dele no mês",
        all(perto(r["pesos"][p_], cont[p_] / (31 * 96)) for p_ in PD.PERIODOS))
confere("o consumo do mês é o do ano × a parte do mês (10 MWh)", perto(r["periodos"]["mwh"].sum(), 10.0))
forma2 = forma.copy()
forma2.loc[(forma2["q"] >= 56) & (forma2["q"] < 68), "coef"] = 3.0          # das 14:00 às 17:00 o triplo
r2 = PD.oferta_contrato(prev1, forma2, plano, partes, 120.0, "MT", "opcional", None)
m2 = r2["meses"].iloc[0]
confere("a forma escala-se para a média do mês dar o preço do mês", perto(m2["preco_pesado"], 100.0, 1e-9))
uteis = sum(1 for x in dias if x.weekday() < 5)
alto = 3 * 100 * 31 * 96 / (31 * 96 + 2 * 12 * 31)                          # o preço dessas horas depois de escalar
confere("a ponta do opcional de verão (14–17 h dos dias úteis) fica com o preço dessas horas",
        perto(r2["omie"]["P"], alto, 1e-9), (r2["omie"]["P"], alto))
confere("custo = Σ consumo × preço", perto(r2["periodos"]["custo"].sum(), m2["custo"]))
r3 = PD.oferta_contrato(prev1.assign(mes=[D(2027, 7, 1)]), forma, plano, partes, 120.0, "MT", "opcional", D(2027, 4, 1))
dias3 = [D(2027, 7, 1) + dt.timedelta(days=k) for k in range(31)]
fer = PD.feriados_nacionais(2027)
pp3 = sum(PD.periodos_do_dia(x, "MT", "opcional", D(2027, 4, 1), fer).count("P") for x in dias3)
confere("em julho de 2027 já valem os períodos novos (ponta das 19:30 às 22:30)", perto(r3["pesos"]["P"], pp3 / (31 * 96))
        and r3["meses"].iloc[0]["periodos"] == "2027")

print("Leitura de uma curva da EGASI")
linhas = ["Curvas de carga;;;;;;", "Data atual:;16/11/2026;;;;;", "CPE;FUNCAO;UNIDADE;VALOR;PERDA;DATA;HORA"]
d0 = D(2026, 9, 7)                                                           # segunda-feira
for k in range(14):
    dd = d0 + dt.timedelta(days=k)
    for qq in range(96):
        fim = (qq + 1) * 15
        hora = f"{fim // 60:02d}{fim % 60:02d}"
        kw = 10 if dd.weekday() < 5 else (5 if dd.weekday() == 5 else 2)
        linhas.append(f"PT0002000000000000XX;A+;kW;{kw},0;0;{dd:%Y%m%d};{hora}")
        linhas.append(f"PT0002000000000000XX;Ri+;kVA;999;0;{dd:%Y%m%d};{hora}")
cv = PD.ler_curva("\n".join(linhas).encode("utf-8"), "curvas.csv")
confere("14 dias, só a energia ativa (kW), «2400» é o último quarto de hora", cv["dias"] == 14 and
        perto(float(cv["perfil"]["util"].mean()), 10.0) and perto(float(cv["perfil"]["sab"][95]), 5.0) and
        perto(float(cv["perfil"]["dom"][0]), 2.0), (cv["dias"], cv["perfil"]["util"].mean()))
confere("menos de um ano: sem sazonalidade própria", cv["partes"] is None and cv["cpes"] == ["PT0002000000000000XX"])
try:
    PD.ler_curva("\n".join(linhas[:200]).encode("utf-8"), "curta.csv")
    confere("uma curva com menos de 7 dias completos pára", False, "não parou")
except ValueError:
    confere("uma curva com menos de 7 dias completos pára", True)
prof = pd.DataFrame([{"cpe": "X", "tipo": t, "q": k, "kw": 1.0 + k} for t in PD.TIPOS_DIA for k in range(96)])
pt = PD.perfil_de_tabela(prof, "cpe", "X")
confere("perfil de uma tabela: 96 valores por tipo de dia", pt is not None and perto(float(pt["sab"][95]), 96.0)
        and PD.perfil_de_tabela(prof[prof["tipo"] != "dom"], "cpe", "X") is None)

print("TAR por nível, opção horária e potência")
TAR = {"MT": {"P": 0.0203, "C": 0.0182, "VN": 0.0146, "SV": 0.0129},
       "BTN": {"SIMPLES": {"P": 0.0607, "C": 0.0607, "VN": 0.0607, "SV": 0.0607},
               "BI": {"P": 0.0835, "C": 0.0835, "VN": 0.0158, "SV": 0.0158},
               "TRI": {"P": 0.2452, "C": 0.0412, "VN": 0.0158, "SV": 0.0158},
               "TRI>20.7": {"P": 0.2457, "C": 0.0524, "VN": 0.0150, "SV": 0.0150}}}
confere("MT: a do nível, seja qual for o horário", PD.tar_omissao(TAR, "MT", "TRI", 300)["P"] == 0.0203)
confere("BTN bi: fora de vazio na ponta e nas cheias, vazio nos dois vazios",
        PD.tar_omissao(TAR, "BTN", "BI", 6.9) == {"P": 0.0835, "C": 0.0835, "VN": 0.0158, "SV": 0.0158})
confere("BTN tri até 20,7 kVA e acima (outra tabela, também para o tetra pedido)",
        PD.tar_omissao(TAR, "BTN", "TRI", 20.7)["C"] == 0.0412 and PD.tar_omissao(TAR, "BTN", "TETRA", 27.6)["C"] == 0.0524)
confere("BTN tetra até 20,7 kVA: vale o tri", PD.tar_omissao(TAR, "BTN", "TETRA", 10.35)["P"] == 0.2452)
confere("sem TAR no config: vazio", PD.tar_omissao({}, "MT", "TETRA") == {} and PD.tar_omissao(TAR, "AT", "TETRA") == {})
TP = {"MT": {"ponta": 0.2435, "contratada": 0.0458}, "BTN": {"6.9": 0.3436, "10.35": 0.5154}}
confere("TAR de potência: MT por kW.dia; BTN o escalão igual ou acima", "0.2435" in PD.tar_potencia_omissao(TP, "MT", 300)
        and "10.35 kVA: 0.5154" in PD.tar_potencia_omissao(TP, "BTN", 8.0))

print(f"\n{len(FALHAS)} falha(s)")
sys.exit(1 if FALHAS else 0)
