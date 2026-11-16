"""
teste_pricing.py — as contas do Dashboard Pricing com números feitos à mão
=========================================================================
Corre sem rede nem Streamlit:  python scripts/teste_pricing.py
Tem de acabar com «0 falha(s)».
"""
from __future__ import annotations

import base64
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

print(f"\n{len(FALHAS)} falha(s)")
sys.exit(1 if FALHAS else 0)
