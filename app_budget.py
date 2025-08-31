# -*- coding: utf-8 -*-
"""
Fitkofer Budžet — kućne finansije (Streamlit)
Autor: Fitkofer
Python 3.12+
Funkcije:
- Unos: Trošak / Priliv / Transfer (sa izračunom i validacijama)
- Mesečni dnevnik: filter po mesecu/korisniku, zbir po kategorijama, graf, izvoz CSV
- Budžet: očekivano vs ostvareno vs GAP (po kategoriji i mesecu)
- Stanja: početno po računima, trošak/priliv, trenutno stanje i "ušteda" (razlika)
- 2 korisnika (Marko/Nevena) – tag i filter, bez naloga
- RSD + EUR (kurs po mesecu, preračun u RSD zbirno)
- CSV skladište u ./data
"""

from __future__ import annotations
import uuid
from pathlib import Path
from datetime import datetime, date
from typing import Any, Dict, List

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------- Page setup -----------------------------------
st.set_page_config(
    page_title="Fitkofer Budžet",
    page_icon="💸",
    layout="wide",
)

# ----------------------------- Paths & constants ----------------------------
ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
BUDGET_CSV = DATA_DIR / "budget.csv"
ACCOUNTS_CSV = DATA_DIR / "accounts.csv"
OPENING_CSV = DATA_DIR / "opening_balances.csv"

USERS = ["Marko", "Nevena"]
TYPES = ["Trošak", "Priliv", "Transfer"]

# CSV schemas
TXN_COLUMNS = [
    "id","date","user","type","account","category","description",
    "quantity","unit_price","amount","currency","src_account","dst_account","pair_id","frequency"
]
BUDGET_COLUMNS = ["month","category","expected"]
ACCOUNTS_COLUMNS = ["account","currency"]
OPENING_COLUMNS = ["month","account","opening_balance"]

# Optional password gate via Streamlit secrets
def _get_secret(key: str):
    try:
        return st.secrets.get(key)  # radi i na cloudu i lokalno (ako postoji secrets.toml)
    except Exception:
        return None

_app_pwd = _get_secret("APP_PASSWORD")
if _app_pwd:
    pwd = st.text_input("Lozinka za pristup", type="password")
    if pwd != _app_pwd:
        st.stop()

# ----------------------------- Helpers --------------------------------------
def normalize_float(x: Any) -> float | None:
    """Prihvata tačku ili zarez kao decimalni separator."""
    if x is None or (isinstance(x, str) and x.strip() == ""):
        return None
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def fmt_rsd(v: float) -> str:
    """Format RSD sa zarezom kao decimalnim i tačkama kao hiljadama."""
    s = f"{float(v):,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def ensure_csv(path: Path, columns: List[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")

def load_df(path: Path, columns: List[str]) -> pd.DataFrame:
    ensure_csv(path, columns)
    df = pd.read_csv(path, encoding="utf-8")
    # garantuj kolone i redosled
    for c in columns:
        if c not in df.columns: df[c] = np.nan
    return df[columns]

def save_df(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, encoding="utf-8")

def month_str(d: date) -> str:
    return f"{d:%Y-%m}"

def load_data():
    acc = load_df(ACCOUNTS_CSV, ACCOUNTS_COLUMNS)
    bud = load_df(BUDGET_CSV, BUDGET_COLUMNS)
    tx = load_df(TRANSACTIONS_CSV, TXN_COLUMNS)
    op = load_df(OPENING_CSV, OPENING_COLUMNS)
    return acc, bud, tx, op

def add_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    df = load_df(path, columns)
    df = pd.concat([df, pd.DataFrame(rows, columns=columns)], ignore_index=True)
    save_df(path, df)

def get_month_filter(df: pd.DataFrame, m: str) -> pd.DataFrame:
    return df[df["date"].astype(str).str.startswith(m)]

def to_rsd(amount: float, currency: str, eur_rate: float) -> float:
    if pd.isna(currency) or currency == "RSD":
        return amount
    if currency == "EUR":
        return amount * eur_rate
    return amount  # fallback

# ----------------------------- Load data ------------------------------------
accounts_df, budget_df, tx_df, opening_df = load_data()

# Fallback accounts if empty
if accounts_df.empty:
    default_acc = pd.DataFrame([
        {"account":"Racun Marko","currency":"RSD"},
        {"account":"Racun Nevena","currency":"RSD"},
        {"account":"Kuca","currency":"RSD"},
        {"account":"Eur","currency":"EUR"},
    ])
    accounts_df = default_acc
    save_df(ACCOUNTS_CSV, accounts_df)

# ----------------------------- Sidebar --------------------------------------
st.sidebar.title("💸 Fitkofer Budžet")
# aktivni mesec
today = datetime.now().date()
months_available = sorted(
    set([month_str(today)])
    | set(budget_df["month"].dropna().astype(str))
    | set(opening_df["month"].dropna().astype(str))
    | set(tx_df["date"].dropna().astype(str).str[:7])
)
default_month = month_str(today)
active_month = st.sidebar.selectbox("Aktivni mesec (YYYY-MM)", options=months_available or [default_month], index=(months_available or [default_month]).index(default_month))

# korisnik filter
user_filter = st.sidebar.multiselect("Filter korisnik", USERS, default=USERS)

# EUR kurs (za zbirne RSD izračune)
eur_rate = normalize_float(st.sidebar.text_input("EUR kurs za mesec (RSD)", value="117"))
if eur_rate is None or eur_rate <= 0:
    eur_rate = 117.0
st.sidebar.caption("Kurs se koristi samo za zbirne prikaze u RSD.")

st.sidebar.markdown("---")
st.sidebar.caption("Saveti: decimale možeš sa zarezom ili tačkom. Transfer pravi dva knjiženja (izvor ➜ destinacija).")

# ----------------------------- Tabs -----------------------------------------
tab_unos, tab_dnevnik, tab_budzet, tab_stanja = st.tabs(["➕ Unos", "📒 Dnevnik (mesec)", "📊 Budžet", "🏦 Stanja"])

# ============================= UNOS =========================================
with tab_unos:
    st.subheader("Novi unos")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        d = st.date_input("Datum", value=today)
    with c2:
        korisnik = st.selectbox("Korisnik", USERS, index=0)
    with c3:
        tip = st.selectbox("Tip", TYPES, index=0)

    if tip in ("Trošak", "Priliv"):
        acc = st.selectbox("Račun", accounts_df["account"].tolist(), index=0)
        currency = accounts_df.set_index("account").loc[acc, "currency"]
        category = st.text_input("Kategorija", value="Hrana" if tip=="Trošak" else "Plata")
        description = st.text_input("Opis / trgovac", value="")
        colq, colp, colf = st.columns([1,1,1])
        with colq:
            qty = normalize_float(st.text_input("Količina", value="1"))
        with colp:
            price = normalize_float(st.text_input("Cena (po jedinici)", value="0"))
        with colf:
            freq = st.selectbox("Učestalost", ["Ad hoc","Nedeljno","Mesečno","Godišnje"], index=0)

        if st.button("💾 Sačuvaj", type="primary", use_container_width=True):
            if qty is None or price is None or qty <= 0 or price < 0:
                st.warning("Unesi ispravnu količinu i cenu.")
            else:
                total = round(qty * price, 2)
                amount = -total if tip=="Trošak" else total
                row = {
                    "id": str(uuid.uuid4()),
                    "date": f"{d:%Y-%m-%d}",
                    "user": korisnik,
                    "type": tip,
                    "account": acc,
                    "category": category.strip() or ("Ostalo" if tip=="Trošak" else "Priliv"),
                    "description": description.strip(),
                    "quantity": qty,
                    "unit_price": price,
                    "amount": amount,
                    "currency": currency,
                    "src_account": "" if tip!="Trošak" else acc,
                    "dst_account": "" if tip!="Priliv" else acc,
                    "pair_id": "",
                    "frequency": freq,
                }
                add_rows(TRANSACTIONS_CSV, [row], TXN_COLUMNS)
                st.success(f"Sačuvano: {tip} {fmt_rsd(abs(total))} {currency} na {acc}.")
                tx_df = load_df(TRANSACTIONS_CSV, TXN_COLUMNS)

    else:  # Transfer
        csrc, cdst = st.columns(2)
        with csrc:
            src = st.selectbox("Izvorni račun", accounts_df["account"].tolist(), index=0)
            src_cur = accounts_df.set_index("account").loc[src, "currency"]
        with cdst:
            dst = st.selectbox("Odredišni račun", accounts_df["account"].tolist(), index=1)
            dst_cur = accounts_df.set_index("account").loc[dst, "currency"]
        amount_txt = st.text_input("Iznos za transfer (izvorna valuta)", value="0")
        freq = st.selectbox("Učestalost", ["Ad hoc","Mesečno"], index=0)
        desc = st.text_input("Opis (opciono)", value="Transfer")

        if st.button("🔁 Izvrši transfer", type="primary", use_container_width=True):
            amount = normalize_float(amount_txt)
            if amount is None or amount <= 0 or src == dst:
                st.warning("Proveri iznos i račune (mora biti pozitivan i različiti računi).")
            else:
                pid = str(uuid.uuid4())
                out_row = {
                    "id": str(uuid.uuid4()),
                    "date": f"{d:%Y-%m-%d}",
                    "user": korisnik,
                    "type": "Transfer",
                    "account": src,
                    "category": "Transfer",
                    "description": desc,
                    "quantity": 1, "unit_price": amount, "amount": -amount,
                    "currency": src_cur, "src_account": src, "dst_account": dst,
                    "pair_id": pid, "frequency": freq,
                }
                in_row = {
                    "id": str(uuid.uuid4()),
                    "date": f"{d:%Y-%m-%d}",
                    "user": korisnik,
                    "type": "Transfer",
                    "account": dst,
                    "category": "Transfer",
                    "description": desc,
                    "quantity": 1, "unit_price": amount, "amount": amount,
                    "currency": dst_cur, "src_account": src, "dst_account": dst,
                    "pair_id": pid, "frequency": freq,
                }
                add_rows(TRANSACTIONS_CSV, [out_row, in_row], TXN_COLUMNS)
                st.success(f"Transfer: {fmt_rsd(amount)} {src_cur} {src} → {dst}.")
                tx_df = load_df(TRANSACTIONS_CSV, TXN_COLUMNS)

# ============================= DNEVNIK =======================================
with tab_dnevnik:
    st.subheader(f"Dnevnik — {active_month}")

    month_tx = get_month_filter(tx_df, active_month)
    if user_filter:
        month_tx = month_tx[month_tx["user"].isin(user_filter)]

    st.caption(f"Ukupno stavki: {len(month_tx)}")
    st.dataframe(month_tx.sort_values("date"), use_container_width=True, height=350)

    # Sažeci
    # Preračunaj u RSD za zbirne vrednosti
    rsd_amounts = [
        to_rsd(a, c if isinstance(c, str) else "RSD", eur_rate) 
        for a,c in zip(month_tx["amount"].fillna(0), month_tx["currency"].fillna("RSD"))
    ]
    month_tx = month_tx.assign(amount_rsd=rsd_amounts)

    total_income = month_tx[month_tx["amount_rsd"]>0]["amount_rsd"].sum()
    total_expense = -month_tx[month_tx["amount_rsd"]<0]["amount_rsd"].sum()
    net = total_income - total_expense

    st.markdown(f"**Prilivi:** +{fmt_rsd(total_income)} RSD | **Troškovi:** -{fmt_rsd(total_expense)} RSD | **Neto:** {fmt_rsd(net)} RSD")

    # Grupisanje po kategoriji (bez dupliranja transfera jer je suma 0 po mesecu)
    by_cat = month_tx.groupby("category")["amount_rsd"].sum().sort_values()
    by_cat_df = by_cat.round(2).to_frame().rename(columns={"amount_rsd": "saldo (RSD)"})
    st.markdown("**Saldo po kategoriji (RSD):**")
    st.dataframe(by_cat_df, use_container_width=True, height=250)

    # Graf (troškovi top 8)
    top_exp = month_tx[month_tx["amount_rsd"]<0].groupby("category")["amount_rsd"].sum().abs().sort_values(ascending=False).head(8)
    if not top_exp.empty:
        fig, ax = plt.subplots()
        ax.bar(top_exp.index.tolist(), top_exp.values.tolist())
        ax.set_title("Top troškovi (RSD)")
        ax.set_ylabel("Iznos")
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig, use_container_width=True)

    # Izvoz
    csv_bytes = month_tx.sort_values("date")[TXN_COLUMNS + ["amount_rsd"]].to_csv(index=False, encoding="utf-8").encode("utf-8")
    st.download_button("⬇️ Preuzmi mesečni dnevnik (CSV)", data=csv_bytes, file_name=f"dnevnik_{active_month}.csv", mime="text/csv", use_container_width=True)

# ============================= BUDŽET =======================================
with tab_budzet:
    st.subheader(f"Budžet — {active_month}")

    # Form za očekivane iznose po kategoriji
    # Skupljamo sve relevantne kategorije iz transakcija + postojećeg budžeta
    month_tx = get_month_filter(tx_df, active_month)
    cats_from_tx = sorted(set(month_tx["category"].dropna()) - {"Transfer"})
    cats_from_budget = sorted(set(budget_df[budget_df["month"]==active_month]["category"].dropna()))
    all_cats = sorted(set(cats_from_tx) | set(cats_from_budget))

    if not all_cats:
        st.info("Još nema kategorija za ovaj mesec. Dodaj troškove/prilive ili unesi budžet ručno ispod.")

    edit_rows = []
    for cat in all_cats or ["Hrana","Računi","Transport","Razno"]:
        expected_existing = budget_df[(budget_df["month"]==active_month) & (budget_df["category"]==cat)]["expected"]
        expected_val = float(expected_existing.iloc[0]) if not expected_existing.empty else 0.0
        col1, col2 = st.columns([2,1])
        with col1:
            st.write(f"**{cat}**")
        with col2:
            new_val = normalize_float(st.text_input(f"Očekivano za {cat} (RSD)", value=str(expected_val), key=f"exp_{cat}")) or 0.0
        edit_rows.append({"month":active_month,"category":cat,"expected":round(new_val,2)})

    if st.button("💾 Sačuvaj budžet za mesec", type="primary"):
        # Obriši postojeće za mesec pa upiši nove
        bd = load_df(BUDGET_CSV, BUDGET_COLUMNS)
        bd = bd[bd["month"] != active_month]
        bd = pd.concat([bd, pd.DataFrame(edit_rows)], ignore_index=True)
        save_df(BUDGET_CSV, bd)
        budget_df = bd
        st.success("Budžet sačuvan.")

    # Pregled: očekivano vs ostvareno
    bd_m = budget_df[budget_df["month"]==active_month].copy()
    mtx = get_month_filter(tx_df, active_month).copy()
    mtx["amount_rsd"] = [
        to_rsd(a, c if isinstance(c,str) else "RSD", eur_rate)
        for a,c in zip(mtx["amount"].fillna(0), mtx["currency"].fillna("RSD"))
    ]
    # Ostvareno trošak (pozitivna cifra)
    realized = mtx[mtx["amount_rsd"]<0].groupby("category")["amount_rsd"].sum().abs().reset_index().rename(columns={"amount_rsd":"realized"})
    table = pd.merge(bd_m, realized, on="category", how="outer").fillna(0)
    table["gap"] = table["expected"] - table["realized"]
    table = table.sort_values("category")
    st.markdown("**Očekivano vs Ostvareno (trošak) i GAP (RSD):**")
    st.dataframe(table, use_container_width=True)

# ============================= STANJA =======================================
with tab_stanja:
    st.subheader(f"Stanja po računima — {active_month}")

    # Urednik početnih stanja
    acc_list = accounts_df["account"].tolist()
    ob_rows = []
    for acc in acc_list:
        cur = accounts_df.set_index("account").loc[acc,"currency"]
        existing = opening_df[(opening_df["month"]==active_month) & (opening_df["account"]==acc)]["opening_balance"]
        val = float(existing.iloc[0]) if not existing.empty else 0.0
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            st.write(f"**{acc}** ({cur})")
        with col2:
            new_val = normalize_float(st.text_input(f"Početno stanje ({acc})", value=str(val), key=f"ob_{acc}")) or 0.0
        with col3:
            st.write("")  # spacer
        ob_rows.append({"month":active_month,"account":acc,"opening_balance":round(new_val,2)})

    if st.button("💾 Sačuvaj početna stanja", type="primary"):
        ob = load_df(OPENING_CSV, OPENING_COLUMNS)
        ob = ob[ob["month"] != active_month]
        ob = pd.concat([ob, pd.DataFrame(ob_rows)], ignore_index=True)
        save_df(OPENING_CSV, ob)
        opening_df = ob
        st.success("Početna stanja sačuvana.")

    # Izračun trenutnih stanja
    mtx = get_month_filter(tx_df, active_month).copy()
    acc_cur = accounts_df.set_index("account")["currency"].to_dict()

    # po računu prilivi/troškovi (u originalnoj valuti)
    pivot = mtx.groupby(["account"])["amount"].sum().to_frame().rename(columns={"amount":"neto"})
    pivot["currency"] = pivot.index.map(acc_cur)
    # dodaj račune koji nemaju transakcije
    for a in acc_list:
        if a not in pivot.index:
            pivot.loc[a, ["neto","currency"]] = [0.0, acc_cur[a]]

    # opening
    open_month = opening_df[opening_df["month"]==active_month].set_index("account")["opening_balance"].to_dict()
    pivot["opening"] = pivot.index.map(lambda a: float(open_month.get(a, 0.0)))
    pivot["current"] = pivot["opening"] + pivot["neto"]
    pivot = pivot.reset_index().rename(columns={"index":"account"})

    # tabela prikaza + preračun u RSD
    pivot["rsd_current"] = [
        to_rsd(v, acc_cur.get(a,"RSD"), eur_rate) for a,v in zip(pivot["account"], pivot["current"])
    ]
    pivot["saving"] = pivot["current"] - pivot["opening"]

    show_cols = ["account","opening","neto","current","currency","saving","rsd_current"]
    st.dataframe(pivot[show_cols], use_container_width=True)

    total_liquidity_rsd = pivot["rsd_current"].sum()
    total_saving_rsd = sum([to_rsd(v, acc_cur.get(a,"RSD"), eur_rate) for a,v in zip(pivot["account"], pivot["saving"])])
    st.markdown(f"**Likvidnost (RSD): {fmt_rsd(total_liquidity_rsd)}** | **Ušteda u mesecu (RSD): {fmt_rsd(total_saving_rsd)}**")
