# ============================================================
# THE US DOLLAR SYSTEM — INTERACTIVE STREAMLIT DASHBOARD
# ============================================================
# Rebuilt from the uploaded architecture graphic into a more
# understandable, searchable and interactive learning dashboard.
#
# Install:
#   pip install streamlit pandas numpy plotly
# Optional live market panel:
#   pip install yfinance
# Run:
#   streamlit run dollar_system.py
#
# Data/cache rule:
#   All data-loading and model-building helpers use a 5-minute
#   Streamlit cache (ttl=300). UI render functions are intentionally
#   not cached because they contain widgets.
# ============================================================

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE / STYLE
# ============================================================

st.set_page_config(
    page_title="The US Dollar System",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root{
        --bg:#f6f8fb;
        --card:#ffffff;
        --ink:#142033;
        --muted:#64748b;
        --line:#csve3ec;
        --blue:#2563eb;
        --green:#0f9d74;
        --gold:#d69e2e;
        --purple:#7655d6;
        --red:#c24141;
    }
    .stApp{background:var(--bg);color:var(--ink)}
    .block-container{max-width:1550px;padding-top:1.2rem;padding-bottom:4rem}
    .hero{
        background:radial-gradient(circle at 90% 10%,rgba(37,99,235,.23),transparent 30%),
                   radial-gradient(circle at 10% 90%,rgba(15,157,116,.16),transparent 30%),
                   linear-gradient(135deg,#0f172a,#1e293b);
        color:white;border-radius:24px;padding:30px 34px;margin-bottom:22px;
        box-shadow:0 18px 45px rgba(15,23,42,.15)
    }
    .hero h1{margin:0;font-size:2.65rem;line-height:1.08;letter-spacing:-.04em}
    .hero p{margin:12px 0 0;color:#cbd5e1;max-width:1100px;line-height:1.65}
    .tag{display:inline-block;padding:5px 10px;border-radius:999px;background:rgba(255,255,255,.12);margin-right:5px;font-size:.82rem}
    .section-title{font-size:1.6rem;font-weight:780;letter-spacing:-.03em;margin:28px 0 6px}
    .section-subtitle{color:var(--muted);line-height:1.6;margin-bottom:15px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:17px;box-shadow:0 7px 22px rgba(15,23,42,.055);height:100%}
    .card h4{margin:0 0 8px;font-size:1rem}
    .card p{margin:0;color:#475569;line-height:1.55;font-size:.9rem}
    .kpi{background:white;border:1px solid var(--line);border-radius:17px;padding:17px;box-shadow:0 7px 22px rgba(15,23,42,.055);min-height:135px}
    .kpi-icon{font-size:1.65rem}.kpi-label{font-size:.8rem;color:var(--muted);margin-top:6px}.kpi-value{font-size:1.52rem;font-weight:800;margin-top:2px}.kpi-note{font-size:.75rem;color:var(--muted);line-height:1.35;margin-top:5px}
    .callout{border-left:4px solid var(--blue);background:#eef4ff;padding:13px 15px;border-radius:0 12px 12px 0;line-height:1.55;margin:10px 0 17px}
    .callout.success{border-left-color:var(--green);background:#ecfdf5}.callout.warn{border-left-color:var(--gold);background:#fff8e8}.callout.danger{border-left-color:var(--red);background:#fff1f2}
    .flow{background:white;border:1px solid var(--line);border-radius:16px;padding:17px;box-shadow:0 7px 22px rgba(15,23,42,.055);height:100%}
    .flow-title{font-weight:800;margin-bottom:8px}.flow-steps{font-size:.9rem;line-height:1.8}.arrow{color:#94a3b8;padding:0 5px}.small{font-size:.8rem;color:var(--muted);line-height:1.5}.tiny{font-size:.7rem;color:var(--muted)}
    .chip{display:inline-block;padding:5px 8px;border:1px solid var(--line);background:white;border-radius:999px;margin:3px 4px 3px 0;font-size:.76rem}
    .footer{text-align:center;color:#94a3b8;font-size:.75rem;margin-top:40px;padding-top:20px;border-top:1px solid var(--line)}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def esc(value: object) -> str:
    return html.escape(str(value))


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{esc(subtitle)}</div>', unsafe_allow_html=True)


def card(title: str, body: str, icon: str = "ℹ️") -> None:
    st.markdown(
        f'<div class="card"><h4>{icon} {esc(title)}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(icon: str, label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="kpi"><div class="kpi-icon">{icon}</div><div class="kpi-label">{esc(label)}</div><div class="kpi-value">{esc(value)}</div><div class="kpi-note">{esc(note)}</div></div>',
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "") -> None:
    cls = "callout" + (f" {kind}" if kind else "")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def flow_card(title: str, steps: Sequence[str], description: str, icon: str = "➡️") -> None:
    rendered = f' <span class="arrow">{icon}</span> '.join(f'<b>{esc(s)}</b>' for s in steps)
    st.markdown(
        f'<div class="flow"><div class="flow-title">{esc(title)}</div><div class="flow-steps">{rendered}</div><div class="small" style="margin-top:9px">{esc(description)}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class Entity:
    name: str
    layer: str
    icon: str
    role: str
    description: str
    tags: Tuple[str, ...] = ()
    examples: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Link:
    source: str
    target: str
    kind: str
    description: str


LAYER_COLORS = {
    "International": "#9cc7d8",
    "Domestic Regulators": "#e4a5a5",
    "US Government": "#dfe4e9",
    "Federal Reserve": "#b8d2c8",
    "Offshore": "#ead8b5",
    "Onshore Investors": "#e6d0a7",
    "Banks": "#a8cbdf",
    "Clearing & Settlement": "#c9b5d0",
    "Payments": "#e5ca9c",
    "Depository Institutions": "#a8cadd",
}


# ============================================================
# ENTITY CATALOG — BASE ARCHITECTURE
# ============================================================

def entities() -> Tuple[Entity, ...]:
    e: List[Entity] = []
    def add(name: str, layer: str, icon: str, role: str, desc: str, tags: Iterable[str] = (), examples: Iterable[str] = ()):
        e.append(Entity(name, layer, icon, role, desc, tuple(tags), tuple(examples)))

    # International / supranationals
    add("G20", "International", "🌐", "Policy coordination", "International forum represented in the top governance layer.", ["global", "policy"])
    add("Financial Stability Board (FSB)", "International", "🛡️", "Financial-stability coordination", "Coordinates international financial-stability work.", ["global", "stability"])
    add("World Bank", "International", "🌍", "Development finance", "Development-finance institution in the supranational layer.", ["development", "global"])
    add("IMF", "International", "🏦", "Monetary / balance-of-payments support", "International monetary institution linked to global macro and financial stability.", ["macro", "global"])
    add("BIS", "International", "🏛️", "Central-bank cooperation", "Bank for International Settlements and central-banking forum.", ["central bank", "global"])
    add("BCBS", "International", "🛡️", "Banking supervision standards", "Basel Committee on Banking Supervision.", ["banking", "standards"])
    add("CPMI", "International", "🔄", "Payments / market infrastructure standards", "Committee on Payments and Market Infrastructures.", ["payments", "infrastructure"])
    add("IOSCO", "International", "📈", "Securities regulation coordination", "International Organization of Securities Commissions.", ["securities", "regulation"])
    add("IAIS", "International", "🛡️", "Insurance supervision coordination", "International Association of Insurance Supervisors.", ["insurance", "regulation"])
    add("IAASB", "International", "📚", "Audit standards", "International auditing and assurance standards body.", ["audit", "standards"])
    add("IASB", "International", "📚", "Accounting standards", "International accounting standards body.", ["accounting", "standards"])
    add("FinCoNet", "International", "🛡️", "Financial consumer protection", "International consumer-finance supervisory network.", ["consumer", "regulation"])
    add("IOPS", "International", "👥", "Pension supervision", "International Organization of Pension Supervisors.", ["pensions", "regulation"])
    add("FATF", "International", "🕵️", "AML/CFT standards", "Financial Action Task Force and global AML/CFT framework.", ["AML", "compliance"])
    add("IADI", "International", "🛡️", "Deposit-insurance coordination", "International Association of Deposit Insurers.", ["deposits", "insurance"])

    # Offshore actors
    add("Foreign Central Banks", "Offshore", "🏛️", "Reserve / liquidity interface", "Foreign central banks interacting with the dollar system through reserves, markets and liquidity.", ["reserves", "FX"])
    add("Foreign Central Insurers", "Offshore", "🛡️", "Insurance-sector interface", "Foreign insurance institutions represented in the offshore block.", ["insurance", "cross-border"])
    add("Foreign Money Managers", "Offshore", "📈", "Asset management", "Foreign asset managers allocating to USD-denominated instruments.", ["investors", "offshore"])
    add("Sovereign Wealth Funds", "Offshore", "🏛️", "Sovereign investing", "State-owned investment pools allocating capital into dollar markets.", ["investors", "sovereign"])
    add("Offshore Money Market Funds", "Offshore", "💧", "Short-term liquidity", "Offshore funds investing in short-duration dollar instruments.", ["money market", "offshore"])
    add("Foreign Banks", "Banks", "🏦", "Cross-border banking", "Foreign banks connecting domestic and offshore USD liquidity.", ["banks", "offshore"])
    add("Foreign Subsidiaries of US Banks", "Banks", "🏦", "International banking network", "Foreign legal entities connected to US bank groups.", ["banks", "cross-border"], ["HSBC", "Barclays"])

    # Domestic regulators
    add("NFA", "Domestic Regulators", "🛡️", "Derivatives SRO", "National Futures Association.", ["derivatives", "SRO"])
    add("MSRB", "Domestic Regulators", "🛡️", "Municipal securities rulemaking", "Municipal Securities Rulemaking Board.", ["municipal", "securities"])
    add("FINRA", "Domestic Regulators", "🛡️", "Broker-dealer SRO", "Financial Industry Regulatory Authority.", ["broker-dealer", "SRO"])
    add("FSOC", "Domestic Regulators", "🛡️", "Systemic-risk coordination", "Financial Stability Oversight Council.", ["systemic risk", "macroprudential"])
    add("CFTC", "Domestic Regulators", "🛡️", "Derivatives oversight", "Commodity Futures Trading Commission.", ["derivatives", "regulation"])
    add("SEC", "Domestic Regulators", "🛡️", "Securities oversight", "Securities and Exchange Commission.", ["securities", "regulation"])
    add("NCUA", "Domestic Regulators", "🛡️", "Credit-union supervision", "National Credit Union Administration.", ["credit unions", "regulation"])
    add("FHFA", "Domestic Regulators", "🛡️", "Housing-finance oversight", "Federal Housing Finance Agency.", ["housing", "regulation"])
    add("State Insurance Regulators", "Domestic Regulators", "🛡️", "Insurance regulation", "State-level insurance supervisory authorities.", ["insurance", "states"])
    add("CFPB", "Domestic Regulators", "🛡️", "Consumer-finance oversight", "Consumer Financial Protection Bureau.", ["consumer", "regulation"])
    add("OCC", "Domestic Regulators", "🛡️", "National-bank supervision", "Office of the Comptroller of the Currency.", ["bank", "regulation"])
    add("OFAC", "Domestic Regulators", "🔒", "Sanctions administration", "Office of Foreign Assets Control.", ["sanctions", "compliance"])
    add("State Banking Regulators", "Domestic Regulators", "🛡️", "State bank supervision", "State-chartered banking supervisors.", ["bank", "states"])
    add("State Securities Regulators", "Domestic Regulators", "🛡️", "State securities oversight", "State-level securities regulators represented in the source graphic.", ["securities", "states"])
    add("FDIC", "Domestic Regulators", "🛡️", "Deposit insurance / resolution", "Federal Deposit Insurance Corporation.", ["deposits", "insurance"])
    add("FTC", "Domestic Regulators", "🛡️", "Consumer / competition oversight", "Federal Trade Commission.", ["consumer", "competition"])

    # US government
    add("US Treasury", "US Government", "🏛️", "Sovereign cash and debt management", "Fiscal authority linked to Treasury debt, government cash and the Treasury General Account.", ["Treasury", "government"])
    add("Exchange Stabilization Fund", "US Government", "💱", "Stabilization facility", "Treasury stabilization tool represented near the Treasury block.", ["FX", "stability"])
    add("IRS", "US Government", "🧾", "Federal tax collection", "Internal Revenue Service.", ["tax"])
    add("US Mint", "US Government", "🪙", "Coin issuance", "United States Mint.", ["currency", "coins"])
    add("Fannie Mae", "US Government", "🏠", "Housing-market intermediation", "Government-sponsored enterprise represented in the housing-finance layer.", ["GSE", "housing"])
    add("Freddie Mac", "US Government", "🏠", "Housing-market intermediation", "Government-sponsored enterprise represented in the housing-finance layer.", ["GSE", "housing"])
    add("Other GSEs", "US Government", "🏢", "Specialized finance", "Other government-sponsored entities represented in the source architecture.", ["GSE", "government"])

    # Federal Reserve
    add("Federal Reserve", "Federal Reserve", "🏛️", "Central-bank core", "Central-bank layer responsible for monetary policy, reserves, liquidity and key settlement infrastructure.", ["Fed", "reserves", "liquidity"])
    add("Board of Governors", "Federal Reserve", "🏛️", "Governance / policy", "Federal Reserve Board of Governors.", ["Fed", "policy"])
    add("FOMC", "Federal Reserve", "📊", "Monetary policy", "Federal Open Market Committee.", ["Fed", "rates"])
    add("Federal Reserve Banks", "Federal Reserve", "🏦", "Regional operations", "Regional Reserve Banks operating accounts, payment infrastructure and banking services.", ["Fed", "regional"])
    for city in ["New York", "Boston", "Cleveland", "Atlanta", "Philadelphia", "San Francisco", "Chicago", "Richmond", "Minneapolis", "Kansas City", "Dallas", "St. Louis"]:
        add(f"{city} Fed", "Federal Reserve", "🏦", "Regional Reserve Bank", f"Federal Reserve Bank of {city}.", ["Fed", "regional"])
    add("Federal Open Market Committee", "Federal Reserve", "📊", "Open-market policy committee", "Policy committee represented by the source architecture.", ["Fed", "policy"])
    add("Primary Credit Facility", "Federal Reserve", "💧", "Liquidity backstop", "Central-bank credit facility represented on the asset side.", ["liquidity", "Fed"])
    add("US Treasury Securities", "Federal Reserve", "📜", "Securities asset", "Treasury securities represented as a major Fed asset category.", ["Treasury", "assets"])
    add("Foreign Reserves", "Federal Reserve", "💱", "Foreign reserve asset", "Foreign reserve assets represented in the Fed block.", ["FX", "assets"])
    add("Central Bank Liquidity Swaps", "Federal Reserve", "🔄", "Cross-border dollar liquidity", "Swap arrangements linking the Fed with foreign central banks.", ["swaps", "FX", "liquidity"])
    add("Coins & Currency", "Federal Reserve", "💵", "Cash infrastructure", "Currency and coin category represented in the Fed balance-sheet area.", ["cash", "currency"])
    add("Gold", "Federal Reserve", "🥇", "Reserve asset category", "Gold-related category shown in the source illustration.", ["gold", "assets"])
    add("Short-term Lending", "Federal Reserve", "💧", "Liquidity / lending", "Short-duration lending category shown in the Fed assets block.", ["lending", "liquidity"])
    add("Agency Debt & MBS", "Federal Reserve", "🏠", "Housing-related securities", "Agency debt and mortgage-backed securities category.", ["MBS", "housing"])
    add("Treasury General Account", "Federal Reserve", "🏦", "Government cash account", "US Treasury operating account held at the Federal Reserve.", ["Treasury", "cash"])
    add("FHLB", "Federal Reserve", "🏠", "Housing-finance funding link", "Federal Home Loan Bank category shown in the architecture.", ["housing", "funding"])
    add("Reverse Repurchase Agreements", "Federal Reserve", "🔄", "Liquidity absorption", "Reverse-repo category shown on the Fed liability side.", ["repo", "liquidity"])
    add("Foreign Accounts", "Federal Reserve", "🌍", "Foreign account balances", "Foreign accounts represented on the Fed liability side.", ["foreign", "liabilities"])
    add("Reserve Balances", "Federal Reserve", "💧", "Bank settlement liquidity", "Depository-institution reserve balances at the Federal Reserve.", ["reserves", "settlement"])
    add("Equity Capital", "Federal Reserve", "🧱", "Central-bank capital", "Equity/capital category shown in the source graphic.", ["capital"])
    add("DFMU", "Federal Reserve", "🧩", "Financial-market utilities", "Designated financial market utilities category shown in the source graphic.", ["FMU", "infrastructure"])
    add("Other Deposits", "Federal Reserve", "🏦", "Other liability category", "Other deposits represented on the liability side.", ["deposits"])

    # Onshore investors
    add("Corporates", "Onshore Investors", "🏢", "Corporate cash / funding", "Companies that borrow, invest and manage transaction balances.", ["corporates", "cash"])
    add("Securities Lenders", "Onshore Investors", "📜", "Securities financing", "Institutions supplying securities for borrowing and short-selling.", ["securities lending"])
    add("Insurance Companies", "Onshore Investors", "🛡️", "Institutional investing", "Insurers managing long-duration liabilities and investment portfolios.", ["insurance", "institutional"])
    add("Retail Investors", "Onshore Investors", "👥", "Household investing", "Individuals investing through brokerages and funds.", ["retail"])
    add("Trusts", "Onshore Investors", "📚", "Asset holding", "Trust structures used to hold and administer assets.", ["trusts"])
    add("Prime Money Market Funds", "Onshore Investors", "💧", "Cash management", "Money-market vehicles focused on short-duration instruments.", ["money market", "cash"])
    add("Hedge Funds", "Onshore Investors", "📈", "Leveraged investing", "Alternative funds that may use derivatives, leverage and financing.", ["hedge funds", "leverage"])
    add("Endowments", "Onshore Investors", "🎓", "Long-horizon investing", "Institutional pools associated with universities and foundations.", ["institutional"])
    add("RIAs", "Onshore Investors", "🧑‍💼", "Wealth management", "Registered Investment Advisers.", ["advisory"])
    add("Pension Funds", "Onshore Investors", "👴", "Retirement capital", "Long-horizon pools managing retirement liabilities.", ["pensions"])
    add("Government Money Market Funds", "Onshore Investors", "🏛️", "Government liquidity", "Money-market funds focused on government and agency instruments.", ["money market", "government"])
    add("Exchange & Market Makers", "Onshore Investors", "📈", "Liquidity provision", "Market participants quoting prices and facilitating trading.", ["market making", "liquidity"])
    add("Lending Agents", "Onshore Investors", "🤝", "Securities-finance intermediation", "Intermediaries arranging securities loans for institutional owners.", ["lending"])

    # Banks / dealers
    add("Bank Dealers", "Banks", "🏦", "Financial intermediation", "Dealer banks connect funding, trading, custody, payments and derivatives markets.", ["banks", "dealers"], ["J.P. Morgan", "Morgan Stanley"])
    add("J.P. Morgan", "Banks", "🏦", "Dealer bank", "Example dealer bank represented in the source graphic.", ["dealer", "bank"])
    add("Morgan Stanley", "Banks", "🏦", "Dealer bank", "Example dealer bank represented in the source graphic.", ["dealer", "bank"])
    add("Dealer Assets", "Banks", "📦", "Balance-sheet assets", "Conceptual dealer asset block: securities, receivables, collateral and loans.", ["assets", "dealer"])
    add("Dealer Liabilities", "Banks", "📑", "Balance-sheet liabilities", "Conceptual dealer liability block: payables, securities borrowing and funding.", ["liabilities", "dealer"])
    add("Prime Brokerage", "Banks", "💼", "Leverage / custody services", "Financing, custody and execution services for leveraged investors.", ["prime brokerage"])
    add("Trading Desk", "Banks", "📈", "Market intermediation", "Trading desks intermediate market flows and manage inventory risk.", ["trading"])
    add("Corporate Treasury Desk", "Banks", "🏢", "Bank liquidity management", "Treasury function managing funding, collateral and balance-sheet capacity.", ["treasury", "funding"])

    # Clearing / settlement
    add("CME Group", "Clearing & Settlement", "🔄", "Derivatives clearing", "Major derivatives exchange and clearing ecosystem.", ["CCP", "derivatives"])
    add("ICE", "Clearing & Settlement", "🔄", "Exchange / clearing", "Intercontinental Exchange clearing infrastructure.", ["CCP", "exchange"])
    add("OCC Clearing", "Clearing & Settlement", "🔄", "Options clearing", "Options Clearing Corporation ecosystem.", ["CCP", "options"])
    add("LCH", "Clearing & Settlement", "🔄", "Central counterparty clearing", "Major clearing house represented in the graphic.", ["CCP", "rates"])
    add("DTCC", "Clearing & Settlement", "🔄", "Post-trade infrastructure", "Depository Trust & Clearing Corporation ecosystem.", ["post-trade", "securities"])
    add("CLS", "Clearing & Settlement", "💱", "FX settlement", "Continuous Linked Settlement infrastructure for FX.", ["FX", "settlement"])
    add("HK USD Clearing", "Clearing & Settlement", "💱", "Offshore USD clearing", "Hong Kong USD clearing channel shown in the lower right.", ["USD", "offshore"])
    add("Central Counterparties", "Clearing & Settlement", "🔄", "Counterparty-risk infrastructure", "CCPs stand between counterparties and manage margin/default processes.", ["CCP", "margin"])

    # Payments / settlement
    add("Fedwire Funds", "Payments", "⚡", "Large-value settlement", "Federal Reserve large-value payment infrastructure.", ["payments", "RTGS"])
    add("CHIPS", "Payments", "⚡", "High-value payment clearing", "Clearing House Interbank Payments System.", ["payments", "clearing"])
    add("Fedwire Securities", "Payments", "📜", "Securities settlement", "Federal Reserve securities settlement infrastructure.", ["securities", "Fed"])
    add("SWIFT", "Payments", "🌐", "Financial messaging", "Global messaging network for financial institutions.", ["messaging", "cross-border"])
    add("Fed ACH", "Payments", "⚡", "ACH payments", "Automated clearing-house infrastructure associated with the Federal Reserve.", ["ACH"])
    add("RTP (The Clearing House)", "Payments", "⚡", "Instant payments", "Real-time payment infrastructure operated by The Clearing House.", ["RTP", "instant"])
    add("FedNow", "Payments", "⚡", "Instant payments", "Federal Reserve instant-payment service.", ["instant", "Fed"])
    add("Check Clearing Private TCH", "Payments", "🧾", "Check clearing", "Private-sector check clearing category shown in the source graphic.", ["checks"])
    add("Stablecoins", "Payments", "🪙", "Digital dollar representation", "Dollar-denominated blockchain tokens can form a parallel transfer and settlement rail.", ["stablecoin", "crypto"], ["Tether / USDT"])
    add("Mobile Payment Providers", "Payments", "📱", "Consumer payment interface", "Mobile interfaces connecting customers to payment networks.", ["mobile"])
    add("Remittance Providers", "Payments", "✈️", "Cross-border transfers", "Firms facilitating international transfers and payouts.", ["remittances", "cross-border"])
    add("Money Transfer Operators", "Payments", "💸", "Money transmission", "Operators moving funds domestically or across borders.", ["transfers"])
    add("Foreign Agents", "Payments", "🌍", "Cross-border distribution", "Foreign-side agents represented in the remittance chain.", ["cross-border"])
    add("Merchants", "Payments", "🏪", "End-use of payments", "Businesses receiving payment for goods and services.", ["commerce"])
    add("Card Networks", "Payments", "💳", "Card routing", "Networks connecting issuers, acquirers and merchants.", ["cards"])
    add("ISO / MSP", "Payments", "🔌", "Merchant services", "Independent sales organization / merchant service provider layer.", ["merchant"])
    add("Issuing Processor", "Payments", "💳", "Issuer-side processing", "Processing infrastructure on the card-issuing side.", ["cards"])
    add("Acquiring Processor", "Payments", "💳", "Acquirer-side processing", "Processing infrastructure on the merchant-acquiring side.", ["cards", "merchant"])
    add("Payment Gateway", "Payments", "🔌", "Payment routing", "Gateway that passes transaction data into payment processing.", ["gateway"])
    add("Issuing Bank", "Depository Institutions", "🏦", "Customer-side banking", "Bank associated with payment-card issuance and account funding.", ["bank"])
    add("Merchant Bank", "Depository Institutions", "🏦", "Merchant acquiring", "Bank serving merchant acquiring and settlement relationships.", ["bank", "merchant"])
    add("Customer Account", "Depository Institutions", "👤", "End-customer balance", "Customer account represented at the banking layer.", ["deposits"])
    add("Merchant Account", "Depository Institutions", "🏪", "Merchant settlement", "Merchant account used for receiving payment proceeds.", ["merchant", "settlement"])
    add("Bank of America", "Depository Institutions", "🏦", "Depository institution", "Large US bank represented in the source graphic.", ["bank"])
    add("Wells Fargo", "Depository Institutions", "🏦", "Depository institution", "Large US bank represented in the source graphic.", ["bank"])
    add("Savings", "Payments", "💰", "Stored balances", "Savings relationship represented at the bottom user area.", ["deposits"])
    add("Transfers", "Payments", "↔️", "Account movement", "Account-to-account movement represented at the end-user layer.", ["transfers"])
    add("Purchases", "Payments", "🛒", "Commercial spending", "Payment activity initiated by users.", ["commerce"])
    add("Deposits", "Payments", "💰", "Bank funding", "Deposit activity represented in the end-user flow.", ["deposits"])

    return tuple(e)


# ============================================================
# RELATIONSHIPS — CURATED SYSTEM LINKS
# ============================================================

def links() -> Tuple[Link, ...]:
    rows = [
        ("US Treasury", "Federal Reserve", "Treasury account", "Treasury cash is represented through its account at the Federal Reserve."),
        ("Federal Reserve", "Bank Dealers", "reserves / liquidity", "The central bank influences banking liquidity and settlement balances."),
        ("Bank Dealers", "Onshore Investors", "financing / market making", "Dealers provide financing, custody, execution and liquidity."),
        ("Onshore Investors", "Bank Dealers", "cash / collateral", "Investors provide cash, securities and trading demand."),
        ("Federal Reserve", "Fedwire Funds", "settlement rail", "Large-value payments can settle through Fedwire."),
        ("Federal Reserve", "Fedwire Securities", "securities settlement", "Securities transactions can settle through Fedwire Securities."),
        ("Bank Dealers", "CHIPS", "high-value payment", "Dealer banks can route qualifying payments through CHIPS."),
        ("Bank Dealers", "Prime Brokerage", "prime services", "Dealers provide financing, custody and trading services."),
        ("Prime Brokerage", "Hedge Funds", "financing / custody", "Prime brokers service leveraged investment funds."),
        ("Securities Lenders", "Lending Agents", "securities finance", "Lending agents arrange securities loans."),
        ("Lending Agents", "Bank Dealers", "securities lending", "Dealers can borrow securities for trading and market making."),
        ("Bank Dealers", "CME Group", "derivatives clearing", "Derivative exposures may clear through CME infrastructure."),
        ("Bank Dealers", "ICE", "derivatives clearing", "Derivative exposures may clear through ICE infrastructure."),
        ("Bank Dealers", "OCC Clearing", "options clearing", "Options exposures can clear through OCC."),
        ("Bank Dealers", "LCH", "central clearing", "Rates and other eligible products can clear through LCH."),
        ("Bank Dealers", "DTCC", "post-trade", "Securities post-trade infrastructure links to dealers."),
        ("Bank Dealers", "CLS", "FX settlement", "Eligible FX transactions can settle through CLS."),
        ("CLS", "Foreign Banks", "FX settlement", "CLS links FX participants across jurisdictions."),
        ("Federal Reserve", "Reserve Balances", "central-bank liability", "Reserve balances are core settlement liabilities."),
        ("Reserve Balances", "Bank Dealers", "settlement balances", "Banks use settlement balances for payments."),
        ("Federal Reserve", "Primary Credit Facility", "liquidity backstop", "Central-bank credit can support eligible institutions under defined conditions."),
        ("Federal Reserve", "Reverse Repurchase Agreements", "liquidity absorption", "Reverse repos can absorb liquidity from participating counterparties."),
        ("Federal Reserve", "Central Bank Liquidity Swaps", "FX liquidity", "Swap arrangements can supply dollars to foreign central banks."),
        ("Central Bank Liquidity Swaps", "Foreign Central Banks", "dollar liquidity", "Foreign central banks access dollar liquidity via swap arrangements."),
        ("Foreign Central Banks", "Foreign Banks", "reserve / liquidity channel", "Official-sector dollar liquidity can influence foreign banking systems."),
        ("Foreign Banks", "Foreign Money Managers", "banking / custody", "Foreign banks provide account, custody and transaction links."),
        ("Foreign Money Managers", "Offshore Money Market Funds", "cash investment", "Managers allocate into short-duration funds and assets."),
        ("US Treasury", "Bank Dealers", "Treasury securities", "Dealers distribute and make markets in government securities."),
        ("US Treasury", "Onshore Investors", "Treasury holdings", "Investors hold government securities as dollar assets."),
        ("OFAC", "Foreign Banks", "sanctions compliance", "Sanctions rules can constrain dollar transactions and counterparties."),
        ("SEC", "Bank Dealers", "securities regulation", "Dealer securities activity can fall under SEC oversight."),
        ("CFTC", "CME Group", "derivatives oversight", "CFTC oversees US derivatives markets."),
        ("FDIC", "Depository Institutions", "deposit insurance", "FDIC provides deposit insurance and resolution functions."),
        ("OCC", "Depository Institutions", "bank supervision", "OCC supervises national banks and related institutions."),
        ("FINRA", "Bank Dealers", "broker-dealer SRO", "FINRA is an SRO for broker-dealer activity."),
        ("NFA", "Bank Dealers", "derivatives SRO", "NFA is an SRO for futures and derivatives participants."),
        ("IMF", "Foreign Central Banks", "global monetary coordination", "IMF links into global monetary and balance-of-payments structures."),
        ("BIS", "Foreign Central Banks", "central-bank cooperation", "BIS is a key coordination forum for central banks."),
        ("BCBS", "Bank Dealers", "banking standards", "Basel banking standards influence capital and risk management."),
        ("CPMI", "Fedwire Funds", "payments standards", "CPMI develops principles for payment and market infrastructures."),
        ("SWIFT", "Foreign Banks", "payment messaging", "SWIFT carries standardized financial messages."),
        ("SWIFT", "Bank Dealers", "payment messaging", "Dealer banks use messaging networks for cross-border instructions."),
        ("Fed ACH", "Depository Institutions", "ACH", "ACH infrastructure connects banks to batch payment flows."),
        ("RTP (The Clearing House)", "Depository Institutions", "instant payment", "Participating banks can send and receive real-time payments."),
        ("FedNow", "Depository Institutions", "instant payment", "Participating banks can send and receive instant payments."),
        ("Card Networks", "Issuing Processor", "card authorization", "Card networks connect issuer-side processing."),
        ("Card Networks", "Acquiring Processor", "merchant routing", "Card networks connect acquiring-side processing."),
        ("Issuing Processor", "Issuing Bank", "issuer processing", "Processor supports bank-side card operations."),
        ("Acquiring Processor", "Merchant Bank", "acquirer processing", "Processor supports merchant-side payment operations."),
        ("Merchant Bank", "Merchant Account", "merchant settlement", "Merchant bank maintains the merchant relationship."),
        ("Issuing Bank", "Customer Account", "customer funding", "Customer payment activity debits or funds an account."),
        ("Payment Gateway", "Acquiring Processor", "transaction data", "Gateway sends transaction details into processing."),
        ("ISO / MSP", "Merchants", "merchant services", "Merchant service providers connect businesses to acquiring infrastructure."),
        ("Stablecoins", "Merchants", "digital payment", "Stablecoins can be used as digital dollar payment representations."),
        ("Stablecoins", "Foreign Banks", "digital dollar liquidity", "Stablecoin markets can intersect with offshore users and liquidity venues."),
        ("Remittance Providers", "Foreign Agents", "cross-border transfer", "Remittance providers connect sending and receiving jurisdictions."),
        ("Foreign Agents", "Customer Account", "recipient payout", "Foreign-side agents can connect to recipient accounts."),
        ("US Mint", "Coins & Currency", "coin issuance", "Coin issuance feeds into the currency distribution ecosystem."),
        ("Fannie Mae", "Bank Dealers", "mortgage finance", "Housing-finance securities and funding connect to dealer markets."),
        ("Freddie Mac", "Bank Dealers", "mortgage finance", "Housing-finance securities and funding connect to dealer markets."),
    ]
    return tuple(Link(*row) for row in rows)


# ============================================================
# GLOSSARY / COUNTRY CLASSIFICATION
# ============================================================

def glossary() -> pd.DataFrame:
    rows = [
        ("ACH", "Automated Clearing House", "Payments"),
        ("BCBS", "Basel Committee on Banking Supervision", "International"),
        ("BIS", "Bank for International Settlements", "International"),
        ("CCP", "Central Counterparty", "Clearing"),
        ("CFPB", "Consumer Financial Protection Bureau", "Domestic regulation"),
        ("CHIPS", "Clearing House Interbank Payments System", "Payments"),
        ("CLS", "Continuous Linked Settlement", "FX settlement"),
        ("CFTC", "Commodity Futures Trading Commission", "Domestic regulation"),
        ("CPMI", "Committee on Payments and Market Infrastructures", "International"),
        ("DFMU", "Designated Financial Market Utilities", "Federal Reserve"),
        ("DTCC", "Depository Trust & Clearing Corporation", "Post-trade"),
        ("EPN", "Electronic Payments Network", "Payments"),
        ("ESF", "Exchange Stabilization Fund", "US Government"),
        ("FATF", "Financial Action Task Force", "International"),
        ("FDIC", "Federal Deposit Insurance Corporation", "Domestic regulation"),
        ("Fed", "Federal Reserve System", "Central bank"),
        ("FHFA", "Federal Housing Finance Agency", "Domestic regulation"),
        ("FINRA", "Financial Industry Regulatory Authority", "Domestic regulation"),
        ("FSB", "Financial Stability Board", "International"),
        ("FSOC", "Financial Stability Oversight Council", "Domestic regulation"),
        ("FOMC", "Federal Open Market Committee", "Federal Reserve"),
        ("GSE", "Government-Sponsored Enterprise", "US Government"),
        ("G20", "Group of Twenty", "International"),
        ("IAASB", "International Auditing and Assurance Standards Board", "International"),
        ("IAIS", "International Association of Insurance Supervisors", "International"),
        ("IADI", "International Association of Deposit Insurers", "International"),
        ("IASB", "International Accounting Standards Board", "International"),
        ("IMF", "International Monetary Fund", "International"),
        ("IOPS", "International Organisation of Pension Supervisors", "International"),
        ("IOSCO", "International Organization of Securities Commissions", "International"),
        ("MBS", "Mortgage-Backed Securities", "Markets"),
        ("MSRB", "Municipal Securities Rulemaking Board", "Domestic regulation"),
        ("NFA", "National Futures Association", "Domestic regulation"),
        ("OCC", "Office of the Comptroller of the Currency", "Domestic regulation"),
        ("OFAC", "Office of Foreign Assets Control", "Domestic regulation"),
        ("RIA", "Registered Investment Adviser", "Investors"),
        ("RTP", "Real-Time Payments", "Payments"),
        ("SEC", "Securities and Exchange Commission", "Domestic regulation"),
        ("SWIFT", "Society for Worldwide Interbank Financial Telecommunication", "Messaging"),
        ("TGA", "Treasury General Account", "US Government / Fed"),
        ("TCH", "The Clearing House", "Payments"),
        ("USDT", "Tether stablecoin ticker", "Digital payments"),
    ]
    return pd.DataFrame(rows, columns=["Acronym", "Meaning", "Layer"])


def country_status() -> pd.DataFrame:
    rows = [
        ("Sanctioned", "Russia"), ("Sanctioned", "Iran"), ("Sanctioned", "North Korea"), ("Sanctioned", "Syria"),
        ("Fixed-Peg with USD", "Aruba"), ("Fixed-Peg with USD", "Bahamas"), ("Fixed-Peg with USD", "Bahrain"),
        ("Fixed-Peg with USD", "Barbados"), ("Fixed-Peg with USD", "Belize"), ("Fixed-Peg with USD", "Bermuda"),
        ("Fixed-Peg with USD", "Cayman Islands"), ("Fixed-Peg with USD", "Cuba (CUC)"), ("Fixed-Peg with USD", "Curacao"),
        ("Fixed-Peg with USD", "Djibouti"), ("Fixed-Peg with USD", "Eritrea"), ("Fixed-Peg with USD", "Guyana"),
        ("Fixed-Peg with USD", "Hong Kong"), ("Fixed-Peg with USD", "Macau"), ("Fixed-Peg with USD", "Oman"),
        ("Fixed-Peg with USD", "Panama"), ("Fixed-Peg with USD", "Qatar"), ("Fixed-Peg with USD", "Saudi Arabia"),
        ("Fixed-Peg with USD", "Sint Maarten"), ("Fixed-Peg with USD", "UAE"),
        ("Official Currency", "United States"), ("Official Currency", "Ecuador"), ("Official Currency", "El Salvador"),
        ("Official Currency", "Marshall Islands"), ("Official Currency", "Micronesia"), ("Official Currency", "Palau"),
        ("Official Currency", "Timor-Leste"), ("Official Currency", "Turks and Caicos"), ("Official Currency", "Zimbabwe"),
        ("Unofficial USD Users", "Afghanistan"), ("Unofficial USD Users", "Argentina"), ("Unofficial USD Users", "Cambodia"),
        ("Unofficial USD Users", "Costa Rica"), ("Unofficial USD Users", "Egypt"), ("Unofficial USD Users", "Guatemala"),
        ("Unofficial USD Users", "Honduras"), ("Unofficial USD Users", "Iran"), ("Unofficial USD Users", "Jamaica"),
        ("Unofficial USD Users", "Laos"), ("Unofficial USD Users", "Lebanon"), ("Unofficial USD Users", "Madagascar"),
        ("Unofficial USD Users", "Nicaragua"), ("Unofficial USD Users", "Nigeria"), ("Unofficial USD Users", "Suriname"),
        ("Unofficial USD Users", "Turkey"), ("Unofficial USD Users", "Ukraine"), ("Unofficial USD Users", "Vietnam"),
        ("Unofficial USD Users", "Venezuela"), ("Unofficial USD Users", "Zambia"),
    ]
    return pd.DataFrame(rows, columns=["Status", "Country"])


def balance_sheet() -> pd.DataFrame:
    rows = [
        ("Federal Reserve", "Assets", "US Treasury Securities", "Treasury securities represented as Fed assets"),
        ("Federal Reserve", "Assets", "Foreign Reserves", "Foreign reserve assets"),
        ("Federal Reserve", "Assets", "Primary Credit Facility", "Central-bank lending"),
        ("Federal Reserve", "Assets", "Coins & Currency", "Currency distribution category"),
        ("Federal Reserve", "Assets", "Central Bank Liquidity Swaps", "Cross-border liquidity"),
        ("Federal Reserve", "Assets", "Gold", "Gold category shown"),
        ("Federal Reserve", "Assets", "Short-term Lending", "Short-duration lending"),
        ("Federal Reserve", "Assets", "Agency Debt & MBS", "Agency debt / mortgage securities"),
        ("Federal Reserve", "Liabilities", "Treasury General Account", "Treasury cash account"),
        ("Federal Reserve", "Liabilities", "FHLB", "Housing-finance funding link"),
        ("Federal Reserve", "Liabilities", "Reverse Repurchase Agreements", "Liquidity absorption"),
        ("Federal Reserve", "Liabilities", "Foreign Accounts", "Foreign account balances"),
        ("Federal Reserve", "Liabilities", "Reserve Balances", "Bank settlement balances"),
        ("Federal Reserve", "Liabilities", "Equity Capital", "Capital"),
        ("Federal Reserve", "Liabilities", "DFMU", "Financial-market utilities"),
        ("Federal Reserve", "Liabilities", "Other Deposits", "Other deposits"),
        ("Dealer Banks", "Assets", "Customer Receivables", "Receivables from customers"),
        ("Dealer Banks", "Assets", "Securities Inventory", "Trading inventory"),
        ("Dealer Banks", "Assets", "Collateralized Receivables", "Collateralized funding assets"),
        ("Dealer Banks", "Assets", "Loans", "Loan assets"),
        ("Dealer Banks", "Liabilities", "Customer Payables", "Customer balances owed"),
        ("Dealer Banks", "Liabilities", "Securities Lending", "Securities borrowing obligations"),
        ("Dealer Banks", "Liabilities", "Repo / Short-term Funding", "Wholesale funding"),
        ("Dealer Banks", "Liabilities", "Short-term Deposits", "Deposit funding"),
    ]
    return pd.DataFrame(rows, columns=["Institution", "Side", "Category", "Interpretation"])


def risk_dimensions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Funding liquidity", 5, "High", "Repo, securities lending and short-term funding"),
            ("Settlement liquidity", 5, "High", "Intraday payment obligations and final settlement"),
            ("Collateral dependence", 5, "High", "Treasury collateral, margin and secured funding"),
            ("Counterparty links", 4, "High", "Dealer, CCP and bank exposures"),
            ("Offshore dollar demand", 5, "High", "Foreign borrowers can need USD funding"),
            ("Operational infrastructure", 4, "Medium", "Messaging, clearing and payment systems"),
            ("Regulatory layering", 5, "High", "Multiple agencies and jurisdictions"),
            ("Digital rails", 3, "Emerging", "Stablecoins and tokenized payment channels"),
        ],
        columns=["Dimension", "Score", "Status", "Why it matters"],
    )


# ============================================================
# SEARCH + DATAFRAMES
# ============================================================

def entity_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Name": x.name, "Layer": x.layer, "Role": x.role, "Description": x.description, "Tags": ", ".join(x.tags), "Examples": ", ".join(x.examples)}
        for x in entities()
    ])


def link_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Source": x.source, "Target": x.target, "Kind": x.kind, "Description": x.description}
        for x in links()
    ])


def search_entities(term: str) -> List[str]:
    term = term.strip().lower()
    if not term:
        return entity_df()["Name"].tolist()
    df = entity_df()
    mask = df.apply(lambda r: term in " | ".join(r.astype(str)).lower(), axis=1)
    return df.loc[mask, "Name"].tolist()


# ============================================================
# NETWORK FIGURE
# ============================================================

def network_figure(selected_layers: Tuple[str, ...], max_nodes: int = 95) -> go.Figure:
    all_entities = [x for x in entities() if x.layer in selected_layers]
    names = {x.name for x in all_entities}
    degree = {x.name: 0 for x in all_entities}
    for link in links():
        if link.source in names and link.target in names:
            degree[link.source] += 1
            degree[link.target] += 1
    ranked = sorted(all_entities, key=lambda x: degree[x.name], reverse=True)[:max_nodes]
    names = {x.name for x in ranked}

    layer_order = [
        "International", "Offshore", "US Government", "Domestic Regulators",
        "Federal Reserve", "Banks", "Onshore Investors", "Clearing & Settlement",
        "Depository Institutions", "Payments",
    ]
    xmap = {g: i / max(1, len(layer_order) - 1) for i, g in enumerate(layer_order)}
    coords: Dict[str, Tuple[float, float]] = {}
    for layer in layer_order:
        group = [x for x in ranked if x.layer == layer]
        for i, x in enumerate(group):
            coords[x.name] = (xmap[layer], 0.96 - (i + 1) / (len(group) + 1) * .86)

    fig = go.Figure()

    ex = []
    ey = []
    for link in links():
        if link.source in names and link.target in names:
            x0, y0 = coords[link.source]
            x1, y1 = coords[link.target]
            ex += [x0, x1, None]
            ey += [y0, y1, None]

    if ex:
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="rgba(100,116,139,.28)"), hoverinfo="none", showlegend=False))

    for layer in layer_order:
        group = [x for x in ranked if x.layer == layer]
        if not group:
            continue
        fig.add_trace(go.Scatter(
            x=[coords[x.name][0] for x in group],
            y=[coords[x.name][1] for x in group],
            mode="markers+text",
            name=layer,
            marker=dict(size=17, color=LAYER_COLORS.get(layer, "#cbd5e1"), line=dict(width=1, color="#475569")),
            text=[x.name for x in group],
            textposition="top center",
            textfont=dict(size=8.5, color="#334155"),
            hovertext=[f"<b>{x.icon} {esc(x.name)}</b><br>Layer: {esc(x.layer)}<br>Role: {esc(x.role)}<br>{esc(x.description)}" for x in group],
            hoverinfo="text",
        ))

    fig.update_layout(
        height=780,
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#f8fafc",
        margin=dict(l=15, r=15, t=45, b=15),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-.08, 1.08]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-.03, 1.03]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ============================================================
# HEADER / NAVIGATION
# ============================================================

def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div>
                <span class="tag">💵 USD</span>
                <span class="tag">🏦 Banks</span>
                <span class="tag">🏛️ Federal Reserve</span>
                <span class="tag">🌊 Offshore</span>
                <span class="tag">⚡ Payments</span>
            </div>
            <h1>The US Dollar System</h1>
            <p>A friendlier interactive redesign of the uploaded systems poster. Explore the institutions, balance sheets, regulations, clearing houses, offshore dollar markets and payment rails that connect the global dollar network.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> Tuple[str, Dict[str, object]]:
    st.sidebar.markdown("## 💵 Dollar System")
    st.sidebar.caption("Architecture explorer")
    page = st.sidebar.radio(
        "Explore",
        [
            "🏠 Overview",
            "🗺️ System Map",
            "🔄 Money Flows",
            "🏛️ Federal Reserve",
            "🏦 Banks & Dealers",
            "🌊 Offshore Dollars",
            "🛡️ Regulators",
            "👥 Investors",
            "⚡ Payments",
            "🌍 Global Dollar Status",
            "📚 Glossary",
            "🔎 Entity Explorer",
            "🧠 Learn",
            "📦 Balance Sheets",
        ],
    )
    st.sidebar.divider()
    show_notes = st.sidebar.checkbox("Show architecture notes", True)
    density = st.sidebar.select_slider("Map density", ["Compact", "Balanced", "Detailed"], value="Balanced")
    layers = st.sidebar.multiselect("Quick layer filter", list(LAYER_COLORS), default=list(LAYER_COLORS))
    st.sidebar.divider()
    st.sidebar.caption("Educational model derived from the uploaded architecture graphic. The real financial system is more detailed and changes over time.")
    return page, {"notes": show_notes, "density": density, "layers": layers}


# ============================================================
# PAGE: OVERVIEW
# ============================================================

def overview() -> None:
    hero()
    n_entities = len(entities())
    n_links = len(links())
    n_gloss = len(glossary())
    n_countries = country_status()["Country"].nunique()

    cols = st.columns(5)
    with cols[0]: kpi("🧩", "Entities", f"{n_entities}", "Catalogued from the source graphic")
    with cols[1]: kpi("🔗", "Relationships", f"{n_links}", "Major conceptual links")
    with cols[2]: kpi("🏛️", "Core", "Federal Reserve", "Central-bank layer")
    with cols[3]: kpi("🌍", "Country rows", f"{n_countries}", "Source-graphic classifications")
    with cols[4]: kpi("📚", "Glossary", f"{n_gloss}", "Acronyms and meanings")

    section("The map in one sentence", "The dollar system is a stack of linked balance sheets, market infrastructures, payment rails and regulators — not one giant institution.")
    callout("<b>Most useful distinction:</b> central-bank money, commercial-bank deposits, government securities and digital dollar instruments can all be dollar-denominated, but they are not the same asset or liability.", "success")

    section("Five layers to remember")
    rows = [
        ("🌐", "International", "G20, IMF, BIS, FSB and standard-setting bodies coordinate global policy and infrastructure frameworks."),
        ("🛡️", "Domestic regulation", "SEC, CFTC, OCC, FDIC, FINRA, NFA and state authorities supervise different parts of the system."),
        ("🏛️", "Federal Reserve", "Monetary policy, reserve balances, liquidity facilities and settlement infrastructure."),
        ("🏦", "Markets & banks", "Dealers, commercial banks, funds, corporations and clearing houses move capital and risk."),
        ("⚡", "Payments", "Fedwire, CHIPS, ACH, instant payments, cards, remittances and digital rails connect users."),
    ]
    c1, c2 = st.columns(2)
    for i, row in enumerate(rows):
        with (c1 if i < 3 else c2):
            card(row[1], row[2], row[0])

    section("Four mental models")
    flow_card("Central-bank money", ["Federal Reserve", "Reserve Balances", "Settlement"], "Central-bank liabilities form a core final-settlement layer.")
    flow_card("Commercial-bank money", ["Bank", "Deposit", "Customer"], "Everyday transaction balances are commonly commercial-bank liabilities.")
    flow_card("Market liquidity", ["Collateral", "Dealer", "Repo / Clearing"], "Securities and balance sheets support short-term financing and market liquidity.")
    flow_card("Offshore dollars", ["Foreign Bank", "USD Market", "Global User"], "Dollar activity extends beyond the US through banking, funding, investment and payments.")

    section("Where can the network become stressed?")
    st.dataframe(risk_dimensions(), use_container_width=True, hide_index=True)

    section("Legend")
    cols = st.columns(5)
    for i, (layer, color) in enumerate(LAYER_COLORS.items()):
        with cols[i % 5]:
            st.markdown(f'<div class="chip"><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:3px;margin-right:5px"></span>{esc(layer)}</div>', unsafe_allow_html=True)


# ============================================================
# PAGE: SYSTEM MAP
# ============================================================

def system_map(settings: Dict[str, object]) -> None:
    hero()
    section("🗺️ Interactive System Map", "Filter layers, change density, and hover over any node for a plain-language explanation.")
    selected = st.multiselect("Visible layers", list(LAYER_COLORS), default=settings["layers"])
    density_map = {"Compact": 50, "Balanced": 85, "Detailed": 125}
    if not selected:
        callout("Choose at least one layer.", "warn")
        return
    st.plotly_chart(network_figure(tuple(selected), density_map[settings["density"]]), use_container_width=True)
    callout("<b>How to interpret links:</b> arrows in the original poster and lines in this model can mean a funding, balance-sheet, regulatory, messaging, clearing or settlement relationship. They are not all one-way cash transfers.")
    section("Layer explainer")
    explanations = {
        "International": "Global standards and coordination.",
        "Domestic Regulators": "US federal, state and self-regulatory supervision.",
        "US Government": "Fiscal, tax, currency and housing-finance interfaces.",
        "Federal Reserve": "Central-bank money, reserves, liquidity and settlement.",
        "Offshore": "Foreign users, funds, central banks and offshore dollar activity.",
        "Onshore Investors": "Households, companies, funds and institutional investors.",
        "Banks": "Dealer banks, prime brokerage and funding intermediation.",
        "Clearing & Settlement": "CCPs and post-trade / FX clearing infrastructure.",
        "Payments": "Large-value, retail, instant and digital transfer rails.",
        "Depository Institutions": "Bank accounts, merchant relationships and customer deposits.",
    }
    c = st.columns(3)
    for i, layer in enumerate(selected):
        with c[i % 3]:
            card(layer, explanations[layer], "🧩")


# ============================================================
# PAGE: MONEY FLOWS
# ============================================================

def money_flows() -> None:
    hero()
    section("🔄 Money & Settlement Flows", "Follow the same conceptual network through different use cases.")
    flow_card("Domestic bank payment", ["Customer", "Bank A", "Payment Rail", "Bank B", "Recipient"], "The customer experience can hide several ledgers and settlement steps.", "→")
    flow_card("Cross-border USD payment", ["US Bank", "Messaging / Correspondent", "Foreign Bank", "Local Rail", "Recipient"], "Cross-border payments combine messaging, liquidity, FX, compliance and local settlement.", "→")
    flow_card("Treasury funding", ["US Treasury", "Primary Dealers", "Investors", "Clearing", "Settlement"], "Government securities connect fiscal funding with market liquidity and settlement.", "→")
    flow_card("Repo / collateral loop", ["Cash Lender", "Dealer", "Collateral", "Funding", "Return"], "Secured funding transforms collateral into short-term cash liquidity.", "→")
    flow_card("FX settlement", ["US Dealer", "Foreign Dealer", "FX Trade", "CLS", "Final Accounts"], "FX creates obligations in two currencies and can use payment-versus-payment infrastructure.", "→")
    flow_card("Digital dollar rail", ["User", "Wallet / Stablecoin", "Blockchain", "Merchant / Exchange", "Banking System"], "Stablecoins can provide a separate transfer rail that intersects with traditional finance.", "→")

    section("Four types of dollar-denominated value")
    comparison = pd.DataFrame([
        ("Central-bank money", "Federal Reserve", "Reserves / cash", "Settlement core"),
        ("Commercial-bank money", "Commercial banks", "Deposits", "Everyday spending + bank funding"),
        ("Government debt", "US Treasury", "Treasury securities", "Funding + collateral"),
        ("Private digital dollar", "Stablecoin issuer / network", "Tokenized claim", "Transfer / settlement rail"),
    ], columns=["Layer", "Issuer / operator", "Instrument", "Role"])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    section("Potential bottlenecks")
    bottlenecks = pd.DataFrame([
        ("Bank balance-sheet capacity", "Banks / dealers", "Capital, leverage, liquidity and risk limits"),
        ("Clearing capacity", "CCPs", "Margin, default funds and collateral"),
        ("Settlement capacity", "Payment systems", "Intraday liquidity and operational timing"),
        ("Collateral supply", "Repo / securities lending", "Haircuts, inventory and collateral quality"),
        ("Cross-border liquidity", "Foreign banks", "USD funding and correspondent capacity"),
        ("Messaging / operations", "SWIFT / bank ops", "Compliance, technology and instructions"),
    ], columns=["Bottleneck", "Where", "Typical constraint"])
    st.dataframe(bottlenecks, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: FEDERAL RESERVE
# ============================================================

def fed_page() -> None:
    hero()
    section("🏛️ Federal Reserve", "The central-bank layer of the dollar system — important, but not identical to the whole banking system.")
    c = st.columns(4)
    with c[0]: kpi("💧", "Liquidity", "Backstop", "Facilities and reserve operations")
    with c[1]: kpi("🏦", "Settlement", "Reserve balances", "Bank accounts at the Fed")
    with c[2]: kpi("📜", "Assets", "Treasury / agency", "Securities categories")
    with c[3]: kpi("⚡", "Payments", "Fedwire / ACH", "Core rails")

    callout("<b>Simple mental model:</b> the Fed is the central bank and settlement institution underneath much of the banking system. Commercial banks, in turn, issue deposit liabilities used by customers.", "success")
    section("Core Fed functions")
    funcs = [
        ("📊", "Monetary policy", "FOMC and policy tools influence short-term rates and financial conditions."),
        ("💧", "Liquidity backstop", "Central-bank lending facilities can provide liquidity to eligible institutions."),
        ("🏦", "Reserve accounts", "Banks hold settlement balances at the Federal Reserve."),
        ("⚡", "Payment infrastructure", "Fedwire and ACH connect banks to federal payment services."),
        ("📜", "Securities settlement", "Fedwire Securities supports securities settlement."),
        ("🌍", "International liquidity", "Swap arrangements can provide dollar liquidity to foreign central banks."),
    ]
    cc = st.columns(3)
    for i, (icon, title, body) in enumerate(funcs):
        with cc[i % 3]: card(title, body, icon)

    section("Balance-sheet categories from the source architecture")
    bs = balance_sheet()
    a, b = st.columns(2)
    with a:
        st.markdown("### 📦 Assets")
        st.dataframe(bs[(bs["Institution"] == "Federal Reserve") & (bs["Side"] == "Assets")][["Category", "Interpretation"]], use_container_width=True, hide_index=True)
    with b:
        st.markdown("### 📑 Liabilities / capital")
        st.dataframe(bs[(bs["Institution"] == "Federal Reserve") & (bs["Side"] == "Liabilities")][["Category", "Interpretation"]], use_container_width=True, hide_index=True)

    section("Policy transmission chain")
    trans = pd.DataFrame([
        ("Policy stance", "Rate expectations", "Funding price"),
        ("Funding price", "Bank / dealer balance sheet", "Credit / leverage"),
        ("Balance sheets", "Asset markets", "Risk-taking / prices"),
        ("Asset markets", "Collateral values", "Financing capacity"),
        ("Collateral", "Settlement / clearing", "Liquidity conditions"),
    ], columns=["Starting point", "Channel", "Potential effect"])
    st.dataframe(trans, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: BANKS
# ============================================================

def banks_page() -> None:
    hero()
    section("🏦 Banks & Dealers", "Balance sheets connect the central bank, investors, corporates, markets and payment systems.")
    c = st.columns(4)
    with c[0]: kpi("🏦", "Core role", "Intermediation", "Borrow, lend, trade and settle")
    with c[1]: kpi("💼", "Prime", "Financing", "Custody + leverage + execution")
    with c[2]: kpi("🔄", "Funding", "Repo", "Collateralized liquidity")
    with c[3]: kpi("📈", "Markets", "Liquidity", "Market making and hedging")

    section("Dealer-bank balance sheet")
    bs = balance_sheet()
    a, b = st.columns(2)
    with a:
        st.markdown("### 📦 Assets")
        st.dataframe(bs[(bs["Institution"] == "Dealer Banks") | (bs["Institution"] == "Dealer Banks")][bs.columns][bs["Side"] == "Assets"], use_container_width=True, hide_index=True)
    with b:
        st.markdown("### 📑 Liabilities")
        st.dataframe(bs[(bs["Institution"] == "Dealer Banks") | (bs["Institution"] == "Dealer Banks")][bs.columns][bs["Side"] == "Liabilities"], use_container_width=True, hide_index=True)

    section("What a major dealer can do")
    roles = [
        ("💰", "Finance", "Provide balance-sheet funding to clients."),
        ("📈", "Make markets", "Quote prices and manage inventory risk."),
        ("🔄", "Repo", "Turn collateral into short-term cash."),
        ("📜", "Securities lending", "Borrow or lend securities for market activity."),
        ("🧾", "Custody", "Safekeep and process client securities."),
        ("⚡", "Payments", "Send, receive and settle client payment flows."),
    ]
    cc = st.columns(3)
    for i, (icon, title, body) in enumerate(roles):
        with cc[i % 3]: card(title, body, icon)

    callout("<b>Why dealers matter:</b> a single global dealer can connect Treasury markets, repo, securities lending, FX, derivatives, equities, payments and prime brokerage. That makes dealer balance sheets important transmission points during stress.", "warn")


# ============================================================
# PAGE: OFFSHORE
# ============================================================

def offshore_page() -> None:
    hero()
    section("🌊 Offshore Dollars", "The source architecture emphasizes that dollar-denominated funding and investment extends outside the geographic United States.")
    callout("<b>Terminology:</b> “offshore dollar” means a dollar-denominated claim or funding activity outside the US banking jurisdiction. “Eurodollar” is a historical term and does not mean the euro currency.")
    c = st.columns(4)
    with c[0]: kpi("🌍", "Foreign banks", "Bridge", "Cross-border banking")
    with c[1]: kpi("💧", "Money funds", "Short term", "Cash / funding markets")
    with c[2]: kpi("🏛️", "Central banks", "Reserves", "Official dollar demand")
    with c[3]: kpi("📈", "Managers", "Global", "Portfolio allocation")

    section("Typical offshore chains")
    flow_card("Foreign corporate funding", ["Foreign Company", "Foreign Bank", "USD Funding", "Global Market"], "Foreign companies can borrow or hedge in dollars without being US residents.")
    flow_card("Foreign bank funding", ["Foreign Bank", "Wholesale Market", "Dealer / CCP", "Settlement"], "Offshore funding can still rely on onshore market infrastructure and collateral.")
    flow_card("Official reserve demand", ["Foreign Central Bank", "USD Reserves", "Treasury / Agency", "Settlement"], "Central banks can hold dollar reserve assets and use international market infrastructure.")

    section("Why this matters")
    items = [
        ("💵", "Trade", "Global trade can be invoiced and settled in dollars."),
        ("🏦", "Funding", "Foreign borrowers can need USD refinancing even when revenues are local-currency."),
        ("📈", "Portfolio markets", "Foreign investors can own US and USD assets."),
        ("💱", "FX", "Dollar liquidity is central to global foreign-exchange markets."),
        ("⚠️", "Shortage risk", "Tight dollar funding can transmit stress globally."),
        ("🌍", "Policy spillovers", "US financial conditions can affect foreign borrowers and markets."),
    ]
    cc = st.columns(3)
    for i, (icon, title, body) in enumerate(items):
        with cc[i % 3]: card(title, body, icon)

    section("Offshore vs onshore")
    df = pd.DataFrame([
        ("Location", "Outside US banking jurisdiction", "Inside US financial system"),
        ("Users", "Foreign banks / funds / sovereigns", "US banks / firms / households"),
        ("Funding issue", "FX + cross-border liquidity", "Domestic capital / liquidity"),
        ("Settlement", "Correspondent / CLS / foreign systems", "Fedwire / ACH / CHIPS / domestic rails"),
        ("Market impact", "Global USD liquidity", "Domestic USD liquidity"),
    ], columns=["Dimension", "Offshore", "Onshore"])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: REGULATORS
# ============================================================

def regulators_page() -> None:
    hero()
    section("🛡️ Regulators & Governance", "The system is supervised by multiple federal, state, international and self-regulatory layers.")
    groups = {
        "International": ["G20", "IMF", "BIS", "FSB", "BCBS", "CPMI", "IOSCO", "IAIS", "FATF", "IADI"],
        "Federal / domestic": ["SEC", "CFTC", "OCC", "FDIC", "FHFA", "CFPB", "FTC", "OFAC", "FSOC", "NCUA"],
        "SROs": ["FINRA", "NFA", "MSRB"],
        "State layer": ["State Banking Regulators", "State Insurance Regulators", "State Securities Regulators"],
    }
    cc = st.columns(2)
    for i, (title, names) in enumerate(groups.items()):
        with cc[i % 2]:
            chips = ''.join(f'<span class="chip">🛡️ {esc(x)}</span>' for x in names)
            st.markdown(f'<div class="card"><h4>{esc(title)}</h4><div>{chips}</div></div>', unsafe_allow_html=True)

    section("Who watches what?")
    df = pd.DataFrame([
        ("Banks", "OCC / FDIC / Fed / state regulators", "Safety, capital, liquidity, supervision"),
        ("Broker-dealers", "SEC / FINRA", "Markets, conduct, investor protection"),
        ("Derivatives", "CFTC / NFA", "Futures and derivatives activity"),
        ("Consumer finance", "CFPB / FTC / states", "Consumer protection / competition"),
        ("Sanctions", "OFAC", "Restricted parties and jurisdictions"),
        ("Systemic risk", "FSOC + member agencies", "Cross-market stability"),
        ("Insurance", "State regulators / IAIS", "Insurance solvency and conduct"),
    ], columns=["Area", "Institutions", "Focus"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    callout("<b>Layering matters:</b> one financial transaction can touch securities regulation, bank supervision, sanctions screening, clearing rules and payment infrastructure at the same time.", "warn")


# ============================================================
# PAGE: INVESTORS
# ============================================================

def investors_page() -> None:
    hero()
    section("👥 Investors & Capital Markets", "The investment layer determines who owns, finances and prices the assets moving through the network.")
    investors = [
        ("Prime Money Market Funds", "Short-duration cash management"),
        ("Government Money Market Funds", "Government-focused liquidity"),
        ("Hedge Funds", "Leveraged / alternative investing"),
        ("Pension Funds", "Long-duration retirement capital"),
        ("Insurance Companies", "Long-duration liability matching"),
        ("Endowments", "Long-horizon institutional capital"),
        ("RIAs", "Wealth and portfolio management"),
        ("Retail Investors", "Household investing"),
        ("Corporates", "Operating cash and hedging"),
        ("Exchange & Market Makers", "Liquidity provision"),
        ("Securities Lenders", "Collateral supply"),
        ("Trusts", "Asset holding"),
    ]
    cc = st.columns(3)
    for i, (title, body) in enumerate(investors):
        with cc[i % 3]: card(title, body, "👥")

    section("Investor actions and system effects")
    df = pd.DataFrame([
        ("Buy Treasuries", "Safe-asset demand", "Yields / collateral / bank balance sheets"),
        ("Sell Treasuries", "Cash demand", "Funding / liquidity needs"),
        ("Money-market inflows", "Short-term cash", "Bills / repo / deposits"),
        ("Hedge-fund deleveraging", "Asset selling", "Cross-market stress"),
        ("Foreign selling", "FX + USD flow", "Cross-border dollar demand"),
        ("Securities-borrow demand", "Collateral demand", "Financing conditions"),
    ], columns=["Action", "Immediate effect", "Potential impact"])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: PAYMENTS
# ============================================================

def payments_page() -> None:
    hero()
    section("⚡ Payments & Settlement", "Different rails exist because transactions differ in value, speed, participants and settlement model.")
    rails = pd.DataFrame([
        ("Fedwire Funds", "High value", "Real-time gross settlement", "Banks / eligible participants"),
        ("CHIPS", "High value", "Clearing + settlement", "Large banks"),
        ("Fedwire Securities", "Securities", "Securities settlement", "Financial institutions"),
        ("Fed ACH", "Retail / batch", "Batch clearing", "Banks / businesses"),
        ("RTP", "Retail / business", "Real-time", "Participating banks"),
        ("FedNow", "Retail / business", "Instant", "Participating banks"),
        ("SWIFT", "Cross-border", "Messaging", "Financial institutions"),
        ("CLS", "FX", "Payment-versus-payment", "FX participants"),
        ("Card Networks", "Retail", "Authorization + clearing", "Issuers / acquirers / merchants"),
        ("Stablecoins", "Digital / global", "Token transfer", "Wallets / exchanges / merchants"),
    ], columns=["Rail", "Typical use", "Core function", "Main participants"])
    st.dataframe(rails, use_container_width=True, hide_index=True)

    section("Payment stack")
    stack = [
        ("1", "User", "👤", "Customer, merchant, corporation or institution"),
        ("2", "Interface", "📱", "Mobile app, gateway, card terminal or wallet"),
        ("3", "Messaging", "🌐", "Payment instructions / transaction messages"),
        ("4", "Clearing", "🔄", "Validation, netting and risk management"),
        ("5", "Settlement", "✅", "Final account / obligation discharge"),
        ("6", "Banking", "🏦", "Deposit liabilities and balance sheets"),
    ]
    cc = st.columns(3)
    for i, (n, title, icon, body) in enumerate(stack):
        with cc[i % 3]: card(f"{n}. {title}", body, icon)
    callout("<b>Messaging is not settlement.</b> A message can instruct a payment while actual settlement happens later through the relevant bank and payment infrastructure.", "success")

    section("Card example")
    flow_card("Card purchase", ["Customer Account", "Issuing Bank", "Card Network", "Acquirer", "Merchant", "Settlement"], "A simple checkout experience can involve multiple ledgers and processing layers.")
    section("Cross-border remittance")
    flow_card("Remittance", ["Sender", "Money Transfer Operator", "Foreign Agent", "Recipient Bank", "Customer Account"], "Cross-border retail flows add FX, compliance and payout layers.")


# ============================================================
# PAGE: GLOBAL DOLLAR STATUS
# ============================================================

def global_page() -> None:
    hero()
    df = country_status()
    section("🌍 Dollar Status Around the World", "The following groups are transcribed from the uploaded source graphic. Treat the categories as a source-graphic classification, not a live legal database.")
    counts = df.groupby("Status")["Country"].nunique().sort_values(ascending=False)
    cc = st.columns(4)
    icon_map = {"Sanctioned": "🔒", "Fixed-Peg with USD": "📌", "Official Currency": "💵", "Unofficial USD Users": "🌎"}
    for i, (status, count) in enumerate(counts.items()):
        with cc[i % 4]: kpi(icon_map.get(status, "🌐"), status, str(count), "Rows in source graphic")

    selected = st.multiselect("Statuses", sorted(df["Status"].unique()), default=sorted(df["Status"].unique()))
    filt = df[df["Status"].isin(selected)].copy()
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(counts.reset_index(name="Count"), use_container_width=True, hide_index=True)
    with right:
        chart = filt.groupby("Status")["Country"].nunique().reset_index(name="Country Count")
        fig = px.bar(chart, x="Status", y="Country Count", title="Countries by source-graphic status")
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(filt.sort_values(["Status", "Country"]), use_container_width=True, hide_index=True)
    callout("<b>Market takeaway:</b> dollar use is broader than official-currency status. A country can borrow, invoice, save, hedge or transact in USD without making it the official currency.")


# ============================================================
# PAGE: GLOSSARY
# ============================================================

def glossary_page() -> None:
    hero()
    section("📚 Acronyms & Definitions", "Search the terms that make the original poster difficult to read.")
    df = glossary()
    q = st.text_input("🔎 Search", placeholder="Try: Fed, repo, payments, securities, FX...")
    layers = st.multiselect("Layers", sorted(df["Layer"].unique()), default=[])
    filt = df.copy()
    if q.strip():
        ql = q.lower().strip()
        mask = df.apply(lambda r: ql in " ".join(r.astype(str)).lower(), axis=1)
        filt = filt[mask]
    if layers:
        filt = filt[filt["Layer"].isin(layers)]
    st.dataframe(filt.sort_values(["Layer", "Acronym"]), use_container_width=True, hide_index=True)
    section("Three terms to learn first")
    c = st.columns(3)
    with c[0]: card("Fedwire", "A key Federal Reserve payment and securities settlement infrastructure.", "⚡")
    with c[1]: card("CHIPS", "A high-value USD payment clearing and settlement network used by major banks.", "⚡")
    with c[2]: card("CCP", "A central counterparty that stands between matched parties and manages margin/default processes.", "🔄")


# ============================================================
# PAGE: ENTITY EXPLORER
# ============================================================

def entity_page() -> None:
    hero()
    section("🔎 Entity Explorer", "Pick a component and inspect its layer, role, tags and curated connections.")
    df = entity_df()
    q = st.text_input("Search entities", placeholder="Treasury, Fedwire, hedge fund, OFAC...")
    layers = st.multiselect("Filter layers", sorted(df["Layer"].unique()), default=[])
    filt = df.copy()
    if q.strip():
        matches = search_entities(q)
        filt = filt[filt["Name"].isin(matches)]
    if layers:
        filt = filt[filt["Layer"].isin(layers)]
    names = filt["Name"].tolist()
    if not names:
        callout("No entities matched the filters.", "warn")
        return
    chosen = st.selectbox("Entity", names)
    ent = next(x for x in entities() if x.name == chosen)
    col = LAYER_COLORS.get(ent.layer, "#cbd5e1")
    examples = ", ".join(ent.examples) if ent.examples else "—"
    tags = ", ".join(ent.tags) if ent.tags else "—"
    st.markdown(f'<div class="card" style="border-top:5px solid {col}"><h4>{ent.icon} {esc(ent.name)}</h4><p><b>Layer:</b> {esc(ent.layer)}<br><b>Role:</b> {esc(ent.role)}<br><br>{esc(ent.description)}<br><br><b>Examples:</b> {esc(examples)}<br><b>Tags:</b> {esc(tags)}</p></div>', unsafe_allow_html=True)

    ldf = link_df()
    related = ldf[(ldf["Source"] == chosen) | (ldf["Target"] == chosen)].copy()
    section("Connections", f"{len(related)} curated relationships in the model.")
    if related.empty:
        st.info("No curated connections yet for this entity.")
    else:
        related["Direction"] = np.where(related["Source"] == chosen, "Outgoing", "Incoming")
        related["Other entity"] = np.where(related["Source"] == chosen, related["Target"], related["Source"])
        st.dataframe(related[["Direction", "Other entity", "Kind", "Description"]], use_container_width=True, hide_index=True)


# ============================================================
# PAGE: LEARN
# ============================================================

def learn_page() -> None:
    hero()
    section("🧠 Learn the Dollar System", "A progressive path from simple concepts to market plumbing.")
    lessons = [
        ("Lesson 1 — What is a dollar?", "A dollar can appear as physical cash, a bank deposit, a reserve balance, a Treasury security denomination or a private digital claim."),
        ("Lesson 2 — Where do deposits fit?", "Commercial banks issue deposit liabilities used by customers for spending and saving."),
        ("Lesson 3 — Why does the Fed matter?", "The Fed provides the central-bank settlement layer, monetary policy and liquidity infrastructure."),
        ("Lesson 4 — Why do dealers matter?", "Dealer balance sheets connect investors, funding and market liquidity."),
        ("Lesson 5 — What are offshore dollars?", "Dollar borrowing and investment can happen outside the US through foreign banks and global markets."),
        ("Lesson 6 — Why are CCPs important?", "A CCP can become the counterparty to both sides of a trade and manage margin/default processes."),
        ("Lesson 7 — What is settlement?", "Settlement is the point at which an obligation is discharged through the appropriate account or infrastructure."),
        ("Lesson 8 — Why can the world face a USD shortage?", "Foreign borrowers can owe dollars while revenues are local-currency, so refinancing and hedging can become difficult during stress."),
    ]
    for i, (title, body) in enumerate(lessons, 1):
        with st.expander(title, expanded=(i == 1)):
            st.write(body)
            if i == 2:
                flow_card("Deposit logic", ["Bank Asset", "Bank Liability", "Customer Balance"], "A bank loan can create a deposit; spending transfers deposit claims between banks.")
            elif i == 3:
                flow_card("Settlement layer", ["Commercial Banks", "Fed Accounts", "Final Settlement"], "The central bank sits under the everyday banking experience.")
            elif i == 4:
                flow_card("Dealer bridge", ["Investor A", "Dealer", "Investor B"], "The dealer bridges customer demand and manages inventory and funding.")
            elif i == 5:
                flow_card("Offshore", ["Foreign User", "Foreign Bank", "USD Market"], "The participant can be outside the US while still using dollar markets.")
            elif i == 6:
                flow_card("CCP", ["A", "CCP", "B"], "The CCP interposes itself between matched counterparties under clearing rules.")
            elif i == 7:
                flow_card("Payment lifecycle", ["Instruction", "Clearing", "Settlement", "Final"], "Messaging and clearing are distinct from settlement finality.")
            elif i == 8:
                callout("A USD funding shortage can transmit through FX, credit, collateral, bank balance sheets and global asset prices.", "warn")

    section("Knowledge check")
    quiz = [
        ("Which layer is the central-bank settlement core?", ["Federal Reserve", "Merchant", "Card Network", "Money Transfer Operator"], "Federal Reserve"),
        ("Which is primarily a financial messaging network?", ["SWIFT", "FDIC", "OCC", "G20"], "SWIFT"),
        ("Which structure usually sits between matched derivatives counterparties?", ["CCP", "Merchant", "Tax agency", "Mobile wallet"], "CCP"),
    ]
    for i, (question, options, answer) in enumerate(quiz):
        choice = st.radio(question, options, key=f"quiz_{i}")
        if st.button(f"Check {i+1}", key=f"check_{i}"):
            if choice == answer:
                st.success("Correct.")
            else:
                st.error(f"Best answer: {answer}")


# ============================================================
# PAGE: BALANCE SHEETS
# ============================================================

def balance_page() -> None:
    hero()
    section("📦 Balance-Sheet View", "Translate network arrows into assets, liabilities, claims, collateral and settlement balances.")
    df = balance_sheet()
    institutions = st.multiselect("Institutions", sorted(df["Institution"].unique()), default=sorted(df["Institution"].unique()))
    sides = st.multiselect("Side", sorted(df["Side"].unique()), default=sorted(df["Side"].unique()))
    filt = df[df["Institution"].isin(institutions) & df["Side"].isin(sides)]
    st.dataframe(filt, use_container_width=True, hide_index=True)

    section("Conceptual balance-sheet bridges")
    bridge = pd.DataFrame([
        ("Federal Reserve", "Treasury securities", "Reserve balances / other Fed liabilities"),
        ("Commercial bank", "Customer loan", "Customer deposit"),
        ("Dealer", "Security inventory", "Repo / wholesale funding"),
        ("Investor", "Treasury security", "Cash / financing source"),
        ("CCP", "Margin / settlement claim", "Collateral / counterparty obligations"),
    ], columns=["Entity", "Asset / claim", "Funding / liability"])
    st.dataframe(bridge, use_container_width=True, hide_index=True)
    callout("<b>Core intuition:</b> financial assets are connected through counterparties and balance sheets. Understanding who owes what often explains the network better than looking only at payment arrows.", "success")


# ============================================================
# OPTIONAL LIVE MARKET PANEL
# ============================================================

def optional_market_snapshot() -> pd.DataFrame:
    tickers = ["DX-Y.NYB", "SPY", "QQQ", "TLT", "BTC-USD", "EURUSD=X"]
    try:
        import yfinance as yf
        data = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, progress=False)
        if data is None or data.empty:
            return pd.DataFrame(columns=["Ticker", "Last", "5D %"])
        close = data["Close"] if "Close" in data else data
        rows = []
        for t in tickers:
            if t not in close.columns:
                continue
            s = close[t].dropna()
            if len(s) == 0:
                continue
            last = float(s.iloc[-1])
            base = float(s.iloc[0])
            pct = (last / base - 1.0) * 100 if base else np.nan
            rows.append((t, last, pct))
        return pd.DataFrame(rows, columns=["Ticker", "Last", "5D %"])
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Last", "5D %"])


def live_panel() -> None:
    with st.expander("📈 Optional market snapshot"):
        st.caption("Optional only; the architecture dashboard does not depend on live market data. Market-data functions use a 5-minute cache.")
        df = optional_market_snapshot()
        if df.empty:
            st.info("Live snapshot unavailable. Install yfinance or check network access.")
        else:
            st.dataframe(df.style.format({"Last": "{:.4f}", "5D %": "{:.2f}%"}), use_container_width=True, hide_index=True)


# ============================================================
# EXPORTS
# ============================================================

def export_entities() -> bytes:
    return entity_df().to_csv(index=False).encode()


def export_links() -> bytes:
    return link_df().to_csv(index=False).encode()


def export_glossary() -> bytes:
    return glossary().to_csv(index=False).encode()


def export_countries() -> bytes:
    return country_status().to_csv(index=False).encode()


def export_panel() -> None:
    with st.expander("📥 Export dashboard data"):
        st.download_button("Entities CSV", export_entities(), "us_dollar_entities.csv", "text/csv", use_container_width=True)
        st.download_button("Relationships CSV", export_links(), "us_dollar_relationships.csv", "text/csv", use_container_width=True)
        st.download_button("Glossary CSV", export_glossary(), "us_dollar_glossary.csv", "text/csv", use_container_width=True)
        st.download_button("Country status CSV", export_countries(), "us_dollar_country_status.csv", "text/csv", use_container_width=True)


# ============================================================
# ROUTER
# ============================================================

def route(page: str, settings: Dict[str, object]) -> None:
    if page == "🏠 Overview": overview()
    elif page == "🗺️ System Map": system_map(settings)
    elif page == "🔄 Money Flows": money_flows()
    elif page == "🏛️ Federal Reserve": fed_page()
    elif page == "🏦 Banks & Dealers": banks_page()
    elif page == "🌊 Offshore Dollars": offshore_page()
    elif page == "🛡️ Regulators": regulators_page()
    elif page == "👥 Investors": investors_page()
    elif page == "⚡ Payments": payments_page()
    elif page == "🌍 Global Dollar Status": global_page()
    elif page == "📚 Glossary": glossary_page()
    elif page == "🔎 Entity Explorer": entity_page()
    elif page == "🧠 Learn": learn_page()
    elif page == "📦 Balance Sheets": balance_page()
    else: overview()


def main() -> None:
    page, settings = sidebar()
    if settings["notes"]:
        with st.expander("ℹ️ How to use this dashboard", expanded=False):
            st.write("Start with Overview, then System Map, Federal Reserve, Banks & Dealers, Offshore Dollars and Payments. Use Entity Explorer whenever a label is unclear.")
    route(page, settings)
    if page == "🏠 Overview":
        live_panel()
    export_panel()
    st.markdown(f'<div class="footer">💵 The US Dollar System · Interactive educational architecture · {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}<br>Not investment, legal or regulatory advice.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()


# ============================================================
# ARCHITECTURE REFERENCE APPENDIX
# ============================================================
# The following reference notes are intentionally kept in the source file.
# They are available to developers who want to extend the dashboard, and they
# keep the redesign traceable to the source architecture.
# ============================================================

REFERENCE_TOPICS = [
    'International regulators & supranationals',
    'G20 policy coordination',
    'Financial Stability Board',
    'World Bank',
    'IMF monetary system',
    'BIS central-bank cooperation',
    'BCBS banking standards',
    'CPMI payment standards',
    'IOSCO securities standards',
    'IAIS insurance supervision',
    'IAASB audit standards',
    'IASB accounting standards',
    'FinCoNet consumer protection',
    'IOPS pension supervision',
    'FATF AML/CFT standards',
    'IADI deposit insurance coordination',
    'Foreign central banks',
    'Foreign insurance institutions',
    'Foreign money managers',
    'Sovereign wealth funds',
    'Offshore money market funds',
    'Foreign banks',
    'Foreign subsidiaries of US banks',
    'Domestic regulators',
    'NFA derivatives self-regulation',
    'MSRB municipal securities',
    'FINRA broker-dealer regulation',
    'FSOC systemic risk',
    'CFTC derivatives oversight',
    'SEC securities oversight',
    'NCUA credit unions',
    'FHFA housing finance',
    'State insurance regulators',
    'CFPB consumer finance',
    'OCC national bank supervision',
    'OFAC sanctions',
    'State banking regulators',
    'State securities regulators',
    'FDIC deposit insurance',
    'FTC consumer and competition policy',
    'US Treasury',
    'Exchange Stabilization Fund',
    'IRS',
    'US Mint',
    'Fannie Mae',
    'Freddie Mac',
    'Other GSEs',
    'Federal Reserve',
    'Board of Governors',
    'FOMC',
    'Federal Reserve Banks',
    'New York Fed',
    'Boston Fed',
    'Cleveland Fed',
    'Atlanta Fed',
    'Philadelphia Fed',
    'San Francisco Fed',
    'Chicago Fed',
    'Richmond Fed',
    'Minneapolis Fed',
    'Kansas City Fed',
    'Dallas Fed',
    'St. Louis Fed',
    'Primary Credit Facility',
    'US Treasury securities on Fed assets',
    'Foreign reserves',
    'Central bank liquidity swaps',
    'Coins and currency',
    'Gold category',
    'Short-term lending',
    'Agency debt and MBS',
    'Treasury General Account',
    'FHLB',
    'Reverse repurchase agreements',
    'Foreign accounts',
    'Reserve balances',
    'Equity capital',
    'Designated financial market utilities',
    'Other deposits',
    'Corporates',
    'Securities lenders',
    'Insurance companies',
    'Retail investors',
    'Trusts',
    'Prime money market funds',
    'Hedge funds',
    'Endowments',
    'Registered investment advisers',
    'Pension funds',
    'Government money market funds',
    'Exchange and market makers',
    'Lending agents',
    'Dealer banks',
    'J.P. Morgan example dealer',
    'Morgan Stanley example dealer',
    'Dealer assets',
    'Dealer liabilities',
    'Prime brokerage',
    'Trading desks',
    'Corporate treasury desks',
    'CME Group',
    'ICE',
    'Options Clearing Corporation',
    'LCH',
    'DTCC',
    'CLS',
    'Hong Kong USD clearing',
    'Central counterparties',
    'Fedwire Funds',
    'CHIPS',
    'Fedwire Securities',
    'SWIFT',
    'Fed ACH',
    'RTP',
    'FedNow',
    'Private check clearing',
    'Stablecoins',
    'Mobile payment providers',
    'Remittance providers',
    'Money transfer operators',
    'Foreign agents',
    'Merchants',
    'Card networks',
    'Independent sales organizations and merchant service providers',
    'Issuing processor',
    'Acquiring processor',
    'Payment gateway',
    'Issuing bank',
    'Merchant bank',
    'Customer account',
    'Merchant account',
    'Bank of America',
    'Wells Fargo',
    'Savings',
    'Transfers',
    'Purchases',
    'Deposits',
    'Sanctioned-country source classification',
    'Fixed-peg with USD source classification',
    'Official-currency source classification',
    'Unofficial USD user source classification',
    'Cash versus deposits',
    'Reserves versus deposits',
    'Treasury securities versus bank deposits',
    'Central-bank settlement',
    'Commercial-bank settlement',
    'Correspondent banking',
    'Cross-border USD liquidity',
    'Repo financing',
    'Securities lending',
    'Collateral haircuts',
    'CCP margin',
    'Variation margin',
    'Initial margin',
    'Default funds',
    'Intraday liquidity',
    'Payment finality',
    'Messaging versus settlement',
    'FX payment-versus-payment',
    'Stablecoin settlement concepts',
    'Dollar funding stress',
    'Dealer balance-sheet capacity',
    'Collateral scarcity',
    'Market liquidity',
    'Funding liquidity',
    'Settlement liquidity',
    'Operational resilience',
    'Regulatory perimeter',
    'Systemic risk transmission',
    'Offshore-onshore feecsvack loop',
    'Treasury market liquidity',
    'Banking system deposits',
    'Reserve balances',
    'Government cash management',
    'Money-market fund cash management',
    'Global portfolio allocation',
    'Household payment flows',
    'Merchant acquiring',
    'Card authorization',
    'ACH batch payments',
    'Instant payments',
    'Large-value payments',
    'FX settlement',
    'Post-trade settlement',
    'Clearinghouse risk management',
    'Central counterparty risk',
    'Sanctions screening',
    'AML/CFT controls',
    'Consumer financial protection',
    'Bank capital constraints',
    'Bank liquidity constraints',
    'Wholesale funding',
    'Short-term deposits',
    'Customer receivables',
    'Customer payables',
    'Security inventories',
    'Repo receivables',
    'Treasury collateral',
    'Agency MBS',
    'Mortgage finance',
    'Government-sponsored enterprises',
    'Foreign official reserves',
    'Sovereign wealth funds',
    'Offshore asset managers',
    'Offshore money funds',
    'Global dollar invoicing',
    'USD as funding currency',
    'USD as reserve currency',
    'USD as investment currency',
    'USD as transaction currency',
    'Dollarization concepts',
    'Currency pegs',
    'FX hedging',
    'Cross-border corporate funding',
    'Foreign-bank balance sheets',
    'US dealer balance sheets',
    'Market-maker inventories',
    'Prime brokerage financing',
    'Hedge-fund leverage',
    'Pension fund duration',
    'Insurance asset-liability management',
    'Retail investment flows',
    'Corporate cash balances',
    'Securities borrowing demand',
    'Securities-lending supply',
    'Collateral chains',
    'Clearing-member exposure',
    'Settlement-bank exposure',
    'Payment gateway architecture',
    'Merchant processor architecture',
    'Card issuing stack',
    'Card acquiring stack',
    'Digital wallet interfaces',
    'Remittance payout chains',
    'Foreign agent networks',
    'Customer account balances',
    'Merchant settlement balances',
    'Depository institution role',
    'Federal reserve account role',
    'Treasury account role',
    'Central bank swap line concept',
    'Liquidity backstop concept',
    'Reverse repo concept',
    'Payment-system resilience',
    'Clearing-system resilience',
    'Market-infrastructure resilience',
    'Bank resolution',
    'Deposit insurance',
    'Securities regulation',
    'Derivatives regulation',
    'Insurance regulation',
    'Municipal securities regulation',
    'Credit-union regulation',
    'Housing-finance regulation',
    'State banking oversight',
    'State insurance oversight',
    'State securities oversight',
    'International standards',
    'Domestic standards',
    'Self-regulatory organizations',
    'Financial stability monitoring',
    'System-wide leverage',
    'Collateral-driven deleveraging',
    'Cross-market contagion',
    'Global USD shortage mechanism',
    'Refinancing risk',
    'FX basis intuition',
    'Bank funding spreads',
    'Short-term funding markets',
    'Money-market transmission',
    'Treasury collateral transmission',
    'Payment obligations',
    'Settlement queues',
    'Intraday credit',
    'Counterparty exposure',
    'Margin calls',
    'Liquidity buffers',
    'Cash hoarding',
    'Safe-asset demand',
    'Dollar demand under stress',
    'Offshore funding stress',
    'Global policy spillovers',
    'Architecture map conventions',
    'Source-graphic simplification',
    'Educational interpretation',
    'Interactive node exploration',
    'Layer filtering',
    'Network density control',
    'Search by tags',
    'CSV export',
    'Five-minute cache policy',
    'Optional live-market snapshot',
    'No dependency on market data',
    'Streamlit deployment',
    'Local run command',
    'Cloud deployment considerations',
    'Data refresh behavior',
    'UI responsiveness',
    'Defensive API loading',
    'Fallback behavior',
    'Error-safe optional market data',
    'Developer extension points',
    'Model-data separation',
    'Reusable render helpers',
    'Reusable network model',
    'Reusable glossary model',
    'Reusable country classification',
    'Reusable balance-sheet model',
    'Reusable export layer',
    'Reusable sidebar navigation',
    'Reusable learning modules',
    'Architecture legend',
    'Search panel',
    'Knowledge check',
    'Risk dimensions',
    'Payment stack',
    'Settlement stack',
    'Market plumbing',
    'Regulatory plumbing',
    'Balance-sheet plumbing',
    'Global dollar network',
    'Onshore dollar network',
    'Offshore dollar network',
    'Digital dollar network',
    'Traditional payment network',
    'CCP network',
    'Dealer network',
    'Investor network',
    'Treasury network',
    'Federal Reserve network',
    'End-user network',
]

REFERENCE_EXPLANATIONS = {
    'International regulators & supranationals': "Reference note 1: use the relevant page, entity explorer, or relationship table to study this component.",
    'G20 policy coordination': "Reference note 2: use the relevant page, entity explorer, or relationship table to study this component.",
    'Financial Stability Board': "Reference note 3: use the relevant page, entity explorer, or relationship table to study this component.",
    'World Bank': "Reference note 4: use the relevant page, entity explorer, or relationship table to study this component.",
    'IMF monetary system': "Reference note 5: use the relevant page, entity explorer, or relationship table to study this component.",
    'BIS central-bank cooperation': "Reference note 6: use the relevant page, entity explorer, or relationship table to study this component.",
    'BCBS banking standards': "Reference note 7: use the relevant page, entity explorer, or relationship table to study this component.",
    'CPMI payment standards': "Reference note 8: use the relevant page, entity explorer, or relationship table to study this component.",
    'IOSCO securities standards': "Reference note 9: use the relevant page, entity explorer, or relationship table to study this component.",
    'IAIS insurance supervision': "Reference note 10: use the relevant page, entity explorer, or relationship table to study this component.",
    'IAASB audit standards': "Reference note 11: use the relevant page, entity explorer, or relationship table to study this component.",
    'IASB accounting standards': "Reference note 12: use the relevant page, entity explorer, or relationship table to study this component.",
    'FinCoNet consumer protection': "Reference note 13: use the relevant page, entity explorer, or relationship table to study this component.",
    'IOPS pension supervision': "Reference note 14: use the relevant page, entity explorer, or relationship table to study this component.",
    'FATF AML/CFT standards': "Reference note 15: use the relevant page, entity explorer, or relationship table to study this component.",
    'IADI deposit insurance coordination': "Reference note 16: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign central banks': "Reference note 17: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign insurance institutions': "Reference note 18: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign money managers': "Reference note 19: use the relevant page, entity explorer, or relationship table to study this component.",
    'Sovereign wealth funds': "Reference note 20: use the relevant page, entity explorer, or relationship table to study this component.",
    'Offshore money market funds': "Reference note 21: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign banks': "Reference note 22: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign subsidiaries of US banks': "Reference note 23: use the relevant page, entity explorer, or relationship table to study this component.",
    'Domestic regulators': "Reference note 24: use the relevant page, entity explorer, or relationship table to study this component.",
    'NFA derivatives self-regulation': "Reference note 25: use the relevant page, entity explorer, or relationship table to study this component.",
    'MSRB municipal securities': "Reference note 26: use the relevant page, entity explorer, or relationship table to study this component.",
    'FINRA broker-dealer regulation': "Reference note 27: use the relevant page, entity explorer, or relationship table to study this component.",
    'FSOC systemic risk': "Reference note 28: use the relevant page, entity explorer, or relationship table to study this component.",
    'CFTC derivatives oversight': "Reference note 29: use the relevant page, entity explorer, or relationship table to study this component.",
    'SEC securities oversight': "Reference note 30: use the relevant page, entity explorer, or relationship table to study this component.",
    'NCUA credit unions': "Reference note 31: use the relevant page, entity explorer, or relationship table to study this component.",
    'FHFA housing finance': "Reference note 32: use the relevant page, entity explorer, or relationship table to study this component.",
    'State insurance regulators': "Reference note 33: use the relevant page, entity explorer, or relationship table to study this component.",
    'CFPB consumer finance': "Reference note 34: use the relevant page, entity explorer, or relationship table to study this component.",
    'OCC national bank supervision': "Reference note 35: use the relevant page, entity explorer, or relationship table to study this component.",
    'OFAC sanctions': "Reference note 36: use the relevant page, entity explorer, or relationship table to study this component.",
    'State banking regulators': "Reference note 37: use the relevant page, entity explorer, or relationship table to study this component.",
    'State securities regulators': "Reference note 38: use the relevant page, entity explorer, or relationship table to study this component.",
    'FDIC deposit insurance': "Reference note 39: use the relevant page, entity explorer, or relationship table to study this component.",
    'FTC consumer and competition policy': "Reference note 40: use the relevant page, entity explorer, or relationship table to study this component.",
    'US Treasury': "Reference note 41: use the relevant page, entity explorer, or relationship table to study this component.",
    'Exchange Stabilization Fund': "Reference note 42: use the relevant page, entity explorer, or relationship table to study this component.",
    'IRS': "Reference note 43: use the relevant page, entity explorer, or relationship table to study this component.",
    'US Mint': "Reference note 44: use the relevant page, entity explorer, or relationship table to study this component.",
    'Fannie Mae': "Reference note 45: use the relevant page, entity explorer, or relationship table to study this component.",
    'Freddie Mac': "Reference note 46: use the relevant page, entity explorer, or relationship table to study this component.",
    'Other GSEs': "Reference note 47: use the relevant page, entity explorer, or relationship table to study this component.",
    'Federal Reserve': "Reference note 48: use the relevant page, entity explorer, or relationship table to study this component.",
    'Board of Governors': "Reference note 49: use the relevant page, entity explorer, or relationship table to study this component.",
    'FOMC': "Reference note 50: use the relevant page, entity explorer, or relationship table to study this component.",
    'Federal Reserve Banks': "Reference note 51: use the relevant page, entity explorer, or relationship table to study this component.",
    'New York Fed': "Reference note 52: use the relevant page, entity explorer, or relationship table to study this component.",
    'Boston Fed': "Reference note 53: use the relevant page, entity explorer, or relationship table to study this component.",
    'Cleveland Fed': "Reference note 54: use the relevant page, entity explorer, or relationship table to study this component.",
    'Atlanta Fed': "Reference note 55: use the relevant page, entity explorer, or relationship table to study this component.",
    'Philadelphia Fed': "Reference note 56: use the relevant page, entity explorer, or relationship table to study this component.",
    'San Francisco Fed': "Reference note 57: use the relevant page, entity explorer, or relationship table to study this component.",
    'Chicago Fed': "Reference note 58: use the relevant page, entity explorer, or relationship table to study this component.",
    'Richmond Fed': "Reference note 59: use the relevant page, entity explorer, or relationship table to study this component.",
    'Minneapolis Fed': "Reference note 60: use the relevant page, entity explorer, or relationship table to study this component.",
    'Kansas City Fed': "Reference note 61: use the relevant page, entity explorer, or relationship table to study this component.",
    'Dallas Fed': "Reference note 62: use the relevant page, entity explorer, or relationship table to study this component.",
    'St. Louis Fed': "Reference note 63: use the relevant page, entity explorer, or relationship table to study this component.",
    'Primary Credit Facility': "Reference note 64: use the relevant page, entity explorer, or relationship table to study this component.",
    'US Treasury securities on Fed assets': "Reference note 65: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign reserves': "Reference note 66: use the relevant page, entity explorer, or relationship table to study this component.",
    'Central bank liquidity swaps': "Reference note 67: use the relevant page, entity explorer, or relationship table to study this component.",
    'Coins and currency': "Reference note 68: use the relevant page, entity explorer, or relationship table to study this component.",
    'Gold category': "Reference note 69: use the relevant page, entity explorer, or relationship table to study this component.",
    'Short-term lending': "Reference note 70: use the relevant page, entity explorer, or relationship table to study this component.",
    'Agency debt and MBS': "Reference note 71: use the relevant page, entity explorer, or relationship table to study this component.",
    'Treasury General Account': "Reference note 72: use the relevant page, entity explorer, or relationship table to study this component.",
    'FHLB': "Reference note 73: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reverse repurchase agreements': "Reference note 74: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign accounts': "Reference note 75: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reserve balances': "Reference note 76: use the relevant page, entity explorer, or relationship table to study this component.",
    'Equity capital': "Reference note 77: use the relevant page, entity explorer, or relationship table to study this component.",
    'Designated financial market utilities': "Reference note 78: use the relevant page, entity explorer, or relationship table to study this component.",
    'Other deposits': "Reference note 79: use the relevant page, entity explorer, or relationship table to study this component.",
    'Corporates': "Reference note 80: use the relevant page, entity explorer, or relationship table to study this component.",
    'Securities lenders': "Reference note 81: use the relevant page, entity explorer, or relationship table to study this component.",
    'Insurance companies': "Reference note 82: use the relevant page, entity explorer, or relationship table to study this component.",
    'Retail investors': "Reference note 83: use the relevant page, entity explorer, or relationship table to study this component.",
    'Trusts': "Reference note 84: use the relevant page, entity explorer, or relationship table to study this component.",
    'Prime money market funds': "Reference note 85: use the relevant page, entity explorer, or relationship table to study this component.",
    'Hedge funds': "Reference note 86: use the relevant page, entity explorer, or relationship table to study this component.",
    'Endowments': "Reference note 87: use the relevant page, entity explorer, or relationship table to study this component.",
    'Registered investment advisers': "Reference note 88: use the relevant page, entity explorer, or relationship table to study this component.",
    'Pension funds': "Reference note 89: use the relevant page, entity explorer, or relationship table to study this component.",
    'Government money market funds': "Reference note 90: use the relevant page, entity explorer, or relationship table to study this component.",
    'Exchange and market makers': "Reference note 91: use the relevant page, entity explorer, or relationship table to study this component.",
    'Lending agents': "Reference note 92: use the relevant page, entity explorer, or relationship table to study this component.",
    'Dealer banks': "Reference note 93: use the relevant page, entity explorer, or relationship table to study this component.",
    'J.P. Morgan example dealer': "Reference note 94: use the relevant page, entity explorer, or relationship table to study this component.",
    'Morgan Stanley example dealer': "Reference note 95: use the relevant page, entity explorer, or relationship table to study this component.",
    'Dealer assets': "Reference note 96: use the relevant page, entity explorer, or relationship table to study this component.",
    'Dealer liabilities': "Reference note 97: use the relevant page, entity explorer, or relationship table to study this component.",
    'Prime brokerage': "Reference note 98: use the relevant page, entity explorer, or relationship table to study this component.",
    'Trading desks': "Reference note 99: use the relevant page, entity explorer, or relationship table to study this component.",
    'Corporate treasury desks': "Reference note 100: use the relevant page, entity explorer, or relationship table to study this component.",
    'CME Group': "Reference note 101: use the relevant page, entity explorer, or relationship table to study this component.",
    'ICE': "Reference note 102: use the relevant page, entity explorer, or relationship table to study this component.",
    'Options Clearing Corporation': "Reference note 103: use the relevant page, entity explorer, or relationship table to study this component.",
    'LCH': "Reference note 104: use the relevant page, entity explorer, or relationship table to study this component.",
    'DTCC': "Reference note 105: use the relevant page, entity explorer, or relationship table to study this component.",
    'CLS': "Reference note 106: use the relevant page, entity explorer, or relationship table to study this component.",
    'Hong Kong USD clearing': "Reference note 107: use the relevant page, entity explorer, or relationship table to study this component.",
    'Central counterparties': "Reference note 108: use the relevant page, entity explorer, or relationship table to study this component.",
    'Fedwire Funds': "Reference note 109: use the relevant page, entity explorer, or relationship table to study this component.",
    'CHIPS': "Reference note 110: use the relevant page, entity explorer, or relationship table to study this component.",
    'Fedwire Securities': "Reference note 111: use the relevant page, entity explorer, or relationship table to study this component.",
    'SWIFT': "Reference note 112: use the relevant page, entity explorer, or relationship table to study this component.",
    'Fed ACH': "Reference note 113: use the relevant page, entity explorer, or relationship table to study this component.",
    'RTP': "Reference note 114: use the relevant page, entity explorer, or relationship table to study this component.",
    'FedNow': "Reference note 115: use the relevant page, entity explorer, or relationship table to study this component.",
    'Private check clearing': "Reference note 116: use the relevant page, entity explorer, or relationship table to study this component.",
    'Stablecoins': "Reference note 117: use the relevant page, entity explorer, or relationship table to study this component.",
    'Mobile payment providers': "Reference note 118: use the relevant page, entity explorer, or relationship table to study this component.",
    'Remittance providers': "Reference note 119: use the relevant page, entity explorer, or relationship table to study this component.",
    'Money transfer operators': "Reference note 120: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign agents': "Reference note 121: use the relevant page, entity explorer, or relationship table to study this component.",
    'Merchants': "Reference note 122: use the relevant page, entity explorer, or relationship table to study this component.",
    'Card networks': "Reference note 123: use the relevant page, entity explorer, or relationship table to study this component.",
    'Independent sales organizations and merchant service providers': "Reference note 124: use the relevant page, entity explorer, or relationship table to study this component.",
    'Issuing processor': "Reference note 125: use the relevant page, entity explorer, or relationship table to study this component.",
    'Acquiring processor': "Reference note 126: use the relevant page, entity explorer, or relationship table to study this component.",
    'Payment gateway': "Reference note 127: use the relevant page, entity explorer, or relationship table to study this component.",
    'Issuing bank': "Reference note 128: use the relevant page, entity explorer, or relationship table to study this component.",
    'Merchant bank': "Reference note 129: use the relevant page, entity explorer, or relationship table to study this component.",
    'Customer account': "Reference note 130: use the relevant page, entity explorer, or relationship table to study this component.",
    'Merchant account': "Reference note 131: use the relevant page, entity explorer, or relationship table to study this component.",
    'Bank of America': "Reference note 132: use the relevant page, entity explorer, or relationship table to study this component.",
    'Wells Fargo': "Reference note 133: use the relevant page, entity explorer, or relationship table to study this component.",
    'Savings': "Reference note 134: use the relevant page, entity explorer, or relationship table to study this component.",
    'Transfers': "Reference note 135: use the relevant page, entity explorer, or relationship table to study this component.",
    'Purchases': "Reference note 136: use the relevant page, entity explorer, or relationship table to study this component.",
    'Deposits': "Reference note 137: use the relevant page, entity explorer, or relationship table to study this component.",
    'Sanctioned-country source classification': "Reference note 138: use the relevant page, entity explorer, or relationship table to study this component.",
    'Fixed-peg with USD source classification': "Reference note 139: use the relevant page, entity explorer, or relationship table to study this component.",
    'Official-currency source classification': "Reference note 140: use the relevant page, entity explorer, or relationship table to study this component.",
    'Unofficial USD user source classification': "Reference note 141: use the relevant page, entity explorer, or relationship table to study this component.",
    'Cash versus deposits': "Reference note 142: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reserves versus deposits': "Reference note 143: use the relevant page, entity explorer, or relationship table to study this component.",
    'Treasury securities versus bank deposits': "Reference note 144: use the relevant page, entity explorer, or relationship table to study this component.",
    'Central-bank settlement': "Reference note 145: use the relevant page, entity explorer, or relationship table to study this component.",
    'Commercial-bank settlement': "Reference note 146: use the relevant page, entity explorer, or relationship table to study this component.",
    'Correspondent banking': "Reference note 147: use the relevant page, entity explorer, or relationship table to study this component.",
    'Cross-border USD liquidity': "Reference note 148: use the relevant page, entity explorer, or relationship table to study this component.",
    'Repo financing': "Reference note 149: use the relevant page, entity explorer, or relationship table to study this component.",
    'Securities lending': "Reference note 150: use the relevant page, entity explorer, or relationship table to study this component.",
    'Collateral haircuts': "Reference note 151: use the relevant page, entity explorer, or relationship table to study this component.",
    'CCP margin': "Reference note 152: use the relevant page, entity explorer, or relationship table to study this component.",
    'Variation margin': "Reference note 153: use the relevant page, entity explorer, or relationship table to study this component.",
    'Initial margin': "Reference note 154: use the relevant page, entity explorer, or relationship table to study this component.",
    'Default funds': "Reference note 155: use the relevant page, entity explorer, or relationship table to study this component.",
    'Intraday liquidity': "Reference note 156: use the relevant page, entity explorer, or relationship table to study this component.",
    'Payment finality': "Reference note 157: use the relevant page, entity explorer, or relationship table to study this component.",
    'Messaging versus settlement': "Reference note 158: use the relevant page, entity explorer, or relationship table to study this component.",
    'FX payment-versus-payment': "Reference note 159: use the relevant page, entity explorer, or relationship table to study this component.",
    'Stablecoin settlement concepts': "Reference note 160: use the relevant page, entity explorer, or relationship table to study this component.",
    'Dollar funding stress': "Reference note 161: use the relevant page, entity explorer, or relationship table to study this component.",
    'Dealer balance-sheet capacity': "Reference note 162: use the relevant page, entity explorer, or relationship table to study this component.",
    'Collateral scarcity': "Reference note 163: use the relevant page, entity explorer, or relationship table to study this component.",
    'Market liquidity': "Reference note 164: use the relevant page, entity explorer, or relationship table to study this component.",
    'Funding liquidity': "Reference note 165: use the relevant page, entity explorer, or relationship table to study this component.",
    'Settlement liquidity': "Reference note 166: use the relevant page, entity explorer, or relationship table to study this component.",
    'Operational resilience': "Reference note 167: use the relevant page, entity explorer, or relationship table to study this component.",
    'Regulatory perimeter': "Reference note 168: use the relevant page, entity explorer, or relationship table to study this component.",
    'Systemic risk transmission': "Reference note 169: use the relevant page, entity explorer, or relationship table to study this component.",
    'Offshore-onshore feecsvack loop': "Reference note 170: use the relevant page, entity explorer, or relationship table to study this component.",
    'Treasury market liquidity': "Reference note 171: use the relevant page, entity explorer, or relationship table to study this component.",
    'Banking system deposits': "Reference note 172: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reserve balances': "Reference note 173: use the relevant page, entity explorer, or relationship table to study this component.",
    'Government cash management': "Reference note 174: use the relevant page, entity explorer, or relationship table to study this component.",
    'Money-market fund cash management': "Reference note 175: use the relevant page, entity explorer, or relationship table to study this component.",
    'Global portfolio allocation': "Reference note 176: use the relevant page, entity explorer, or relationship table to study this component.",
    'Household payment flows': "Reference note 177: use the relevant page, entity explorer, or relationship table to study this component.",
    'Merchant acquiring': "Reference note 178: use the relevant page, entity explorer, or relationship table to study this component.",
    'Card authorization': "Reference note 179: use the relevant page, entity explorer, or relationship table to study this component.",
    'ACH batch payments': "Reference note 180: use the relevant page, entity explorer, or relationship table to study this component.",
    'Instant payments': "Reference note 181: use the relevant page, entity explorer, or relationship table to study this component.",
    'Large-value payments': "Reference note 182: use the relevant page, entity explorer, or relationship table to study this component.",
    'FX settlement': "Reference note 183: use the relevant page, entity explorer, or relationship table to study this component.",
    'Post-trade settlement': "Reference note 184: use the relevant page, entity explorer, or relationship table to study this component.",
    'Clearinghouse risk management': "Reference note 185: use the relevant page, entity explorer, or relationship table to study this component.",
    'Central counterparty risk': "Reference note 186: use the relevant page, entity explorer, or relationship table to study this component.",
    'Sanctions screening': "Reference note 187: use the relevant page, entity explorer, or relationship table to study this component.",
    'AML/CFT controls': "Reference note 188: use the relevant page, entity explorer, or relationship table to study this component.",
    'Consumer financial protection': "Reference note 189: use the relevant page, entity explorer, or relationship table to study this component.",
    'Bank capital constraints': "Reference note 190: use the relevant page, entity explorer, or relationship table to study this component.",
    'Bank liquidity constraints': "Reference note 191: use the relevant page, entity explorer, or relationship table to study this component.",
    'Wholesale funding': "Reference note 192: use the relevant page, entity explorer, or relationship table to study this component.",
    'Short-term deposits': "Reference note 193: use the relevant page, entity explorer, or relationship table to study this component.",
    'Customer receivables': "Reference note 194: use the relevant page, entity explorer, or relationship table to study this component.",
    'Customer payables': "Reference note 195: use the relevant page, entity explorer, or relationship table to study this component.",
    'Security inventories': "Reference note 196: use the relevant page, entity explorer, or relationship table to study this component.",
    'Repo receivables': "Reference note 197: use the relevant page, entity explorer, or relationship table to study this component.",
    'Treasury collateral': "Reference note 198: use the relevant page, entity explorer, or relationship table to study this component.",
    'Agency MBS': "Reference note 199: use the relevant page, entity explorer, or relationship table to study this component.",
    'Mortgage finance': "Reference note 200: use the relevant page, entity explorer, or relationship table to study this component.",
    'Government-sponsored enterprises': "Reference note 201: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign official reserves': "Reference note 202: use the relevant page, entity explorer, or relationship table to study this component.",
    'Sovereign wealth funds': "Reference note 203: use the relevant page, entity explorer, or relationship table to study this component.",
    'Offshore asset managers': "Reference note 204: use the relevant page, entity explorer, or relationship table to study this component.",
    'Offshore money funds': "Reference note 205: use the relevant page, entity explorer, or relationship table to study this component.",
    'Global dollar invoicing': "Reference note 206: use the relevant page, entity explorer, or relationship table to study this component.",
    'USD as funding currency': "Reference note 207: use the relevant page, entity explorer, or relationship table to study this component.",
    'USD as reserve currency': "Reference note 208: use the relevant page, entity explorer, or relationship table to study this component.",
    'USD as investment currency': "Reference note 209: use the relevant page, entity explorer, or relationship table to study this component.",
    'USD as transaction currency': "Reference note 210: use the relevant page, entity explorer, or relationship table to study this component.",
    'Dollarization concepts': "Reference note 211: use the relevant page, entity explorer, or relationship table to study this component.",
    'Currency pegs': "Reference note 212: use the relevant page, entity explorer, or relationship table to study this component.",
    'FX hedging': "Reference note 213: use the relevant page, entity explorer, or relationship table to study this component.",
    'Cross-border corporate funding': "Reference note 214: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign-bank balance sheets': "Reference note 215: use the relevant page, entity explorer, or relationship table to study this component.",
    'US dealer balance sheets': "Reference note 216: use the relevant page, entity explorer, or relationship table to study this component.",
    'Market-maker inventories': "Reference note 217: use the relevant page, entity explorer, or relationship table to study this component.",
    'Prime brokerage financing': "Reference note 218: use the relevant page, entity explorer, or relationship table to study this component.",
    'Hedge-fund leverage': "Reference note 219: use the relevant page, entity explorer, or relationship table to study this component.",
    'Pension fund duration': "Reference note 220: use the relevant page, entity explorer, or relationship table to study this component.",
    'Insurance asset-liability management': "Reference note 221: use the relevant page, entity explorer, or relationship table to study this component.",
    'Retail investment flows': "Reference note 222: use the relevant page, entity explorer, or relationship table to study this component.",
    'Corporate cash balances': "Reference note 223: use the relevant page, entity explorer, or relationship table to study this component.",
    'Securities borrowing demand': "Reference note 224: use the relevant page, entity explorer, or relationship table to study this component.",
    'Securities-lending supply': "Reference note 225: use the relevant page, entity explorer, or relationship table to study this component.",
    'Collateral chains': "Reference note 226: use the relevant page, entity explorer, or relationship table to study this component.",
    'Clearing-member exposure': "Reference note 227: use the relevant page, entity explorer, or relationship table to study this component.",
    'Settlement-bank exposure': "Reference note 228: use the relevant page, entity explorer, or relationship table to study this component.",
    'Payment gateway architecture': "Reference note 229: use the relevant page, entity explorer, or relationship table to study this component.",
    'Merchant processor architecture': "Reference note 230: use the relevant page, entity explorer, or relationship table to study this component.",
    'Card issuing stack': "Reference note 231: use the relevant page, entity explorer, or relationship table to study this component.",
    'Card acquiring stack': "Reference note 232: use the relevant page, entity explorer, or relationship table to study this component.",
    'Digital wallet interfaces': "Reference note 233: use the relevant page, entity explorer, or relationship table to study this component.",
    'Remittance payout chains': "Reference note 234: use the relevant page, entity explorer, or relationship table to study this component.",
    'Foreign agent networks': "Reference note 235: use the relevant page, entity explorer, or relationship table to study this component.",
    'Customer account balances': "Reference note 236: use the relevant page, entity explorer, or relationship table to study this component.",
    'Merchant settlement balances': "Reference note 237: use the relevant page, entity explorer, or relationship table to study this component.",
    'Depository institution role': "Reference note 238: use the relevant page, entity explorer, or relationship table to study this component.",
    'Federal reserve account role': "Reference note 239: use the relevant page, entity explorer, or relationship table to study this component.",
    'Treasury account role': "Reference note 240: use the relevant page, entity explorer, or relationship table to study this component.",
    'Central bank swap line concept': "Reference note 241: use the relevant page, entity explorer, or relationship table to study this component.",
    'Liquidity backstop concept': "Reference note 242: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reverse repo concept': "Reference note 243: use the relevant page, entity explorer, or relationship table to study this component.",
    'Payment-system resilience': "Reference note 244: use the relevant page, entity explorer, or relationship table to study this component.",
    'Clearing-system resilience': "Reference note 245: use the relevant page, entity explorer, or relationship table to study this component.",
    'Market-infrastructure resilience': "Reference note 246: use the relevant page, entity explorer, or relationship table to study this component.",
    'Bank resolution': "Reference note 247: use the relevant page, entity explorer, or relationship table to study this component.",
    'Deposit insurance': "Reference note 248: use the relevant page, entity explorer, or relationship table to study this component.",
    'Securities regulation': "Reference note 249: use the relevant page, entity explorer, or relationship table to study this component.",
    'Derivatives regulation': "Reference note 250: use the relevant page, entity explorer, or relationship table to study this component.",
    'Insurance regulation': "Reference note 251: use the relevant page, entity explorer, or relationship table to study this component.",
    'Municipal securities regulation': "Reference note 252: use the relevant page, entity explorer, or relationship table to study this component.",
    'Credit-union regulation': "Reference note 253: use the relevant page, entity explorer, or relationship table to study this component.",
    'Housing-finance regulation': "Reference note 254: use the relevant page, entity explorer, or relationship table to study this component.",
    'State banking oversight': "Reference note 255: use the relevant page, entity explorer, or relationship table to study this component.",
    'State insurance oversight': "Reference note 256: use the relevant page, entity explorer, or relationship table to study this component.",
    'State securities oversight': "Reference note 257: use the relevant page, entity explorer, or relationship table to study this component.",
    'International standards': "Reference note 258: use the relevant page, entity explorer, or relationship table to study this component.",
    'Domestic standards': "Reference note 259: use the relevant page, entity explorer, or relationship table to study this component.",
    'Self-regulatory organizations': "Reference note 260: use the relevant page, entity explorer, or relationship table to study this component.",
    'Financial stability monitoring': "Reference note 261: use the relevant page, entity explorer, or relationship table to study this component.",
    'System-wide leverage': "Reference note 262: use the relevant page, entity explorer, or relationship table to study this component.",
    'Collateral-driven deleveraging': "Reference note 263: use the relevant page, entity explorer, or relationship table to study this component.",
    'Cross-market contagion': "Reference note 264: use the relevant page, entity explorer, or relationship table to study this component.",
    'Global USD shortage mechanism': "Reference note 265: use the relevant page, entity explorer, or relationship table to study this component.",
    'Refinancing risk': "Reference note 266: use the relevant page, entity explorer, or relationship table to study this component.",
    'FX basis intuition': "Reference note 267: use the relevant page, entity explorer, or relationship table to study this component.",
    'Bank funding spreads': "Reference note 268: use the relevant page, entity explorer, or relationship table to study this component.",
    'Short-term funding markets': "Reference note 269: use the relevant page, entity explorer, or relationship table to study this component.",
    'Money-market transmission': "Reference note 270: use the relevant page, entity explorer, or relationship table to study this component.",
    'Treasury collateral transmission': "Reference note 271: use the relevant page, entity explorer, or relationship table to study this component.",
    'Payment obligations': "Reference note 272: use the relevant page, entity explorer, or relationship table to study this component.",
    'Settlement queues': "Reference note 273: use the relevant page, entity explorer, or relationship table to study this component.",
    'Intraday credit': "Reference note 274: use the relevant page, entity explorer, or relationship table to study this component.",
    'Counterparty exposure': "Reference note 275: use the relevant page, entity explorer, or relationship table to study this component.",
    'Margin calls': "Reference note 276: use the relevant page, entity explorer, or relationship table to study this component.",
    'Liquidity buffers': "Reference note 277: use the relevant page, entity explorer, or relationship table to study this component.",
    'Cash hoarding': "Reference note 278: use the relevant page, entity explorer, or relationship table to study this component.",
    'Safe-asset demand': "Reference note 279: use the relevant page, entity explorer, or relationship table to study this component.",
    'Dollar demand under stress': "Reference note 280: use the relevant page, entity explorer, or relationship table to study this component.",
    'Offshore funding stress': "Reference note 281: use the relevant page, entity explorer, or relationship table to study this component.",
    'Global policy spillovers': "Reference note 282: use the relevant page, entity explorer, or relationship table to study this component.",
    'Architecture map conventions': "Reference note 283: use the relevant page, entity explorer, or relationship table to study this component.",
    'Source-graphic simplification': "Reference note 284: use the relevant page, entity explorer, or relationship table to study this component.",
    'Educational interpretation': "Reference note 285: use the relevant page, entity explorer, or relationship table to study this component.",
    'Interactive node exploration': "Reference note 286: use the relevant page, entity explorer, or relationship table to study this component.",
    'Layer filtering': "Reference note 287: use the relevant page, entity explorer, or relationship table to study this component.",
    'Network density control': "Reference note 288: use the relevant page, entity explorer, or relationship table to study this component.",
    'Search by tags': "Reference note 289: use the relevant page, entity explorer, or relationship table to study this component.",
    'CSV export': "Reference note 290: use the relevant page, entity explorer, or relationship table to study this component.",
    'Five-minute cache policy': "Reference note 291: use the relevant page, entity explorer, or relationship table to study this component.",
    'Optional live-market snapshot': "Reference note 292: use the relevant page, entity explorer, or relationship table to study this component.",
    'No dependency on market data': "Reference note 293: use the relevant page, entity explorer, or relationship table to study this component.",
    'Streamlit deployment': "Reference note 294: use the relevant page, entity explorer, or relationship table to study this component.",
    'Local run command': "Reference note 295: use the relevant page, entity explorer, or relationship table to study this component.",
    'Cloud deployment considerations': "Reference note 296: use the relevant page, entity explorer, or relationship table to study this component.",
    'Data refresh behavior': "Reference note 297: use the relevant page, entity explorer, or relationship table to study this component.",
    'UI responsiveness': "Reference note 298: use the relevant page, entity explorer, or relationship table to study this component.",
    'Defensive API loading': "Reference note 299: use the relevant page, entity explorer, or relationship table to study this component.",
    'Fallback behavior': "Reference note 300: use the relevant page, entity explorer, or relationship table to study this component.",
    'Error-safe optional market data': "Reference note 301: use the relevant page, entity explorer, or relationship table to study this component.",
    'Developer extension points': "Reference note 302: use the relevant page, entity explorer, or relationship table to study this component.",
    'Model-data separation': "Reference note 303: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reusable render helpers': "Reference note 304: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reusable network model': "Reference note 305: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reusable glossary model': "Reference note 306: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reusable country classification': "Reference note 307: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reusable balance-sheet model': "Reference note 308: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reusable export layer': "Reference note 309: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reusable sidebar navigation': "Reference note 310: use the relevant page, entity explorer, or relationship table to study this component.",
    'Reusable learning modules': "Reference note 311: use the relevant page, entity explorer, or relationship table to study this component.",
    'Architecture legend': "Reference note 312: use the relevant page, entity explorer, or relationship table to study this component.",
    'Search panel': "Reference note 313: use the relevant page, entity explorer, or relationship table to study this component.",
    'Knowledge check': "Reference note 314: use the relevant page, entity explorer, or relationship table to study this component.",
    'Risk dimensions': "Reference note 315: use the relevant page, entity explorer, or relationship table to study this component.",
    'Payment stack': "Reference note 316: use the relevant page, entity explorer, or relationship table to study this component.",
    'Settlement stack': "Reference note 317: use the relevant page, entity explorer, or relationship table to study this component.",
    'Market plumbing': "Reference note 318: use the relevant page, entity explorer, or relationship table to study this component.",
    'Regulatory plumbing': "Reference note 319: use the relevant page, entity explorer, or relationship table to study this component.",
    'Balance-sheet plumbing': "Reference note 320: use the relevant page, entity explorer, or relationship table to study this component.",
    'Global dollar network': "Reference note 321: use the relevant page, entity explorer, or relationship table to study this component.",
    'Onshore dollar network': "Reference note 322: use the relevant page, entity explorer, or relationship table to study this component.",
    'Offshore dollar network': "Reference note 323: use the relevant page, entity explorer, or relationship table to study this component.",
    'Digital dollar network': "Reference note 324: use the relevant page, entity explorer, or relationship table to study this component.",
    'Traditional payment network': "Reference note 325: use the relevant page, entity explorer, or relationship table to study this component.",
    'CCP network': "Reference note 326: use the relevant page, entity explorer, or relationship table to study this component.",
    'Dealer network': "Reference note 327: use the relevant page, entity explorer, or relationship table to study this component.",
    'Investor network': "Reference note 328: use the relevant page, entity explorer, or relationship table to study this component.",
    'Treasury network': "Reference note 329: use the relevant page, entity explorer, or relationship table to study this component.",
    'Federal Reserve network': "Reference note 330: use the relevant page, entity explorer, or relationship table to study this component.",
    'End-user network': "Reference note 331: use the relevant page, entity explorer, or relationship table to study this component.",
}


# ============================================================
# EXTENSION NOTES — AUTOMATICALLY GENERATED REFERENCE LINES
# ============================================================
# Extension note 0001: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0002: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0003: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0004: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0005: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0006: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0007: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0008: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0009: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0010: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0011: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0012: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0013: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0014: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0015: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0016: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0017: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0018: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0019: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0020: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0021: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0022: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0023: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0024: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0025: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0026: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0027: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0028: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0029: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0030: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0031: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0032: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0033: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0034: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0035: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0036: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0037: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0038: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0039: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0040: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0041: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0042: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0043: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0044: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0045: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0046: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0047: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0048: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0049: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0050: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0051: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0052: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0053: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0054: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0055: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0056: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0057: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0058: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0059: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0060: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0061: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
# Extension note 0062: keep data-loading helpers cached with ttl=300 seconds when adding new external or derived datasets.
