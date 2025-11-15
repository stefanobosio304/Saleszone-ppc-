# ==========================
# INTESTAZIONE / CONFIG
# ==========================
import streamlit as st
import pandas as pd
import numpy as np
import re, csv, io, unicodedata
from io import BytesIO

# Deve essere la PRIMA chiamata Streamlit del file
st.set_page_config(
    page_title="Saleszone Ads Optimizer",
    page_icon="📈",
    layout="wide"
)

# ==========================
# MENU DI NAVIGAZIONE (unico)
# ==========================
MENU_VOCI = [
    "PPC Optimizer",
    "Brand Analytics Insights",
    "Generazione Corrispettivi",
    "Controllo Inventario FBA",
    "Funnel Audit",
]

# Usa una selectbox con chiave UNICA per evitare conflitti
menu = st.sidebar.selectbox("Naviga", MENU_VOCI, index=0, key="main_menu_v3")

# =========================================================
# PAGINA 1: PPC OPTIMIZER
# =========================================================
if menu == "PPC Optimizer":
    st.title("📊 Saleszone Ads Optimizer")
    st.write("Carica i report Amazon PPC, analizza KPI e genera suggerimenti intelligenti.")

    # === UPLOAD FILE ===
    st.subheader("📂 Carica i tuoi report")
    col1, col2 = st.columns(2)
    with col1:
        search_term_file = st.file_uploader("Carica Report Search Term (Obbligatorio)", type=["csv", "xlsx"])
    with col2:
        placement_file = st.file_uploader("Carica Report Placement (Opzionale)", type=["csv", "xlsx"])

    # === FILTRI GLOBALI ===
    acos_target = st.number_input("🎯 ACOS Target (%)", min_value=1, max_value=100, value=30)
    click_min = st.number_input("⚠️ Click minimo per Search Terms senza vendite", min_value=1, value=10)
    percent_threshold = st.number_input("📊 % Spesa per segnalazione critica", min_value=1, max_value=100, value=10)

    if search_term_file:
        # Lettura file
        if search_term_file.name.endswith(".csv"):
            df = pd.read_csv(search_term_file)
        else:
            df = pd.read_excel(search_term_file)

        df.columns = df.columns.str.strip()

        # Mapping colonne
        mapping = {
            'Nome portafoglio': 'Portfolio',
            'Nome campagna': 'Campaign',
            'Targeting': 'Keyword',
            'Termine ricerca cliente': 'Search Term',
            'Impressioni': 'Impressions',
            'Clic': 'Clicks',
            'Spesa': 'Spend',
            'Vendite totali (€) 7 giorni': 'Sales',
            'Totale ordini (#) 7 giorni': 'Orders'
        }
        df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)

        # Colonne mancanti
        required_cols = ['Impressions', 'Clicks', 'Spend', 'Sales', 'Orders']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        df.fillna(0, inplace=True)

        # KPI per riga
        df['CPC'] = df['Spend'] / df['Clicks'].replace(0, 1)
        df['CTR'] = (df['Clicks'] / df['Impressions'].replace(0, 1)) * 100
        df['CR'] = (df['Orders'] / df['Clicks'].replace(0, 1)) * 100
        df['ACOS'] = df.apply(lambda r: (r['Spend'] / r['Sales'] * 100) if r['Sales'] > 0 else None, axis=1)

        # KPI globali
        total_spend = df['Spend'].sum()
        total_sales = df['Sales'].sum()
        total_clicks = df['Clicks'].sum()
        total_impressions = df['Impressions'].sum()
        total_orders = df['Orders'].sum()

        avg_acos = (total_spend / total_sales * 100) if total_sales > 0 else 0
        ctr_global = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        cr_global = (total_orders / total_clicks * 100) if total_clicks > 0 else 0

        threshold_spesa = total_spend * (percent_threshold / 100)

        # KPI VISUAL
        st.markdown("### 📌 KPI Principali")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Spesa Totale", f"€{total_spend:,.2f}")
        col2.metric("Vendite Totali", f"€{total_sales:,.2f}")
        col3.metric("ACOS Medio", f"{avg_acos:.2f}%")
        col4.metric("CTR Totale", f"{ctr_global:.2f}%")
        col5.metric("CR Totale", f"{cr_global:.2f}%")

        # PANORAMICA PORTAFOGLI
        st.subheader("📦 Panoramica per Portafoglio")
        portfolio_group = df.groupby('Portfolio', as_index=False).agg({
            'Impressions': 'sum', 'Clicks': 'sum', 'Spend': 'sum', 'Sales': 'sum', 'Orders': 'sum'
        })
        portfolio_group['CPC'] = portfolio_group['Spend'] / portfolio_group['Clicks'].replace(0, 1)
        portfolio_group['CTR'] = (portfolio_group['Clicks'] / portfolio_group['Impressions'].replace(0, 1)) * 100
        portfolio_group['CR'] = (portfolio_group['Orders'] / portfolio_group['Clicks'].replace(0, 1)) * 100
        portfolio_group['ACOS'] = portfolio_group.apply(lambda r: (r['Spend'] / r['Sales'] * 100) if r['Sales'] > 0 else None, axis=1)
        st.dataframe(portfolio_group.style.format({
            'Spend': '€{:.2f}', 'Sales': '€{:.2f}', 'CPC': '€{:.2f}', 'CTR': '{:.2f}%', 'CR': '{:.2f}%', 'ACOS': lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        }))

        # PANORAMICA CAMPAGNE con filtro
        st.subheader("📊 Panoramica per Campagna")
        portfolio_options = ["Tutti"] + sorted(df['Portfolio'].dropna().unique().tolist())
        selected_portfolio_for_campaign = st.selectbox("Filtra per Portafoglio", portfolio_options, key="portfolio_campaign")
        df_campaign_filtered = df.copy()
        if selected_portfolio_for_campaign != "Tutti":
            df_campaign_filtered = df_campaign_filtered[df_campaign_filtered['Portfolio'] == selected_portfolio_for_campaign]

        campaign_group = df_campaign_filtered.groupby('Campaign', as_index=False).agg({
            'Impressions': 'sum', 'Clicks': 'sum', 'Spend': 'sum', 'Sales': 'sum', 'Orders': 'sum'
        })
        campaign_group['CPC'] = campaign_group['Spend'] / campaign_group['Clicks'].replace(0, 1)
        campaign_group['CTR'] = (campaign_group['Clicks'] / campaign_group['Impressions'].replace(0, 1)) * 100
        campaign_group['CR'] = (campaign_group['Orders'] / campaign_group['Clicks'].replace(0, 1)) * 100
        campaign_group['ACOS'] = campaign_group.apply(lambda r: (r['Spend'] / r['Sales'] * 100) if r['Sales'] > 0 else None, axis=1)

        st.dataframe(campaign_group.style.format({
            'Spend': '€{:.2f}', 'Sales': '€{:.2f}', 'CPC': '€{:.2f}', 'CTR': '{:.2f}%', 'CR': '{:.2f}%', 'ACOS': lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
        }))

        # DETTAGLIO SEARCH TERMS
        st.subheader("🔍 Dettaglio Search Terms per Campagna")
        portfolio_filter = st.selectbox("Seleziona Portafoglio", ["Tutti"] + sorted(df['Portfolio'].unique()), key="portfolio_terms")
        if portfolio_filter != "Tutti":
            campaign_options = df[df['Portfolio'] == portfolio_filter]['Campaign'].unique().tolist()
        else:
            campaign_options = df['Campaign'].unique().tolist()
        campaign_filter = st.selectbox("Seleziona Campagna", ["Tutte"] + sorted(campaign_options), key="campaign_terms")

        df_filtered_terms = df.copy()
        if portfolio_filter != "Tutti":
            df_filtered_terms = df_filtered_terms[df_filtered_terms['Portfolio'] == portfolio_filter]
        if campaign_filter != "Tutte":
            df_filtered_terms = df_filtered_terms[df_filtered_terms['Campaign'] == campaign_filter]

        if not df_filtered_terms.empty:
            cols_to_show = ['Search Term', 'Keyword', 'Campaign', 'Impressions', 'Clicks', 'Spend', 'Sales', 'Orders', 'CPC', 'CTR', 'CR', 'ACOS']
            st.dataframe(df_filtered_terms[cols_to_show].head(50).style.format({
                'Spend': '€{:.2f}', 'Sales': '€{:.2f}', 'CPC': '€{:.2f}', 'CTR': '{:.2f}%', 'CR': '{:.2f}%', 'ACOS': lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            }), height=500)

        # SEARCH TERMS SENZA VENDITE
        st.subheader(f"⚠️ Search Terms senza vendite (>{click_min} click)")
        portfolio_options_waste = ["Tutti"] + sorted(df['Portfolio'].unique().tolist())
        selected_portfolio_waste = st.selectbox("Filtra per Portafoglio", portfolio_options_waste, key="portfolio_waste")
        waste_terms = df[(df['Sales'] == 0) & (df['Clicks'] >= click_min)]
        if selected_portfolio_waste != "Tutti":
            waste_terms = waste_terms[waste_terms['Portfolio'] == selected_portfolio_waste]
        st.dataframe(waste_terms[['Portfolio', 'Search Term', 'Keyword', 'Campaign', 'Clicks', 'Spend']])

        # SUGGERIMENTI AI
        st.subheader("🤖 Suggerimenti AI")
        suggestions = []
        for _, row in df.groupby('Campaign', as_index=False).agg({'Spend': 'sum', 'Sales': 'sum'}).iterrows():
            if row['Sales'] == 0 and row['Spend'] >= threshold_spesa:
                suggestions.append(f"🔴 Blocca campagna **{row['Campaign']}**: spesa €{row['Spend']:.2f} senza vendite")
            elif row['Sales'] == 0 and row['Spend'] >= 5:
                suggestions.append(f"🟠 Valuta campagna **{row['Campaign']}**: spesa €{row['Spend']:.2f} senza vendite")
            elif row['Sales'] > 0 and (row['Spend'] / row['Sales'] * 100) > acos_target:
                suggestions.append(f"🟡 Ottimizza bid in **{row['Campaign']}**: ACOS {(row['Spend'] / row['Sales'] * 100):.2f}% > target {acos_target}%")
        for s in suggestions:
            st.markdown(f"- {s}")

        # TOP 3 OTTIMIZZAZIONI
        st.subheader("🔥 Cosa ottimizzare subito")
        st.markdown("**Portafogli peggiori (Top 3)**")
        pf_sorted = portfolio_group.copy()
        pf_sorted['ACOS_value'] = pf_sorted['ACOS'].fillna(9999)
        pf_sorted = pf_sorted.sort_values(by=['Sales', 'Spend'], ascending=[True, False]).head(3)
        for _, row in pf_sorted.iterrows():
            acos_display = f"{row['ACOS']:.2f}%" if pd.notna(row['ACOS']) else "N/A"
            st.markdown(f"- **{row['Portfolio']}** → Spesa: €{row['Spend']:.2f}, Vendite: €{row['Sales']:.2f}, ACOS: {acos_display}")

        st.markdown("**Campagne peggiori (Top 3)**")
        camp_sorted = campaign_group.copy()
        camp_sorted['ACOS_value'] = camp_sorted['ACOS'].fillna(9999)
        camp_sorted = camp_sorted.sort_values(by=['Sales', 'Spend'], ascending=[True, False]).head(3)
        for _, row in camp_sorted.iterrows():
            acos_display = f"{row['ACOS']:.2f}%" if pd.notna(row['ACOS']) else "N/A"
            st.markdown(f"- **{row['Campaign']}** → Spesa: €{row['Spend']:.2f}, Vendite: €{row['Sales']:.2f}, ACOS: {acos_display}")

# PAGINA 2: BRAND ANALYTICS — COMPLETA, ORDINATA, CON CRUSCOTTO OTTIMIZZATO
# =================================================================
if menu == "Brand Analytics Insights":
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.title("Brand Analytics — Semplificato (Metriche chiave)")

    brand_file = st.file_uploader(
        "Carica il file Brand Analytics (CSV/XLSX)", type=["csv", "xlsx"]
    )

    # ---------- helpers ----------
    def norm(s: str) -> str:
        """Normalizza header: minuscolo, toglie simboli, spazi->underscore."""
        return (
            str(s).strip().lower()
            .replace("%", "")
            .replace(":", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", " ")
            .replace("-", " ")
            .replace("  ", " ")
            .replace(" ", "_")
        )

    def safe_div(a, b):
        """Divisione sicura: funziona con scalari e Series"""
        if np.isscalar(a) and np.isscalar(b):
            return float(a) / float(b) if b not in [0, None, np.nan] else 0.0
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        b = b.replace({0: np.nan})
        return (a / b).fillna(0)

    @st.cache_data
    def load_df(file):
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        # alcuni export hanno una riga di header extra
        if not any(c.lower().startswith("query") for c in df.columns):
            file.seek(0)
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file, skiprows=1)
            else:
                df = pd.read_excel(file, skiprows=1)
        return df

    def build_index(df):
        return {norm(c): c for c in df.columns}

    def pick(col_index, *aliases):
        for a in aliases:
            n = norm(a)
            if n in col_index:
                return col_index[n]
        return None

    if brand_file:
        df_raw = load_df(brand_file)
        idx = build_index(df_raw)

        # ------------ risoluzione colonne necessarie ------------
        c_query     = pick(idx, "Query di ricerca", "search_query", "query_di_ricerca")
        c_volume    = pick(idx, "Volume query di ricerca", "search_query_volume", "volume_query_di_ricerca")
        c_imp_tot   = pick(idx, "Impressioni: conteggio totale", "search_funnel_impressions_total", "impressioni_conteggio_totale")
        c_imp_asin  = pick(idx, "Impressioni: numero ASIN", "impressioni_numero_asin", "impressioni_conteggio_asin")
        c_clk_tot   = pick(idx, "Clic: conteggio totale", "search_funnel_clicks_total", "clic_conteggio_totale")
        c_clk_asin  = pick(idx, "Clic: numero di ASIN", "clic_numero_asin", "clic_numero_di_asin")
        c_add_tot   = pick(idx, "Aggiunte al carrello: conteggio totale", "search_funnel_add_to_carts_total", "aggiunte_al_carrello_conteggio_totale")
        c_add_asin  = pick(idx, "Aggiunte al carrello: numero ASIN", "search_funnel_add_to_carts_brand_asin_count", "aggiunte_al_carrello_numero_asin")
        c_buy_tot   = pick(idx, "Acquisti: conteggio totale", "search_funnel_purchases_total", "acquisti_conteggio_totale")
        c_buy_asin  = pick(idx, "Acquisti: numero ASIN", "search_funnel_purchases_brand_asin_count", "acquisti_numero_asin")

        needed = {
            "Query di ricerca": c_query,
            "Volume query di ricerca": c_volume,
            "Impr tot": c_imp_tot,
            "Impr #ASIN": c_imp_asin,
            "Click tot": c_clk_tot,
            "Click #ASIN": c_clk_asin,
            "ATC tot": c_add_tot,
            "ATC #ASIN": c_add_asin,
            "Buy tot": c_buy_tot,
            "Buy #ASIN": c_buy_asin,
        }
        missing = [k for k, v in needed.items() if v is None]
        if missing:
            st.error("Mancano colonne nel file: " + ", ".join(missing))
            st.stop()

        # ------------ base + conversione numerica ------------
        base = df_raw[
            [
                c_query, c_volume, c_imp_tot, c_imp_asin,
                c_clk_tot, c_clk_asin, c_add_tot, c_add_asin,
                c_buy_tot, c_buy_asin
            ]
        ].copy()
        base.columns = [
            "Query di ricerca",
            "Volume query di ricerca",
            "Impr_tot",
            "Impr_num_ASIN",
            "Click_tot",
            "Click_num_ASIN",
            "ATC_tot",
            "ATC_num_ASIN",
            "Buy_tot",
            "Buy_num_ASIN",
        ]
        for c in base.columns:
            if c not in ["Query di ricerca", "Volume query di ricerca"]:
                base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)
        base["Volume query di ricerca"] = pd.to_numeric(base["Volume query di ricerca"], errors="coerce").fillna(0)

        # ------------ calcolo metriche richieste (per riga) ------------
        out = pd.DataFrame()
        out["Query di ricerca"] = base["Query di ricerca"]
        out["Volume query di ricerca"] = base["Volume query di ricerca"]

        out["Impression Share Asin"]       = safe_div(base["Impr_num_ASIN"], base["Impr_tot"])
        out["CTR Market"]                  = safe_div(base["Click_tot"], base["Impr_tot"])
        out["CTR Asin"]                    = safe_div(base["Click_num_ASIN"], base["Impr_num_ASIN"])
        out["Add To Cart Market"]          = safe_div(base["ATC_tot"], base["Click_tot"])
        out["Add To Cart Asin"]            = safe_div(base["ATC_num_ASIN"], base["Click_num_ASIN"])
        out["Carrelli abbandonati Market"] = safe_div(base["ATC_tot"], base["Buy_tot"])
        out["Carrelli abbandonati Asin"]   = safe_div(base["ATC_num_ASIN"], base["Buy_num_ASIN"])
        out["CR Market"]                   = safe_div(base["Buy_tot"], base["Click_tot"])
        out["CR Asin"]                     = safe_div(base["Buy_num_ASIN"], base["Click_num_ASIN"])

        # ------------ formattazione a 2 decimali ------------
        display = out.copy()
        for col in display.columns:
            if col not in ["Query di ricerca", "Volume query di ricerca"]:
                display[col] = display[col].round(2)

        st.subheader("Risultati")
        st.dataframe(display, use_container_width=True)

        st.download_button(
            "Scarica risultati (CSV)",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="brand_analytics_metriche.csv",
        )

        # ---------- Target opzionali ----------
        st.sidebar.header("Target KPI (facoltativi)")
        t_ctr = st.sidebar.number_input("Target CTR Market (%)", min_value=0.0, value=2.0, step=0.1) / 100.0
        t_atc = st.sidebar.number_input("Target Add To Cart Market (%)", min_value=0.0, value=20.0, step=0.5) / 100.0
        t_cr  = st.sidebar.number_input("Target CR Market (%)", min_value=0.0, value=8.0, step=0.5) / 100.0

        # ---------- CRUSCOTTO TOTALE ORDINATO ----------
        sum_imp_tot   = base["Impr_tot"].sum()
        sum_imp_asin  = base["Impr_num_ASIN"].sum()
        sum_clk_tot   = base["Click_tot"].sum()
        sum_clk_asin  = base["Click_num_ASIN"].sum()
        sum_atc_tot   = base["ATC_tot"].sum()
        sum_atc_asin  = base["ATC_num_ASIN"].sum()
        sum_buy_tot   = base["Buy_tot"].sum()
        sum_buy_asin  = base["Buy_num_ASIN"].sum()

        agg = {
            "Impression Share Asin":       safe_div(sum_imp_asin, sum_imp_tot),
            "CTR Market":                  safe_div(sum_clk_tot,  sum_imp_tot),
            "CTR Asin":                    safe_div(sum_clk_asin, sum_imp_asin),
            "Add To Cart Market":          safe_div(sum_atc_tot,  sum_clk_tot),
            "Add To Cart Asin":            safe_div(sum_atc_asin, sum_clk_asin),
            "Carrelli abbandonati Market": safe_div(sum_atc_tot,  sum_buy_tot),
            "Carrelli abbandonati Asin":   safe_div(sum_atc_asin, sum_buy_asin),
            "CR Market":                   safe_div(sum_buy_tot,  sum_clk_tot),
            "CR Asin":                     safe_div(sum_buy_asin, sum_clk_asin),
            "Acquisti su ATC Market":      safe_div(sum_buy_tot,  sum_atc_tot),
            "Acquisti su ATC Asin":        safe_div(sum_buy_asin, sum_atc_asin),
        }

        def pct(x): 
            try:
                return f"{float(x)*100:.2f}%"
            except Exception:
                return "0.00%"

        def delta_pp(curr, target):
            if target is None or target == 0:
                return None
            return f"{(float(curr - target)*100):.2f} pp"

        st.subheader("Cruscotto totale")

        # --- Domanda ---
        st.markdown("### Domanda")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Query analizzate", f"{len(base):,}")
        c2.metric("Volume totale", f"{int(base['Volume query di ricerca'].sum()):,}")
        c3.metric("Impression totali", f"{int(sum_imp_tot):,}")
        c4.metric("Impression Share Asin", pct(agg["Impression Share Asin"]))

        # --- Interazione (CTR) ---
        st.markdown("### Interazione · CTR")
        c5, c6 = st.columns(2)
        c5.metric("CTR Market", pct(agg["CTR Market"]), delta=delta_pp(agg["CTR Market"], t_ctr))
        c6.metric("CTR Asin", pct(agg["CTR Asin"]))

        # --- Add To Cart ---
        st.markdown("### Interazione · Add To Cart")
        c7, c8 = st.columns(2)
        c7.metric("ATC Market", pct(agg["Add To Cart Market"]), delta=delta_pp(agg["Add To Cart Market"], t_atc))
        c8.metric("ATC Asin", pct(agg["Add To Cart Asin"]))

        # --- Conversione ---
        st.markdown("### Conversione · CR")
        c9, c10, c11 = st.columns(3)
        c9.metric("Click totali", f"{int(sum_clk_tot):,}")
        c10.metric("ATC totali", f"{int(sum_atc_tot):,}")
        c11.metric("Acquisti totali", f"{int(sum_buy_tot):,}")

        c12, c13 = st.columns(2)
        c12.metric("CR Market", pct(agg["CR Market"]), delta=delta_pp(agg["CR Market"], t_cr))
        c13.metric("CR Asin", pct(agg["CR Asin"]))

        # --- Carrello / Attrito ---
        st.markdown("### Carrello · Attrito e Resa")
        c14, c15 = st.columns(2)
        c14.metric("Carrelli abbandonati Market (ATC/Ordini)", f"{float(agg['Carrelli abbandonati Market']):.2f}")
        c15.metric("Carrelli abbandonati Asin (ATC/Ordini)", f"{float(agg['Carrelli abbandonati Asin']):.2f}")

        c16, c17 = st.columns(2)
        c16.metric("Acquisti su ATC Market (Ordini/ATC)", pct(agg["Acquisti su ATC Market"]))
        c17.metric("Acquisti su ATC Asin (Ordini/ATC)", pct(agg["Acquisti su ATC Asin"]))

    else:
        st.info("Carica un file per procedere.")
# PAGINA 3: GENERAZIONE CORRISPETTIVI
# =========================================================
if menu == "Generazione Corrispettivi":
    st.title("📄 Generazione Corrispettivi Mensili")
    corrispettivi_file = st.file_uploader("Carica il report Transazioni con IVA (.csv)", type=["csv"])

    if corrispettivi_file:
        df_corr = pd.read_csv(corrispettivi_file, encoding="utf-8")
        df_corr.columns = df_corr.columns.str.strip()

        if 'TRANSACTION_TYPE' in df_corr.columns:
            df_corr = df_corr[df_corr['TRANSACTION_TYPE'].str.upper() == 'SALE']

        df_corr['TRANSACTION_COMPLETE_DATE'] = pd.to_datetime(df_corr['TRANSACTION_COMPLETE_DATE'], errors='coerce')
        df_corr = df_corr.dropna(subset=['TRANSACTION_COMPLETE_DATE'])
        df_corr = df_corr.sort_values('TRANSACTION_COMPLETE_DATE')

        df_group = df_corr.groupby(df_corr['TRANSACTION_COMPLETE_DATE'].dt.date).agg({
            'TOTAL_ACTIVITY_VALUE_AMT_VAT_EXCL': 'sum',
            'TOTAL_ACTIVITY_VALUE_VAT_AMT': 'sum',
            'TOTAL_ACTIVITY_VALUE_AMT_VAT_INCL': 'sum'
        }).reset_index()

        st.subheader("📊 Riepilogo Giornaliero")
        st.dataframe(df_group.style.format({
            'TOTAL_ACTIVITY_VALUE_AMT_VAT_EXCL': '€{:.2f}',
            'TOTAL_ACTIVITY_VALUE_VAT_AMT': '€{:.2f}',
            'TOTAL_ACTIVITY_VALUE_AMT_VAT_INCL': '€{:.2f}'
        }))

        totale_netto = df_group['TOTAL_ACTIVITY_VALUE_AMT_VAT_EXCL'].sum()
        totale_iva = df_group['TOTAL_ACTIVITY_VALUE_VAT_AMT'].sum()
        totale_lordo = df_group['TOTAL_ACTIVITY_VALUE_AMT_VAT_INCL'].sum()
        st.markdown(f"**Totale Mese:** Netto: €{totale_netto:.2f}, IVA: €{totale_iva:.2f}, Lordo: €{totale_lordo:.2f}")

        st.subheader("📋 Dettaglio Completo (Ordine Cronologico)")
        colonne_finali = [
            'TRANSACTION_COMPLETE_DATE', 'MARKETPLACE', 'SELLER_SKU', 'ITEM_DESCRIPTION', 'QTY',
            'TOTAL_ACTIVITY_VALUE_AMT_VAT_EXCL', 'TOTAL_ACTIVITY_VALUE_VAT_AMT', 'TOTAL_ACTIVITY_VALUE_AMT_VAT_INCL'
        ]
        df_dettaglio = df_corr[colonne_finali]
        st.dataframe(df_dettaglio.style.format({
            'TOTAL_ACTIVITY_VALUE_AMT_VAT_EXCL': '€{:.2f}',
            'TOTAL_ACTIVITY_VALUE_VAT_AMT': '€{:.2f}',
            'TOTAL_ACTIVITY_VALUE_AMT_VAT_INCL': '€{:.2f}'
        }), height=500)

        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_group.to_excel(writer, index=False, sheet_name="Riepilogo Giornaliero")
            df_dettaglio.to_excel(writer, index=False, sheet_name="Dettaglio Cronologico")
        st.download_button(
            "📥 Scarica Corrispettivi (Excel)",
            data=buffer.getvalue(),
            file_name="corrispettivi_mensili.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =========================================================
# PAGINA 4: CONTROLLO INVENTARIO FBA
# =========================================================

if menu == "Controllo Inventario FBA":
    import streamlit as st
    import pandas as pd
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    st.title("📦 Controllo Inventario FBA")
    st.write("Carica il report Contabilità Inventario per identificare anomalie, KPI e generare report per reclami Amazon.")

    inventory_file = st.file_uploader("Carica il report di Contabilità Inventario (CSV o XLSX)", type=["csv", "xlsx"])

    if inventory_file:
        # === LETTURA FILE ===
        if inventory_file.name.endswith(".csv"):
            df_inv = pd.read_csv(inventory_file)
        else:
            df_inv = pd.read_excel(inventory_file)

        # ✅ Normalizza nomi colonne
        df_inv.columns = (
            df_inv.columns.str.strip()
            .str.replace("\ufeff", "", regex=False)
            .str.lower()
        )

        # ✅ Converte colonne numeriche in modo sicuro
        numeric_cols = [
            'starting warehouse balance','in transit between warehouses','receipts','customer shipments',
            'customer returns','vendor returns','warehouse transfer in/out','found','lost',
            'damaged','disposed','other events','ending warehouse balance','unknown events'
        ]
        for col in numeric_cols:
            if col in df_inv.columns:
                df_inv[col] = pd.to_numeric(df_inv[col], errors='coerce').fillna(0)

        # ✅ Conversione date
        if 'date' in df_inv.columns:
            df_inv['date'] = pd.to_datetime(df_inv['date'], errors='coerce')

        # === FILTRI ===
        st.sidebar.subheader("Filtri")
        if 'date' in df_inv.columns and df_inv['date'].notna().any():
            min_date = pd.to_datetime(df_inv['date'].min()).date()
            max_date = pd.to_datetime(df_inv['date'].max()).date()
            date_range = st.sidebar.date_input("Intervallo Date", [min_date, max_date])
        else:
            date_range = []

        asin_filter = st.sidebar.text_input("Filtra per ASIN")
        fnsku_filter = st.sidebar.text_input("Filtra per FNSKU")
        title_filter = st.sidebar.text_input("Filtra per Titolo")
        location_filter = st.sidebar.text_input("Filtra per Magazzino (Location)")
        unit_cost = st.sidebar.number_input("Costo unitario stimato (€) – opzionale", min_value=0.0, value=0.0, step=0.01)

        # ⚙️ Impostazioni “Distributor Damaged durante trasferimento”
        detect_dd = st.sidebar.checkbox("Segnala 'Damaged' durante trasferimento", value=True)
        dd_window = st.sidebar.slider("Compensazioni 'found' entro (giorni)", 0, 7, 3)

        df_filtered = df_inv.copy()
        if len(date_range) == 2 and 'date' in df_filtered.columns:
            d0 = pd.to_datetime(date_range[0])
            d1 = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_filtered = df_filtered[(df_filtered['date'] >= d0) & (df_filtered['date'] <= d1)]
        if asin_filter and 'asin' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['asin'].astype(str).str.contains(asin_filter, case=False)]
        if fnsku_filter and 'fnsku' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['fnsku'].astype(str).str.contains(fnsku_filter, case=False)]
        if title_filter and 'title' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['title'].astype(str).str.contains(title_filter, case=False)]
        if location_filter and 'location' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['location'].astype(str).str.contains(location_filter, case=False)]

        # === KPI GLOBALI ===
        def safe_sum(col):
            return df_filtered[col].sum() if col in df_filtered.columns else 0

        total_starting = safe_sum('starting warehouse balance')
        total_receipts = safe_sum('receipts')
        total_shipments = safe_sum('customer shipments')
        total_returns = safe_sum('customer returns')
        total_vendor_returns = safe_sum('vendor returns')
        total_transfers = safe_sum('warehouse transfer in/out')
        total_ending = safe_sum('ending warehouse balance')

        st.subheader("📊 KPI Globali")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Starting Totale", f"{total_starting}")
        col2.metric("Ending Totale", f"{total_ending}")
        col3.metric("Differenza", f"{total_ending - total_starting}")
        col4.metric("Totale Receipts", f"{total_receipts}")

        # === Analisi per Disposition (se presente) ===
        if 'disposition' in df_filtered.columns:
            st.subheader("📌 Analisi per Disposition")
            disp_summary = (
                df_filtered.groupby('disposition')
                .agg({
                    'starting warehouse balance': 'sum',
                    'ending warehouse balance': 'sum'
                })
                .reset_index()
            )
            st.dataframe(disp_summary, use_container_width=True)

        # === Supporto calcolo ending teorico e root cause ===
        present = set(df_filtered.columns)
        inc_cols = [c for c in ['receipts','customer returns','vendor returns','warehouse transfer in/out',
                                'found','other events','unknown events'] if c in present]
        dec_cols = [c for c in ['lost','damaged','disposed'] if c in present]
        ship_col = 'customer shipments' if 'customer shipments' in present else None

        def shipments_adjust(value):
            if ship_col is None or pd.isna(value):
                return 0
            return value if value < 0 else -abs(value)

        def expected_ending_row(row):
            start = row.get('starting warehouse balance', 0) or 0
            inc = sum((row.get(c, 0) or 0) for c in inc_cols)
            dec = sum((row.get(c, 0) or 0) for c in dec_cols)
            ship = shipments_adjust(row.get(ship_col, 0) or 0)
            return start + inc - dec + ship

        # === Calcolo ending teorico e anomalie riga per riga (Tipo reclamo: Anomalia Delta) ===
        df_calc = df_filtered.copy()
        if 'ending warehouse balance' in df_calc.columns:
            df_calc['ending_teorico'] = df_calc.apply(expected_ending_row, axis=1)
            df_calc['delta'] = (df_calc['ending warehouse balance'] - df_calc['ending_teorico']).round(2)
        else:
            df_calc['ending_teorico'] = pd.NA
            df_calc['delta'] = pd.NA

        tol = 0.01
        row_anomalies = df_calc[(df_calc['delta'].abs() > tol)].copy() if 'delta' in df_calc.columns else pd.DataFrame()

        key_candidates = [k for k in ['asin','fnsku','msku','location','disposition'] if k in present]
        if not key_candidates:
            key_candidates = [k for k in ['title'] if k in present]  # fallback

        guide_rows = []
        if 'date' in df_calc.columns and 'starting warehouse balance' in df_calc.columns and 'ending warehouse balance' in df_calc.columns:
            sort_cols = [k for k in ['asin','fnsku','location','disposition'] if k in present] + ['date']
            df_calc = df_calc.sort_values(sort_cols)
            for keys, g in df_calc.groupby(key_candidates, dropna=False):
                g = g.sort_values('date').copy()
                g['prev_ending'] = g['ending warehouse balance'].shift(1)
                g['continuity_break'] = (g['starting warehouse balance'].round(2) != g['prev_ending'].round(2))
                g['ending_teorico'] = g.apply(expected_ending_row, axis=1)
                g['delta'] = (g['ending warehouse balance'] - g['ending_teorico']).round(2)

                mask = g['continuity_break'] | (g['delta'].abs() > tol)
                if mask.any():
                    r = g[mask].iloc[0]
                    anomaly_date = pd.to_datetime(r['date']).date() if pd.notna(r['date']) else None
                    window_start = pd.to_datetime(r['date']).date() - pd.Timedelta(days=2) if pd.notna(r['date']) else None
                    window_end = pd.to_datetime(r['date']).date() + pd.Timedelta(days=2) if pd.notna(r['date']) else None

                    start = r.get('starting warehouse balance', 0) or 0
                    receipts = r.get('receipts', 0) or 0
                    returns = r.get('customer returns', 0) or 0
                    vendor_returns = r.get('vendor returns', 0) or 0
                    transfers = r.get('warehouse transfer in/out', 0) or 0
                    found = r.get('found', 0) or 0
                    lost = r.get('lost', 0) or 0
                    damaged = r.get('damaged', 0) or 0
                    disposed = r.get('disposed', 0) or 0
                    other_events = r.get('other events', 0) or 0
                    unknown_events = r.get('unknown events', 0) or 0
                    shipments_raw = r.get(ship_col, 0) or 0
                    shipments_adj = shipments_adjust(shipments_raw)

                    ending_teorico = r.get('ending_teorico', start + receipts + returns + vendor_returns + transfers + found + other_events + unknown_events - lost - damaged - disposed + shipments_adj)
                    ending_reale = r.get('ending warehouse balance', 0) or 0
                    delta_val = r.get('delta', ending_reale - ending_teorico)
                    refund_units = max(round(ending_teorico - ending_reale, 2), 0)

                    event_cols = [c for c in (inc_cols + dec_cols + ([ship_col] if ship_col else [])) if c in g.columns]
                    events_day = []
                    for c in event_cols:
                        val = r.get(c, 0) or 0
                        if abs(val) > 0:
                            if c == ship_col:
                                val = shipments_adj
                            events_day.append(f"{c}: {int(val) if float(val).is_integer() else round(float(val),2)}")
                    events_str = ", ".join(events_day) if events_day else "nessun evento rilevante"

                    row = {
                        'Tipo reclamo': "Anomalia Delta",
                        'ASIN': r.get('asin', None),
                        'FNSKU': r.get('fnsku', None),
                        'MSKU': r.get('msku', None),
                        'Title': r.get('title', None),
                        'Location': r.get('location', None),
                        'Disposition': r.get('disposition', None),
                        'Data sospetta': anomaly_date,
                        'Ending teorico': ending_teorico,
                        'Ending reale': ending_reale,
                        'Delta (Ending - Teorico)': delta_val,
                        'Proposta rimborso (unità)': refund_units,
                        'Eventi nel giorno': events_str,
                        'Intervallo da controllare': f"{window_start} → {window_end}" if window_start and window_end else None,
                        'Starting': start,
                        'Receipts': receipts,
                        'Customer returns': returns,
                        'Vendor returns': vendor_returns,
                        'Warehouse transfer in/out': transfers,
                        'Found': found,
                        'Lost': lost,
                        'Damaged': damaged,
                        'Disposed': disposed,
                        'Other events': other_events,
                        'Unknown events': unknown_events,
                        'Spedizioni clienti (uscite)': shipments_adj,
                    }
                    guide_rows.append(row)

        df_guide = pd.DataFrame(guide_rows)

        # === NUOVO: “Damaged durante trasferimento” (senza delta) ===
        dd_rows = []
        if detect_dd and all(c in df_filtered.columns for c in ['date','damaged']):
            # chiavi per raggruppare
            dd_keys = [k for k in ['asin','fnsku','msku'] if k in df_filtered.columns]
            if not dd_keys:
                dd_keys = ['title'] if 'title' in df_filtered.columns else None

            if dd_keys:
                for keys, sub in df_filtered.sort_values(dd_keys + (['date'] if 'date' in df_filtered.columns else [])).groupby(dd_keys, dropna=False):
                    sub = sub.sort_values('date').copy()
                    if 'location' in sub.columns:
                        sub['loc_prev'] = sub['location'].shift(1)
                        sub['loc_change'] = sub['location'].fillna('') != sub['loc_prev'].fillna('')
                    else:
                        sub['loc_prev'] = None
                        sub['loc_change'] = False

                    if 'warehouse transfer in/out' in sub.columns:
                        sub['transfer_flag'] = sub['warehouse transfer in/out'].fillna(0).abs() != 0
                    else:
                        sub['transfer_flag'] = False

                    for idx, r in sub.iterrows():
                        damaged = float(r.get('damaged', 0) or 0)
                        if damaged <= 0:
                            continue

                        transfer_near = bool(r['transfer_flag']) or bool(r.get('loc_change', False))
                        prev_transfer = bool(sub['transfer_flag'].shift(1).fillna(False).loc[idx]) if 'transfer_flag' in sub.columns else False
                        if not (transfer_near or prev_transfer):
                            continue

                        d0 = pd.to_datetime(r['date']) if pd.notna(r.get('date')) else None
                        if d0 is None:
                            continue

                        # Compensazioni: sottraggo "found" entro dd_window giorni
                        found_after = 0
                        if 'found' in sub.columns:
                            mask = (sub['date'] > d0) & (sub['date'] <= d0 + pd.Timedelta(days=dd_window))
                            found_after = float(sub.loc[mask, 'found'].sum() or 0)

                        refund = max(damaged - found_after, 0)
                        if refund <= 0:
                            continue

                        row = {
                            'Tipo reclamo': "Damaged durante trasferimento",
                            'ASIN': r.get('asin', None),
                            'FNSKU': r.get('fnsku', None),
                            'MSKU': r.get('msku', None),
                            'Title': r.get('title', None),
                            'Da Location': r.get('loc_prev', None),
                            'A Location': r.get('location', None),
                            'Data sospetta': d0.date(),
                            'Damaged registrato': damaged,
                            'Found entro finestra': found_after,
                            'Proposta rimborso (unità)': round(refund, 2),
                            'Intervallo da controllare': f"{(d0 - pd.Timedelta(days=1)).date()} → {(d0 + pd.Timedelta(days=dd_window)).date()}",
                        }
                        dd_rows.append(row)

        df_dd = pd.DataFrame(dd_rows)

        # === Unifica i casi rimborsabili/analizzabili ===
        # (df_guide: Anomalia Delta ; df_dd: Damaged durante trasferimento)
        df_claims = pd.concat([df_guide, df_dd], ignore_index=True) if not df_dd.empty else df_guide.copy()

        # Valore economico stimato (opzionale)
        if not df_claims.empty and unit_cost > 0 and 'Proposta rimborso (unità)' in df_claims.columns:
            df_claims['Valore rimborso stimato (€)'] = (df_claims['Proposta rimborso (unità)'].fillna(0).astype(float) * unit_cost).round(2)

        # === Mostra risultati ===
        if not row_anomalies.empty or not df_claims.empty:
            st.subheader(f"📌 Anomalie rilevate (delta): {len(row_anomalies)}")
            if not row_anomalies.empty:
                st.dataframe(row_anomalies, use_container_width=True)

            st.subheader("🕵️ Reclami suggeriti (delta + damaged/transfer)")
            if not df_claims.empty:
                st.dataframe(df_claims, use_container_width=True)
            else:
                st.info("Nessun reclamo suggerito in base ai criteri selezionati.")

            # === DOWNLOAD EXCEL ===
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                if not row_anomalies.empty:
                    row_anomalies.to_excel(writer, index=False, sheet_name="Dettaglio Anomalie")
                if not df_claims.empty:
                    df_claims.to_excel(writer, index=False, sheet_name="Reclami Suggeriti")
                df_filtered.to_excel(writer, index=False, sheet_name="Inventario Filtrato")

            st.download_button(
                "📥 Scarica Report (Excel)",
                data=buffer.getvalue(),
                file_name="controllo_inventario_fba.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # === PDF: SOLO casi con Proposta rimborso > 0 ===
            def genera_pdf():
                def fnum(x):
                    try:
                        xf = float(x)
                        return str(int(xf)) if abs(xf - int(xf)) < 1e-9 else f"{xf:.2f}"
                    except Exception:
                        return str(x)

                pdf_buffer = BytesIO()
                c = canvas.Canvas(pdf_buffer, pagesize=A4)
                width, height = A4

                src = df_claims.copy()
                if 'Proposta rimborso (unità)' in src.columns:
                    src = src[src['Proposta rimborso (unità)'] > 0]
                else:
                    src = src.iloc[0:0]

                if src.empty:
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(2 * cm, height - 2 * cm, "Nessun reclamo rimborsabile (> 0 unità).")
                    c.save()
                    pdf_buffer.seek(0)
                    return pdf_buffer

                # Pagina 1: riepilogo
                c.setFont("Helvetica-Bold", 14)
                c.drawString(2 * cm, height - 2 * cm, "Report Reclami FBA (solo rimborsabili)")
                c.setFont("Helvetica", 12)
                c.drawString(2 * cm, height - 3 * cm, f"Reclami rimborsabili: {len(src)}")
                c.drawString(2 * cm, height - 4 * cm, f"Periodo: {str(date_range[0]) if len(date_range)==2 else '-'} → {str(date_range[1]) if len(date_range)==2 else '-'}")
                if 'Valore rimborso stimato (€)' in src.columns and src['Valore rimborso stimato (€)'].sum() > 0:
                    c.drawString(2 * cm, height - 5 * cm, f"Valore stimato complessivo: € {fnum(src['Valore rimborso stimato (€)'].sum())}")
                c.showPage()

                # Pagine reclamo
                for _, r in src.iterrows():
                    tipo = str(r.get('Tipo reclamo') or 'Anomalia Delta')

                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(2 * cm, height - 2 * cm, f"{tipo} – ASIN: {str(r.get('ASIN') or '')}")
                    c.setFont("Helvetica", 10)
                    c.drawString(2 * cm, height - 3 * cm, f"FNSKU: {str(r.get('FNSKU') or '')} | MSKU: {str(r.get('MSKU') or '')}")
                    title = str(r.get('Title') or '')
                    c.drawString(2 * cm, height - 4 * cm, f"Title: {title[:90]}")
                    c.drawString(2 * cm, height - 5 * cm, f"Data sospetta: {r.get('Data sospetta')}")
                    c.drawString(2 * cm, height - 6 * cm, f"Intervallo da controllare: {str(r.get('Intervallo da controllare') or '')}")

                    # Importi
                    refund_units = r.get('Proposta rimborso (unità)', 0) or 0
                    c.setFont("Helvetica-Bold", 11)
                    c.drawString(2 * cm, height - 7.5 * cm, f"Proposta rimborso: n. {fnum(refund_units)} unità")
                    if unit_cost and unit_cost > 0 and refund_units:
                        est = float(refund_units) * float(unit_cost)
                        c.drawString(2 * cm, height - 8.5 * cm, f"Valore stimato: € {fnum(est)} (costo unitario € {fnum(unit_cost)})")

                    # Motivazioni
                    y = height - 10 * cm
                    c.setFont("Helvetica", 10)
                    if tipo == "Damaged durante trasferimento":
                        da_loc = str(r.get('Da Location') or '')
                        a_loc = str(r.get('A Location') or '')
                        damaged_rec = r.get('Damaged registrato', 0) or 0
                        found_after = r.get('Found entro finestra', 0) or 0
                        c.drawString(2 * cm, y, "Motivazione:")
                        y -= 0.6 * cm
                        c.drawString(2 * cm, y, f"Unità segnate come 'damaged' contestualmente a trasferimento inter-magazzino ({da_loc} → {a_loc}).")
                        y -= 0.6 * cm
                        c.drawString(2 * cm, y, f"Damaged registrato: {fnum(damaged_rec)}  |  Found (compensazioni) entro {dd_window} gg: {fnum(found_after)}")
                        y -= 0.6 * cm
                        c.drawString(2 * cm, y, f"Richiesta rimborso = {fnum(damaged_rec)} - {fnum(found_after)} = {fnum(refund_units)} unità.")
                        y -= 0.6 * cm
                        c.drawString(2 * cm, y, "Causa imputabile a movimentazione FC durante trasferimento.")
                    else:
                        ending_teo = r.get('Ending teorico', 0) or 0
                        ending_real = r.get('Ending reale', 0) or 0
                        delta_val = r.get('Delta (Ending - Teorico)', 0) or 0
                        c.drawString(2 * cm, y, "Motivazione:")
                        y -= 0.6 * cm
                        c.drawString(2 * cm, y, f"Ending reale = {fnum(ending_real)}  vs  Ending teorico = {fnum(ending_teo)}")
                        y -= 0.6 * cm
                        c.drawString(2 * cm, y, f"Differenza (mancanti) = {fnum(max(ending_teo - ending_real, 0))}  (Delta = {fnum(delta_val)}).")
                        y -= 0.9 * cm
                        c.setFont("Helvetica-Bold", 10)
                        c.drawString(2 * cm, y, "Dettaglio formula (giorno anomalo):")
                        y -= 0.6 * cm
                        c.setFont("Helvetica", 10)
                        parts = [
                            ("Starting", r.get('Starting', 0)),
                            ("+ Receipts", r.get('Receipts', 0)),
                            ("+ Customer returns", r.get('Customer returns', 0)),
                            ("+ Vendor returns", r.get('Vendor returns', 0)),
                            ("+ Warehouse transfer in/out", r.get('Warehouse transfer in/out', 0)),
                            ("+ Found", r.get('Found', 0)),
                            ("- Lost", r.get('Lost', 0)),
                            ("- Damaged", r.get('Damaged', 0)),
                            ("- Disposed", r.get('Disposed', 0)),
                            ("+ Other events", r.get('Other events', 0)),
                            ("+ Unknown events", r.get('Unknown events', 0)),
                            ("+ Spedizioni clienti (uscite)", r.get('Spedizioni clienti (uscite)', 0)),
                        ]
                        for label, val in parts:
                            c.drawString(2 * cm, y, f"{label}: {fnum(val)}")
                            y -= 0.5 * cm
                            if y < 3 * cm:
                                c.showPage()
                                y = height - 2 * cm
                        y -= 0.2 * cm
                        c.setFont("Helvetica-Bold", 10)
                        c.drawString(2 * cm, y, f"⇒ Proposta rimborso: {fnum(refund_units)} unità")
                    c.showPage()

                c.save()
                pdf_buffer.seek(0)
                return pdf_buffer

            has_claims = (not df_claims.empty) and (df_claims.get('Proposta rimborso (unità)', 0) > 0).any()
            if has_claims:
                st.download_button(
                    "📄 Scarica PDF Reclami (solo rimborsabili)",
                    data=genera_pdf().getvalue(),
                    file_name="reclami_fba_rimborsabili.pdf",
                    mime="application/pdf"
                )
            else:
                st.info("Nessun reclamo rimborsabile (> 0 unità) nel periodo/filtri selezionati.")
        else:
            st.success("✅ Nessuna anomalia rilevata nel periodo e filtri selezionati.")


# ==========================
# FUNZIONE PAGINA: FUNNEL AUDIT (Macro → STR)
# ==========================
def render_funnel_audit():
    import re, csv, io, unicodedata
    import numpy as np
    import pandas as pd
    import streamlit as st

    # ---- titolo e intro della pagina (OK qui dentro) ----
    st.header("🧭 PPC Funnel Audit")
    st.caption("Carica un **File Macro** (campagne) e, opzionalmente, gli **Search Term Report** (SP/SB).")

    # ---------------- Utilità ----------------
    def _norm_cols(cols):
        return (pd.Series(cols).astype(str)
                .str.replace("\ufeff","", regex=False)
                .str.replace("\xa0"," ", regex=False)
                .str.replace(r"\s+"," ", regex=True)
                .str.strip())

    def _num_locale(s):
        x = pd.Series(s).astype(str)
        x = (x.str.replace("\u2212","-", regex=False)
               .str.replace("€","", regex=False)
               .str.replace("\u00a0","", regex=False)
               .str.replace(r"[^\d,.\-]","", regex=True))
        def conv(v):
            if v in ("","-") or v is None: return 0.0
            d = "," if v.rfind(",") > v.rfind(".") else "."
            if d == ",": v = v.replace(".","").replace(",",".")
            else: v = v.replace(",","")
            try: return float(v)
            except: return 0.0
        return pd.to_numeric(x.map(conv), errors="coerce").fillna(0.0)

    def _safe_str(x):
        s = "" if x is None else str(x)
        return "" if re.match(r"^\s*(nan|none|null|na)\s*$", s, flags=re.I) else s

    def _normalize_name(name: str) -> str:
        s = unicodedata.normalize("NFKD", _safe_str(name))
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = s.replace("\xa0", " ")
        return re.sub(r"\s+"," ", s).strip()

    def _read_csv_smart(file) -> pd.DataFrame:
        try:
            if hasattr(file, "seek"): file.seek(0)
            raw = file.read()
            if not raw: raise ValueError("file vuoto o già letto")
            text = None
            for enc in ("utf-8-sig","utf-8","latin-1"):
                try: text = raw.decode(enc); break
                except Exception: pass
            if text is None: text = raw.decode("latin-1", errors="ignore")
            try:
                dialect = csv.Sniffer().sniff(text[:4096], delimiters=";, \t")
                sep = dialect.delimiter
            except Exception:
                sep = ";" if text.count(";") >= text.count(",") else ","
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
            if df.shape[1] == 0: raise ValueError("nessuna colonna rilevata")
            return df
        finally:
            if hasattr(file, "seek"): file.seek(0)

    def _read_excel_smart(file) -> pd.DataFrame:
        if hasattr(file, "seek"): file.seek(0)
        df = pd.read_excel(file)
        if hasattr(file, "seek"): file.seek(0)
        return df

    def _read_any(file) -> pd.DataFrame:
        name = getattr(file, "name", "").lower()
        return _read_csv_smart(file) if name.endswith(".csv") else _read_excel_smart(file)

    # ---------------- Parsing nome campagna → layer ----------------
    SPLIT_RX = re.compile(r"\s*\|\s*|\s*[-–—]\s*|\s*>\s*|\s+to\s+|_", flags=re.I)

    def parse_campaign_name(name: str) -> dict:
        raw = _normalize_name(name)
        up_all = raw.upper()
        parts  = [p.strip() for p in SPLIT_RX.split(raw) if p and p.strip()]
        up     = [p.upper() for p in parts]
        out = {"campaign_name": raw, "ad_type":"SP", "match_type":None,
               "product_targeting":False, "is_video":False, "objective":None}

        if any(p=="SBV" or "VIDEO" in p for p in up): out["ad_type"]="SB"; out["is_video"]=True
        elif any(p=="SB" for p in up): out["ad_type"]="SB"
        elif "DISPLAY" in up_all or any(p=="SD" for p in up): out["ad_type"]="SD"
        elif any(p=="SP" for p in up): out["ad_type"]="SP"

        if re.search(r"\b(EXACT|ESATTA)\b", up_all): out["match_type"]="exact"
        elif re.search(r"\b(PHRASE|FRASE)\b", up_all): out["match_type"]="phrase"
        elif re.search(r"\b(BROAD|GENERICA|AMPIA)\b", up_all): out["match_type"]="broad"
        elif re.search(r"\bAUTO\b|_AUTO\b|-AUTO\b", up_all): out["match_type"]="auto"

        if (re.search(r"\b(PAT|CATEGORY)\b", up_all) or
            re.search(r"PRODUCT\s*TARGET(ING)?", up_all) or
            re.search(r"\bPT\s*COMPETITOR\b", up_all) or
            re.search(r"\bCOMPETITOR(S)?\b", up_all)):
            out["product_targeting"]=True

        if re.search(r"BRAND\s*(PROTECTION|DEFEN[CS]E)", up_all): out["objective"]="Brand Protection"
        elif re.search(r"(MARKET\s*)?ACQUISITION", up_all): out["objective"]="Market Acquisition"
        return out

    def layer_from_parsed(p: dict, sb_headline_as_mofu: bool) -> tuple[str,str]:
        ad = p["ad_type"]; mt = (p["match_type"] or "").lower()
        name_up = p["campaign_name"].upper()
        if ad=="SB":
            if p["is_video"] or "VIDEO" in name_up: return "MOFU","SB Video"
            if re.search(r"PRODUCT\s*COLLECTION|COLLECTION", name_up):
                return ("MOFU" if sb_headline_as_mofu else "TOFU", "SB Product Collection")
            return ("MOFU" if sb_headline_as_mofu else "TOFU", "SB Headline")
        if ad=="SP":
            if mt=="exact": return "BOFU","SP Exact"
            if p["product_targeting"]: return "BOFU","SP Product Targeting"
            if mt in ("phrase","broad","auto"): return "MOFU", f"SP {mt.capitalize()}"
            return "MOFU","SP (fallback)"
        if ad=="SD":
            if re.search(r"(VIEW|REMARKETING|RETARGET|PURCHASES)", name_up): return "BOFU","SD Remarketing"
            if re.search(r"(IN-?MARKET|AFFINITY|VCP)", name_up): return "TOFU","SD In-market/Affinity/VCP"
            if "CONTEXT" in name_up or "CONTEXTUAL" in name_up: return "MOFU","SD Contextual"
            return "MOFU","SD (fallback)"
        return "MOFU","Default"

    # =========================================================
    # STEP 1 — File Macro → mappatura + funnel
    # =========================================================
    sb_non_video_as_mofu = st.checkbox(
        "Tratta le Sponsored Brands non-video come MOFU (se disattivo → TOFU)",
        value=True, key="fa__sb_headline_toggle"
    )

    st.markdown("### Step 1 — Carica il **File Macro** (CSV/XLSX)")
    macro_file = st.file_uploader("File Macro unico", type=["csv","xlsx"], key="fa__macro_only")

    m2 = pd.DataFrame(); layer_kpi = pd.DataFrame(); names = []

    if macro_file:
        MACRO_COLS = {
            "campaign": ["Campagne","Nome campagna","Campaign Name","Campaign","Nome Campagna"],
            "impressions": ["Impressions","Impressioni","Impressioni visualizzabili"],
            "clicks": ["Clicks","Clic"],
            "spend": ["Spesa","Spend","Costo","Spesa (convertito)"],
            "sales": [
                "Vendite totali (€) 30 giorni","Vendite totali (€) 14 giorni","Vendite totali (€) 7 giorni",
                "Sales","Vendite","Vendite totali in 14 giorni – (clic)","Vendite totali in 30 giorni – (clic)"
            ],
            "orders": [
                "Totale ordini (#) 30 giorni","Totale ordini (#) 14 giorni","Totale ordini (#) 7 giorni",
                "Orders","Ordini","Ordini totali in 14 giorni (n.) – (clic)","Ordini totali in 30 giorni (n.) – (clic)"
            ],
        }
        def _pick_any(df, candidates):
            df_cols = _norm_cols(df.columns)
            low2orig = {c.lower(): c for c in df_cols}
            for k in candidates:
                kk = str(k).lower()
                if kk in low2orig: return low2orig[kk]
            return None

        df_macro = _read_any(macro_file)
        df_macro.columns = _norm_cols(df_macro.columns)

        c_name = _pick_any(df_macro, MACRO_COLS["campaign"])
        if not c_name:
            st.error("Nel File Macro non trovo la colonna *Nome campagna/Campaign*. Correggi e ricarica.")
        else:
            macro_metrics = pd.DataFrame()
            macro_metrics["campaign_name"] = df_macro[c_name].astype(str).map(_normalize_name)
            for key, store in [("impressions","impressions"),("clicks","clicks"),
                               ("spend","spend"),("sales","sales"),("orders","orders")]:
                col = _pick_any(df_macro, MACRO_COLS[key])
                macro_metrics[store] = _num_locale(df_macro[col]) if col else 0.0

            names = macro_metrics["campaign_name"].dropna().unique().tolist()
            parsed  = [parse_campaign_name(n) for n in names]
            mapping = pd.DataFrame(parsed)
            lr      = [layer_from_parsed(p, sb_non_video_as_mofu) for p in parsed]
            mapping["layer"]  = [x[0] for x in lr]
            mapping["reason"] = [x[1] for x in lr]
            mapping["match_type"] = mapping["match_type"].fillna("—")

            m2 = mapping.merge(
                macro_metrics.groupby("campaign_name", as_index=False)[["impressions","clicks","spend","sales","orders"]].sum(),
                on="campaign_name", how="left"
            ).fillna(0.0)

            st.markdown("#### Mappa campagne → livello funnel (dal nome)")
            sel = st.selectbox("Seleziona una campagna", ["—"] + names, index=0, key="fa__sel_campaign")
            if sel != "—":
                r = m2[m2["campaign_name"] == sel].head(1).iloc[0]
                a,b,c,d = st.columns(4)
                a.metric("Layer", r["layer"])
                b.metric("Tipo annuncio", r["ad_type"])
                c.metric("Match type", (r["match_type"] or "—").upper())
                d.metric("SB Video?", "Sì" if bool(r["is_video"]) else "No")
                st.caption(f"Motivo: {r['reason']} — Product targeting: {'Sì' if bool(r['product_targeting']) else 'No'}")

            st.dataframe(
                m2[["campaign_name","ad_type","match_type","product_targeting","is_video","objective","layer","reason","impressions","clicks","orders","spend","sales"]]
                  .rename(columns={"ad_type":"kind","objective":"objective_hint","product_targeting":"pt_product"}),
                use_container_width=True, height=360
            )

            st.markdown("#### 📊 Funnel (TOFU / MOFU / BOFU) — dal Macro")
            layer_order = ["TOFU","MOFU","BOFU"]
            layer_kpi = (m2.groupby("layer", dropna=False)[["impressions","clicks","spend","sales","orders"]]
                           .sum().reindex(layer_order).fillna(0.0))
            tot_sp = float(layer_kpi["spend"].sum())
            tot_sa = float(layer_kpi["sales"].sum())
            layer_kpi["budget_%"] = np.where(tot_sp>0, layer_kpi["spend"]/tot_sp*100, 0.0)
            layer_kpi["sales_%"]  = np.where(tot_sa>0, layer_kpi["sales"]/tot_sa*100, 0.0)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Budget BOFU", f"€{layer_kpi.loc['BOFU','spend']:.2f} — {layer_kpi.loc['BOFU','budget_%']:.0f}%")
            c2.metric("Budget MOFU", f"€{layer_kpi.loc['MOFU','spend']:.2f} — {layer_kpi.loc['MOFU','budget_%']:.0f}%")
            c3.metric("Budget TOFU", f"€{layer_kpi.loc['TOFU','spend']:.2f} — {layer_kpi.loc['TOFU','budget_%']:.0f}%")
            roas_all = (layer_kpi["sales"].sum()/layer_kpi["spend"].sum()) if layer_kpi["spend"].sum()>0 else np.nan
            c4.metric("ROAS complessivo", f"{roas_all:.2f}" if np.isfinite(roas_all) else "—")

            try:
                import plotly.graph_objects as go
                metric_choice = st.radio("Metrica del funnel", ["% Spesa", "% Vendite"], horizontal=True, index=0, key="fa__metric")
                vals = layer_kpi["budget_%"].values if metric_choice == "% Spesa" else layer_kpi["sales_%"].values
                text_labels = ([f"€{layer_kpi.loc[l,'spend']:.2f} — {layer_kpi.loc[l,'budget_%']:.0f}%" for l in layer_order]
                               if metric_choice=="% Spesa"
                               else [f"€{layer_kpi.loc[l,'sales']:.2f} — {layer_kpi.loc[l,'sales_%']:.0f}%" for l in layer_order])
                hover = ([f"Layer: {l}<br>Spesa: €{layer_kpi.loc[l,'spend']:.2f}<br>Quota: {layer_kpi.loc[l,'budget_%']:.0f}%" for l in layer_order]
                         if metric_choice=="% Spesa"
                         else [f"Layer: {l}<br>Vendite: €{layer_kpi.loc[l,'sales']:.2f}<br>Quota: {layer_kpi.loc[l,'sales_%']:.0f}%" for l in layer_order])
                fig = go.Figure(go.Funnel(y=layer_order, x=vals, text=text_labels, textinfo="text",
                                          hovertext=hover, hoverinfo="text"))
                fig.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
    else:
        st.info("Carica il **File Macro** per vedere mappatura e funnel.")

    st.markdown("---")

    # =========================================================
    # STEP 2 — Search Term Report (SP e SB) → idee funnel (opzionale)
    # =========================================================
    st.markdown("### Step 2 — Carica i **Search Term Report** (opzionale)")
    c1, c2 = st.columns(2)
    with c1:
        sp_files = st.file_uploader("Report **SP** (CSV/XLSX) — multipli", type=["csv","xlsx"], accept_multiple_files=True, key="fa__str_sp")
    with c2:
        sb_files = st.file_uploader("Report **SB** (CSV/XLSX) — multipli", type=["csv","xlsx"], accept_multiple_files=True, key="fa__str_sb")

    brand_hint = st.text_input("(Opzionale) Nome brand per distinguere query branded", value="", key="fa__brand_hint")
    min_clicks = st.number_input("Click minimi (filtri idee)", min_value=1, value=10, key="fa__min_clicks")
    acos_target = st.number_input("ACOS target (%)", min_value=1, max_value=100, value=30, key="fa__acos_target")

    def _pick(df, keys):
        df_cols = _norm_cols(df.columns); low2orig = {c.lower(): c for c in df_cols}
        for k in keys:
            kk = str(k).lower()
            if kk in low2orig: return low2orig[kk]
        return None

    def _load_terms(files, kind):
        if not files: return None, f"Nessun file {kind} caricato."
        COLS = {
            "campaign": ["Nome campagna","Campaign Name","Campagne"],
            "term": ["Termine ricerca cliente","Customer Search Term","Search Term"],
            "impr": ["Impressioni","Impressions","Impressioni visualizzabili"],
            "clicks": ["Clic","Clicks","Click"],
            "spend": ["Spesa","Spend","Costo","Cost"],
            "sales": ["Vendite totali (€) 7 giorni","Vendite totali (€) 14 giorni","Vendite","Sales"],
            "orders": ["Totale ordini (#) 7 giorni","Totale ordini (#) 14 giorni","Ordini","Orders"]
        }
        frames, issues = [], []
        for f in files:
            try:
                df = _read_any(f); df.columns = _norm_cols(df.columns)
                need = ["campaign","term","impr","clicks","spend","sales","orders"]
                cols = {k:_pick(df, COLS[k]) for k in need}
                if any(v is None for v in cols.values()):
                    issues.append(f"{getattr(f,'name','?')}: colonne mancanti.")
                    continue
                out = pd.DataFrame()
                out["source"] = kind
                out["campaign_name"] = df[cols["campaign"]].astype(str).map(_normalize_name)
                out["search_term"] = df[cols["term"]].astype(str)
                out["impressions"] = _num_locale(df[cols["impr"]])
                out["clicks"] = _num_locale(df[cols["clicks"]])
                out["spend"] = _num_locale(df[cols["spend"]])
                out["sales"] = _num_locale(df[cols["sales"]])
                out["orders"] = _num_locale(df[cols["orders"]])
                frames.append(out)
            except Exception as e:
                issues.append(f"{getattr(f,'name','?')}: errore lettura → {e}")
        if not frames: return None, "; ".join(issues) if issues else f"Nessun dato {kind} leggibile."
        return pd.concat(frames, ignore_index=True), "; ".join(issues)

    sp_terms, sp_issues = _load_terms(sp_files, "SP")
    sb_terms, sb_issues = _load_terms(sb_files, "SB")
    if sp_issues and sp_files: st.caption(f"Note SP: {sp_issues}")
    if sb_issues and sb_files: st.caption(f"Note SB: {sb_issues}")

    terms = None
    if (sp_terms is not None) or (sb_terms is not None):
        terms = pd.concat([x for x in [sp_terms, sb_terms] if isinstance(x, pd.DataFrame)], ignore_index=True)
        if terms.empty: terms = None

    bofu = mofu = tofu = None
    if terms is not None:
        terms["ctr"]  = np.where(terms["impressions"]>0, terms["clicks"]/terms["impressions"], 0.0)
        terms["cvr"]  = np.where(terms["clicks"]>0,      terms["orders"]/terms["clicks"], 0.0)
        terms["acos"] = np.where(terms["sales"]>0,       terms["spend"]/terms["sales"], np.nan)
        terms["roas"] = np.where(terms["spend"]>0,       terms["sales"]/terms["spend"], np.nan)

        if brand_hint.strip():
            patt = re.compile(rf"\b{re.escape(brand_hint.strip())}\b", flags=re.I)
            terms["is_branded"] = terms["search_term"].astype(str).str.contains(patt)
        else:
            terms["is_branded"] = False

        with st.expander("Anteprima STR (prime 20 righe)"):
            st.dataframe(terms.head(20), use_container_width=True)

        st.subheader("💡 Idee campagne da termini di ricerca")
        t = terms[(terms["clicks"]>=min_clicks)].copy()

        bofu = (t[(~t["is_branded"]) & (t["sales"]>0) & (t["acos"]<=acos_target/100)]
                .groupby("search_term", as_index=False)[["impressions","clicks","spend","sales","orders"]]
                .sum().sort_values(["sales","clicks"], ascending=False).head(50))
        st.markdown("**BOFU (SP Exact / PAT) — query vincenti**")
        if bofu.empty: st.info("Nessuna query BOFU con i filtri correnti.")
        else: st.dataframe(bofu.style.format({"impressions":"{:,.0f}","clicks":"{:,.0f}","orders":"{:,.0f}","spend":"€{:,.2f}","sales":"€{:,.2f}"}), use_container_width=True)

        mofu = (t[(~t["is_branded"]) & (t["ctr"]>=0.0035) & (t["orders"]>0) & (t["acos"]<=(acos_target/100)*1.5)]
                .groupby("search_term", as_index=False)[["impressions","clicks","spend","sales","orders"]]
                .sum().sort_values(["orders","clicks"], ascending=False).head(50))
        st.markdown("**MOFU (SP Phrase/Broad + SB Video/Product Collection) — query promettenti**")
        if mofu.empty: st.info("Nessuna query MOFU con i filtri correnti.")
        else: st.dataframe(mofu.style.format({"impressions":"{:,.0f}","clicks":"{:,.0f}","orders":"{:,.0f}","spend":"€{:,.2f}","sales":"€{:,.2f}"}), use_container_width=True)

        tofu = (t[(~t["is_branded"]) & (t["impressions"]>=1000) & (t["ctr"]>=0.002) & (t["orders"]<=1)]
                .groupby("search_term", as_index=False)[["impressions","clicks","spend","sales","orders"]]
                .sum().sort_values(["impressions","clicks"], ascending=False).head(50))
        st.markdown("**TOFU (SB Headline/Product Collection + SD Contextual) — query di copertura/test**")
        if tofu.empty: st.info("Nessuna query TOFU con i filtri correnti.")
        else: st.dataframe(tofu.style.format({"impressions":"{:,.0f}","clicks":"{:,.0f}","orders":"{:,.0f}","spend":"€{:,.2f}","sales":"€{:,.2f}"}), use_container_width=True)

        st.markdown("#### Raccomandazioni rapide")
        recs = []
        if (bofu is not None) and (not bofu.empty): recs.append("Apri/espandi SP Exact con le query BOFU (ToS +60–120%).")
        if (mofu is not None) and (not mofu.empty): recs.append("Per le query MOFU usa SP Phrase/Broad e SB Video/Product Collection.")
        if (tofu is not None) and (not tofu.empty): recs.append("Per le query TOFU attiva SB Headline/Product Collection e SD Contextual in test.")
        if brand_hint.strip(): recs.append("Valuta campagne Branded dedicate separando i termini che contengono il brand.")
        if recs:
            for r in recs: st.markdown(f"- {r}")
        else:
            st.info("Caricando gli STR potrai vedere suggerimenti operativi per ogni livello del funnel.")

    # ---------------------------------------------------------
    # Roadmap operativa 4 settimane
    # ---------------------------------------------------------
    st.subheader("Roadmap campagne (4 settimane)")
    roadmap = [
        ("Settimana 1", "Attiva SB Video su 1–3 SKU hero; 10–20 keyword generiche con headline orientate al beneficio.", "7–10 giorni"),
        ("Settimana 1", "SD Remarketing (Views/Purchases 14–30g); escludi acquirenti; CPC di partenza ~0,45€.", "7–10 giorni"),
        ("Settimana 2", "Promuovi le query BOFU a SP Exact; aumenta Top of Search di +60–120%; soglia minima ROAS 2,5.", "10–14 giorni"),
        ("Settimana 2", "Amplia SP Product Targeting: PAT/Category su 20–40 ASIN competitor e complementari.", "10–14 giorni"),
        ("Settimana 3", "SB Product Collection verso Brand Store; testa 3 varianti di headline e creatività.", "7–10 giorni"),
        ("Settimana 3", "SD Contextual su cluster affini; escludi segmenti con bassa viewability e alto CPC.", "7–10 giorni"),
        ("Settimana 4", "Rialloca budget: +20–40% ai rami vincenti; pausa rami con ROAS < 1,2 (ultimi 7g).", "7 giorni"),
        ("Settimana 4", "Pulizia keyword: negative exact su termini con ≥20 click e 0 ordini; -20% bid su CPC inefficiente.", "7 giorni"),
    ]
    df_roadmap = pd.DataFrame(roadmap, columns=["Quando","Cosa fare","Periodo di test consigliato"])
    st.dataframe(df_roadmap, use_container_width=True)

    # ---------------------------------------------------------
    # Download pack
    # ---------------------------------------------------------
    st.markdown("---")
    pack = io.BytesIO()
    with pd.ExcelWriter(pack, engine="openpyxl") as w:
        if not m2.empty: m2.to_excel(w, sheet_name="campaigns_mapped", index=False)
        if not layer_kpi.empty: layer_kpi.reset_index().to_excel(w, sheet_name="funnel_kpi", index=False)
        if terms is not None: terms.to_excel(w, sheet_name="search_terms_all", index=False)
        if (bofu is not None) and (not bofu.empty): bofu.to_excel(w, sheet_name="ideas_BOFU", index=False)
        if (mofu is not None) and (not mofu.empty): mofu.to_excel(w, sheet_name="ideas_MOFU", index=False)
        if (tofu is not None) and (not tofu.empty): tofu.to_excel(w, sheet_name="ideas_TOFU", index=False)
        df_roadmap.to_excel(w, sheet_name="roadmap_4w", index=False)

    st.download_button(
        "📦 Scarica pacchetto analisi (Excel)",
        data=pack.getvalue(),
        file_name="funnel_audit_pack.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="fa__dl_pack"
    )

    return None  # esplicito

# ==========================
# ROUTING DELLE PAGINE
# ==========================
if menu == "PPC Optimizer":
    st.title("📊 Saleszone Ads Optimizer")

elif menu == "Brand Analytics Insights":
    st.title("📈 Brand Analytics - Analisi Strategica")

elif menu == "Generazione Corrispettivi":
    st.title("📄 Generazione Corrispettivi Mensili")

elif menu == "Controllo Inventario FBA":
    st.title("📦 Controllo Inventario FBA")

elif menu == "Funnel Audit":
    render_funnel_audit()


