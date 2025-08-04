import streamlit as st
import pandas as pd
from io import BytesIO

# ==========================
# CONFIGURAZIONE BASE
# ==========================
st.set_page_config(page_title="Saleszone Ads Optimizer", layout="wide")

# MENU DI NAVIGAZIONE
menu = st.sidebar.radio("Naviga", ["PPC Optimizer", "Brand Analytics Insights", "Generazione Corrispettivi"])

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

# =========================================================
# PAGINA 2: BRAND ANALYTICS
# =========================================================
if menu == "Brand Analytics Insights":
    st.title("📈 Brand Analytics - Analisi Strategica")
    brand_file = st.file_uploader("Carica il file Brand Analytics", type=["csv", "xlsx"])

    if brand_file:
        if brand_file.name.endswith(".csv"):
            df_brand = pd.read_csv(brand_file, skiprows=1)
        else:
            df_brand = pd.read_excel(brand_file, skiprows=1)

        df_brand.columns = df_brand.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(":", "")

        required_cols = [
            'query_di_ricerca', 'volume_query_di_ricerca',
            'impressioni_conteggio_totale', 'impressioni_conteggio_marchio',
            'clic_conteggio_totale', 'clic_conteggio_marchio',
            'acquisti_conteggio_totale', 'acquisti_conteggio_marchio'
        ]
        extra_cols_price = ['clic_prezzo_valore_medio', 'clic_prezzo_del_marchio_valore_medio']
        extra_cols_shipping = [
            'clic_velocità_di_spedizione_in_giornata',
            'clic_velocità_di_spedizione_1g',
            'clic_velocità_di_spedizione_2g'
        ]

        for col in required_cols[1:] + extra_cols_price + extra_cols_shipping:
            if col in df_brand.columns:
                df_brand[col] = pd.to_numeric(df_brand[col], errors='coerce').fillna(0)

        # KPI
        df_brand['impression_share'] = df_brand['impressioni_conteggio_marchio'] / df_brand['impressioni_conteggio_totale']
        df_brand['click_share'] = df_brand['clic_conteggio_marchio'] / df_brand['clic_conteggio_totale']
        df_brand['purchase_share'] = df_brand['acquisti_conteggio_marchio'] / df_brand['acquisti_conteggio_totale']
        df_brand['ctr_market'] = df_brand['clic_conteggio_totale'] / df_brand['impressioni_conteggio_totale']
        df_brand['ctr_brand'] = df_brand['clic_conteggio_marchio'] / df_brand['impressioni_conteggio_marchio']
        df_brand['cnvr_market'] = df_brand['acquisti_conteggio_totale'] / df_brand['clic_conteggio_totale']
        df_brand['cnvr_brand'] = df_brand['acquisti_conteggio_marchio'] / df_brand['clic_conteggio_marchio']

        # Azioni suggerite
        def suggerisci(row):
            actions = []
            if row['impression_share'] < 0.10: actions.append("Spingi PPC")
            if row['ctr_brand'] < row['ctr_market'] * 0.5: actions.append("Ottimizza contenuti")
            if row['cnvr_brand'] < row['cnvr_market'] * 0.5: actions.append("Migliora scheda/prezzo")
            return ", ".join(actions)
        df_brand['Azioni Suggerite'] = df_brand.apply(suggerisci, axis=1)

        # Analisi globale
        st.subheader("📊 Analisi Globale")
        st.markdown(f"""
        - **Impression Share totale:** {df_brand['impressioni_conteggio_marchio'].sum() / df_brand['impressioni_conteggio_totale'].sum():.2%}
        - **Click Share totale:** {df_brand['clic_conteggio_marchio'].sum() / df_brand['clic_conteggio_totale'].sum():.2%}
        - **Purchase Share totale:** {df_brand['acquisti_conteggio_marchio'].sum() / df_brand['acquisti_conteggio_totale'].sum():.2%}
        """)

        # Opportunità PPC
        st.subheader("✅ TOP 10 Opportunità PPC")
        opp_pcc = df_brand[(df_brand['volume_query_di_ricerca'] > 1000) & (df_brand['impression_share'] < 0.10)]
        opp_pcc = opp_pcc.sort_values(by='volume_query_di_ricerca', ascending=False).head(10)
        for _, row in opp_pcc.iterrows():
            st.markdown(f"- **{row['query_di_ricerca']}** | Impression Share {row['impression_share']:.2%}, CTR Brand {row['ctr_brand']:.2%} vs Market {row['ctr_market']:.2%}")

        # Criticità CTR
        st.subheader("⚠ Criticità CTR")
        ctr_issues = df_brand[(df_brand['ctr_brand'] < df_brand['ctr_market'] * 0.5) & (df_brand['clic_conteggio_marchio'] > 0)]
        for _, row in ctr_issues.head(10).iterrows():
            st.markdown(f"- {row['query_di_ricerca']} | CTR Brand {row['ctr_brand']:.2%} vs Market {row['ctr_market']:.2%}")

        # Analisi spedizione
        st.subheader("⚠ Analisi Velocità di Spedizione")
        total_clicks_brand = df_brand['clic_conteggio_marchio'].sum()
        if total_clicks_brand > 0:
            in_giornata = df_brand['clic_velocità_di_spedizione_in_giornata'].sum() / total_clicks_brand
            one_day = df_brand['clic_velocità_di_spedizione_1g'].sum() / total_clicks_brand
            two_day = df_brand['clic_velocità_di_spedizione_2g'].sum() / total_clicks_brand
        else:
            in_giornata = one_day = two_day = 0
        total_percent = in_giornata + one_day + two_day
        st.markdown(f"In giornata: {(in_giornata/total_percent):.2%}, 1G: {(one_day/total_percent):.2%}, 2G: {(two_day/total_percent):.2%}")

        suggestions = []
        if in_giornata < 0.05: suggestions.append("⚠ Aumenta spedizioni in giornata (Prime consigliato)")
        if one_day < 0.20: suggestions.append("⚠ Migliora disponibilità 1 giorno")
        if two_day > 0.50: suggestions.append("⚠ Troppe spedizioni lente (2 giorni)")
        if suggestions:
            for s in suggestions:
                st.markdown(f"- {s}")
        else:
            st.success("✅ Spedizione veloce ottimale")

# =========================================================
# PAGINA 3: GENERAZIONE CORRISPETTIVI
# =========================================================
if menu == "Generazione Corrispettivi":
    st.title("📄 Generazione Corrispettivi Mensili")
    corrispettivi_file = st.file_uploader("Carica il report Transazioni con IVA (.csv)", type=["csv"])

    if corrispettivi_file:
        # Lettura e pulizia
        df_corr = pd.read_csv(corrispettivi_file, encoding="utf-8")
        df_corr.columns = df_corr.columns.str.strip()

        # Filtriamo solo le vendite
        if 'TRANSACTION_TYPE' in df_corr.columns:
            df_corr = df_corr[df_corr['TRANSACTION_TYPE'].str.upper() == 'SALE']

        # Conversione data e ordinamento cronologico
        df_corr['TRANSACTION_COMPLETE_DATE'] = pd.to_datetime(df_corr['TRANSACTION_COMPLETE_DATE'], errors='coerce')
        df_corr = df_corr.dropna(subset=['TRANSACTION_COMPLETE_DATE'])
        df_corr = df_corr.sort_values('TRANSACTION_COMPLETE_DATE')

        # Riepilogo per giorno
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

        # Totali del mese
        totale_netto = df_group['TOTAL_ACTIVITY_VALUE_AMT_VAT_EXCL'].sum()
        totale_iva = df_group['TOTAL_ACTIVITY_VALUE_VAT_AMT'].sum()
        totale_lordo = df_group['TOTAL_ACTIVITY_VALUE_AMT_VAT_INCL'].sum()
        st.markdown(f"**Totale Mese:** Netto: €{totale_netto:.2f}, IVA: €{totale_iva:.2f}, Lordo: €{totale_lordo:.2f}")

        # Dettaglio cronologico
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

        # Download Excel (ordinato)
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
