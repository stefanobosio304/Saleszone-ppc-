import pandas as pd
import streamlit as st

# === CONFIGURAZIONE PAGINA ===
st.set_page_config(page_title="Amazon PPC Analyzer", layout="wide")

st.title("📊 Amazon PPC Analyzer + AI Suggerimenti")
st.write("Analizza KPI e ottieni 3 suggerimenti chiave per ottimizzare le tue campagne.")

# === FILTRI GLOBALI ===
acos_target = st.number_input("🎯 ACOS Target (%)", min_value=1, max_value=100, value=30)
click_min = st.number_input("⚠️ Click minimo per Search Terms senza vendite", min_value=1, value=10)

# === UPLOAD FILE ===
uploaded_file = st.file_uploader("Carica il file CSV o Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Legge file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Mapping colonne italiane → inglesi
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
    df.rename(columns=mapping, inplace=True)
    df.fillna(0, inplace=True)

    # KPI base
    df['CPC'] = df['Spend'] / df['Clicks'].replace(0, 1)
    df['CTR'] = (df['Clicks'] / df['Impressions'].replace(0, 1)) * 100
    df['CR'] = (df['Orders'] / df['Clicks'].replace(0, 1)) * 100
    df['ACOS'] = df.apply(lambda row: (row['Spend'] / row['Sales'] * 100) if row['Sales'] > 0 else None, axis=1)
    df['ROAS'] = df.apply(lambda row: (row['Sales'] / row['Spend']) if row['Spend'] > 0 else 0, axis=1)

    # === FUNZIONI DI FORMATTAZIONE ===
    def highlight_acos(val):
        if pd.isna(val):
            return ''
        try:
            val = float(val)
            if val > 100:
                return 'background-color: #ffcccc; color: red; font-weight: bold'
            elif val > 50:
                return 'background-color: #fff5cc'
        except:
            return ''
        return ''

    def format_acos(x):
        return "N/A" if pd.isna(x) else f"{x:.2f}%"

    def format_roas(x):
        return f"{x:.2f}"

    # === PANORAMICA PER PORTAFOGLIO ===
    st.subheader("📦 Panoramica per Portafoglio")
    portfolio_group = df.groupby('Portfolio', as_index=False).agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Spend': 'sum',
        'Sales': 'sum',
        'Orders': 'sum'
    })
    portfolio_group['CPC'] = portfolio_group['Spend'] / portfolio_group['Clicks'].replace(0, 1)
    portfolio_group['CTR'] = (portfolio_group['Clicks'] / portfolio_group['Impressions'].replace(0, 1)) * 100
    portfolio_group['CR'] = (portfolio_group['Orders'] / portfolio_group['Clicks'].replace(0, 1)) * 100
    portfolio_group['ACOS'] = portfolio_group.apply(lambda row: (row['Spend'] / row['Sales'] * 100) if row['Sales'] > 0 else None, axis=1)
    portfolio_group['ROAS'] = portfolio_group.apply(lambda row: (row['Sales'] / row['Spend']) if row['Spend'] > 0 else 0, axis=1)
    portfolio_group = portfolio_group.round({'Spend': 2, 'Sales': 2, 'CPC': 2, 'CTR': 2, 'CR': 2, 'ROAS': 2})
    st.dataframe(portfolio_group.style.format({
        'Spend': '€{:.2f}', 'Sales': '€{:.2f}', 'CPC': '€{:.2f}',
        'CTR': '{:.2f}%', 'CR': '{:.2f}%', 'ACOS': format_acos, 'ROAS': format_roas
    }).applymap(highlight_acos, subset=['ACOS']), use_container_width=True)

    # === PANORAMICA PER CAMPAGNA ===
    st.subheader("📊 Panoramica per Campagna")
    campaign_group = df.groupby('Campaign', as_index=False).agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Spend': 'sum',
        'Sales': 'sum',
        'Orders': 'sum'
    })
    campaign_group['CPC'] = campaign_group['Spend'] / campaign_group['Clicks'].replace(0, 1)
    campaign_group['CTR'] = (campaign_group['Clicks'] / campaign_group['Impressions'].replace(0, 1)) * 100
    campaign_group['CR'] = (campaign_group['Orders'] / campaign_group['Clicks'].replace(0, 1)) * 100
    campaign_group['ACOS'] = campaign_group.apply(lambda row: (row['Spend'] / row['Sales'] * 100) if row['Sales'] > 0 else None, axis=1)
    campaign_group['ROAS'] = campaign_group.apply(lambda row: (row['Sales'] / row['Spend']) if row['Spend'] > 0 else 0, axis=1)
    campaign_group = campaign_group.round({'Spend': 2, 'Sales': 2, 'CPC': 2, 'CTR': 2, 'CR': 2, 'ROAS': 2})
    st.dataframe(campaign_group.style.format({
        'Spend': '€{:.2f}', 'Sales': '€{:.2f}', 'CPC': '€{:.2f}',
        'CTR': '{:.2f}%', 'CR': '{:.2f}%', 'ACOS': format_acos, 'ROAS': format_roas
    }).applymap(highlight_acos, subset=['ACOS']), use_container_width=True)

    # === TOP 3 PORTAFOGLI CRITICI ===
    st.subheader("🔥 I 3 Portafogli da Ottimizzare Subito")
    portfolio_group['Problema'] = portfolio_group.apply(
        lambda row: "⚠️ Spesa senza vendite" if row['Sales'] == 0 and row['Spend'] > 0 else
        ("Riduci bid (ACOS > 100%)" if row['ACOS'] and row['ACOS'] > 100 else
         ("Ottimizza targeting (ACOS > target)" if row['ACOS'] and row['ACOS'] > acos_target else
          "Buona performance")), axis=1)
    critical_portfolios = portfolio_group.sort_values(by=['Sales', 'ACOS', 'Spend'], ascending=[True, False, False]).head(3)
    st.dataframe(critical_portfolios[['Portfolio', 'Spend', 'Sales', 'ACOS', 'CTR', 'Problema']].style.format({
        'Spend': '€{:.2f}', 'Sales': '€{:.2f}', 'ACOS': format_acos, 'CTR': '{:.2f}%'
    }).applymap(highlight_acos, subset=['ACOS']), use_container_width=True)

    # === TOP 3 CAMPAGNE CRITICHE ===
    st.subheader("🚨 Le 3 Campagne da Ottimizzare Subito")
    campaign_group['Problema'] = campaign_group.apply(
        lambda row: "⚠️ Spesa senza vendite" if row['Sales'] == 0 and row['Spend'] > 0 else
        ("Riduci bid (ACOS > 100%)" if row['ACOS'] and row['ACOS'] > 100 else
         ("Ottimizza targeting (ACOS > target)" if row['ACOS'] and row['ACOS'] > acos_target else
          "Buona performance")), axis=1)
    critical_campaigns = campaign_group.sort_values(by=['Sales', 'ACOS', 'Spend'], ascending=[True, False, False]).head(3)
    st.dataframe(critical_campaigns[['Campaign', 'Spend', 'Sales', 'ACOS', 'CTR', 'Problema']].style.format({
        'Spend': '€{:.2f}', 'Sales': '€{:.2f}', 'ACOS': format_acos, 'CTR': '{:.2f}%'
    }).applymap(highlight_acos, subset=['ACOS']), use_container_width=True)

    # === SEARCH TERMS SENZA VENDITE FILTRATI PER PORTAFOGLIO ===
    st.subheader(f"⚠️ Search Terms senza vendite (almeno {click_min} click)")
    portfolios = df['Portfolio'].unique()
    selected_portfolio = st.selectbox("Seleziona il Portafoglio", ["Tutti"] + list(portfolios))

    if selected_portfolio != "Tutti":
        waste_terms = df[(df['Portfolio'] == selected_portfolio) & (df['Sales'] == 0) & (df['Clicks'] >= click_min)]
    else:
        waste_terms = df[(df['Sales'] == 0) & (df['Clicks'] >= click_min)]

    if waste_terms.empty:
        st.write("✅ Nessun search term con click elevati e 0 vendite per il filtro selezionato.")
    else:
        st.dataframe(waste_terms[['Search Term', 'Keyword', 'Campaign', 'Clicks', 'Spend', 'ACOS']].style.format({
            'Spend': '€{:.2f}', 'ACOS': format_acos
        }).applymap(highlight_acos, subset=['ACOS']), use_container_width=True)

        # === DOWNLOAD CSV ===
        csv = waste_terms[['Search Term', 'Keyword', 'Campaign', 'Clicks', 'Spend']].to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Scarica CSV Search Terms Negativi", data=csv, file_name="search_terms_negativi.csv", mime="text/csv")

    # === DETTAGLIO SEARCH TERMS ===
    st.subheader("🔍 Dettagli Search Terms")
    selected_keyword = st.selectbox("Seleziona una Keyword per vedere i Search Terms", df['Keyword'].unique())
    search_terms = df[df['Keyword'] == selected_keyword][[
        'Search Term', 'Campaign', 'Impressions', 'Clicks', 'Spend', 'Sales', 'Orders', 'CPC', 'CTR', 'CR', 'ACOS', 'ROAS'
    ]]
    st.write(f"Search Terms per Keyword: **{selected_keyword}**")
    st.dataframe(search_terms.style.format({
        'Spend': '€{:.2f}', 'Sales': '€{:.2f}', 'CPC': '€{:.2f}',
        'CTR': '{:.2f}%', 'CR': '{:.2f}%', 'ACOS': format_acos, 'ROAS': format_roas
    }).applymap(highlight_acos, subset=['ACOS']), use_container_width=True)

    # === AI SUGGERIMENTI (3 PRIORITARI) ===
    st.subheader("🤖 AI: 3 Suggerimenti Chiave")
    suggestions = []

    # Regole: spesa senza vendite > ACOS alto > CTR basso
    for _, row in campaign_group.iterrows():
        if len(suggestions) >= 3:
            break
        if row['Sales'] == 0 and row['Spend'] > 5:
            suggestions.append(f"Blocca la campagna **{row['Campaign']}**: spesa {row['Spend']}€ senza vendite.")
        elif row['ACOS'] and row['ACOS'] > 100:
            suggestions.append(f"Riduci drasticamente il bid in **{row['Campaign']}** (ACOS {row['ACOS']:.2f}%).")
        elif row['ACOS'] and row['ACOS'] > acos_target:
            suggestions.append(f"Ottimizza targeting per **{row['Campaign']}**: ACOS {row['ACOS']:.2f}% > target {acos_target}%.")

    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.write("✅ Nessun suggerimento critico: performance sotto controllo.")
