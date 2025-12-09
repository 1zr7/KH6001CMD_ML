import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import base64
import time
import os

# ====== CONFIG ======
# Theme Palette
COLOR_1 = "#003f5c"  # Deep Navy (Background Base)
COLOR_2 = "#58508d"  # Royal Purple
COLOR_3 = "#bc5090"  # Magenta
COLOR_4 = "#ff6361"  # Coral
COLOR_5 = "#ffa600"  # Gold

st.set_page_config(page_title="Gym Churn Lab", layout="wide", initial_sidebar_state="collapsed")

# ====== IMAGE LOADER ======
import base64
import os

def load_bg_image():
    path = r"c:\Users\h1zr7\OneDrive\Desktop\anti gravity\test\background_magma.jpg"
    
    if not os.path.exists(path):
        st.error(f"CRITICAL: Image file missing at {path}")
        return ""
    
    try:
        with open(path, "rb") as f:
            data = f.read()
        enc = base64.b64encode(data).decode()
        return enc
    except Exception as e:
        st.error(f"CRITICAL: Error reading image: {str(e)}")
        return ""

bin_str = load_bg_image()

if not bin_str:
    st.warning("⚠️ Background image failed to load. Using fallback color.")

# ====== STYLING ======
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">', unsafe_allow_html=True)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;800&display=swap');

/* Global Variables */
:root {{
    --c1: {COLOR_1};
    --c2: {COLOR_2};
    --c3: {COLOR_3};
    --c4: {COLOR_4};
    --c5: {COLOR_5};
    --bg-dark: #002b40;
    --text-main: #ffffff;
}}

/* Body & Background */
@media screen {{
    .stApp {{
        background-color: var(--c1);
        background-image: url("data:image/jpg;base64,{bin_str}") !important;
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        font-family: 'Outfit', sans-serif;
        color: var(--text-main);
    }}
}}
/* Overlay */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 63, 92, 0.7); /* Slightly lighter overlay */
    z-index: -1;
}}

/* Headers */
h1, h2, h3, h4, h5, h6 {{
    color: #ffffff !important;
    font-weight: 800;
    letter-spacing: -0.5px;
}}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: rgba(0, 63, 92, 0.4) !important; /* Semi-transparent */
    backdrop-filter: blur(20px) !important; /* The requested blur */
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
    transition: transform 0.2s;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color: var(--c3) !important;
    transform: translateY(-2px);
}}

/* Widget Labels */
.stRadio > label, .stSlider > label, .stToggle > label, .stSelectbox > label, .stMultiSelect > label {{
    color: var(--c5) !important;
    font-weight: 600;
    font-size: 0.95rem;
}}

/* Accent Colors for Widgets */
div[data-baseweb="slider"] div[role="slider"] {{
    background-color: var(--c4) !important;
    box-shadow: 0 0 10px var(--c4);
}}
div[data-baseweb="slider"] div[style*="width"] {{
    background-color: var(--c3) !important;
}}
.stToggle div[role="switch"][aria-checked="true"] {{
    background-color: var(--c3) !important;
}}

/* Buttons */
button[kind="primary"] {{
    background: linear-gradient(45deg, var(--c4), var(--c5)) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(255, 99, 97, 0.4);
    transition: 0.3s;
}}
button[kind="primary"]:hover {{
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(255, 166, 0, 0.5);
}}
button[kind="secondary"] {{
    background-color: rgba(255,255,255,0.1) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}}

/* Metrics */
div[data-testid="stMetricValue"] {{
    background: -webkit-linear-gradient(left, var(--c4), var(--c5));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 10px;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 10px 20px;
    color: #ccc;
    border: none;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(90deg, var(--c3), var(--c2)) !important;
    color: white !important;
    font-weight: bold;
}}

/* Custom Header Badge */
.header-tag {{
    display: inline-block;
    background: linear-gradient(90deg, var(--c2), var(--c3));
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(188, 80, 144, 0.3);
}}
</style>
""", unsafe_allow_html=True)

# ====== HELPERS ======
@st.cache_resource
def load_model():
    try:
        model = joblib.load('final_mlp_pipeline.pkl')
        return model
    except:
        return None

@st.cache_data
def load_data():
    if os.path.exists('gym_churn_us.csv'):
        return pd.read_csv('gym_churn_us.csv')
    return None

def create_pdf(prediction, probability, input_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_page()
    # PDF Theme
    pdf.set_fill_color(0, 63, 92)
    pdf.rect(0, 0, 210, 297, "F")
    
    # Header
    pdf.set_text_color(255, 166, 0) # Gold
    pdf.set_font('Arial', 'B', 28)
    pdf.cell(0, 20, "GYM CHURN LAB", ln=True, align='C')
    
    pdf.set_text_color(188, 80, 144) # Magenta
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, "Business Analytics & Prediction Report", ln=True, align='C')
    pdf.ln(10)
    
    # Result Box
    pdf.set_fill_color(0, 63, 92)
    
    # Risk Logic
    if probability < 0.4:
        r, g, b = 0, 255, 136
        status_text = "LOW RISK"
    elif probability < 0.7:
        r, g, b = 255, 166, 0
        status_text = "MEDIUM RISK"
    else:
        r, g, b = 255, 99, 97
        status_text = "HIGH RISK"

    pdf.set_draw_color(r, g, b)
    pdf.set_line_width(1.5)
    pdf.rect(20, 50, 170, 45, "DF")
    
    pdf.set_xy(20, 60)
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(r, g, b)
    pdf.cell(170, 10, status_text, ln=True, align='C')
        
    pdf.set_xy(20, 75)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', '', 14)
    pdf.cell(170, 10, f"Churn Probability: {probability:.1%}", ln=True, align='C')
    
    # Details
    pdf.ln(30)
    pdf.set_text_color(255, 166, 0) # Gold
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, "Evaluation Metrics", ln=True)
    pdf.ln(5)
    
    pdf.set_text_color(220, 220, 220)
    pdf.set_font('Arial', '', 12)
    
    col_width = 85
    y_start = pdf.get_y()
    left_margin = 20
    
    for i, (k, v) in enumerate(input_data.items()):
        col = i % 2
        row = i // 2
        x = left_margin + (col * col_width)
        y = y_start + (row * 10)
        pdf.set_xy(x, y)
        pdf.cell(col_width, 8, f"{k}: {v}")
        
    return pdf.output(dest='S').encode('latin-1')

# ====== LOADING ======
model = load_model()
df_full = load_data()

# ====== LAYOUT ======
st.markdown("<div style='text-align: center; margin-bottom: 20px;'><div class='header-tag'>aa2201459</div></div>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center;">
    <h1 style="display: inline-flex; align-items: center; gap: 15px;">
        <i class="fa-solid fa-dumbbell" style="color: #ffa600; font-size: 2.5rem;"></i>
        Gym Churn Lab
    </h1>
    <p style="color: #e0e0e0; font-size: 1.1rem; margin-top: -10px;">Predict retention. Visualize trends. Optimize revenue.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction Engine", "📊 Analytics Dashboard", "📂 Batch Processor"])

# --- TAB 1: PREDICTION ENGINE ---
with tab1:
    if not model:
        st.error("Model unavailable.")
    else:
        with st.form("engine_form"):
            c1, c2, c3 = st.columns(3)
            
            
            # Profile Column
            with c1:
                st.markdown("#### User Profile")
                with st.container(border=True):
                    gender_disp = st.radio("Gender", ["Male", "Female"], horizontal=True, label_visibility="visible")
                    age = st.slider("Age", 18, 100, 29)
                    c_inner1, c_inner2 = st.columns(2)
                    with c_inner1:
                        partner = st.toggle("Partner", value=False)
                    with c_inner2:
                        promo = st.toggle("Promo Code", value=False)
            
            # Engagement Column
            with c2:
                st.markdown("#### Engagement")
                with st.container(border=True):
                    lifetime = st.slider("Lifetime (Mo)", 0, 60, 2)
                    contract = st.select_slider("Contract (Mo)", options=[1, 6, 12], value=1)
                    group = st.toggle("Group Visits", value=True)
                    phone = st.toggle("Phone Service", value=True)
            
            # Behavior Column
            with c3:
                st.markdown("#### Behavioral Metrics")
                with st.container(border=True):
                    charges = st.slider("Extra Charges", 0.0, 500.0, 50.0)
                    freq_total = st.slider("Freq/Week (Total)", 0.0, 7.0, 1.5)
                    freq_curr = st.slider("Freq/Week (Curr)", 0.0, 7.0, 1.5)
                    month_end = st.slider("Months to End", 1.0, 12.0, 1.0)
                    near = st.toggle("Near Gym", value=True)

            st.write("")
            submit = st.form_submit_button("GENERATE PREDICTION", type="primary", use_container_width=True)

        if submit:
            # Map Inputs
            data_map = {
                'gender': 1 if gender_disp == "Male" else 0,
                'Near_Location': int(near),
                'Partner': int(partner),
                'Promo_friends': int(promo),
                'Phone': int(phone),
                'Contract_period': contract,
                'Group_visits': int(group),
                'Age': age,
                'Avg_additional_charges_total': charges,
                'Month_to_end_contract': month_end,
                'Lifetime': lifetime,
                'Avg_class_frequency_total': freq_total,
                'Avg_class_frequency_current_month': freq_curr
            }
            
            # Predict
            idf = pd.DataFrame([data_map])
            pred = model.predict(idf)[0]
            prob = model.predict_proba(idf)[0][1]

            # Result Dialog
            @st.dialog("Analysis Result")
            def show_result(p, pr, d):
                # Risk Logic
                if pr < 0.4:
                    risk_level = "LOW RISK"
                    color = "#00ff88"
                elif pr < 0.7:
                    risk_level = "MEDIUM RISK"
                    color = "#ffa600"
                else:
                    risk_level = "HIGH RISK"
                    color = "#ff6361"
                
                st.markdown(f"""
                <div style='text-align: center; padding: 30px; border-radius: 12px; background: rgba(0, 63, 92, 0.6); backdrop-filter: blur(20px); border: 2px solid {color}; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);'>
                    <h1 style='color: {color}; margin:0; font-size: 2.5rem; text-shadow: 0 0 20px {color}40;'>{risk_level}</h1>
                    <p style='color: #fff; margin-top: 10px; font-size: 1.2rem; opacity: 0.8;'>Churn Probability</p>
                    <div style='font-size: 3rem; font-weight: 800; color: #fff;'>{pr:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.write("### Actions")
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    pdf_bytes = create_pdf(p, pr, d)
                    st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="churn_report.pdf", mime="application/pdf", use_container_width=True)
                with c_d2:
                    json_str = pd.Series(d).to_json()
                    st.download_button("💾 Export JSON Data", data=json_str, file_name="churn_data.json", mime="application/json", use_container_width=True)

            show_result(pred, prob, data_map)

# --- TAB 2: ANALYTICS DASHBOARD ---
with tab2:
    if df_full is None:
        st.warning("No dataset found. Please upload a Batch in Tab 3 to visualize data.")
    else:
        st.markdown(f"### <span style='color:{COLOR_5}'>Business Intelligence Overview</span>", unsafe_allow_html=True)
        
        
        k1, k2, k3, k4 = st.columns(4)
        total_customers = len(df_full)
        churn_rate = df_full['Churn'].mean()
        avg_ltv = df_full['Lifetime'].mean()
        est_revenue = df_full['Avg_additional_charges_total'].sum() # Simple proxy
        
        k1.metric("Total Customers", f"{total_customers:,}")
        k2.metric("Churn Rate", f"{churn_rate:.1%}", delta=f"{-2.5 if churn_rate < 0.3 else 1.2}%", delta_color="inverse")
        k3.metric("Avg Lifetime", f"{avg_ltv:.1f} Months")
        k4.metric("Est. Addt Revenue", f"${est_revenue:,.0f}")
        
        st.markdown("---")
        
        st.markdown("---")
        
        r1_c1, r1_c2 = st.columns(2)
        
        with r1_c1:
            st.markdown("#### Churn Distribution")
            churn_counts = df_full['Churn'].value_counts()
            fig_donut = px.pie(
                values=churn_counts, 
                names=['Retained', 'Churned'], 
                hole=0.6,
                color_discrete_sequence=[COLOR_3, COLOR_4] # Magenta, Coral
            )
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=True
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with r1_c2:
            st.markdown("#### Churn by Contract Period")
            # Group by Contract and Churn
            # 1, 6, 12 months
            contract_churn = df_full.groupby(['Contract_period', 'Churn']).size().reset_index(name='Count')
            contract_churn['Churn'] = contract_churn['Churn'].map({0: 'Retained', 1: 'Churned'})
            
            fig_bar = px.bar(
                contract_churn, 
                x="Contract_period", 
                y="Count", 
                color="Churn",
                barmode='group',
                color_discrete_map={'Retained': COLOR_1, 'Churned': COLOR_4},
                text_auto=True
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                xaxis_title="Contract Duration (Months)",
                yaxis_title="Customer Count"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
        r2_c1, r2_c2 = st.columns(2)
        
        with r2_c1:
            st.markdown("#### Age vs Lifetime Density")
            fig_dens = px.density_heatmap(
                df_full, 
                x="Age", 
                y="Lifetime", 
                z="Churn", 
                histfunc="avg",
                color_continuous_scale=[COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]
            )
            fig_dens.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                title="Avg Churn Probability Heatmap",
            )
            st.plotly_chart(fig_dens, use_container_width=True)
            
        with r2_c2:
            st.markdown("#### Revenue Distribution")
            fig_box = px.box(
                df_full, 
                x="Churn", 
                y="Avg_additional_charges_total",
                color="Churn",
                color_discrete_sequence=[COLOR_3, COLOR_4]
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                xaxis_title="Churn Status (0=Stay, 1=Leave)",
                yaxis_title="Additional Charges ($)"
            )
            st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 3: BATCH PROCESSOR ---
with tab3:
    st.markdown("### Bulk Prediction Engine")
    uploaded = st.file_uploader("Upload Batch CSV", type="csv")
    
    if uploaded:
        bdf = pd.read_csv(uploaded)
        st.info(f"Loaded {len(bdf)} rows for processing.")
        
        if st.button("RUN BATCH ANALYSIS", type="primary"):
            with st.spinner("Processing..."):
                try:
                    preds = model.predict(bdf)
                    probs = model.predict_proba(bdf)[:, 1]
                    bdf['Churn_Prediction'] = preds
                    bdf['Churn_Confidence'] = probs
                    
                    st.success("Analysis Complete!")
                    st.dataframe(bdf.head(), use_container_width=True)
                    
                    csv = bdf.to_csv(index=False)
                    st.download_button(
                        f"Download Results ({len(bdf)} rows)", 
                        data=csv, 
                        file_name="batch_results.csv", 
                        mime="text/csv", 
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"Batch Error: {e}")
