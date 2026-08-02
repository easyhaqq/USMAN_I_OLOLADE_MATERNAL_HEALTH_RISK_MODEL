import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb
import os
from google import genai

st.set_page_config(
    page_title="Maternal Triage System | MedAI", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS Injection ---
st.markdown("""
<style>
    /* Soften the main background color */
    .stApp {
        background-color: #F8FAFC; 
    }
    
    /* Make headers a professional dark slate */
    h1, h2, h3 {
        color: #0F172A;
    }
    
    /* Style the Form and Data Containers as floating white cards */
    [data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background-color: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Style the st.metric items (Patient Vitals) to look like hospital monitors */
    [data-testid="stMetric"] {
        background-color: #F1F5F9;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    /* Add a smooth hover effect to the Primary Button */
    [data-testid="baseButton-primary"] {
        background-color: #3B82F6;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    [data-testid="baseButton-primary"]:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
</style>
""", unsafe_allow_html=True)

API_KEY = st.secrets["GEMINI_API_KEY"] 
client = genai.Client(api_key=API_KEY)

# --- 2. Dynamic Prompt Engine ---
RISK_GUIDANCE = {
    "low": {
        "urgency": "Reassure her that these readings fall within a healthy range and no urgent action is needed.",
        "next_step_tone": "routine, preventive",
        "closing": "Please share these readings with your provider at your next scheduled visit.",
    },
    "mid": {
        "urgency": "Let her know some readings need closer monitoring, and follow-up isn't optional even though this isn't an emergency.",
        "next_step_tone": "monitoring and timely follow-up",
        "closing": "Please contact your provider within the next few days to review these readings.",
    },
    "high": {
        "urgency": "Be direct that these readings require prompt medical attention. Do not soften this to avoid alarming her.",
        "next_step_tone": "urgent",
        "closing": "Please contact your provider today, or go to the nearest emergency care if you feel unwell before then.",
    },
}

def _get_risk_guidance(risk_level: str) -> dict:
    key = risk_level.strip().lower()
    for tier in ("high", "mid", "low"):  # check most urgent tier first
        if tier in key:
            return RISK_GUIDANCE[tier]
    return RISK_GUIDANCE["mid"]  # unrecognized label: fail toward caution, not complacency

def build_consultation_prompt(display_name, systolic, diastolic, bs_mgdl, temp, heart_rate, risk_level):
    guidance = _get_risk_guidance(risk_level)

    prompt = f"""You are an AI maternal-health assistant. Write in the voice of an experienced, warm OB/GYN talking to a patient: plain language, no jargon, calm and direct. You are an AI, not a licensed physician, and must never imply otherwise.

Patient: {display_name}
Vitals: BP {systolic}/{diastolic} mmHg, Blood glucose {bs_mgdl} mg/dL, Temp {temp}°F, Heart rate {heart_rate} BPM
Triage risk level (already determined upstream — report it, do not recalculate, soften, or contradict it): {risk_level.upper()}

Write a well explained, empathetic, and actionable consultation note for {display_name}. Include the following sections:
[addressing {display_name} by name. Reference her specific numbers in plain language. Explain what her risk level means for her, practically. {guidance['urgency']} Do not name a specific diagnosis (e.g. preeclampsia, gestational diabetes) — a single reading can't establish one.]

**Recommendation:**
Explain what she should do next, in plain language. Include a clear recommendation for follow-up with her healthcare provider. Emphasize that she should not ignore these readings, and that {guidance['next_step_tone']} care is important.

**Please Note:**
*As an AI health assistant, I'm not a substitute for medical care. {guidance['closing']}*

Rules:
- Base everything only on the vitals and risk level above — don't invent symptoms, history, or reference ranges.
- Never adopt a name, title, or years-of-experience for yourself.
- Calm delivery is not the same as minimizing — for mid/high risk, clarity about urgency comes first."""
    return prompt

# --- 3. Load ML Assets (Cached for Performance) ---
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    
    if os.path.exists('maternal_health_model.json'):
        ml_model = xgb.XGBClassifier()
        ml_model.load_model('maternal_health_model.json')
    else:
        ml_model = joblib.load('maternal_health_model.pkl')
    return scaler, le, ml_model

scaler, le, ml_model = load_models()

# --- 4. UI Layout: Header & Clinical Disclaimer ---
st.title("🏥 Usman Maternal Health Clinical Triage")
st.markdown("---")

with st.expander("⚠️ Clinical AI Disclaimer (Please Read)"):
    st.write("""
        *This system is an experimental digital health tool powered by machine learning and generative AI. 
        It is designed to assist in triaging physiological vitals but does **not** replace professional medical advice, diagnosis, or treatment. 
        Always consult with a qualified healthcare provider for medical decisions.*
    """)

# --- 5. Input Form: Patient Vitals ---
st.markdown("### Patient Intake Form")
with st.form("vitals_form", border=True):
    
    patient_name = st.text_input("Patient Full Name", placeholder="e.g., Jane Doe")
    
    st.markdown("#### Physiological Vitals")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (Years)", min_value=10, max_value=100, value=25)
        # Updated to mg/dL with appropriate limits and default value
        bs_mgdl = st.number_input("Random Blood Sugar (mg/dL)", min_value=0, max_value=500, value=126, step=1)
    with col2:
        systolic = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=200, value=120)
        diastolic = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
    with col3:
        temp = st.number_input("Body Temp (°F)", min_value=90.0, max_value=110.0, value=98.6, step=0.1)
        heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=150, value=75)
    
    submitted = st.form_submit_button("Run Clinical Analysis", type="primary", use_container_width=True)

# --- 6. Processing & Output Pipeline ---
if submitted:
    display_name = patient_name.strip() if patient_name.strip() else "our patient"
    
    # CONVERSION: Convert mg/dL back to mmol/L for the ML Model
    bs_mmol = round(bs_mgdl / 18.0, 1) 
    
    # Pass the converted bs_mmol to the ML model
    input_df = pd.DataFrame([[age, systolic, diastolic, bs_mmol, temp, heart_rate]], 
                            columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
    
    input_scaled = scaler.transform(input_df)
    pred_encoded = ml_model.predict(input_scaled)
    risk_level = le.inverse_transform(pred_encoded)[0].lower()
    
    st.markdown("---")
    st.markdown("### 📊 Triage Results & Consultation")
    
    metric_cols = st.columns(5)
    metric_cols[0].metric("Blood Pressure", f"{systolic}/{diastolic}")
    metric_cols[1].metric("Blood Sugar", f"{bs_mgdl} mg/dL")  # Display the local mg/dL metric
    metric_cols[2].metric("Body Temp", f"{temp} °F")
    metric_cols[3].metric("Heart Rate", f"{heart_rate} BPM")
    metric_cols[4].metric("Age", f"{age} Yrs")
    
    if "high" in risk_level:
        st.error(f"**Automated Triage Assessment: {risk_level.upper()}**")
    elif "mid" in risk_level:
        st.warning(f"**Automated Triage Assessment: {risk_level.title()}**")
    else:
        st.success(f"**Automated Triage Assessment: {risk_level.title()}**")
    
    with st.spinner("Consulting Clinical AI..."):
        
        # Call your dynamic prompt builder function
        prompt = build_consultation_prompt(
            display_name=display_name,
            systolic=systolic,
            diastolic=diastolic,
            bs_mgdl=bs_mgdl,  # AI gets the mg/dL value
            temp=temp,
            heart_rate=heart_rate,
            risk_level=risk_level
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt
            )
            
            with st.container(border=True):
                st.markdown("#### 🩺 Physician's Notes")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Clinical AI Connection Error: {e}")