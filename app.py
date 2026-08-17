import streamlit as st
import pandas as pd
import joblib
from openai import OpenAI

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="MaternalCare AI | Nigeria",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Helpers & Model Loader
# ---------------------------------------------------------
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load("maternal_model.pkl")
    except Exception:
        return None

model = load_ml_model()

def get_grok_client():
    # Retrieve key from Streamlit secrets
    api_key = st.secrets.get("XAI_API_KEY", None)
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

# ---------------------------------------------------------
# Sidebar: Navigation & Settings
# ---------------------------------------------------------
st.sidebar.title("🩺 MaternalCare AI")
st.sidebar.caption("Intelligent Maternal Health Risk Assessment")

menu = st.sidebar.radio(
    "Navigation",
    ["Patient Assessment", "Project Mission & Technical Architecture"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Localization")
selected_language = st.sidebar.selectbox(
    "Preferred Language for AI Assistant",
    ["English", "Nigerian Pidgin", "Yoruba", "Hausa", "Igbo"]
)

# ---------------------------------------------------------
# Tab 1: Patient Assessment Form
# ---------------------------------------------------------
if menu == "Patient Assessment":
    st.title("Maternal Health Risk Triage")
    st.write(
        "Enter the patient's basic vitals below to generate a real-time risk classification and clinical explanation."
    )

    with st.form("patient_form"):
        st.subheader("1. Patient Profile")
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Full Name / ID", placeholder="e.g. Amina Bello")
        with col2:
            age = st.number_input("Age (Years)", min_value=12, max_value=60, value=25)

        st.subheader("2. Physiological Vitals (Nigerian Standards)")
        col3, col4, col5 = st.columns(3)
        with col3:
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=220, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=140, value=80)
        with col4:
            bs = st.number_input("Blood Sugar (BS) [mmol/L]", min_value=3.0, max_value=30.0, value=7.0, step=0.1)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=160, value=75)
        with col5:
            body_temp_c = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)

        st.subheader("3. Clinical Red-Flag Symptoms (Doctor Checklist)")
        symptoms_col1, symptoms_col2 = st.columns(2)
        with symptoms_col1:
            has_headache = st.checkbox("Severe persistent headache or vision changes")
            has_bleeding = st.checkbox("Any vaginal bleeding or spotting")
        with symptoms_col2:
            has_swelling = st.checkbox("Sudden swelling in face, hands, or feet")
            fetal_movement_drop = st.checkbox("Noticed decrease in baby's movement")

        submitted = st.form_submit_state = st.form_submit_button("Run Risk Assessment", use_container_width=True)

    if submitted:
        if not patient_name.strip():
            st.error("Please enter the patient's name or identifier.")
        elif model is None:
            st.error("ML Model artifact (`maternal_model.pkl`) not found. Please train and place the model in the root directory.")
        else:
            # 1. Unit Conversion: Celsius -> Fahrenheit for dataset alignment
            body_temp_f = (body_temp_c * 9/5) + 32

            # 2. Machine Learning Inference
            input_df = pd.DataFrame([{
                'Age': age,
                'SystolicBP': systolic_bp,
                'DiastolicBP': diastolic_bp,
                'BS': bs,
                'BodyTemp': body_temp_f,
                'HeartRate': heart_rate
            }])

            prediction = model.predict(input_df)[0]
            
            # Prediction Probabilities if supported
            probabilities = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                classes = model.classes_
                probabilities = dict(zip(classes, probs))

            st.markdown("---")
            st.subheader("Assessment Results")

            # Visual Risk Badge
            if prediction.lower() == "high risk":
                st.error(f"### Predicted Risk Level: HIGH RISK")
            elif prediction.lower() == "mid risk":
                st.warning(f"### Predicted Risk Level: MID RISK")
            else:
                st.success(f"### Predicted Risk Level: LOW RISK")

            if probabilities:
                st.write("**Confidence Breakdown:**")
                prob_cols = st.columns(len(probabilities))
                for idx, (cls, prob) in enumerate(probabilities.items()):
                    prob_cols[idx].metric(label=f"{cls.title()}", value=f"{prob * 100:.1f}%")

            # 3. Grok AI Generation Layer
            st.subheader(f"Digital Health Assistant Note ({selected_language})")
            
            red_flags = []
            if has_headache: red_flags.append("Severe headache/blurred vision")
            if has_bleeding: red_flags.append("Vaginal bleeding")
            if has_swelling: red_flags.append("Sudden swelling")
            if fetal_movement_drop: red_flags.append("Decreased fetal movement")

            prompt = f"""
You are an empathetic, professional maternal health AI assistant operating in Nigeria.
Explain the following triage assessment directly to the patient ({patient_name}).

Patient Vitals:
- Age: {age}
- Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg
- Blood Glucose: {bs} mmol/L
- Temperature: {body_temp_c} °C
- Heart Rate: {heart_rate} bpm
- Additional Reported Symptoms: {', '.join(red_flags) if red_flags else 'None reported'}

Machine Learning Risk Output: {prediction.upper()}

Instructions:
1. Speak in a respectful, warm, reassuring, and clear tone.
2. Communicate in {selected_language}.
3. Explain what the vitals and risk level mean in simple language without medical jargon.
4. Highlight any vital signs or reported symptoms that need close attention.
5. Emphasize clearly that you are an AI assistant and that this is not a final medical diagnosis.
6. Provide concrete, actionable next steps (e.g., visit the nearest primary healthcare centre or antenatal clinic).
"""
            client = get_grok_client()
            if client:
                with st.spinner("Generating AI explanation via Grok..."):
                    try:
                        response = client.chat.completions.create(
                            model="grok-2-mini",
                            messages=[
                                {"role": "system", "content": "You are a specialized maternal health AI assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_tokens=600
                        )
                        st.info(response.choices[0].message.content)
                    except Exception as e:
                        st.warning(f"Could not connect to Grok API: {e}")
            else:
                st.info(
                    "*(Grok API key not configured in `.streamlit/secrets.toml`. ML inference completed successfully above.)*"
                )

# ---------------------------------------------------------
# Tab 2: Project Mission & Technical Documentation
# ---------------------------------------------------------
elif menu == "Project Mission & Technical Architecture":
    st.title("Project Overview: Combating Maternal Mortality in Nigeria")
    
    st.markdown("""
    ### The Challenge
    Nigeria accounts for a significant portion of global maternal deaths. Delays in identifying complications—such as gestational hypertension, preeclampsia, and gestational diabetes—often occur due to limited access to specialized triage in rural and underserved primary health centres.

    ### System Architecture
    This application integrates predictive machine learning with conversational Generative AI:
    
    1. **Predictive Core (Scikit-Learn Ensemble):** 
       * Evaluates 6 vital signs: Age, Systolic/Diastolic BP, Blood Sugar (`mmol/L`), Body Temperature, and Heart Rate.
       * Generates probabilistic triage classification (*Low*, *Mid*, or *High Risk*).
    
    2. **Generative AI Layer (xAI Grok API):**
       * Ingests both tabular predictions and qualitative symptom markers.
       * Synthesizes clinical metrics into culturally accessible, empathetic feedback in English, Pidgin, Yoruba, Hausa, or Igbo.
    
    3. **Data Protection & Privacy:**
       * Operates on a stateless session model—patient identifiable information is not stored in a persistent database.
    """)