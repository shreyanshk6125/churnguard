
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ChurnGuard",
    page_icon="🛡️",
    layout="wide"
)


# ==================================================
# 2. LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("models/churnguard_model.joblib")


model_bundle = load_model()

pipeline = model_bundle["pipeline"]
threshold = model_bundle["threshold"]


# ==================================================
# 3. LOAD CUSTOMER DATASET
# ==================================================

@st.cache_data
def load_customer_data():

    df = pd.read_csv(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df


customer_data = load_customer_data()


# ==================================================
# 4. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

h1 {
    color: #142D4E;
    font-weight: 800;
}

h2, h3 {
    color: #203B5C;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}

.stFormSubmitButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stFormSubmitButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🛡️ ChurnGuard")

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Predict Churn"]
)

st.sidebar.divider()

st.sidebar.caption(
    "Customer Retention Intelligence Platform"
)


# ==================================================
# 6. MAIN HEADER
# ==================================================

st.title("🛡️ ChurnGuard")

st.markdown(
    "### Customer Retention Intelligence Platform"
)

st.markdown(
    "Predict customer churn using machine learning "
    "and identify customers who may need attention."
)

st.divider()


# ==================================================
# 7. OVERVIEW PAGE
# ==================================================

if page == "Overview":

    # --------------------------------------------------
    # BUSINESS OVERVIEW - KPI CARDS
    # --------------------------------------------------

    total_customers = len(customer_data)

    churned_customers = (
        customer_data["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    avg_monthly_charges = (
        customer_data["MonthlyCharges"].mean()
    )

    st.subheader("Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}"
        )

    with col2:
        st.metric(
            label="Churned Customers",
            value=f"{churned_customers:,}"
        )

    with col3:
        st.metric(
            label="Churn Rate",
            value=f"{churn_rate:.2f}%"
        )

    with col4:
        st.metric(
            label="Avg. Monthly Charges",
            value=f"${avg_monthly_charges:.2f}"
        )

    st.divider()

    # --------------------------------------------------
    # CUSTOMER CHURN ANALYTICS
    # --------------------------------------------------

    st.subheader("Customer Churn Analytics")

    # Chart 1: Churn Distribution

    churn_counts = (
        customer_data["Churn"]
        .value_counts()
        .reset_index()
    )

    churn_counts.columns = [
        "Churn",
        "Customers"
    ]

    fig_churn = px.pie(
        churn_counts,
        names="Churn",
        values="Customers",
        hole=0.55,
        title="Customer Churn Distribution",
        color="Churn",
        color_discrete_map={
            "Yes": "#EF4444",
            "No": "#2563EB"
        }
    )

    fig_churn.update_traces(
        textinfo="percent+label"
    )

    # Chart 2: Churn by Contract

    contract_churn = (
        customer_data.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )

    fig_contract = px.bar(
        contract_churn,
        x="Contract",
        y="Churn Rate",
        title="Churn Rate by Contract Type",
        color="Churn Rate",
        color_continuous_scale="Blues",
        text_auto=".1f"
    )

    fig_contract.update_layout(
        yaxis_title="Churn Rate (%)",
        xaxis_title="Contract Type",
        coloraxis_showscale=False
    )

    # Display first two charts side by side

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            fig_churn,
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            fig_contract,
            use_container_width=True
        )

    # --------------------------------------------------
    # Chart 3: Churn by Tenure
    # --------------------------------------------------

    tenure_data = customer_data.copy()

    tenure_data["Tenure Group"] = pd.cut(
        tenure_data["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-36 months",
            "37-48 months",
            "49-60 months",
            "61-72 months"
        ]
    )

    tenure_churn = (
        tenure_data.groupby(
            "Tenure Group",
            observed=False
        )["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )

    fig_tenure = px.line(
        tenure_churn,
        x="Tenure Group",
        y="Churn Rate",
        markers=True,
        title="Churn Rate Across Customer Tenure",
        text="Churn Rate"
    )

    fig_tenure.update_traces(
        line=dict(width=3),
        texttemplate="%{text:.1f}%",
        textposition="top center"
    )

    fig_tenure.update_layout(
        xaxis_title="Customer Tenure",
        yaxis_title="Churn Rate (%)"
    )

    st.plotly_chart(
        fig_tenure,
        use_container_width=True
    )

    # --------------------------------------------------
    # OVERVIEW INSIGHTS
    # --------------------------------------------------

    st.divider()

    st.subheader("Key Business Insights")

    highest_churn_contract = (
        contract_churn.loc[
            contract_churn["Churn Rate"].idxmax(),
            "Contract"
        ]
    )

    highest_churn_rate = (
        contract_churn["Churn Rate"].max()
    )

    st.markdown(
        f"""
        - **Overall churn rate:** {churn_rate:.2f}% of customers have left.
        - **Highest-risk contract:** {highest_churn_contract}.
        - **Churn rate for this contract:** {highest_churn_rate:.2f}%.
        - **Average monthly charges:** ${avg_monthly_charges:.2f}.
        """
    )


# ==================================================
# 8. PREDICT CHURN PAGE
# ==================================================

elif page == "Predict Churn":

    st.subheader("Predict Customer Churn")

    st.markdown(
        "Enter customer details below to estimate "
        "their probability of leaving the service."
    )

    # --------------------------------------------------
    # CUSTOMER INPUT FORM
    # --------------------------------------------------

    with st.form("customer_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                [
                    "Yes",
                    "No",
                    "No phone service"
                ]
            )

        with col2:

            internet = st.selectbox(
                "Internet Service",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

            online_security = st.selectbox(
                "Online Security",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            online_backup = st.selectbox(
                "Online Backup",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            device_protection = st.selectbox(
                "Device Protection",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            tech_support = st.selectbox(
                "Tech Support",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        with col3:

            streaming_movies = st.selectbox(
                "Streaming Movies",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        submit = st.form_submit_button(
            "Predict Churn",
            use_container_width=True
        )

    # --------------------------------------------------
    # PREDICTION LOGIC
    # --------------------------------------------------

    if submit:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        probabilities = pipeline.predict_proba(customer)[0]

        yes_index = list(
            pipeline.classes_
        ).index("Yes")

        churn_probability = probabilities[yes_index]

        prediction = (
            "Yes"
            if churn_probability >= threshold
            else "No"
        )

        # --------------------------------------------------
        # DISPLAY PREDICTION RESULTS
        # --------------------------------------------------

        st.divider()

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Churn Prediction",
                prediction
            )

        st.progress(
            float(churn_probability)
        )

        if prediction == "Yes":

            st.error(
                "⚠️ This customer is at risk of churning."
            )

        else:

            st.success(
                "✅ This customer is unlikely to churn."
            )

        # Risk categorization

        if churn_probability < threshold:

            risk = "Low"

        elif churn_probability < 0.6:

            risk = "Moderate"

        else:

            risk = "High"

        st.write(
            f"**Risk Category:** {risk}"
        )

        st.caption(
            f"Classification threshold: {threshold:.2f}"
        )
