import streamlit as st
import pandas as pd
import pickle
import numpy as np
import hashlib
import os
import plotly.express as px

# --- Constants for files ---
USERS_DB = 'users.csv'          # Store username and hashed passwords
PREDICTIONS_DB = 'predictions.csv'  # Store user prediction history

# --- Password hashing ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- User data management ---
def load_users():
    if os.path.exists(USERS_DB):
        return pd.read_csv(USERS_DB)
    else:
        return pd.DataFrame(columns=['username', 'password_hash'])

def save_user(username, password_hash):
    users = load_users()
    if username not in users['username'].values:
        users = pd.concat([users, pd.DataFrame([{'username': username, 'password_hash': password_hash}])], ignore_index=True)
        users.to_csv(USERS_DB, index=False)
        return True
    return False

def check_user(username, password):
    users = load_users()
    user = users[users['username'] == username]
    if not user.empty:
        return user.iloc[0]['password_hash'] == hash_password(password)
    return False

# --- Load model and expected columns ---
try:
    with open('churn_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    pipeline = model_data['pipeline']
    expected_columns = model_data['columns']  # Columns from your training data
except FileNotFoundError:
    st.error("Fatal Error: 'churn_model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop()


# --- Session state defaults ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'df_upload' not in st.session_state:
    st.session_state.df_upload = pd.DataFrame()


# --- Login function ---
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_user(username, password):
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            # FIX 1: Use st.rerun() to immediately reload the app after successful login
            st.rerun()
        else:
            st.error("Invalid username or password")

# --- Signup function ---
def signup():
    st.title("📝 Sign Up")
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    password_confirm = st.text_input("Confirm Password", type="password")
    if st.button("Sign Up"):
        if password != password_confirm:
            st.error("Passwords do not match")
        elif len(password) < 6:
            st.error("Password must be at least 6 characters")
        else:
            success = save_user(username, hash_password(password))
            if success:
                st.success("User registered! Please proceed to the Login page.")
            else:
                st.error("Username already exists")

# --- Logout function ---
def logout():
    # Clear session state on logout
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # FIX 1: Use st.rerun() for a clean logout and redirect
    st.rerun()

# --- Save prediction history ---
def save_prediction_history(username, input_data, prediction, prob):
    record = input_data.copy()
    record['Churn Prediction'] = prediction
    record['Churn Probability'] = prob
    record['username'] = username
    if os.path.exists(PREDICTIONS_DB):
        df = pd.read_csv(PREDICTIONS_DB)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(PREDICTIONS_DB, index=False)

# --- Main application after login ---
def main_app():
    st.sidebar.title(f"Welcome, {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        logout()

    st.title("🔮 AI-Powered Customer Churn Prediction System")
    st.markdown("This tool helps businesses identify customers likely to leave and suggests retention strategies.")

    tabs = st.tabs(["📝 Manual Input", "📂 Upload CSV File", "📈 Churn Insights", "🗂 Prediction History"])

    # --- Manual Input Tab ---
    with tabs[0]:
        with st.form("customer_form"):
            st.subheader("Customer Information")
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                tenure = st.slider("Tenure (months)", 0, 72, 12)

            with col2:
                monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
                contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

            with col3:
                total_charges = st.number_input("Total Charges", min_value=0.0, value=800.0)
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

            submitted = st.form_submit_button("Predict Churn")

        if submitted:
            input_df = pd.DataFrame([{
                'Age': age, 'Gender': gender, 'Tenure': tenure, 'MonthlyCharges': monthly_charges,
                'ContractType': contract_type, 'InternetService': internet_service,
                'TotalCharges': total_charges, 'TechSupport': tech_support
            }])
            input_df = input_df.reindex(columns=expected_columns, fill_value=0)

            try:
                prediction = pipeline.predict(input_df)[0]
                prediction_proba = pipeline.predict_proba(input_df)[0][1]

                save_prediction_history(st.session_state['username'], input_df.iloc[0].to_dict(), prediction, prediction_proba)

                st.markdown("---")
                st.subheader("📢 Prediction Result")

                if prediction == "Yes":
                    st.error(f"⚠️ This customer is likely to churn! (Probability: {prediction_proba:.2%})")
                    st.markdown("### 💡 Suggested Retention Strategies:")
                    st.markdown("- Offer personalized discount plans or loyalty programs.\n- Upgrade their plan with better features.\n- Contact with personalized support.")
                else:
                    st.success(f"✅ This customer is not likely to churn. (Probability: {prediction_proba:.2%})")
                    st.markdown("Keep up the good service and maintain engagement!")

            except Exception as e:
                st.error(f"Prediction error: {e}")

    # --- CSV Upload Bulk Prediction Tab ---
    with tabs[1]:
        st.subheader("📂 Upload CSV File for Bulk Prediction")
        file = st.file_uploader("Upload a customer CSV file", type=["csv"])

        if file:
            try:
                df_upload = pd.read_csv(file)
                df_upload_original = df_upload.copy()
                df_upload_processed = df_upload.reindex(columns=expected_columns, fill_value=0)

                preds = pipeline.predict(df_upload_processed)
                probs = pipeline.predict_proba(df_upload_processed)[:, 1]

                df_upload_original['Churn Prediction'] = preds
                df_upload_original['Churn Probability'] = np.round(probs, 4)
                
                # Store the results in session state for the insights tab
                st.session_state.df_upload = df_upload_original

                # Save predictions to history file
                df_to_save = df_upload_original.copy()
                df_to_save['username'] = st.session_state['username']
                if os.path.exists(PREDICTIONS_DB):
                    hist_df = pd.read_csv(PREDICTIONS_DB)
                    hist_df = pd.concat([hist_df, df_to_save], ignore_index=True)
                else:
                    hist_df = df_to_save
                hist_df.to_csv(PREDICTIONS_DB, index=False)

                st.success("✅ Prediction Complete")
                st.dataframe(df_upload_original)

                churned_df = df_upload_original[df_upload_original['Churn Prediction'] == "Yes"]
                st.markdown("---")
                st.subheader("📉 Customers Predicted to Churn")
                st.dataframe(churned_df)

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # --- Churn Insights Tab ---
    with tabs[2]:
        st.subheader("📈 Churn Insights Dashboard")
        st.markdown("Visual analysis of the most recently uploaded customer data.")
        
        # FIX 2: Check for data in session_state
        if not st.session_state.df_upload.empty:
            churn_data = st.session_state.df_upload

            # --- Row 1: Pie Chart and Probability Distribution ---
            col1, col2 = st.columns(2)
            with col1:
                # NEW: Pie chart for churn percentage
                st.markdown("#### Churn vs. No Churn")
                churn_counts = churn_data['Churn Prediction'].value_counts()
                pie_fig = px.pie(
                    values=churn_counts.values,
                    names=churn_counts.index,
                    color=churn_counts.index,
                    color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"}
                )
                st.plotly_chart(pie_fig, use_container_width=True)

            with col2:
                st.markdown("#### Distribution of Churn Probabilities")
                hist_fig = px.histogram(
                    churn_data,
                    x='Churn Probability',
                    nbins=20,
                    color='Churn Prediction',
                    marginal='box',
                    color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"}
                )
                st.plotly_chart(hist_fig, use_container_width=True)

            st.markdown("---")
            
            # --- Row 2: Detailed Feature Analysis ---
            st.markdown("#### Feature Analysis")
            
            # Identify categorical columns for analysis (you can customize this list)
            categorical_cols = [col for col in expected_columns if churn_data[col].dtype == 'object' or churn_data[col].nunique() < 10]
            
            feature_to_analyze = st.selectbox(
                "Select a feature to analyze:",
                options=categorical_cols
            )

            if feature_to_analyze:
                # NEW: Bar chart for feature distribution
                bar_fig = px.histogram(
                    churn_data,
                    x=feature_to_analyze,
                    color='Churn Prediction',
                    barmode='group',
                    title=f'Distribution of {feature_to_analyze} by Churn Status',
                    color_discrete_map={"Yes": "#EF553B", "No": "#636EFA"}
                )
                st.plotly_chart(bar_fig, use_container_width=True)

        else:
            st.info("Upload a CSV file in the 'Upload CSV File' tab to see churn insights.")

    # --- Prediction History Tab ---
    with tabs[3]:
        st.subheader("🗂 Your Past Predictions")
        if os.path.exists(PREDICTIONS_DB):
            history_df = pd.read_csv(PREDICTIONS_DB)
            user_history = history_df[history_df['username'] == st.session_state['username']]
            if not user_history.empty:
                st.dataframe(user_history.drop(columns=['username']))
            else:
                st.info("You have no saved predictions yet.")
        else:
            st.info("No prediction history found.")

# --- App entry point ---
if not st.session_state.get('authenticated', False):
    auth_mode = st.sidebar.selectbox("Choose Action", ["Login", "Sign Up"])
    if auth_mode == "Login":
        login()
    else:
        signup()
else:
    main_app()
