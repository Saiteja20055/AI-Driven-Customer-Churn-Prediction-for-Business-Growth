# "Kindly save all the provided files in a single folder to ensure proper execution before running the code."

# 🔮 AI-Powered Customer Churn Prediction System

## 📌 1. Problem Statement
Customer retention is more cost-effective than acquisition, yet many businesses—especially in subscription-based sectors like telecommunications and SaaS—struggle with **customer churn** (when customers stop doing business with them).  
This project addresses the challenge by building an **intelligent machine learning system** that:
- Predicts churn with high accuracy.
- Provides actionable insights and retention strategies.
- Empowers businesses to proactively reduce customer loss.

---

## 📋 2. Project Overview
This is an **end-to-end Customer Churn Prediction System** deployed as an interactive **Streamlit web app**.  
Key capabilities:
- Predict churn for **individual customers** or **bulk datasets**.
- **Interactive dashboards** for churn insights.
- **Strategic recommendations** for at-risk customers.
- **Secure login system** with personalized prediction history.

---

## ⚙️ 3. Technologies & Libraries
**Core Language:** Python  
**Machine Learning & Data Handling:**  
- Scikit-learn (Random Forest Classifier, pipeline)  
- Pandas, NumPy  

**Web App Framework:**  
- Streamlit  

**Data Visualization:**  
- Plotly Express (interactive charts)  

**Model & Data Persistence:**  
- Pickle (save/load trained ML pipeline)  

---

## 🧠 4. Workflow: From Data to Deployment
1. **Data Collection & Exploration**
   - Dataset: `customer_churn_data.csv`
   - Key features: Tenure, Contract Type, Monthly Charges
   - Target variable: `Churn`

2. **Data Preprocessing**
   - Handle missing values.
   - Drop non-essential columns (`CustomerID`).
   - One-hot encode categorical variables.
   - Integrated into a **Scikit-learn pipeline**.

3. **Model Training & Selection**
   - Train-test split: 80% / 20%.
   - Algorithm: **Random Forest Classifier**.
   - Pipeline ensures consistent preprocessing during prediction.

4. **Model Evaluation & Saving**
   - Evaluate with Accuracy, Precision, Recall, F1-score.
   - Save pipeline as `churn_model.pkl` using `pickle`.

5. **Web Application Development**
   - **Secure authentication** (hashed passwords, SHA-256).
   - Tabs: Manual Prediction, Bulk Upload, Insights, History.

6. **Deployment**
   - Load model at app startup.
   - Save prediction results in `predictions.csv`.

---

## 🌟 5. Key Features
- **🔐 Secure User Authentication**
  - Login/Sign-up system with hashed passwords.
  - Private, account-specific prediction history.

- **📊 Dual Prediction Modes**
  - Manual Input (single customer).
  - Bulk CSV Upload (multiple customers).

- **💡 Actionable Recommendations**
  - Display churn probability (e.g., 85% likely to churn).
  - Suggest retention strategies (discounts, support).

- **📈 Churn Insights Dashboard**
  - Pie chart: churn vs. non-churn distribution.
  - Probability histogram: model confidence.
  - Feature analysis: correlation between attributes & churn.

---

## 🖥 6. How to Run the Project
**Prerequisites**
- Python 3.7+
- `pip` installed

**Steps**
1. **Create a Project Folder**  
   Example: `C:\Projects\Churn-Prediction`

2. **Download Files**  
   Place `app.py`, `churn_model.pkl`, and all other project files in the **same directory**.

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
4. **Run the Application**
```bash
streamlit run app.py
```
5.**Access the App**
Once the command runs successfully, your browser will open automatically.
Sign up for an account.
Start predicting customer churn instantly.

📊 6.**Model Performance**
Algorithm: Random Forest Classifier
Accuracy: **~94%**
Metrics: Precision, Recall, F1-score, Confusion Matrix




