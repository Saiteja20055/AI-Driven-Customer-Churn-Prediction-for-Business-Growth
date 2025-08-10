"Kindly save all the provided files in a single folder to ensure proper execution before running the code."

#🔮 AI-Powered Customer Churn Prediction System
1. Problem Statement
In the competitive business landscape, customer retention is more cost-effective than customer acquisition. Businesses across various sectors, especially in subscription-based services like telecommunications and SaaS, face the challenge of customer churn—the phenomenon where customers cease to do business with a company. Identifying which customers are at a high risk of churning is a critical business problem. This project aims to solve this by building an intelligent system that not only predicts customer churn with high accuracy but also provides actionable insights and retention strategies, empowering businesses to proactively reduce customer loss.

2. Project Overview
This project is a comprehensive, end-to-end Customer Churn Prediction System built as an interactive web application. It leverages a machine learning model to analyze customer data and predict the likelihood of churn. The system is designed for business users, providing a user-friendly interface to get predictions for individual customers or in bulk, visualize churn patterns, and receive strategic recommendations to retain at-risk customers.

3. Technologies & Libraries Used
This project was built using a robust stack of open-source technologies:

Core Language:

Python: The primary language for data analysis, model training, and application logic.

Machine Learning & Data Handling:

Scikit-learn: For building the machine learning pipeline, training the Random Forest model, and making predictions.

Pandas: For efficient data loading, manipulation, and analysis.

NumPy: For numerical operations, especially during data processing.

Web Application Framework:

Streamlit: For creating and deploying the interactive, user-friendly web interface with minimal code.

Model & Data Persistence:

Pickle: For serializing and saving the trained Scikit-learn pipeline, allowing it to be loaded and used in the application without retraining.

Data Visualization:

Plotly Express: For creating interactive charts and graphs (pie charts, histograms, bar charts) in the "Churn Insights" dashboard.

4. How the Project Works: From Data to Deployment
Here is a step-by-step breakdown of the entire project lifecycle:

Data Collection & Exploration: The project started with the customer_churn_data.csv dataset. The data was explored to understand its structure, identify key features (like tenure, contract type, monthly charges), and determine the target variable (Churn).

Data Preprocessing: Before training, the data was cleaned and prepared. This involved:

Converting the TotalCharges column to a numeric type and handling missing values.

Dropping non-essential columns like CustomerID.

Using One-Hot Encoding to convert categorical variables (e.g., Gender, ContractType) into a numerical format that the model can understand. This was integrated into a Scikit-learn pipeline.

Model Training & Selection:

The dataset was split into an 80% training set and a 20% testing set.

A Random Forest Classifier was chosen for its high performance and ability to handle complex relationships in the data.

The entire process—preprocessing and modeling—was encapsulated in a Scikit-learn Pipeline. This ensures that the same steps are applied consistently during training and prediction.

Model Evaluation & Saving:

The trained model was evaluated on the test set to ensure its accuracy and reliability.

Once validated, the entire pipeline object was saved to a file named churn_model.pkl using Python's pickle library.

Web Application Development (Streamlit):

A user-friendly web interface was built using Streamlit.

A secure user authentication system (Login/Sign Up) was created to manage access and personalize the user experience. User credentials are saved locally in users.csv with hashed passwords for security.

The application was structured into tabs for different functionalities: Manual Input, Bulk Upload, Insights, and History.

Integration & Deployment:

The saved churn_model.pkl file was loaded into the Streamlit app at startup.

When a user inputs data (manually or via CSV), the app uses the loaded pipeline to make predictions in real-time.

Prediction results and user activity are saved to predictions.csv, creating a persistent history for each user.

5. Features of the Project
Secure User Authentication:

A robust Login and Sign Up system.

Passwords are never stored in plain text; they are hashed using the SHA-256 algorithm for security.

Each user's prediction history is private and tied to their account.

Dual Prediction Modes:

📝 Manual Input: An intuitive form to enter the details of a single customer and get an instant prediction.

📂 Bulk CSV Upload: Allows users to upload a CSV file containing data for multiple customers and receive predictions for the entire batch at once.

Actionable Insights & Recommendations:

The system doesn't just predict churn; it provides the churn probability (e.g., 85% likely to churn).

For customers predicted to churn, the app automatically suggests concrete retention strategies like offering discounts or personalized support.

📈 Interactive Churn Insights Dashboard:

After a bulk prediction, this dashboard comes to life with interactive visualizations.

Pie Chart: Shows the overall percentage of customers predicted to churn versus those who are not.

Probability Histogram: Visualizes the distribution of churn probabilities, helping to identify how confident the model is in its predictions.

Feature Analysis Bar Chart: Allows users to select a customer attribute (like Contract Type) and see how it correlates with churn, providing deeper business insights.

🗂 Personalized Prediction History:

Every prediction made by a logged-in user is saved.

This tab allows users to review their past prediction activities, helping them track trends or revisit previous analyses.

6. How to Set Up and Run the Project
Follow these steps to get the project running on your local machine.

Prerequisites
Python 3.7+ installed on your system.

pip (Python package installer).

Installation
Create a Project Folder:
First, create a new folder on your computer where you will store all the project files. For example, C:\Projects\Churn-Prediction.

Download and Place Files:
Download all the project files (app.py, churn_model.pkl, etc.) and save them directly inside the folder you just created.

Important Note: For the application to work, all files must be in the same directory. The app.py script needs to be able to find the churn_model.pkl file to load the machine learning model.

Install Required Libraries:
Create a file named requirements.txt inside your project folder with the following content:

streamlit
pandas
scikit-learn
numpy
plotly
Then, open your terminal or command prompt, navigate to your project folder, and run the following command:

Bash

pip install -r requirements.txt
Running the Application
Open your terminal and navigate to the project folder you created.

Bash

# Example for Windows
cd C:\Projects\Churn-Prediction

# Example for macOS/Linux
cd /path/to/your/Churn-Prediction
Run the Streamlit command:

Bash

streamlit run app.py
Your web browser will automatically open a new tab with the application running. You can now sign up for a new account and start predicting churn!
