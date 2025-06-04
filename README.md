Telco Customer Churn Prediction 📉📈

This project presents a data-driven approach to predicting customer churn for a fictional telecom company. Using real-world-style data, the analysis focuses on identifying high-risk customers and offering actionable insights to support customer retention strategies.
🔍 Objective

To develop a machine learning model that effectively predicts customer churn and provides business-oriented insights to support intervention efforts.
📂 Project Structure

    Telco_churn.html: Final business-style interactive report.

    Business_Style_Telco_Churn_Report.ipynb: Full annotated notebook used to generate the report.

📊 Key Insights

    Top Predictors of Churn:
    MonthlyCharges, TotalCharges, Tenure, and Contract type.

    Model Performance:

        AUC: 0.96

        Cohen’s Kappa: 0.75

        Optimal threshold: 0.6 to 0.7, balancing recall and precision.

    Cumulative Gains Insight:
    Targeting the top 30% most at-risk customers captures ~90% of churners, enabling focused retention strategies.

    SHAP Interpretability:
    Features like TechSupport, OnlineSecurity, and Contract showed strong directional influence on churn. For instance, churn was highest among customers with no tech support and monthly contracts.

📈 Tools Used

    Python (Pandas, NumPy, Scikit-learn, CatBoost)

    SHAP for interpretability

    Matplotlib / Seaborn for visualizations

    Optuna for hyperparameter tuning

    Jupyter Notebook for development

    HTML report output for stakeholders

🧠 Business Recommendations

    Consider offering discounts or incentives to customers approaching high monthly charge thresholds.

    Encourage adoption of tech support and online security add-ons to reduce churn risk.

    Focus retention campaigns on month-to-month contract holders, especially those using fiber optic internet.
