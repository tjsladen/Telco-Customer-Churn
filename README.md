# Telco Customer Churn Prediction 📉📈

This project presents a data-driven approach to predicting customer churn for a fictional telecom company. Using real-world-style data, the analysis focuses on identifying high-risk customers and offering actionable insights to support customer retention strategies.
🔍 Objective

To develop a machine learning model that effectively predicts customer churn and provides business-oriented insights to support intervention efforts.

📂 Project Structure

    Telco_churn.html: Final business-style interactive report.
    churn_eda and churn_model: python code scripts outlining code used during exploratory data analysis and modelling.
    Business_STyle_Telco_Churn_Report: jupyter notebook version of HTML file.

    Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

📊 Key Insights

    Top Predictors of Churn:
    MonthlyCharges, TotalCharges, Tenure, and Contract type.

    Model Performance:

        AUC: 0.96

        Cohen’s Kappa: 0.73

        Recall: 0.96

        Optimal threshold: 0.6, balancing recall and precision.

    Cumulative Gains Insight:
    Targeting the top 30% most at-risk customers captures ~90% of churners, enabling focused retention strategies.

    SHAP Interpretability:
    Features like Contract Type, Monthly Charges and Tenure showed strong directional influence on churn. For instance, churn was highest among customers with monthly contracts and high monthly charges.

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
 
 🔗 View the Project   
[Download HTML](https://github.com/tjsladen/Telco-Customer-Churn/blob/578cdf949739061e7c7e870f9ccdaa9976ee76e9/Telco_churn.html) (599KB)

[Jupyter notebook version](https://github.com/tjsladen/Telco-Customer-Churn/blob/25b85c5b796452fc8fc6173cb36a6f83e5472e30/Business_Style_Telco_Churn_Report.ipynb) (Can be previewed following link.)
