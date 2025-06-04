# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:31:53 2025

@author: tjsla
"""

import os
os.chdir(r"C:\Users\tjsla\OneDrive\Desktop\Personal projects\Telco cutomer churn\Scripts") 
from check_zeros_nas import check_zeros_nas
from custom_one_hot_encoder_final import CustomOneHotEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, cv, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from evaluate_classification_binary_BC import evaluate_model
import optuna
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix
import shap


churn_original = pd.read_csv(r"C:\Users\tjsla\OneDrive\Desktop\Personal projects\Telco cutomer churn\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
churn_copy = churn_original.copy()

entry_info = check_zeros_nas(churn_copy, verbose=True)

churn_copy['TotalCharges'] = churn_copy['TotalCharges'].astype(str).str.strip()
churn_copy['TotalCharges'] = churn_copy['TotalCharges'].replace("",np.nan)

na_row_dict = entry_info['row_indexes']['nas']
drop_rows = sorted(set(index for indices in na_row_dict.values() for index in indices))
churn_copy = churn_copy.drop(index=drop_rows)
churn_copy['TotalCharges'] = churn_copy['TotalCharges'].astype(float)
churn_copy = churn_copy.drop('customerID', axis = 1)
churn_copy['TotalCharges'] = np.log(churn_copy['TotalCharges'])
churn_copy['SeniorCitizen'] = churn_copy['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
churn_copy['Churn'] = churn_copy['Churn'].map({'Yes': 1, 'No': 0})

churn_logisticR = churn_copy.copy()
numerical_columns = ['tenure', 'MonthlyCharges','TotalCharges']
churn_logisticR[numerical_columns] = StandardScaler().fit_transform(churn_logisticR[numerical_columns])

encoder = encoder = CustomOneHotEncoder(max_categories = 5, verbose=True)
encoded_xgboost = encoder.fit_transform(churn_copy)
encoded_logisticR = encoder.fit_transform(churn_logisticR)
churn_catboost = churn_copy.copy()

###########################################################catboost
target_cat = churn_catboost['Churn']
feature_cat = churn_catboost.drop('Churn', axis = 1)

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(feature_cat, target_cat, 
                                                    shuffle = True, 
                                                    test_size=0.2, 
                                                    random_state=1)
cat_features = X_train_cat.select_dtypes(include=['object', 'category']).columns.tolist()

catC = CatBoostClassifier(random_state=1)
catC.fit(X_train_cat, y_train_cat, cat_features = cat_features)

cat_eval = evaluate_model(catC, X_test_cat, y_test_cat)
#########################################################
#########################################################Xgboost
target_xgb = encoded_xgboost['Churn']
feature_xgb = encoded_xgboost.drop('Churn', axis=1)

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(feature_xgb, target_xgb, 
                                                    shuffle = True, 
                                                    test_size=0.2, 
                                                    random_state=1)

xgbC = XGBClassifier(random_state=1)
xgbC.fit(X_train_xgb, y_train_xgb)

xgb_eval = evaluate_model(xgbC, X_test_xgb, y_test_xgb)
######################################################
######################################################LogisticR
target_logR = encoded_logisticR['Churn']
feature_logR = encoded_logisticR.drop('Churn', axis=1)

X_train_logR, X_test_logR, y_train_logR, y_test_logR = train_test_split(feature_logR, target_logR, 
                                                    shuffle = True, 
                                                    test_size=0.2, 
                                                    random_state=1)

logRC = LogisticRegression(penalty='l1', solver='saga', random_state=1)
logRC.fit(X_train_logR, y_train_logR)

logR_eval = evaluate_model(logRC, X_test_logR, y_test_logR)
######################################################
################################################################oob table
eval_dict = {
    'CatBoost': {k: v for k, v in cat_eval.items() if k != 'confusion_matrix'},
    'XGBoost': {k: v for k, v in xgb_eval.items() if k != 'confusion_matrix'},
    'Logistic Regression': {k: v for k, v in logR_eval.items() if k != 'confusion_matrix'},
}

# First: Round all values in eval_dict
for model in eval_dict:
    # Round class-specific metrics
    for metric in ['precision_per_class', 'recall_per_class', 'f1_score_per_class']:
        for cls in ['class_0', 'class_1']:
            eval_dict[model][metric][cls] = round(eval_dict[model][metric][cls], 3)

    # Round overall metrics
    for overall_metric in ['accuracy', 'auc', 'cohen_kappa']:
        eval_dict[model][overall_metric] = round(eval_dict[model][overall_metric], 3)

# Prepare data for the table
rows = []
row_labels = []

for model_name, metrics in eval_dict.items():
    # Class 0 row (no overall metrics)
    row_labels.append(f"{model_name} - Class 0")
    rows.append([
        metrics['precision_per_class']['class_0'],
        metrics['recall_per_class']['class_0'],
        metrics['f1_score_per_class']['class_0'],
        "", "", ""
    ])
    # Class 1 row (with overall metrics)
    row_labels.append(f"{model_name} - Class 1")
    rows.append([
        metrics['precision_per_class']['class_1'],
        metrics['recall_per_class']['class_1'],
        metrics['f1_score_per_class']['class_1'],
        metrics['accuracy'],
        metrics['auc'],
        metrics['cohen_kappa']
    ])

df = pd.DataFrame(
    rows,
    columns=['Precision', 'Recall', 'F1 Score', 'Accuracy', 'AUC', 'Cohen Kappa'],
    index=row_labels
)

# Plotting the table
fig, ax = plt.subplots(figsize=(10, len(row_labels)*0.5))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    rowLabels=df.index,
    cellLoc='center',
    loc='center'
)

# Highlight max values in Class 1 rows only per column
for col_idx, col_name in enumerate(df.columns):
    class_1_values = []
    class_1_row_indices = []

    for row_idx, row_label in enumerate(row_labels):
        if "Class 1" in row_label:
            val = df.iloc[row_idx, col_idx]
            # Ignore empty cells ("")
            if val != "":
                class_1_values.append(val)
                class_1_row_indices.append(row_idx)

    if class_1_values:
        max_val = max(class_1_values)

        for row_idx, val in zip(class_1_row_indices, class_1_values):
            cell = table[row_idx + 1, col_idx]  # +1 for header offset
            if val == max_val:
                cell.set_facecolor('#c8e6c9')  # Light green
            else:
                cell.set_facecolor('white')

    # Set Class 0 cells background to white (optional cleanup)
    for row_idx, row_label in enumerate(row_labels):
        if "Class 0" in row_label:
            cell = table[row_idx + 1, col_idx]
            cell.set_facecolor('white')

# Adjust font size and scale
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("OOB Model Evaluation Metric Scores by Class", fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()
################################################################
################################################################ optuna bayesian optimisation
######### no need to re-run when model saved
def objective(trial):
    weight_ratio = trial.suggest_float('class_weight_ratio', 1.0, 5.0)
    class_weights = [1.0, weight_ratio]

    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 1, 10, log=True),
        'od_type': trial.suggest_categorical('od_type', ["Iter", "IncToDec"]),
        'od_wait': trial.suggest_int('od_wait', 20, 100),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ["Bayesian", "Bernoulli", "MVS"]),
        'grow_policy': trial.suggest_categorical('grow_policy', ["SymmetricTree", "Depthwise", "Lossguide"]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 40),
        'verbose': False,
        'task_type': "CPU",
        'eval_metric': "F1",
        'use_best_model': True,
        'class_weights': class_weights,
        'loss_function': 'Logloss'
    }

    if params["bootstrap_type"] in ["Bernoulli", "MVS"]:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    if params["grow_policy"] == "Lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 31, 64)

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 1)

    train_pool = Pool(data=feature_cat, label=target_cat, cat_features=cat_features)

    cv_result = cv(
        params=params,
        pool=train_pool,
        fold_count=5,
        partition_random_seed=1,
        shuffle=True,
        stratified=True,
        early_stopping_rounds=50,
        verbose=False
    )

    # Return best test Recall mean
    return np.max(cv_result['test-F1-mean'])

study_cat = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=1))
study_cat.optimize(objective, n_trials=100)
print(study_cat.best_value)
#remove class weights from best params and manually redefine it for use
best_params_cat = study_cat.best_params
weight_ratio = best_params_cat.pop('class_weight_ratio')
class_weights = [1.0, weight_ratio]

best_model_cat = CatBoostClassifier(**best_params_cat, class_weights=class_weights, random_state=1)
best_model_cat.fit(feature_cat, target_cat, cat_features=cat_features)
best_model_cat.save_model("best_catboost_model.cbm")

best_trial = study_cat.best_trial

print("Best Trial:")
print(f"  Value: {best_trial.value}")
print(f"  Params: {best_trial.params}")

# Optional: save to text file
with open("optuna_best_trial.txt", "w") as f:
    f.write(f"Best value: {best_trial.value}\n")
    for key, value in best_trial.params.items():
        f.write(f"{key}: {value}\n")
################################################### run from here - skip optim
best_model_cat = CatBoostClassifier()
best_model_cat.load_model("best_catboost_model.cbm")

cat_eval_optim = evaluate_model(best_model_cat, X_test_cat, y_test_cat)

preds = best_model_cat.predict(X_test_cat)
print(classification_report(y_test_cat, preds))
#########different thresholds
probs = best_model_cat.predict_proba(X_test_cat)[:, 1]

thresholds = np.arange(0.1, 0.95, 0.05)

for thresh in thresholds:
    preds_thresh = (probs >= thresh).astype(int)
    
    print(f"--- Threshold: {thresh:.2f} ---")
    print(classification_report(y_test_cat, preds_thresh, digits=2))
    print(f"AUC: {roc_auc_score(y_test_cat, probs):.2f}")
    print(f"Cohen Kappa: {cohen_kappa_score(y_test_cat, preds_thresh):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_cat, preds_thresh)}\n")
############## final threshold
probs = best_model_cat.predict_proba(X_test_cat)[:, 1]
threshold = 0.6
preds_custom = (probs >= threshold).astype(int)

print(classification_report(y_test_cat, preds_custom))
from evaluate_classification_binary_custom_threshold import evaluate_model_custom_threshold

cat_eval_optim_06 = evaluate_model_custom_threshold(best_model_cat, X_test_cat, y_test_cat,threshold=0.6)
#####################################################final model table

class_metrics = {
    "Precision": cat_eval_optim_06["precision_per_class"],
    "Recall": cat_eval_optim_06["recall_per_class"],
    "F1 Score": cat_eval_optim_06["f1_score_per_class"]
}

# Extract overall metrics
overall_metrics = {
    "Accuracy": round(cat_eval_optim_06["accuracy"], 2),
    "AUC": round(cat_eval_optim_06["auc"], 2),
    "Cohen Kappa": round(cat_eval_optim_06["cohen_kappa"], 2)
}

# Build rows manually
data = [
    [
        class_metrics["Precision"]["class_0"],
        class_metrics["Recall"]["class_0"],
        class_metrics["F1 Score"]["class_0"],
        "", "", ""  # Leave overall metrics blank for class 0
    ],
    [
        class_metrics["Precision"]["class_1"],
        class_metrics["Recall"]["class_1"],
        class_metrics["F1 Score"]["class_1"],
        overall_metrics["Accuracy"],
        overall_metrics["AUC"],
        overall_metrics["Cohen Kappa"]
    ]
]

# Column labels
columns = [
    "Precision", "Recall", "F1 Score",
    "Accuracy", "AUC", "Cohen Kappa"
]
row_labels = ["Class 0", "Class 1"]

# Create DataFrame
df = pd.DataFrame(data, columns=columns, index=row_labels).round(2)

# Plot the table
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    rowLabels=df.index,
    colLabels=df.columns,
    cellLoc='center',
    rowLoc='center',
    loc='center'
)

# Format table appearance
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("CatBoost Final Model: Per-Class and Overall Metric Scores", fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()
#######################################
def plot_probability_distribution(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(8, 5))
    sns.histplot(y_prob[y_test == 0], color='green', label='No Churn', kde=True, stat="density", bins=25)
    sns.histplot(y_prob[y_test == 1], color='red', label='Churn', kde=True, stat="density", bins=25)
    plt.axvline(0.5, color='black', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Predicted Churn Probability Distribution')
    plt.legend()
    plt.show()

plot_probability_distribution(best_model_cat, X_test_cat, y_test_cat)

def plot_feature_importance(model, feature_names, top_n=10):
    importances = model.get_feature_importance()
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.show()
    
plot_feature_importance(best_model_cat, feature_cat.columns, top_n=19)

from sklearn.metrics import roc_curve
def plot_cumulative_gain(y_true, y_prob, pos_label=1):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)

    df['cumulative_positives'] = df['y_true'].cumsum()
    df['total_positives'] = df['y_true'].sum()
    df['gain'] = df['cumulative_positives'] / df['total_positives']

    pct_samples = np.arange(1, len(df)+1) / len(df)

    plt.figure(figsize=(8,6))
    plt.plot(pct_samples, df['gain'], label='Model')
    plt.plot([0, 1], [0, 1], '--', label='Random', color='gray')
    plt.xlabel('% of Sample')
    plt.ylabel('% of Positive Cases Captured')
    plt.title('Cumulative Gains Chart')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Return values so we can use them for precise calculations
    return pct_samples, df['gain'].values

pct_samples, gains = plot_cumulative_gain(y_test_cat, probs)

idx_10 = (np.abs(pct_samples - 0.1)).argmin()
idx_20 = (np.abs(pct_samples - 0.2)).argmin()
idx_30 = (np.abs(pct_samples - 0.3)).argmin()
idx_34 = (np.abs(pct_samples - 0.34)).argmin()

print(f"At top 20% of customers (by predicted risk), you capture {gains[idx_20]*100:.2f}% of all churners.")
print(f"At top 10% of customers (by predicted risk), you capture {gains[idx_10]*100:.2f}% of all churners.")
print(f"At top 30% of customers (by predicted risk), you capture {gains[idx_30]*100:.2f}% of all churners.")
print(f"At top 34% of customers (by predicted risk), you capture {gains[idx_34]*100:.2f}% of all churners")

def plot_lift_chart(y_true, y_prob, pos_label=1, bins=10):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    
    df['bucket'] = pd.qcut(df.index, q=bins, labels=False)
    grouped = df.groupby('bucket')['y_true'].agg(['sum', 'count'])
    grouped['response_rate'] = grouped['sum'] / grouped['count']
    
    baseline = df['y_true'].mean()
    grouped['lift'] = grouped['response_rate'] / baseline

    plt.figure(figsize=(8,6))
    plt.bar(range(1, bins+1), grouped['lift'], color='steelblue')
    plt.xlabel('Decile (1 = Highest Probability)')
    plt.ylabel('Lift')
    plt.title('Lift Chart')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_lift_chart(y_test_cat, probs)
######################shap
explainer = shap.TreeExplainer(best_model_cat)
shap_values = explainer(X_test_cat)

shap.summary_plot(shap_values, X_test_cat,show=False)
