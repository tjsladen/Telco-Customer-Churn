# -*- coding: utf-8 -*-
"""
Created on Mon May 19 11:23:02 2025

@author: tjsla
"""

import os
os.chdir(r"C:\Users\tjsla\OneDrive\Desktop\Personal projects\Telco cutomer churn\Scripts") 
from check_zeros_nas import check_zeros_nas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

churn_original = pd.read_csv(r"C:\Users\tjsla\OneDrive\Desktop\Personal projects\Telco cutomer churn\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
churn_copy = churn_original.copy()

entry_info = check_zeros_nas(churn_copy, verbose=True)

churn_copy['TotalCharges'] = churn_copy['TotalCharges'].astype(str).str.strip()
churn_copy['TotalCharges'] = churn_copy['TotalCharges'].replace("",np.nan)

#extract na rows dictionary from entry_info
na_row_dict = entry_info['row_indexes']['nas']
#flatten row indexes into a list
drop_rows = sorted(set(index for indices in na_row_dict.values() for index in indices))
#remove rows with missing values
churn_copy = churn_copy.drop(index=drop_rows)
#ensure TotalCharges is float64 instead of object
churn_copy['TotalCharges'] = churn_copy['TotalCharges'].astype(float)
#remove customer id as its not required
churn_copy = churn_copy.drop('customerID', axis = 1)

#plot class counts
palette = ["blue", "orange"]
plt.figure(figsize = (14,6))
ax = sns.countplot(x=churn_copy['Churn'],\
                   palette=palette, hue=churn_copy['Churn'],\
                       legend=False)
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn Counts")
plt.tight_layout()
plt.show()

#produce count plots for categorical features
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                    'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaymentMethod']

# Number of plots per grid
n_per_grid = 8

# Split into two lists
cat_split_1 = categorical_cols[:n_per_grid]
cat_split_2 = categorical_cols[n_per_grid:]

def plot_count_grid(columns, df, grid_num):
    n_cols = 4  # 4 plots per row
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        # Create the countplot
        plot = sns.countplot(data=df, x=col, ax=ax, order=df[col].value_counts().index, palette='pastel',\
                             hue=col, legend=False)
        ax.set_title(f'Count Plot: {col}')
        ax.tick_params(axis='x', rotation=45)
        
        # Add count labels
        for container in ax.containers:
            ax.bar_label(container)

    # Turn off unused plots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Categorical Count Plots - Grid {grid_num}', y=1.02, fontsize=16)
    plt.show()

# Plot first 8
plot_count_grid(cat_split_1, churn_copy, grid_num=1)

# Plot next 8
plot_count_grid(cat_split_2, churn_copy, grid_num=2)

plt.figure(figsize=(14,6))
ax = sns.countplot(data=churn_copy, x="Contract", hue="Churn")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn by Contract Type")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
ax = sns.countplot(data=churn_copy, x="PaymentMethod", hue="Churn")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn by Customers Payment Method")
plt.tight_layout()
plt.show()

#statistically test relationship between churn and payment method
from scipy.stats import chi2_contingency

contingency = pd.crosstab(churn_copy['PaymentMethod'], churn_copy['Churn'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi2 = {chi2:.2f}, p-value = {p:.4f}")

contingency = pd.crosstab(churn_copy['Contract'], churn_copy['Churn'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi2 = {chi2:.2f}, p-value = {p:.4f}")

contingency = pd.crosstab(churn_copy['Dependents'], churn_copy['Churn'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi2 = {chi2:.2f}, p-value = {p:.4f}")

from scipy.stats import ttest_ind

churn_yes = churn_copy[churn_copy['Churn'] == 'Yes']['MonthlyCharges']
churn_no = churn_copy[churn_copy['Churn'] == 'No']['MonthlyCharges']

t_stat, p_value = ttest_ind(churn_yes, churn_no, equal_var=False)  # Welch's t-test
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1)*x.std(ddof=1)**2 + (ny - 1)*y.std(ddof=1)**2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_std

d = cohens_d(churn_yes, churn_no)
print(f"Cohen's d: {d:.3f}")

sns.violinplot(data=churn_copy, x='Churn', y='MonthlyCharges', palette='pastel',\
               inner="box", hue='Churn', legend=False)
plt.title('Monthly Charges Distribution by Churn Status')
plt.ylabel("Monthly Charges ($)")
plt.text(0.05, 157, f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
plt.text(0.3, 150, f"Cohen's d: {d:.2f}")
plt.tight_layout()
plt.show()


churn_yes = churn_copy[churn_copy['Churn'] == 'Yes']['tenure']
churn_no = churn_copy[churn_copy['Churn'] == 'No']['tenure']

t_stat, p_value = ttest_ind(churn_yes, churn_no, equal_var=False)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

d = cohens_d(churn_yes, churn_no)
print(f"Cohen's d: {d:.3f}")

sns.violinplot(data=churn_copy, x='Churn', y='tenure', palette='pastel',\
               inner="box", hue='Churn', legend=False)
plt.title('Tenure Distribution by Churn Status')
plt.ylabel("Tenure (Months)")
plt.text(0.05, 100, f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
plt.text(0.3, 95, f"Cohen's d: {d:.2f}")
plt.tight_layout()
plt.show()

churn_yes = churn_copy[churn_copy['Churn'] == 'Yes']['TotalCharges']
churn_no = churn_copy[churn_copy['Churn'] == 'No']['TotalCharges']

t_stat, p_value = ttest_ind(churn_yes, churn_no, equal_var=False)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

d = cohens_d(churn_yes, churn_no)
print(f"Cohen's d: {d:.3f}")

sns.violinplot(data=churn_copy, x='Churn', y='TotalCharges', palette='pastel',\
               inner="box", hue='Churn', legend=False)
plt.title('Total Charges Distribution by Churn Status')
plt.ylabel("Total Charges ($)")
plt.text(0.05, 11700, f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
plt.text(0.3, 11200, f"Cohen's d: {d:.2f}")
plt.tight_layout()
plt.show()

from scipy.stats import skew

monthly_charges = churn_copy['MonthlyCharges']
skew_value = skew(monthly_charges, bias=False)

sns.displot(data = churn_copy, x='MonthlyCharges', hue='Churn')
plt.axvline(churn_copy['MonthlyCharges'].mean(), c='k', ls='-', lw=2.5)
plt.axvline(churn_copy['MonthlyCharges'].median(), c='orange', ls='--', lw=2.5)
plt.text(130, 400, f"Skew: {skew(churn_copy['MonthlyCharges'], bias=False):.2f}",\
             ha='right', fontsize=10)

plt.title("Distribution of Monthly Charges ($)")
plt.tight_layout()
plt.show()

tenure = churn_copy['tenure']
skew_value = skew(tenure, bias=False)

g=sns.displot(data = churn_copy, x='tenure', hue='Churn')
plt.axvline(churn_copy['tenure'].mean(), c='k', ls='-', lw=2.5)
plt.axvline(churn_copy['tenure'].median(), c='orange', ls='--', lw=2.5)
plt.text(80, 200, f"Skew: {skew(churn_copy['tenure'], bias=False):.2f}",\
             ha='center', fontsize=10)
plt.title("Distribution of Tenure (Months)")
plt.tight_layout()
g._legend.set_bbox_to_anchor((0.83, 0.5))
g._legend.set_loc('upper left')
plt.show()

total_charges = churn_copy['TotalCharges']
skew_value = skew(tenure, bias=False)

sns.displot(data = churn_copy, x='TotalCharges', hue='Churn')
plt.axvline(churn_copy['TotalCharges'].mean(), c='k', ls='-', lw=2.5)
plt.axvline(churn_copy['TotalCharges'].median(), c='orange', ls='--', lw=2.5)
plt.text(9000, 350, f"Skew: {skew(churn_copy['TotalCharges'], bias=False):.2f}",\
             ha='right', fontsize=10)

plt.title("Distribution of Total Charges ($)")
plt.tight_layout()
plt.show()
##########################################extra
plt.figure(figsize=(14,6))
ax = sns.countplot(data=churn_copy, x="TechSupport", hue="Churn")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn by Possessing Tech Support")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
ax = sns.countplot(data=churn_copy, x="InternetService", hue="Churn")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn by Possessing Internet Service")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
ax = sns.countplot(data=churn_copy, x="OnlineSecurity", hue="Churn")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Churn by Possessing Online Security")
plt.tight_layout()
plt.show()