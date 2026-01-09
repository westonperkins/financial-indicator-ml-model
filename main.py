import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018?resource=download&select=2018_Financial_Data.csv

# LOAD DATA
# each csv contains finanical indicators from 10-K filings for each year
data_2014 = pd.read_csv("data/2014_Financial_Data.csv")
data_2015 = pd.read_csv("data/2015_Financial_Data.csv")
data_2016 = pd.read_csv("data/2016_Financial_Data.csv")
data_2017 = pd.read_csv("data/2017_Financial_Data.csv")
data_2018 = pd.read_csv("data/2018_Financial_Data.csv")

# ADD YEAR LABEL AND COMBINE DATASETS INTO ONE
data_2014["year"] = 2014
data_2015["year"] = 2015
data_2016["year"] = 2016
data_2017["year"] = 2017
data_2018["year"] = 2018

# consolidate data
data = pd.concat(
    [data_2014, data_2015, data_2016, data_2017, data_2018], ignore_index=True
)

# SPLIT DATA INTO TRAINING AND TESTING SETS
# we will be training on data from 2014-2017 and testing our findings with data from 2018
train_data = data[data["year"] <= 2017]
test_data = data[data["year"] == 2018]

# sanity check - checking if overlap between training and test data
set(train_data.index).intersection(set(test_data.index))

# DEFINE GOAL AND WHAT WE ARE TRYING TO PREDICT
# y = results, or what actually happened in the following year
# class is either 0 or 1, 1 = stock price increased the following year, buy signal; 0 = stock price decreased the following year, sell signal;
y_train = train_data["Class"]
y_test = test_data["Class"]

# REMOVE COLUMNS THAT WOULD LEAK FUTURE INFORMATION
# these columns hold data that would not be avaliable at the time of prediction and would corrupt our findings if left in
price_var_columns = [c for c in data.columns if "PRICE VAR [%]" in c]
leakage_columns = ["Class", "year"] + price_var_columns


# not numerical data, ticker and sector names
id_columns = ["Unnamed: 0", "Sector"]

# BUILD FEATURE MATRICES
# X = information used to make decision - what we know before investing
# build feature matrix but exclude leakage colums and id columns - same structure as CSV file
X_train = train_data.drop(
    columns=[c for c in leakage_columns + id_columns if c in train_data.columns]
)

X_test = test_data.drop(
    columns=[c for c in leakage_columns + id_columns if c in test_data.columns]
)

# restrict feature matrices to numerical indicators only
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# HANDLE MISSING VALUES
# locate missing values in training
missing_summary = X_train.isna().mean().sort_values(ascending=False)

# fill missing data with averages from each indicator to avoid discarding companies
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

# RESCALE FINANCIAL INDICATORS
# rescale values to make them easier to compare across indicators, some indicators hold values > 1,000,000,000, and others hold values < 0
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# sanity check to make sure we have no more missing data points
np.isnan(X_train_scaled).sum()

# CREATE AND TRAIN LOGICAL REGRESSION MODEL - ASSIGNS WEIGHT TO EACH INDICATOR
# create model
logical_regression = LogisticRegression(max_iter=1000, random_state=42)

# train model
# model is shown indicators from 2014-2017, and then whether they went up or down, then finds patterns which will predict outcome
logical_regression.fit(X_train_scaled, y_train)

# MAKE PREDICTIONS ON 2018 DATA
# predict buy / sell signals for 2018
y_predict = logical_regression.predict(X_test_scaled)

# EVALUATE MODEL PERFORMANCE
# compare what happened in 2018 to what the model predicted as a percentage
accuracy = accuracy_score(y_test, y_predict)

# INTERPRET THE MODEL
# get names of all financial indicators used by the model
feature_names = X_train.columns
# get the coefficients learned by the regression model
coefficients = logical_regression.coef_[0]

# combine indicator names and their coefficients into table
feature_importance = pd.DataFrame(
    {"Feature": feature_names, "Coefficient": coefficients}
)

# RANK INDICATORS BY IMPORTANCE
# use absolute value - magnitude of importance
feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
# sort by most influenctial ot least influential
feature_importance = feature_importance.sort_values(
    by="Abs_Coefficient", ascending=False
)

# SECTOR LEVEL PERFORMANCE ANALYSIS
# create a copy of the 2018 test data to store predictions
test_results = test_data.copy()
# add the models predicted buy/sell decision for each stock
test_results["Prediction"] = y_predict
# mark whether the prediction was correct or not
test_results["Correct"] = test_results["Prediction"] == test_results["Class"]

# find prediciton accuracy within each sector
sector_accuracy = (
    test_results.groupby("Sector")["Correct"].mean().sort_values(ascending=False)
)

# EXPORT FINDINGS TO SPREADSHEETS
# FEATURE IMPORTANCE
feature_importance.to_csv("feature_importance_logistic_regression.csv", index=False)

# STOCK LEVEL PREDICTIONS
test_results_export = test_results[
    ["Unnamed: 0", "Sector", "Class", "Prediction", "Correct"]
].rename(columns={"Unamed: 0": "Ticker", "Class": "Actual"})

test_results_export.to_csv("2018_stock_predictions.csv", index=False)

# SECTOR LEVEL ACCURACY
sector_accuracy_df = sector_accuracy.reset_index()
sector_accuracy_df.columns = ["Sector", "Accuracy"]

sector_accuracy_df.to_csv("sector_accuracy_2018.csv", index=False)

# PERFORMANCE SUMMARY
performance_summary = pd.DataFrame({"Metric": ["Accuracy"], "Value": [accuracy]})

performance_summary.to_csv("model_performance_summary.csv", index=False)



# PLOTS
# -----

# VISUALIZE RESULTS WITH A CONFUSION MATRIX
# visualize correct buys, correct sells, false positives (model predicted buy, when stock went down ), and false negatives (model predicted sell, when stock went up)
cm = confusion_matrix(y_test, y_predict)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (2018 predictions)")
plt.show()

# TOP 15 MOST IMPORTANT FINANCIAL INDICATORS
top_features = feature_importance.head(15)

plt.figure(figsize=(8, 6))
sns.barplot(data=top_features, x="Abs_Coefficient", y="Feature", palette="viridis")

plt.title("Top 15 Most Important Financial Indicators")
plt.xlabel("Importance (magnitude)")
plt.ylabel("Indicator")
plt.tight_layout()
plt.show()

# MODEL ACCURACY BY SECTOR
plt.figure(figsize=(8, 5))
sector_accuracy.plot(kind="bar", color="steelblue")

plt.title("Model Accuracy by Sector (2018)")
plt.ylabel("Accuracy")
plt.xlabel("Sector")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ACTUAL VS PREDICTED CLASS DISTRIBUTION
distribution_df = pd.DataFrame(
    {"Actual": y_test.value_counts(), "Predicted": pd.Series(y_predict).value_counts()}
).fillna(0)

distribution_df.plot(kind="bar", figsize=(6, 4))
plt.title("Actual vs Predicted Buy/Sell Results (2018)")
plt.ylabel("Number of Stocks")
plt.xlabel("Class (0 = Sell, 1 = Buy)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# CLASS RESULTS IN 2018
plt.figure(figsize=(4, 4))
y_test.value_counts().plot(kind="bar", color=["green", "red"])

plt.title("Actual Stock Outcomes (2018)")
plt.xlabel("Class (0 = Sell, 1 = Buy)")
plt.ylabel("Number of Stocks")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ROC CURVE
y_prob = logical_regression.predict_proba(X_test_scaled)[:, 1]

# false positive rate (fpr) true positive rate (tpr)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.title("ROC Curve – Buy vs Sell")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# POSITIVE VS NEGATIVE INDICATOR INFLUENCE
top_signed = feature_importance.head(15).copy()
top_signed["Direction"] = top_signed["Coefficient"].apply(
    lambda x: "Buy Signal" if x > 0 else "Sell Signal"
)

plt.figure(figsize=(8, 6))
sns.barplot(data=top_signed, x="Coefficient", y="Feature", hue="Direction", dodge=False)

plt.title("Buy/Sell Influence for Top Financial Indicators")
plt.xlabel("Model Weight")
plt.ylabel("Financial Indicator")
plt.tight_layout()
plt.show()
