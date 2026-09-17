# Problem Statement

## Business Problem

Customer churn — when a subscriber cancels their service — is expensive to
recover from: acquiring a new customer typically costs far more than
retaining an existing one. This project builds a binary classification
model that predicts which telecom customers are likely to churn based on
their account details, subscribed services, and billing history. A
retention team can use these predictions to proactively target at-risk
customers with tailored offers or outreach before they cancel, rather than
reacting after the fact.

## Dataset

The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(IBM sample dataset) contains **7,043 customer records** and **21 columns**:

- **Target**: `Churn` (Yes/No) — whether the customer left within the last month.
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
- **Account info**: `tenure` (months), `Contract` type, `PaperlessBilling`,
  `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- **Services subscribed**: `PhoneService`, `MultipleLines`, `InternetService`,
  `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
  `StreamingTV`, `StreamingMovies`.
- **Identifier**: `customerID` (dropped before modeling — not predictive).

Class distribution: **73.5% No-churn / 26.5% Churn** — moderately imbalanced,
enough to require stratified sampling and metrics beyond accuracy.

`TotalCharges` is stored as a string and has 11 blank values corresponding to
customers with 0 months of tenure (new customers who haven't been billed
yet) — these need explicit handling rather than a naive numeric cast.

## Why This Dataset Fits the ML Lifecycle

This dataset is a good fit for demonstrating end-to-end ML engineering
skills because it combines several realistic challenges in a small,
fast-iterating package:

- **Mixed feature types**: a blend of binary, categorical, and continuous
  numeric features, requiring real encoding and preprocessing decisions.
- **Data quality issues**: hidden missing values (blank strings, not NaN)
  that must be found and handled deliberately, not just dropped blindly.
- **Class imbalance**: churn is the minority class, so evaluation must go
  beyond accuracy (precision/recall/F1/ROC-AUC) and training may need
  class weighting or resampling.
- **Business-relevant framing**: the prediction target maps directly to an
  actionable business decision (who to target for retention), which
  motivates thinking about precision/recall tradeoffs in terms of cost —
  not just optimizing a metric in isolation.
- **Small enough to iterate fast**: ~7K rows trains in seconds, keeping the
  focus on data preparation, evaluation rigor, and service-building rather
  than on managing training infrastructure.
