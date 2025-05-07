# 🏥 Healthcare Appointment No-Show Prediction

This project predicts whether a patient will miss a scheduled healthcare appointment using demographic and appointment-related data. Built for IA651 (Spring 2025) by **Bhavani Chendrayan**, the project combines EDA, feature engineering, and classification models.

---

## 📁 Dataset Description

- **Source**: Brazilian public healthcare system  
- **Dataset**: `healthcare_noshows.csv`  
- **Observations**: 110,527  
- **Fields**: 14 (reduced to ~9 features post-cleaning)

### Key Features

| Field            | Description                                        |
|------------------|----------------------------------------------------|
| `Gender`         | Male/Female indicator                              |
| `Age`            | Patient’s age                                      |
| `ScheduledDay`   | When appointment was booked                        |
| `AppointmentDay` | Actual appointment date                            |
| `Scholarship`    | Indicates welfare enrollment                       |
| `Hypertension`   | 0/1 indicator                                      |
| `Diabetes`       | 0/1 indicator                                      |
| `Alcoholism`     | 0/1 indicator                                      |
| `SMS_received`   | Whether a reminder SMS was sent                    |
| `No-show`        | Target variable: 1 = missed, 0 = attended          |

This data allows health centers to anticipate no-shows and optimize their scheduling systems.

---

## 🎯 Prediction Goal

We aim to predict the likelihood of a no-show. This could:
- Enable double-booking strategies
- Reduce idle time in clinics
- Target reminders to high-risk patients

---

## 🔄 Process Overview

The project followed an iterative structure:
- Initial exploration with `pd.get_dummies` led to pipeline issues — corrected by using `OneHotEncoder`.
- Accuracy was misleading due to class imbalance — switched to precision, recall, and F1-score.
- Decision Tree and Logistic Regression showed weaknesses — Random Forest and Neural Networks used.
- Grid search with cross-validation tuned model parameters.

---

## 📊 EDA

### Target: `no_show` (binary classification)

- **X variables**: All features excluding `no_show`
- **Y variable**: `no_show`
- **Type**: Classification
- **Class distribution**: ~20% No-show, 80% Show
- **Observations**: 110,527
- **Features**: 8–10 post-processing

### Notable Distributions

- `days_waited` was negatively skewed — clipped to 0.
- `Age`: right-skewed with outliers (age 0 and 100+)
- `SMS_received`: mostly zero

### Correlation

- Minimal collinearity found
- Used feature importance from Random Forest and Decision Tree

### Feature Importance

| Rank | Feature           |
|------|-------------------|
| 1    | `days_waited`     |
| 2    | `age`             |
| 3    | `sms_received`    |
| 4    | `hypertension`    |

All features were retained due to their independent utility and performance gain.

---

## 🧠 Feature Engineering

### Engineered Features

- `days_waited = AppointmentDay - ScheduledDay`
- `scheduled_dayofweek` and `appointment_dayofweek`
- `age_group` via binning (`Child`, `Teen`, etc.)

### Encoding

- `OneHotEncoder` for categorical fields
- Avoided `pd.get_dummies` for pipeline compatibility

---

## ⚙️ Model Fitting

### Train/Test Split

- 80/20 split using `train_test_split`
- Stratified on `no_show`
- Scaling and encoding embedded in `Pipeline` to avoid leakage

### Data Leakage

- Avoided by dropping ID/date columns
- `StandardScaler` applied only within training folds

### Models Tried

| Model               | Reason                                       |
|---------------------|----------------------------------------------|
| Logistic Regression | Baseline, interpretable                      |
| Decision Tree       | Handles non-linearity, fast                  |
| Random Forest       | Robust and accurate                          |
| Neural Network (MLP)| Captures complex feature interactions        |

---

## 🔍 Hyperparameter Tuning with Grid Search

Used `GridSearchCV` with 5-fold cross-validation:

- **LogReg**: `C`
- **DecisionTree**: `max_depth`, `min_samples_split`
- **Random Forest**: `n_estimators`, `max_depth`
- **MLPClassifier**: `hidden_layer_sizes`, `alpha`, `learning_rate_init`

All models were embedded in pipelines, ensuring preprocessing integrity.

---

## 📏 Validation & Metrics

Given imbalance, we used:
- **Precision**
- **Recall**
- **F1-score**
- `classification_report` and confusion matrix

| Metric     | Reason                                 |
|------------|----------------------------------------|
| Precision  | Avoid false alarms                     |
| Recall     | Catch as many no-shows as possible     |
| F1-score   | Balanced metric                        |

---

## 🧪 Predictions

### Real Examples

| Patient Summary                            | Predicted | Actual |
|--------------------------------------------|-----------|--------|
| 45 y/o, 0-day wait, SMS received           | Show      | Show   |
| 79 y/o, 15-day wait, no SMS                | No-show   | No-show|

### Synthetic Examples

| Hypothetical Patient                      | Prediction |
|-------------------------------------------|------------|
| 25 y/o, 3-day wait, SMS received          | Show       |
| 60 y/o, 20-day wait, chronic illness      | No-show    |

---

## ⚖️ Overfitting / Underfitting

| Model         | Behavior         | Fixes Applied                 |
|---------------|------------------|-------------------------------|
| Logistic Reg. | Underfit         | Used interactions, scaling    |
| Decision Tree | Overfit          | Pruned with `max_depth`       |
| Random Forest | Balanced         | Ensemble effect               |
| Neural Net    | Slight overfit   | Dropout + early stopping      |

---

## 🚀 Production Considerations

- Set up retraining every month with new data
- Threshold tuning required for real-time deployment
- Alert doctors/admins about likely no-shows
- Model fairness: monitor for age/gender bias

---

## 🔭 Going Further

- Include `Neighbourhood` as geospatial feature
- Try advanced models (XGBoost, LightGBM)
- Use SMOTE or Focal Loss for imbalance
- Add historical appointment behavior features

---

## 👩‍💻 Author

**Rishi Chendrayan**
**Lonika Bhavani**  
IA651 – Spring 2025  
Department of Data Science

---
