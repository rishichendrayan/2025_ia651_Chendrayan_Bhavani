
# 🏥 Predicting Patient No-Shows for Healthcare Appointments

## 📊 Dataset Description

This project utilizes the **"No-show appointments" dataset**, which includes information on **medical appointment records in Brazil**. Each record indicates whether a patient showed up for a scheduled appointment and includes demographics, health-related variables, and SMS reminder information.

### 📌 Key Fields:
- `Gender`: Binary gender (F/M)
- `Age`: Patient age
- `Scholarship`: Indicates welfare program participation
- `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`: Health conditions (binary)
- `SMS_received`: Whether the patient received an SMS reminder
- `ScheduledDay`, `AppointmentDay`: Date fields to compute appointment delay
- `Showed_up`: Target variable (Yes/No → 1/0)

## 🎯 Project Objective

We aim to **predict whether a patient will attend a medical appointment** using demographic and health-related features.

### 📈 Practical Use:
- Helps clinics **optimize scheduling**
- Enables targeted **reminders or interventions**
- Reduces **resource waste** due to no-shows

## 🔄 Process Overview

The project followed a structured ML workflow:
1. Data loading and cleaning
2. Feature engineering (`date.diff`, binary encoding)
3. Exploratory Data Analysis (EDA)
4. Model training using multiple classifiers (Logistic Regression, Random Forest, Neural Network)
5. Evaluation via confusion matrix, ROC, and PR curves

## 🧭 Narrative & Iterations

- Initially attempted full feature set
- Created `days_waited` as the difference between `AppointmentDay` and `ScheduledDay`
- Pivoted from simple classifiers to MLP due to nonlinear relationships
- Handled boolean and datetime types explicitly
- Added dropout layers to reduce overfitting in Neural Network

## 🔍 Exploratory Data Analysis (EDA)

- **Target (Y)**: `Showed_up` (0 = No, 1 = Yes)
- **Features (X)**: Age, Gender, Scholarship, Medical conditions, `days_waited`, SMS
- **Classification task**
- **Observations**: ~110,000, **Features**: ~10 after cleanup
- **Imbalanced features**:
  - `Alcoholism`, `Handcap`: Heavily skewed toward 0
  - `SMS_received`: 1 ≈ 30%
- **Target Distribution**:
  - Showed Up: ~80%
  - No-Show: ~20%

### 📌 Correlations
- Moderate correlation between `Hipertension` and `Diabetes`
- `days_waited` negatively correlated with attendance

## 🧰 Feature Engineering

- Dropped ID/date fields after extracting useful info (`days_waited`)
- Converted booleans to 0/1
- Label encoded `Gender`
- No need for One-Hot due to binary structure

## 🤖 Model Fitting

- Used train/test split: 80/20
- Scaled features using `StandardScaler`
- Models evaluated:
  - Logistic Regression
  - Random Forest
  - **Neural Network (MLPClassifier, Keras Sequential)**

### ⚠️ Leakage Consideration:
- Dates used only to compute delay, then dropped
- No patient ID or neighborhood used

## 🔬 Hyperparameter Tuning

- Used `GridSearchCV` on `MLPClassifier`
  - Explored activation functions, `alpha`, solver, layer sizes
- Keras model used:
  - 2 hidden layers (64, 32)
  - Dropout (0.3, 0.2)
  - Sigmoid output, binary crossentropy loss

## 📏 Validation & Metrics

- **Primary metrics**: Accuracy, Precision, Recall, F1
- **ROC-AUC** and **PR curves** plotted
- Confusion matrix revealed:
  - High true positive rate
  - Moderate false positive rate (some no-shows predicted as present)

### Sample Metrics:
- Accuracy: ~82%
- Precision: ~84%
- Recall: ~90%
- F1: ~87%

## 🧪 Prediction Examples

### Real:
- [✓] 45-year-old female, diabetic, received SMS → Showed up
- [✗] 19-year-old male, long wait time, no SMS → No-show

### Synthesized:
- [✓] 65-year-old hypertensive, short wait → Showed up
- [✗] 23-year-old no conditions, 20-day wait → No-show

## ⚖️ Overfitting / Underfitting

- Early signs of overfitting in the base MLP
- Mitigated using:
  - Dropout layers
  - Validation split
  - Reduced epochs + EarlyStopping

## 🚀 Production & Deployment

### Cautions:
- Must retrain regularly as **patient behavior changes**
- SMS impact could vary across **regions or times**

### Deployment advice:
- Deploy behind an API endpoint for hospital systems
- Use explainability tools (e.g., SHAP) in clinical settings

## 🔮 Going Further

- Incorporate location, prior appointment history
- Use time-of-day or day-of-week features
- More advanced models: XGBoost, ensemble stacking
- Balance classes using SMOTE or class weights
