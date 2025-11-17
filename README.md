# 🌲 EcoType: Forest Cover Type Classification

A Machine Learning project that predicts **forest cover type** (7 classes) using U.S. Forest Service cartographic data.

---

## 📌 Project Overview  
The objective of this project is to build a classification model that predicts the type of forest cover based on features like elevation, hydrology distances, slope, soil type, and wilderness area information.

This supports:
- 🌿 Environmental monitoring  
- 🔥 Wildfire risk assessment  
- 📡 Land cover mapping  
- 🌎 Forest resource management  

---

## 📊 Dataset Information
- **Rows:** 145,891  
- **Columns:** 55  
- **Target Variable:** `Cover_Type` (7 classes)  
- **Features:**  
  - 10 numerical columns  
  - 4 Wilderness Areas (binary)  
  - 40 Soil Types (binary)

Dataset includes:  
Elevation, Aspect, Slope, Distances to hydrology/roadways/fire points, Soil & Wilderness categories.

---

## 🛠️ Workflow
1. Data Collection  
2. Data Understanding  
3. Data Cleaning  
4. Exploratory Data Analysis  
5. Feature Engineering  
6. Handling Class Imbalance  
7. Model Building  
8. Hyperparameter Tuning  
9. Model Evaluation  
10. Streamlit App Deployment  

---

## 🧠 Models Used
The following models were trained & compared:

- Logistic Regression  
- Decision Tree  
- Random Forest (Best)  
- XGBoost  
- KNN  

---

## 🏆 Best Model  
**Random Forest Classifier**

Saved under `models/model.pkl`

Supporting files:
- `scaler.pkl`
- `label_encoder.pkl`

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- Feature Importance  

---

## 🚀 Streamlit Application  
### How to run locally:

```bash
pip install -r requirements.txt
streamlit run app/app.py
