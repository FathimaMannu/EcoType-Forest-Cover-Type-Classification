# 🌲 EcoType: Forest Cover Classification

This project predicts **Forest Cover Type** using Machine Learning based on geographical and cartographic features such as elevation, soil type, wilderness area, slope, and more.

The project includes:
- Complete ML workflow (EDA → Preprocessing → Modeling → Evaluation)
- Model comparison (RF, DT, KNN, LR, XGBoost)
- Hyperparameter tuning
- Streamlit web application for prediction
- Deployment-ready folder structure

## 📌 Features
- Predicts 7 forest cover types
- Uses a trained ML model (`model.pkl`)
- Uses a scaler (`scaler.pkl`) and label encoder (`label_encoder.pkl`)
- Interactive Streamlit UI
- Includes important plots and model evaluation metrics

## 📁 Folder Structure
```
EcoType/
│── data/
│── notebooks/
│── streamlit_app/
│── models/
│── README.md
│── requirements.txt
│── project_description.docx
```

## 🚀 How to Run
Install dependencies:
```
pip install -r requirements.txt
```

Run Streamlit app:
```
streamlit run app.py
```

## 🧠 Model Details
- Best Model: Random Forest Classifier
- Accuracy: ~85–90% (varies by dataset split)
- Hyperparameter tuning performed with GridSearchCV

## 📦 Files Included
- `app.py` – Streamlit UI
- `model.pkl`, `scaler.pkl`, `label_encoder.pkl`
- `cover_type.csv` – dataset
- `requirements.txt` – dependency list
- `project_description.docx` – full documentation

## ✨ Author
Fathima — EcoType ML Project
