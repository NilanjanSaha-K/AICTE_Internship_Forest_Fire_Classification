# 🔥 Fire Types Classification - India (MODIS Data)

This repository contains a complete pipeline for classifying different types of fire events in India using MODIS satellite data from the years 2021, 2022, and 2023. It includes data preprocessing, model training, evaluation, and deployment through a user-friendly Streamlit interface.

---

## 📁 Contents

- `Classification_of_fire_types_Nilanjan.ipynb`  
  Main Jupyter Notebook for preprocessing, EDA, feature engineering, model training, SMOTE balancing, and evaluation.

- `modis_2021_India.csv`  
  MODIS fire data for India (Year: 2021)

- `modis_2022_India.csv`  
  MODIS fire data for India (Year: 2022)

- `modis_2023_India.csv`  
  MODIS fire data for India (Year: 2023)

- `best_fire_detection_model.pkl`  
  Saved classification model using `joblib`.

- `scaler.pkl`  
  StandardScaler used during training for feature normalization.

- `app.py`  
  Streamlit frontend script for real-time fire type prediction.

---

## 🔍 Objective

To classify fire types detected via satellite imagery into meaningful categories such as:
- 🌲 Vegetation Fires  
- 🏭 Static Land Sources  
- 🌊 Offshore Fires  

using machine learning models trained on remote sensing data.

---

## 🧠 Technologies Used

- **Python 3**
- **Pandas, NumPy** – Data preprocessing
- **Matplotlib, Seaborn** – Data visualization
- **Scikit-learn** – Model training and evaluation
- **SMOTE** – Handling imbalanced datasets
- **Joblib** – Model serialization
- **Streamlit** – Frontend deployment

---

## 📊 Key Features

- ✅ MODIS fire data aggregation (2021–2023)
- ✅ Outlier removal using IQR method
- ✅ Label encoding and feature scaling
- ✅ Addressed class imbalance with SMOTE
- ✅ Trained classification model (e.g., Random Forest)
- ✅ Streamlit web app for live prediction

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NilanjanSaha-K/AICTE_Internship_Forest_Fire_Classification.git
cd AICTE_Internship_Forest_Fire_Classification

2. Install required packages
pip install -r requirements.txt

If requirements.txt is missing, install manually:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib streamlit

3. Launch the Streamlit app
streamlit run app.py