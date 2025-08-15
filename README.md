# Travel Insurance Claim Prediction

## Project Overview
Predict travel insurance claims to help companies **identify valid claims** and minimize missed valid claims (False Negatives). Includes **EDA, preprocessing, handling class imbalance, and model evaluation**.

---

## Objectives
- Identify travelers likely to file claims.  
- **Minimize False Negatives** – reduce missed valid claims.  
- Understand key features affecting claims.  
- Evaluate various ML models and resampling techniques.

---

## Notebook Contents
1. **EDA & Data Loading** – Explore data, check missing values, visualize distributions.  
2. **Preprocessing** – Handle missing values, encode categorical features, scale numerical features.  
3. **Handling Imbalance** – Apply **SMOTE** for minority class.  
4. **Modeling** – Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, LightGBM.  
5. **Threshold Tuning** – Adjust thresholds (e.g., 0.4) to **prioritize recall and reduce False Negatives**.  
6. **Evaluation** – Confusion matrix, classification report, ROC-AUC, feature importance.  
7. **Conclusion** – Model insights and recommendations.

---

## Key Results
- Most valid claims correctly identified, with focus on **reducing False Negatives**.  
- Features like travel duration, age, and policy price are most influential.  
- Further improvement: alternative algorithms, hyperparameter tuning, and different resampling techniques to enhance recall.

---

## How to Run
1. Clone/download this repository.  
2. Install required libraries:  
   `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `lightgbm`, `shap`  
3. Open and run the notebook in **Jupyter Lab/Notebook** sequentially.

---

## Notes
- Dataset contains traveler info and claims. Ensure **sensitive fields (Gender, Age) are complete**.  
- Prediction thresholds can be adjusted to **prioritize recall** and reduce False Negatives.
