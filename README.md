# My Data Science Journey 🚀

This repository documents my daily progress in mastering Data Science and Machine Learning. 

---

## # Day 1: Data Science - Basics of Classification 🚻

For **Day 1**, I focused on understanding the **fundamentals of Classification** in Machine Learning. I explored the `scikit-learn` library to build a simple "Gender Predictor" based on physical attributes.

### 📋 Project Overview
My goal was to compare how different classification algorithms handle the same small dataset. I manually created a dataset and trained 5 different models to see if they would agree on the prediction for a new user.

* **Features:** `[Height (cm), Weight (kg), Shoe Size]`
* **Target:** `Gender` ('male' or 'female')

### 🛠️ Tech Stack
* **Language:** Python
* **Library:** Scikit-Learn (sklearn)

### 🧠 Models Implemented
* **Decision Tree Classifier:** Uses a flowchart-like structure to make decisions.
* **K-Nearest Neighbors (KNN):** Classifies based on the closest "k" data points.
* **Support Vector Machine (SVM):** Finds the optimal boundary to separate the classes.
* **Random Forest Classifier:** Uses multiple Decision Trees for better accuracy (Ensemble).
* **Gaussian Naive Bayes:** Uses probability to predict.

### 📝 Learning Outcomes
* Learned how to import specific models from sklearn.
* Understood the `.fit()` method for training and `.predict()` for prediction.
* Observed that different models might give different results depending on data complexity.

---

## # Day 2: Linear Regression - The Slope

For **Day 2**, I moved from Classification to **Regression**. While classification categorizes data (A vs B), Regression predicts continuous values (like prices, temperature, or scores).

### 📋 Project: Score Predictor
I built a simple logic that predicts a student's test score based on the number of hours they studied.

### 🧠 The Math Behind Linear Regression
The model tries to fit a straight line through the data points using the equation:
$$y = mx + c$$
* **y**: The Score (Target)
* **x**: The Hours Studied (Feature)
* **m**: The Slope (How much the score goes up per hour)
* **c**: The Intercept (The score if you study 0 hours)

### 📊 Visuals
The code generates a graph showing the actual student scores (Blue Dots) and the machine's learned pattern (Red Line).

---

## # Day 3: Logistic Regression - The "S" Curve 

For **Day 3**, I learned **Logistic Regression**. Despite the name "Regression," this is actually a **Classification** algorithm used to predict binary outcomes (Yes/No, True/False, 0/1).

### 📋 Project: Insurance Predictor
I built a model that predicts whether a customer will buy insurance based on their age.

### 🧠 The Math: Sigmoid Function
Unlike Linear Regression which fits a straight line, Logistic Regression applies the **Sigmoid Function** to squash the output between 0 and 1:
$$S(x) = \frac{1}{1 + e^{-x}}$$
* If probability > 50%, it predicts **1 (Yes)**.
* If probability < 50%, it predicts **0 (No)**.

### 📊 Visuals
The red line in the graph represents the **Probability**.
* **Bottom of S:** Low probability (Younger people).
* **Top of S:** High probability (Older people).

---

## # Day 4: K-Nearest Neighbors (KNN) - Based on your Surroundings

For **Day 4**, I explored **KNN**. Unlike other models that "learn" a complex math formula, KNN is a "Lazy Learner"—it simply memorizes the entire dataset. When we ask for a prediction, it looks for the 'K' most similar data points (neighbors) and takes a vote.

### 📋 Project: Interactive T-Shirt Size Predictor
I built a CLI (Command Line Interface) tool that predicts whether a user needs a Small or Large T-shirt.
1.  The program asks the user for their Height and Weight.
2.  It compares the user to a dataset of 18 existing customers.
3.  It predicts the size and plots the user as a **Star** on the graph.

### 🧠 Key Concept: Euclidean Distance
KNN calculates the straight-line distance between the new point and every other point to find the closest matches.
* **Small K (e.g., 1):** Highly sensitive to noise.
* **Large K (e.g., 5):** More stable, but might miss local details.
* *I used K=3 for this project.*

### 📊 Visuals
* 🔵 **Blue Dots:** Small Size
* 🔴 **Red Dots:** Large Size
* ⭐ **Yellow Star:** Entered value (You)

---

## # Day 5: Random Forest 

For **Day 5**, I learned **Random Forest**, an "Ensemble" method. Instead of relying on a single Decision Tree (which can be biased), Random Forest creates hundreds of trees and makes them vote on the final answer.

### 📋 Project: HR Hiring Predictor
I built an AI that mimics a Hiring Manager.
* **Input:** Experience, Test Score, Interview Score.
* **Output:** Hired vs. Rejected.

### 🧠 Key Concept: Feature Importance
One of the best features of Random Forest is that it is "interpretable." It can calculate exactly which input feature had the biggest impact on the decision.
* *Example:* The model revealed that **Interview Score** might be more important than **Experience**.

---

## # Day 6: SVM (Support Vector Machine) - Breast Cancer Prediction 

For **Day 6**, I tackled a high-stakes classification problem: Breast Cancer Detection.

### 📋 Project Overview
The system looks at patient details like age, tumor size, and medical history to predict whether a breast tumor is **Malignant** (Dangerous) or **Benign** (Not Dangerous).

### 🌟 Why This Project Is Important
* Early detection saves lives.
* Reduces human error in diagnosis.

### 📊 The Data
* **Source:** [Kaggle Breast Cancer Prediction Dataset](https://www.kaggle.com/datasets/fatemehmehrparvar/breast-cancer-prediction)
* **Process:** I cleaned the data, converted text labels to numbers, and split the data for training vs testing.

### 📈 Results
* Generated a **Confusion Matrix** to visualize False Positives vs. False Negatives.
* Prioritized reducing False Negatives (missing a cancer diagnosis) over raw accuracy.

---

## # Day 7: K-Means Clustering - Unsupervised Learning 

For **Day 7**, I shifted from Supervised Learning to **Unsupervised Learning**. I used the **K-Means Clustering** algorithm to find hidden patterns in raw data without any labels or guidance.

### 🧠 The Concept: "The Messy Room Party"
Imagine a party where strangers naturally separate into groups (Dancers, Business people, Foodies).
* **Supervised Learning:** The host tells everyone where to stand.
* **Unsupervised Learning:** The guests group themselves based on similarity.

### 📋 Project: Mall Customer Segmentation
* **Dataset:** [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
* **Goal:** Identify specific "Customer Personas" (e.g., High Income + High Spenders) for marketing.

### 🛠️ The Pipeline
1.  **The Elbow Method:** Calculated WCSS to scientifically find that **K=5** was the optimal number of groups.
2.  **Model Training:** Trained K-Means with 5 clusters.
3.  **Visualization:** Plotted the 5 distinct market segments and their Centroids.
4.  **Prediction:** Classifies new customers into these established personas.

---

## # Day 8: XGBoost - The Ferrari of Machine Learning 

For **Day 8**, I mastered **XGBoost (Extreme Gradient Boosting)**, the algorithm that dominates Kaggle competitions. I moved from "Bagging" (Random Forest) to "Boosting" (Sequential Learning).

### 📋 Project: Bank Customer Churn Prediction
I built an AI system for a Bank to identify customers who are at high risk of leaving ("Churning").
* **Goal:** Predict `Exited` (1 = Left, 0 = Stayed).
* **Business Value:** Identifying at-risk customers allows the bank to offer incentives *before* they leave.

### 🧠 Key Concept: Sequential Learning
XGBoost works like a student taking practice exams:
1.  **Model 1** takes the test and fails hard questions.
2.  **Model 2** studies *only* those hard questions (mistakes).
3.  **Model 3** fixes the mistakes of Model 2.
This **Gradient Boosting** approach achieves higher accuracy than standard trees.

### 📊 Model Performance
* **Accuracy:** **86.55%** (Outperforming Random Forest).
* **Confusion Matrix:** Successfully identified **193** high-risk customers who were about to leave.
* **Feature Importance:** Discovered that **Age** and **Number of Products** are the biggest drivers of churn.

### 🛠️ Tech Stack
* **Library:** `xgboost`

---

# Day 9: LightGBM - The Tesla of Machine Learning ⚡

For **Day 9**, I mastered **LightGBM (Light Gradient Boosting Machine)**, developed by Microsoft.
While XGBoost is powerful (the "Ferrari"), LightGBM is designed for the Big Data era. It is **10x faster**, consumes less memory, and handles categorical data automatically.

## 📋 Project: Airline Passenger Satisfaction ✈️
I built an AI model to analyze passenger feedback and predict customer satisfaction.
* **Dataset:** 100,000+ real passenger records (Kaggle).
* **Goal:** Predict `Satisfaction` (1 = Satisfied, 0 = Dissatisfied/Neutral).
* **Challenge:** Processing massive data with mixed text and numbers instantly.

## 🧠 Key Concept: Leaf-Wise Growth 🌿
Unlike XGBoost, which grows trees level-by-level (horizontally), LightGBM grows **Leaf-Wise** (vertically).
* It finds the most promising branch and digs deep immediately.
* **Result:** Faster training and higher accuracy on large datasets.
* **Native Handling:** LightGBM can read text categories (like "Business Class") directly without needing One-Hot Encoding.

## 🛠️ The Tech Stack
* **Library:** `lightgbm` (Native API).
* **Data Structure:** `lgb.Dataset` (Binary optimized format).
* **Configuration:** Used a `params` dictionary for granular control.

## ⚙️ The "Cockpit" Configuration
Instead of default settings, I tuned the model manually:
```python
params = {
    'objective': 'binary',        # Predicting Yes/No
    'metric': 'binary_logloss',   # Confidence scoring
    'boosting_type': 'gbdt',      # Standard Gradient Boosting
    'num_leaves': 31,             # Complexity control
    'learning_rate': 0.05,        # Slow & Steady learning
    'feature_fraction': 0.9       # Prevent overfitting
}
```
---


# DAY-10 : 🌍 Air Quality Index (AQI) Predictor

An End-to-End Machine Learning project that predicts the **Air Quality Index (AQI)** based on pollutant levels and city data.
This project moves beyond simple analysis by deploying the model as a live **FastAPI** backend, ready for integration into web or mobile apps.

## 🚀 Key Features
* **Machine Learning:** Uses **Random Forest Regressor** to predict continuous AQI values.
* **Optimization:** Implements **GridSearchCV** to find the perfect hyperparameters automatically.
* **City-Aware:** Uses **Label Encoding** to adjust predictions based on the specific city (e.g., Delhi vs. Bangalore).
* **Visualization:** Automatically generates charts for Feature Importance and Prediction Error.
* **Deployment:** A professional **FastAPI** server that classifies health risk (Good 🟢 to Severe ☠️).

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **ML Library:** Scikit-Learn (Random Forest, GridSearch)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **API Framework:** FastAPI, Uvicorn
* **Serialization:** Joblib (for saving/loading models)

## 📂 Project Structure
```text
├── city_day.csv            # The Dataset (Kaggle)
├── train_model.py          # Script to Train, Tune, and Save the Model
├── main.py                 # FastAPI Server script
├── aqi_model.pkl           # Saved Model + Encoder (Generated after training)
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```
⚙️ Installation
Clone the repository (or download files).
Install dependencies:
Bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib fastapi uvicorn
Get the Data: Ensure you have the city_day.csv file in the folder.
Source: Kaggle: Air Quality Data in India

🧠 Phase 1: Training the Model
Run the training script to process data, tune the model, and generate visualizations.
Bash
python train_model.py
What happens?
Cleans missing values using Median Imputation.
Encodes City names into numbers.
Runs GridSearchCV to find the best n_estimators and max_depth.
Saves three plots: pollutant_effect.png, Actual_vs_Predicted.png, Truth_Vs_Predict.png.
Saves the final artifact: aqi_model.pkl.

🌐 Phase 2: Running the API
Once aqi_model.pkl is created, launch the server:

Bash
python -m uvicorn main:app --reload
Server URL: http://127.0.0.1:8000
Interactive Docs: http://127.0.0.1:8000/docs
🧪 How to Test (API usage)
Go to http://127.0.0.1:8000/docs
Click POST /predict_aqi → Try it out.
Paste this JSON:
```
JSON
{
  "pm25": 180.0,
  "pm10": 250.0,
  "no2": 60.0,
  "co": 1.5,
  "so2": 10.0,
  "o3": 45.0,
  "city": "Delhi"
}
```
Response:
```
JSON
{
  "City": "Delhi",
  "Predicted_AQI": 312,
  "Status": "Very Poor 🔴"
}
```
📊 Model Performance
Baseline Accuracy: ~88% (R2 Score)
Tuned Accuracy: ~91% (after GridSearchCV)
Key Insight: PM2.5 and PM10 were identified as the most critical drivers of AQI.
