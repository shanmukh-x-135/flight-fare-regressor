# Flight Fare Prediction using Machine Learning

This project predicts airline ticket prices based on flight and passenger features using multiple regression models and ensemble learning. The pipeline includes feature extraction, transformation, and model optimization via cross-validation.

## Dataset
- Source: [Kaggle / Airline dataset]
- Samples: 10,000+
- Features: Categorical & datetime (e.g., source, destination, departure time, stops)

## ML Pipeline
- Data Preprocessing
  - Label Encoding, DateTime Feature Engineering
- Feature Importance (Mutual Information)
- Model Training & Evaluation
  - Models: RandomForest, GradientBoosting, KNN, Lasso
  - Hyperparameter Tuning (RandomizedSearchCV)
- Evaluation: R² Score, RMSE, MAE

## Results
- Best Model: **RandomForest**
- R² Score: **0.86**
- RMSE & MAE: (used but not emphasized in presentation)

## Tech Stack
`Python` `Pandas` `Scikit-learn` `Matplotlib` `Seaborn`

## Key Highlights
- Extracted 10+ new features from datetime columns
- Compared 7 different models with tuning
- Ranked features using mutual information scores
