# Machine Learning-Based Lung Cancer Risk Assessment and Classification

A machine learning project that classifies lung cancer risk from lifestyle and symptom survey data, with a simple Flask web app for making predictions.

## Project overview

The project covers the full workflow:

1. **Data preprocessing:** cleaning, duplicate removal and encoding.
2. **Model training and evaluation:** comparing several classifiers.
3. **Model interpretation:** permutation feature importance.
4. **Deployment:** a Flask web app that loads the saved model.

## Dataset

- Source: [Lung Cancer survey dataset (Kaggle)](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- Features: gender, age, smoking, yellow fingers, anxiety, peer pressure, chronic disease, fatigue, allergy, wheezing, alcohol consumption, coughing, shortness of breath, swallowing difficulty, chest pain.
- Target: `LUNG_CANCER` (YES/NO). The classes are imbalanced, with far more YES than NO records.
- Yes/no features are coded 1 = No, 2 = Yes.

## Method

- 80/20 stratified train/test split (`random_state=42`).
- Class imbalance handled with **SMOTENC**, applied to the training set only so the test set stays untouched.
- Models compared: Dummy baseline, Logistic Regression, Linear SVC, k-NN, Random Forest.
- Hyperparameters of Logistic Regression and Random Forest tuned with `GridSearchCV` (5-fold stratified CV, macro F1).
- Metrics: accuracy, macro precision/recall/F1, weighted F1, confusion matrices, ROC-AUC.

## Results

Tuned models evaluated on the held-out test set (56 records, 8 negative and 48 positive):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest (selected) | 0.946 | 0.905 |
| Logistic Regression | 0.893 | 0.755 |

The selected Random Forest reaches a ROC-AUC of 0.964.

**Feature importance (permutation, macro F1):** alcohol consumption, yellow fingers and coughing contributed most. Smoking, gender, age, peer pressure and chronic disease had negligible effect on test performance.

## Repository structure

```
lung-cancer-classification/
├── lung-cancer-risk-classification.ipynb
├── dataset/
│   └── survey_lung_cancer.csv
├── requirements.txt
└── web_app/
    ├── predict_lung_cancer_app.py
    ├── lung_cancer_random_forest.pkl
    ├── templates/
    │   └── index.html
    └── static/
        └── css/
            └── styles.css
```

## Getting started

### 1. Set up the environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the notebook

Open `lung-cancer-risk-classification.ipynb` and run all cells. The last cells save the trained model as `lung_cancer_random_forest.pkl`

### 3. Run the web app

```powershell
cd web_app
pip install -r requirements.txt
python predict_lung_cancer_app.py
```

Open http://127.0.0.1:5000 in your browser, answer the questions and select **Check my risk**.

## Acknowledgements

- Dataset: Kaggle, "Lung Cancer" survey dataset.
- Web app structure based on the [ML-For-Beginners web app tutorial](https://microsoft.github.io/ML-For-Beginners/#/3-Web-App/1-Web-App/README) by Microsoft.
- Parts of the notebook code were developed with assistance from ChatGPT, and the web app and feature importance code with assistance from Claude (Anthropic).
