# Machine Learning Based Lung Cancer Risk Assessment and Classification

A machine learning project that classifies lung cancer risk from simple, self-reported demographic, lifestyle and symptom information. It covers data preprocessing, class-imbalance handling, model comparison, feature importance analysis and deployment through a small web app.

## Problem

Lung cancer is much easier to treat when it is detected early.

Imaging based screening such as low dose CT can catch the disease early, but it needs specialist equipment and expertise, which are expensive and not available everywhere. That leaves a gap. If a short questionnaire covering things like smoking, age, and symptoms can estimate someone's risk, it could help decide who gets referred for the more expensive tests first. That is the question this project explores.

- **Target:** LUNG_CANCER (YES / NO), a binary classification task
- **Features (15):** GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL CONSUMING, COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN

AGE is the only continuous feature. The rest are binary, coded 1/2 and GENDER is M/F.

## Dataset

- **Source:** [Lung Cancer, Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) (survey_lung_cancer.csv)
- **Raw size:** 309 rows x 16 columns, no missing values
- **33 duplicates:**
- **Split:** 80/20 stratified train/test 

## Methodology

1. **Preprocessing:** check for missing values, drop duplicates, inspect data types, and encode GENDER and LUNG_CANCER with LabelEncoder.
2. **Exploration:** age distribution, gender distribution and class balance plots.
3. **Imbalance handling:** SMOTENC oversampling on the training split only (190 YES / 190 NO), so the test set stays untouched and there is no data leakage.
4. **Model comparison:** Dummy classifier (baseline), Logistic regression, Linear SVC, k-NN and Random forest, evaluated with classification reports and confusion matrices.
5. **Interpretation:** permutation feature importance (drop in macro F1 when a feature is shuffled) and a ROC curve for the selected model.
6. **Deployment:** the selected model is saved with **joblib** to **web_app/lung_cancer_risk_model.pkl** for use by a web app.

## Results

Models trained on the SMOTENC resampled training data and evaluated on test data

| Model               | Accuracy | Macro F1 |
|---------------------|---------:|---------:|
| Dummy classifier    | 0.14     | 0.12     |
| **Logistic regression** | **0.93** | **0.85** |
| Linear SVC          | 0.91     | 0.81     |
| k-NN (k=30)         | 0.73     | 0.63     |
| Random forest       | 0.91     | 0.81     |

Logistic regression performed best and is the model currently saved for deployment. 

## limitations

- **Small dataset, tiny minority class.** 
- **Synthetic samples.** 

## Project structure

```
lung-cancer-classification/
├── dataset/
│   └── survey_lung_cancer.csv
├── lung-cancer-risk-classification.ipynb
├── web_app/
│   └── lung_cancer_risk_model.pkl
├── requirements.txt
└── README.md
```

## Getting started

### 1. Set up the environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the notebook

Open `lung-cancer-risk-classification.ipynb` and run all cells.

### 3. Run the web app

```powershell
cd web_app
python predict_lung_cancer_app.py
```

Open http://127.0.0.1:5000 in your browser, answer the questions and select **Check my risk**.

## Acknowledgements

- Web app structure based on the [ML-For-Beginners web app tutorial](https://microsoft.github.io/ML-For-Beginners/#/3-Web-App/1-Web-App/README) by Microsoft.
- Parts of the notebook code were developed with assistance from ChatGPT, and the web app and feature importance code with assistance from Claude (Anthropic).
