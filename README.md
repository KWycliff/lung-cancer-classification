# Machine Learning Based Lung Cancer Risk Assessment and Classification

A machine learning project that classifies lung cancer risk from demographic, lifestyle and symptom information. It covers data preprocessing, class-imbalance handling, model comparison and evaluation, feature importance analysis and deployment through a small web app.

## Problem

Lung cancer is much easier to treat when it is detected early.

Imaging based screening such as low dose CT can catch the disease early, but it needs specialist equipment and expertise, which are expensive and not available everywhere. That leaves a gap. If a short questionnaire covering things like smoking, age, and symptoms can estimate someone's risk, it could help decide who gets referred for the more expensive tests first. That is the question this project explores.

- **Target:** LUNG_CANCER (YES / NO), a binary classification task
- **Features (15):** GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL CONSUMING, COUGHING, SHORTNESS OF BREATH, SWALLOWING DIFFICULTY, CHEST PAIN

AGE is the only continuous feature. The rest are binary, coded 1/2, and GENDER is M/F.

## Dataset

- **Source:** [Lung Cancer, Kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer) (survey_lung_cancer.csv)
- **Raw size:** 309 rows x 16 columns, no missing values
- **Duplicates:** 33 duplicate rows, removed, leaving 276 rows
- **Class balance:** strongly imbalanced towards YES
- **Split:** 80/20 stratified train/test
  - Train: 220 rows (190 YES / 30 NO)
  - Test: 56 rows (48 YES / 8 NO)

## Methodology

1. **Preprocessing:** check for missing values, drop duplicates, inspect data types, and encode GENDER (F = 0, M = 1) and LUNG_CANCER (NO = 0, YES = 1) with explicit mappings.
2. **Exploration:** age distribution, gender distribution and class balance plots.
3. **Imbalance handling:** SMOTENC oversampling on the training split only (190 YES / 190 NO), so the test set stays untouched and there is no data leakage. AGE is treated as continuous and all other features as categorical.
4. **Model comparison:** Logistic regression, Linear SVC, k-NN (k=5) and Random forest (200 trees), trained on the resampled training data and evaluated on the test set with classification reports and confusion matrices.
5. **Cross validation:** because the test set holds only 8 NO cases, the models are also evaluated with repeated stratified 5-fold cross validation (5 repeats) on the full dataset. SMOTENC is applied inside each training fold through an imblearn pipeline, so synthetic samples never leak into the validation folds. Metrics: accuracy, macro F1 and ROC AUC.
6. **Model selection and interpretation:** the final model is chosen on cross validated macro F1 and ROC AUC. A ROC curve is plotted on the test set, and permutation feature importance (computed on the original training split) shows which features the model relies on.
7. **Deployment:** the selected model is saved with **joblib** to **web_app/lung_cancer_risk_model.pkl** for use by a web app.

## Results

Models trained on the SMOTENC resampled training data and evaluated on the test data.

| Model                   | Accuracy | Macro F1 |
|-------------------------|---------:|---------:|
| **Logistic regression** | **0.93** | **0.85** |
| Linear SVC              | 0.91     | 0.78     |
| k-NN (k=5)              | 0.86     | 0.77     |
| Random forest           | 0.91     | 0.81     |

### Repeated stratified 5-fold cross validation (5 repeats, SMOTENC inside each fold)

| Model                   | Accuracy | Macro F1 | ROC AUC |
|-------------------------|---------:|---------:|--------:|
| **Logistic regression** | **0.91** | **0.81** | **0.94** |
| Random forest           | 0.89     | 0.77     | 0.92    |
| Linear SVC              | 0.88     | 0.77     | 0.91    |
| k-NN (k=5)              | 0.80     | 0.68     | 0.82    |

Logistic regression performed best on both evaluations and is the model saved for deployment. On the test set it reaches a ROC AUC of 0.956. The saved model is the one trained on the SMOTENC resampled training split.

### Feature importance

Permutation importance on the logistic regression model gave the largest drops in macro F1 for PEER_PRESSURE, ALLERGY, CHRONIC DISEASE, FATIGUE and ALCOHOL CONSUMING. AGE, SWALLOWING DIFFICULTY and GENDER had the smallest effect. The standard deviations are large relative to the means for most features, so the ranking should be read as indicative rather than exact. These are associations in this dataset, not causal effects.

## Limitations

- **Small dataset, tiny minority class.** Only 276 rows remain after removing duplicates, with 38 NO cases in total (30 in training, 8 in the test set). A single test split is therefore noisy, which is why cross validation is also reported.
- **Synthetic samples.** SMOTENC generates the extra NO training records, so the model is trained on partly artificial data.
- **Feature importance.** Some of the highest ranked features (for example PEER_PRESSURE and ALLERGY) are not established clinical risk factors, while SMOKING ranks low. This points to dataset-specific patterns, so the model should not be expected to generalise to real screening populations without external validation.

## Project structure

```
lung-cancer-classification/
├── dataset/
│   └── survey_lung_cancer.csv
├── lung-cancer-risk-classification.ipynb
├── web_app/
│   ├── lung_cancer_risk_model.pkl
│   ├── predict_lung_cancer_app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── css/
│           └── styles.css
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

Open `lung-cancer-risk-classification.ipynb` and run all cells. The last cell saves the model to `web_app/lung_cancer_risk_model.pkl`.

### 3. Run the web app

```powershell
cd web_app
python predict_lung_cancer_app.py
```

Open http://127.0.0.1:5000 in your browser, answer the questions and select **Check my risk**.

## Acknowledgements

- Web app structure based on the [ML-For-Beginners web app tutorial](https://microsoft.github.io/ML-For-Beginners/#/3-Web-App/1-Web-App/README) by Microsoft.
- Parts of the notebook code (the model evaluation function) were developed with assistance from ChatGPT. The cross validation, permutation feature importance and web app code were developed with assistance from Claude (Anthropic).