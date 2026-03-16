# Credit Risk Classification

Binary classification on the German Credit Risk dataset — predicting whether a loan applicant is a good or bad credit risk.

---

## Objective

The core question is simple: given a loan applicant's financial profile, can we predict whether they'll default? Getting this wrong in the wrong direction (approving a bad applicant) costs the bank money, so the model needs to be evaluated with that asymmetry in mind.

---

## Dataset

- **Source:** [UCI Machine Learning Repository – Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Records:** 1,000 applicants
- **Features:** 20 (mix of numerical and categorical — account status, loan duration, credit history, employment, etc.)
- **Target:** `1` = Good credit risk, `2` = Bad credit risk (re-mapped to `0` and `1` respectively)
- **Class split:** 70% good, 30% bad

> The dataset is not included in this repo. Download `german.data` from the UCI link above and place it in the `data/` folder.

---

## Project Structure

```
credit-risk-project/
├── data/                   # Raw dataset (not tracked by git)
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis
├── src/
│   ├── preprocess.py       # Data loading and feature engineering
│   ├── train.py            # Model training (Logistic Regression + Random Forest)
│   └── evaluate.py         # Evaluation metrics and plots
├── README.md
└── requirements.txt
```

---

## Methodology

1. **Preprocessing** — one-hot encode categoricals, scale numericals, stratified train/test split, `class_weight='balanced'` to handle the 70/30 imbalance
2. **EDA** — class distribution, feature distributions by target, correlation with credit risk
3. **Logistic Regression** — baseline, easy to interpret
4. **Random Forest** — step up in performance, also gives feature importances
5. **Evaluation** — ROC AUC, precision, recall, F1, confusion matrix

---

## Results

| Model               | ROC AUC | Precision (Bad) | Recall (Bad) | F1 (Bad) |
|---------------------|---------|-----------------|--------------|----------|
| Logistic Regression | ~0.77   | ~0.62           | ~0.68        | ~0.65    |
| Random Forest       | ~0.80   | ~0.65           | ~0.65        | ~0.65    |

*Results will vary slightly depending on random seed and train/test split.*

---

## Business Interpretation

The metric that matters most here is **recall on the bad class** — missing a bad applicant means approving a loan that defaults. Random Forest gets a slightly better AUC, but Logistic Regression is easier to explain to a compliance team and its coefficients map directly to applicant features. In a real setting you'd probably also tune the decision threshold below 0.5 to catch more bad applicants, accepting more false positives in exchange.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess + train + evaluate
python src/train.py
python src/evaluate.py
```

---

## Next Steps

- Tune the decision threshold using a cost matrix (FN is more expensive than FP in this context)
- Try XGBoost and compare against the Random Forest
- Add SHAP values to explain individual predictions

---

## Stack

Python 3.10+, scikit-learn, pandas, matplotlib, seaborn
