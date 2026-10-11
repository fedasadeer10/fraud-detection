# fraud-detection
# Streamlit demo app, built with the help of Claude (Anthropic).
# The model itself (training and evaluation) is in creditcard.ipynb.
# Bank Transaction Fraud Detection

A machine learning model that detects fraudulent credit card transactions, with a small Streamlit app to try it.

## Dataset

[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- The data is highly imbalanced: only about 0.17% of transactions are fraud (98 fraud cases out of 56,962 in the test set).
- Columns `V1` to `V28` are the result of a PCA transformation, so their original meaning is hidden. `Time` and `Amount` are the only original columns.

## How I handled the class imbalance

- Used `stratify=Y` in `train_test_split`, so train and test keep the same fraud ratio.
- Did not rely on accuracy (a model that always predicts "not fraud" would score about 99.8%). I evaluated with precision, recall and f1 of the fraud class.
- Tested `class_weight='balanced'` with two models and compared the results.

## Results (fraud class, test set)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression + `class_weight='balanced'` | 0.07 | 0.90 | 0.14 |
| Logistic Regression | 0.87 | 0.62 | 0.73 |
| **Random Forest (100 trees)** | **0.96** | **0.77** | **0.85** |

Random Forest gave the best balance. It caught about 75 of 98 fraud cases with very few false alarms. `class_weight='balanced'` made Logistic Regression catch more fraud, but with too many false alarms (precision 0.07), so it was not practical.

The scaler and the model are saved together as one scikit-learn `Pipeline` in `fraud_detection_model.pkl`.

## Limitations

- The model misses about 23% of fraud cases. Some of them are close to the decision line (for example a fraud transaction scored 45%), so it should not be used alone in a real bank.
- It was trained only on this Kaggle dataset. Because the columns are anonymized (PCA), the app works on sample transactions from the same data, not on new real transactions.

## Future improvements

- Add a "needs review" zone for medium-risk transactions instead of a strict yes/no decision.
- Retrain the model regularly on newly confirmed fraud cases.
- Try gradient boosting models such as XGBoost or LightGBM.

## Note about the app

My work in this project is the model: data analysis, training, comparing models, evaluating, and saving it. The Streamlit app was built with the help of Claude (Anthropic) just to demonstrate the model.

## Files

- `creditcard.ipynb`: the notebook with all experiments
- `fraud_detection_model.pkl`: the saved model (scaler + Random Forest)
- `sample.csv`: all fraud cases from the test set plus 500 normal ones, used by the app
- `app.py`: the Streamlit app

## Run it

```
pip install -r requirements.txt
python -m streamlit run app.py
```
