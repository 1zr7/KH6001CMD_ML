# Gym Churn Prediction

Project: end‑to‑end machine learning workflow to predict gym membership churn and explore customer segmentation.

## Contents

- `ML_notebook.ipynb` — Main notebook: data loading, EDA, preprocessing, baseline & advanced models, SHAP explainability, K‑Means clustering, model tuning and saving artifacts.
- `gym_churn_us.csv` — Expected dataset (place in the notebook working directory).
- `data/` — Output directory created by the notebook (saved model, metrics, README, requirements).

## Quickstart

1. Clone or copy this repository to your machine.
2. Place the dataset `gym_churn_us.csv` in the repository root or update `DATA_PATH` in the notebook.
3. Create and activate a Python environment (recommended):

   - python 3.8+
   - pip install -r data/requirements.txt (or manually install packages below)

4. Launch the notebook:
   - jupyter lab
   - Open `ML_notebook.ipynb` and run cells in order.

## Required packages (examples)

- numpy, pandas, matplotlib, seaborn
- scikit-learn, shap, joblib
- scipy

For an exact environment run (after first notebook run) `pip freeze > data/requirements.txt`.

## Notebook highlights

- Data inspection, missing value analysis and EDA.
- Preprocessing pipeline using ColumnTransformer (scaling, one‑hot encoding).
- Baseline models: Logistic Regression, Decision Tree.
- Advanced models: Random Forest, Gradient Boosting, MLP; cross‑validation and ROC‑AUC evaluation.
- Explainability: SHAP for tree models; permutation importance for others.
- K‑Means clustering on numeric features with elbow & silhouette evaluation and PCA visualization.
- Hyperparameter tuning for MLP (RandomizedSearchCV) and saving the final pipeline and metrics to `data/`.

## Outputs

- `data/final_mlp_pipeline.pkl` — saved sklearn pipeline (example).
- `data/final_model_metrics.json` — metrics for the selected model.
- `data/MODEL_README.txt` — brief usage snippet.
- `data/requirements.txt` — environment requirements (optional).

## Reproducing results

- Run the notebook from top to bottom to reproduce training, evaluation, SHAP plots and clustering.
- For long steps (Kernel SHAP, RandomizedSearchCV) consider toggling or reducing sample sizes in the notebook parameters.

## Contact

For questions about the notebook structure or reproducing results, open an issue or contact the project owner.
