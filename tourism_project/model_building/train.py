"""
Model Building, Training and Experimentation Tracking
Downloads train/test data from Hugging Face, builds a preprocessing +
classification pipeline, trains several candidate models, logs every run
to MLflow, picks the best model by test-set ROC-AUC, and uploads the
winning pipeline to a Hugging Face Hub model repo.
"""
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo, hf_hub_download
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction-model"
TARGET_COL = "ProdTaken"

# 1. Load train/test data
train_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="train.csv", repo_type="dataset", token=HF_TOKEN)
test_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="test.csv", repo_type="dataset", token=HF_TOKEN)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 2. Preprocessing pipeline
numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", numeric_transformer, numerical_cols), ("cat", categorical_transformer, categorical_cols)])

# 3. Candidate models
candidates = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"),
    "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42),
}

mlflow.set_experiment("tourism-wellness-package-prediction")
best_model_name, best_score, best_pipeline = None, -1.0, None

for name, clf in candidates.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        mlflow.log_param("model_type", name)
        mlflow.log_params(clf.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="pickle")
        print(f"[{name}] {metrics}")
        if metrics["roc_auc"] > best_score:
            best_score, best_model_name, best_pipeline = metrics["roc_auc"], name, pipeline

print(f"\nBest model: {best_model_name} (ROC-AUC = {best_score:.4f})")

# 4. Save and register the best model to the Hugging Face Hub
os.makedirs("tourism_project/model_building/artifacts", exist_ok=True)
model_path = "tourism_project/model_building/artifacts/best_model.joblib"
joblib.dump(best_pipeline, model_path)

api = HfApi(token=HF_TOKEN)
create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True, token=HF_TOKEN)
api.upload_file(path_or_fileobj=model_path, path_in_repo="best_model.joblib",
                 repo_id=MODEL_REPO_ID, repo_type="model", token=HF_TOKEN)
print(f"Best model uploaded to: https://huggingface.co/{MODEL_REPO_ID}")
