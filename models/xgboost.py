from xgboost import XGBClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from random_forest import compute_features, author_jaccard, title_sim


df = pd.read_csv("dblp_cleaned.csv")
df = df.drop(columns=["peditor", "paddress", "ppublisher", "pseries"])

train_pairs = pd.read_csv("train_pairs_features.csv")
val_pairs = pd.read_csv("data/validation_hidden.csv", index_col=0)
test_pairs = pd.read_csv("data/test_hidden.csv", index_col=0)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(df["ptitle"].fillna(""))
key_to_index = {k: i for i, k in enumerate(df["pkey"])}

# Training Random Forest
features = [
    "title_sim",
    "author_jaccard",
    "year_diff",
    "same_journal",
    "same_booktitle",
    "same_ptype"
]

X = train_pairs[features].fillna(0)
y = train_pairs["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

# Predict on validation set
preds_xgb = xgb.predict_proba(X_val)[:, 1]

print("XGBoost Results:")
print(f"  ROC AUC:   {roc_auc_score(y_val, preds_xgb):.4f}")
print(f"  Accuracy:  {accuracy_score(y_val, preds_xgb.round()):.4f}")
print(f"  Precision: {precision_score(y_val, preds_xgb.round()):.4f}")
print(f"  Recall:    {recall_score(y_val, preds_xgb.round()):.4f}")
print(f"  F1 Score:  {f1_score(y_val, preds_xgb.round()):.4f}")

# Feature importance
importances_xgb = pd.Series(xgb.feature_importances_, index=features).sort_values()
print("\nFeature Importances:")
print(importances_xgb.sort_values(ascending=False))

# For Submission

xgb_full = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb_full.fit(X, y)

X_val_submit = compute_features(val_pairs, df, tfidf, key_to_index, features)
X_test_submit = compute_features(test_pairs, df, tfidf, key_to_index, features)

val_pairs["label"] = xgb_full.predict(X_val_submit)
test_pairs["label"] = xgb_full.predict(X_test_submit)

val_pairs[["key1", "key2", "label"]].to_csv("validation_predictions.csv", index=False)
test_pairs[["key1", "key2", "label"]].to_csv("test_predictions.csv", index=False)

print(f"Validation predictions: {val_pairs['label'].value_counts().to_dict()}")
print(f"Test predictions:       {test_pairs['label'].value_counts().to_dict()}")