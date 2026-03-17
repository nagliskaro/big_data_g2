import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


def author_jaccard(a1, a2):
    if pd.isna(a1) or pd.isna(a2):
        return 0
    
    s1 = set(a1.split("|"))
    s2 = set(a2.split("|"))
    
    return len(s1 & s2) / len(s1 | s2)


def title_sim(row, tfidf_matrix, key_index):
    i = key_index.get(row["key1"])
    j = key_index.get(row["key2"])
    if i is None or j is None:
        return 0
    return cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]


def compute_features(pairs_df, dblp_df, tfidf_matrix, key_index, feature_list):
    merged = pairs_df.merge(dblp_df, left_on="key1", right_on="pkey", how="left")
    merged = merged.merge(dblp_df, left_on="key2", right_on="pkey", how="left", suffixes=("_1", "_2"))

    merged["title_sim"] = merged.apply(title_sim, axis=1, tfidf_matrix=tfidf_matrix, key_index=key_index)
    merged["author_jaccard"] = merged.apply(lambda x: author_jaccard(x["pauthor_1"], x["pauthor_2"]), axis=1)
    merged["year_diff"] = (merged["pyear_1"] - merged["pyear_2"]).abs()
    merged["same_journal"] = (merged["pjournal_id_1"] == merged["pjournal_id_2"]).astype(int)
    merged["same_booktitle"] = (merged["pbooktitle_id_1"] == merged["pbooktitle_id_2"]).astype(int)
    merged["same_ptype"] = (merged["ptype_id_1"] == merged["ptype_id_2"]).astype(int)

    return merged[feature_list].fillna(0)


if __name__ == "__main__":
    
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

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Predict on validation set
    preds_rf = rf.predict_proba(X_val)[:, 1]

    print("Random Forest Results:")
    print(f"  ROC AUC:   {roc_auc_score(y_val, preds_rf):.4f}")
    print(f"  Accuracy:  {accuracy_score(y_val, preds_rf.round()):.4f}")
    print(f"  Precision: {precision_score(y_val, preds_rf.round()):.4f}")
    print(f"  Recall:    {recall_score(y_val, preds_rf.round()):.4f}")
    print(f"  F1 Score:  {f1_score(y_val, preds_rf.round()):.4f}")

    # Importance of features
    importances = pd.Series(rf.feature_importances_, index=features).sort_values()
    print("\nFeature Importances:")
    print(importances.sort_values(ascending=False))

    # For Submission
    rf_full = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf_full.fit(X, y)

    X_val_submit = compute_features(val_pairs, df, tfidf, key_to_index, features)
    X_test_submit = compute_features(test_pairs, df, tfidf, key_to_index, features)

    val_pairs["label"] = rf_full.predict(X_val_submit)
    test_pairs["label"] = rf_full.predict(X_test_submit)

    val_pairs[["key1", "key2", "label"]].to_csv("validation_predictions.csv", index=False)
    test_pairs[["key1", "key2", "label"]].to_csv("test_predictions.csv", index=False)

    print(f"Validation predictions: {val_pairs['label'].value_counts().to_dict()}")
    print(f"Test predictions:       {test_pairs['label'].value_counts().to_dict()}")
    