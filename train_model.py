import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
def train_autonomous_model():
    csv_path = os.path.join('dataset', 'dark-patterns-v2.csv')
    if not os.path.exists(csv_path):
        print(f"[-] Dataset not found: {csv_path}")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(columns=['Pattern String', 'Pattern Category']).to_csv(csv_path, index=False)
        return False
    print(f"[*] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    df['Pattern String'] = df['Pattern String'].fillna('').astype(str)
    # Determine target column
    if 'Pattern Category' in df.columns:
        target_col = 'Pattern Category'
    elif 'Category' in df.columns:
        target_col = 'Category'
    else:
        target_col = df.columns[2]
    df[target_col] = df[target_col].fillna('Not Dark Pattern').astype(str).str.strip()
    df = df[df['Pattern String'].str.strip() != '']
    # KEY FIX: Use the Deceptive? column to label non-deceptive entries properly
    if 'Deceptive?' in df.columns:
        deceptive_col = df['Deceptive?'].fillna('').astype(str).str.strip().str.lower()
        not_deceptive = deceptive_col.isin(['no', 'depends', ''])
        df.loc[not_deceptive, target_col] = 'Not Dark Pattern'
        print(f"[*] Marked {not_deceptive.sum()} non-deceptive entries as 'Not Dark Pattern'")
    # Merge classes with fewer than 5 samples into nearest larger class
    min_class_size = 5
    counts = df[target_col].value_counts()
    tiny_classes = counts[counts < min_class_size].index.tolist()
    if tiny_classes:
        print(f"[*] Merging rare classes into 'Other Dark Pattern': {tiny_classes}")
        df.loc[df[target_col].isin(tiny_classes), target_col] = 'Other Dark Pattern'
        # If 'Other Dark Pattern' is still too small, drop those rows
        if df[target_col].value_counts().get('Other Dark Pattern', 0) < min_class_size:
            df = df[df[target_col] != 'Other Dark Pattern']
    X = df['Pattern String']
    y = df[target_col]
    print(f"[*] Records: {len(df)}")
    print(y.value_counts())
    if len(df) < 5:
        print("[-] Too few records to train.")
        return False
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    # Evaluate on held-out split
    can_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42,
        stratify=y if can_stratify else None
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
    model.fit(X_train_vec, y_train)
    print("\n=== Evaluation ===")
    print(classification_report(y_test, model.predict(X_test_vec), zero_division=0))
    # Final model on all data
    print("[*] Training final model on full dataset...")
    X_full_vec = vectorizer.fit_transform(X)
    model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
    model.fit(X_full_vec, y)
    # Save
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(models_dir, 'dark_pattern_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"[+] Model saved to {models_dir}/")
    return True
if __name__ == '__main__':
    train_autonomous_model()