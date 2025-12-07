import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

data = pd.read_csv("TCGA-BRCA.star_tpm.tsv", sep="\t", index_col=0).T
target = pd.read_csv("TCGA-BRCA.clinical.tsv", sep="\t", index_col=0)["ajcc_pathologic_stage.diagnoses"]

def group_stage(stage): # group data to either early stage or late stage BC
    if pd.isna(stage):
        return np.nan
    stage = stage.upper().replace('STAGE ', '').strip()
    if stage in ['I', 'IA', 'IB', 'II', 'IIA', 'IIB']:
        return 0
    elif stage in ['III', 'IIIA', 'IIIB', 'IIIC', 'IV']:
        return 1
    else:
        return np.nan
    
y = target.apply(group_stage).dropna()

common_samples = data.index.intersection(y.index)
X = data.loc[common_samples]
y = y.loc[common_samples]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

K = 50
selector = SelectKBest(f_classif, k=K)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
selected_genes = X_train.columns[selector.get_support()]
X_test_selected = X_test_scaled[selected_genes]

rf_param_grid = {
    'n_estimators': [100],
    'max_depth': [10, None],
}

rf_model = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(
    rf_model, rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)

rf_grid_search.fit(X_train_scaled, y_train) # RF w/o feature selection
best_rf_model_no_fs = rf_grid_search.best_estimator_

rf_grid_search.fit(X_train_selected, y_train) # RF w/ feature selection
best_rf_model_with_fs = rf_grid_search.best_estimator_

# SVM 

svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear']
}

svm_model = SVC(random_state=42, probability=True)
svm_grid_search = GridSearchCV(
    svm_model, svm_param_grid, cv=5, scoring='roc_auc', n_jobs=-1
)

svm_grid_search.fit(X_train_scaled, y_train)
best_svm_model_no_fs = svm_grid_search.best_estimator_

svm_grid_search.fit(X_train_selected, y_train)
best_svm_model_with_fs = svm_grid_search.best_estimator_

models = {
    "RF_NoFS": best_rf_model_no_fs,
    "RF_WithFS": best_rf_model_with_fs,
    "SVM_NoFS": best_svm_model_no_fs,
    "SVM_WithFS": best_svm_model_with_fs,
}

test_results = {}
for name, model in models.items():
    # Prepare data based on whether FS was used in training
    if "NoFS" in name:
        X_eval = X_test_scaled
    else:
        # Filter test data using the genes selected during training
        X_eval = X_test_selected

    # Get predictions
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    # Calculate metrics
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    
    test_results[name] = {'AUROC': auc, 'Accuracy': acc}
    
# Display final comparison
results_df = pd.DataFrame(test_results).T
print("\nFINAL MODEL COMPARISON (Test Set)")
print(results_df.sort_values(by='AUROC', ascending=False))