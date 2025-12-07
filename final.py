import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

data = pd.read_csv("TCGA-BRCA.star_tpm.tsv", sep="\t", index_col=0).T # transposing b/c file has rows=genes, col=patients
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

common_samples = data.index.intersection(y.index) # filter data based on patients in clinical & gene expression files
X = data.loc[common_samples]
y = y.loc[common_samples]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# scaling data to prevent model from thinking certain genes are more important than others b/c of high expression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# filtering to top K genes to correct for potential noise from 1000s of other genes
K = 300
selector = SelectKBest(score_func=mutual_info_classif, k=K)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
selected_genes = X_train.columns[selector.get_support()]
X_test_selected = X_test_scaled[selected_genes]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # repeat testing while forcing same % of early and developed patients

smote = SMOTE(random_state=42) # trying to fix data imbalance w/ smote (idk i looked it up)
X_train_smote, y_train_smote = smote.fit_resample(X_train_selected, y_train)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_smote, y_train_smote)

svm = SVC(kernel="linear", random_state=42, probability=True)
svm.fit(X_train_smote, y_train_smote)

test_results = {} # display results
for name, model in [("RF_Balanced", rf), ("SVM_Balanced", svm)]:

    # Get predictions
    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)[:, 1]
    
    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred))