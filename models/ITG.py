# %% [markdown]
# # Submission for ITG Speech Communication

# %%
#Import libraries
import pandas as pd
from text_features import get_all_text_features
import os

# %% [markdown]
# Loading MT and ATS datasets

# %%
size = "big" # "small" or "big"
type = "averaged" # "averaged" or "stratified"
dataset_name = "{}_data_{}_correct_format".format(size, type)

#Load dataset
train_data = pd.read_csv("CSV/"+ dataset_name +"/train.csv", header=0, delimiter=';')
val_data = pd.read_csv("CSV/"+ dataset_name +"/val.csv", header=0, delimiter=';')
test_data = pd.read_csv("CSV/"+ dataset_name +"/test.csv", header=0, delimiter=';')

train_df, val_df, test_df = pd.DataFrame(train_data), pd.DataFrame(val_data), pd.DataFrame(test_data)

#Groupby sentence and average the labels if necessary
train_df = train_df.groupby('sentence').mean().reset_index()
val_df = val_df.groupby('sentence').mean().reset_index()
test_df = test_df.groupby('sentence').mean().reset_index()

ats_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

#shuffle dataset
ats_df = ats_df.sample(frac=1, random_state=42).reset_index(drop=True)

# %%
# Load dataset
mt_data = pd.read_csv("CSV/crowdee-mt-ratings-for-model.csv", header=0, delimiter=';')
mt_df = pd.DataFrame(mt_data)

# Groupby sentence and average the labels
mt_df = mt_df.groupby('sentence').mean().reset_index()

# Shuffle dataset
mt_df = mt_df.sample(frac=1, random_state=23).reset_index(drop=True)


# %%
mt_df.head(3)

# %%
ats_df.head(3)

# %% [markdown]
# Compute textual features for each text

# %%
# Extract feature names
feature_functions = get_all_text_features()
feature_names = list(feature_functions.keys())

# %%
#Load the dataframes
ats_feature_df = pd.read_csv("CSV/ats_features.csv", header=0, delimiter=';')
mt_feature_df = pd.read_csv("CSV/mt_features.csv", header=0, delimiter=';')

# Remove the problematic feature completely, they contain some NaN or inf values
ats_feature_df = ats_feature_df.drop(columns=['dugast_uber_index'], errors='ignore')
mt_feature_df = mt_feature_df.drop(columns=['dugast_uber_index'], errors='ignore')
ats_feature_df = ats_feature_df.drop(columns=['herdans_vm'], errors='ignore')
mt_feature_df = mt_feature_df.drop(columns=['herdans_vm'], errors='ignore')
ats_feature_df = ats_feature_df.drop(columns=['yules_i'], errors='ignore')
mt_feature_df = mt_feature_df.drop(columns=['yules_i'], errors='ignore')

ats_feature_with_sentences = ats_feature_df.copy()
mt_feature_with_sentences = mt_feature_df.copy()
ats_feature_df = ats_feature_df.drop(columns=["sentence"])
mt_feature_df = mt_feature_df.drop(columns=["sentence"])

# %%
ats_feature_df.head(3)

# %%
mt_feature_df.head(3)

# %%
def get_top_10_correlated_features(df, label_column, all_labels):
    """
    Get top 10 features correlated with a specific label (with values), excluding all other labels.
    
    Parameters:
    df (pd.DataFrame): The feature dataframe
    label_column (str): The target label column
    all_labels (list): List of all label columns to exclude from features
    
    Returns:
    pd.Series: Top 10 features and their absolute correlation values with the label
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Get correlations with the target label and take absolute values
    corr_with_label = corr_matrix[label_column].abs()
    
    # Exclude all label columns
    features_to_consider = [col for col in corr_with_label.index if col not in all_labels]
    corr_with_label = corr_with_label[features_to_consider]
    
    # Sort and get top 10 features (excluding the label itself)
    top_10 = corr_with_label.sort_values(ascending=False).head(10)
    
    return top_10

# For ATS dataframe
ats_labels = ["sprachliche logik", "komplexität", "eindeutigkeit", "vorhersehbarkeit"]
ats_top_features = {}
for label in ats_labels:
    top_features = get_top_10_correlated_features(ats_feature_df, label, ats_labels)
    ats_top_features[label] = top_features

# For MT dataframe
mt_labels = ["präzision", "komplexität", "transparenz", "grammatikalität"]
mt_top_features = {}
for label in mt_labels:
    top_features = get_top_10_correlated_features(mt_feature_df, label, mt_labels)
    mt_top_features[label] = top_features

# Print results with nice formatting
def print_correlation_results(feature_dict, title):
    print(f"\n{title} Results:")
    print("="*50)
    for label, features in feature_dict.items():
        print(f"\nTop 10 features correlated with '{label}':")
        print("-"*40)
        for feature, value in features.items():
            print(f"{feature}: {value:.4f}")
        print()

print_correlation_results(ats_top_features, "ATS")
print_correlation_results(mt_top_features, "MT")

# %%
# Remove the problematic feature completely, they contain some NaN or inf values
ats_feature_df = ats_feature_df.drop(columns=['dugast_uber_index'], errors='ignore')
mt_feature_df = mt_feature_df.drop(columns=['dugast_uber_index'], errors='ignore')
ats_feature_df = ats_feature_df.drop(columns=['herdans_vm'], errors='ignore')
mt_feature_df = mt_feature_df.drop(columns=['herdans_vm'], errors='ignore')
ats_feature_df = ats_feature_df.drop(columns=['yules_i'], errors='ignore')
mt_feature_df = mt_feature_df.drop(columns=['yules_i'], errors='ignore')


# %%
# Create dfs for each dimension quality excluding the other dimensions
ats_feature_logik = ats_feature_df.drop(columns=["komplexität", "eindeutigkeit", "vorhersehbarkeit"], errors='ignore')
ats_feature_komplexitaet = ats_feature_df.drop(columns=["sprachliche logik", "eindeutigkeit", "vorhersehbarkeit"], errors='ignore')
ats_feature_eindeutigkeit = ats_feature_df.drop(columns=["sprachliche logik", "komplexität", "vorhersehbarkeit"], errors='ignore')
ats_feature_vorhersehbarkeit = ats_feature_df.drop(columns=["sprachliche logik", "komplexität", "eindeutigkeit"], errors='ignore')

mt_feature_praezision = mt_feature_df.drop(columns=["komplexität", "transparenz", "grammatikalität"], errors='ignore')
mt_feature_komplexitaet = mt_feature_df.drop(columns=["präzision", "transparenz", "grammatikalität"], errors='ignore')
mt_feature_transparenz = mt_feature_df.drop(columns=["präzision", "komplexität", "grammatikalität"], errors='ignore')
mt_feature_grammatikalitaet = mt_feature_df.drop(columns=["präzision", "komplexität", "transparenz"], errors='ignore')

# %%
from sklearn.model_selection import KFold

#define cv split
skf = KFold(n_splits=6, shuffle=False)

# %% [markdown]
# Filter-based Feature Selection: pearson correlation and mutual information

# %%
def filter_low_correlation(feature_df, target_col, threshold=0.2):
    """
    Filter features with absolute Pearson correlation < threshold for a specific target.
    
    Parameters:
    feature_df (pd.DataFrame): DataFrame containing features and target column
    target_col (str): Name of the target column
    threshold (float): Absolute correlation threshold (default: 0.15)
    
    Returns:
    pd.DataFrame: DataFrame with filtered features
    """
    # Calculate correlations
    corr = feature_df.corr()[target_col].abs()
    
    # Select features meeting threshold
    selected_features = corr[corr >= threshold].index.tolist()
    
    # Always keep the target column
    if target_col not in selected_features:
        selected_features.append(target_col)
    
    return feature_df[selected_features]

# Apply to ATS dimensions
ats_logik_pearson = filter_low_correlation(ats_feature_logik, "sprachliche logik")
ats_komplex_pearson = filter_low_correlation(ats_feature_komplexitaet, "komplexität")
ats_eindeutig_pearson = filter_low_correlation(ats_feature_eindeutigkeit, "eindeutigkeit")
ats_vorherseh_pearson = filter_low_correlation(ats_feature_vorhersehbarkeit, "vorhersehbarkeit")

# Apply to MT dimensions
mt_praezision_pearson = filter_low_correlation(mt_feature_praezision, "präzision")
mt_komplex_pearson = filter_low_correlation(mt_feature_komplexitaet, "komplexität")
mt_transparenz_pearson = filter_low_correlation(mt_feature_transparenz, "transparenz")
mt_grammatikal_pearson = filter_low_correlation(mt_feature_grammatikalitaet, "grammatikalität")

# %%
ats_eindeutig_pearson.head(3)

# %%
from sklearn.feature_selection import mutual_info_regression
import numpy as np

def select_top_mi_features(df, target_col, n_features=20):
    """
    Select top N features by Mutual Information.
    Returns DataFrame with target column as first column.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Calculate MI
    mi = mutual_info_regression(X, y, random_state=42)
    
    # Get top N features (excluding target)
    top_indices = np.argsort(mi)[-n_features:]
    selected_cols = X.columns[top_indices].tolist()
    
    # Reorder with target first
    return df[[target_col] + selected_cols]

# Apply to ATS dimensions
ats_logik_mi = select_top_mi_features(ats_feature_logik, "sprachliche logik")
ats_komplex_mi = select_top_mi_features(ats_feature_komplexitaet, "komplexität")
ats_eindeutig_mi = select_top_mi_features(ats_feature_eindeutigkeit, "eindeutigkeit")
ats_vorherseh_mi = select_top_mi_features(ats_feature_vorhersehbarkeit, "vorhersehbarkeit")

# Apply to MT dimensions
mt_praezision_mi = select_top_mi_features(mt_feature_praezision, "präzision")
mt_komplex_mi = select_top_mi_features(mt_feature_komplexitaet, "komplexität")
mt_transparenz_mi = select_top_mi_features(mt_feature_transparenz, "transparenz")
mt_grammatikal_mi = select_top_mi_features(mt_feature_grammatikalitaet, "grammatikalität")

# %%
ats_eindeutig_mi.head(3)

# %% [markdown]
# Wrapper-based feature selection: RFE, SFS

# %%
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# Initialize models with StandardScaler
scaler = StandardScaler()
lr = LinearRegression()
lasso = LassoCV(cv=skf, max_iter=20000)
elastic = ElasticNetCV(cv=skf, l1_ratio=.5, max_iter=20000)

# %%
def run_wrappers(X, y, cv):
    """Run RFECV and SFS with StandardScaler"""
    # RFECV Pipeline
    rfecv = RFECV(
        estimator=Pipeline([('scaler', scaler), ('lr', lr)]),
        step=0.1,
        cv=cv,
        scoring='r2',
        n_jobs=-1
    )
    rfecv.fit(X, y)
    
    # SFS Forward Pipeline
    sfs = SequentialFeatureSelector(
        Pipeline([('scaler', scaler), ('lr', lr)]),
        n_features_to_select='auto',
        direction='forward',
        cv=cv,
        n_jobs=-1
    )
    sfs.fit(X, y)
    
    return {
        'RFECV': X.columns[rfecv.support_],
        'SFS': X.columns[sfs.get_support()] 
    }

# %% [markdown]
# Embedded Methods: Lasso, ElasticNet

# %%
def run_embedded(X, y, cv):
    """Run Lasso and ElasticNet with StandardScaler"""
    # Lasso Pipeline
    lasso_pipe = Pipeline([('scaler', scaler), ('lasso', lasso)])
    lasso_pipe.fit(X, y)
    
    # ElasticNet Pipeline
    elastic_pipe = Pipeline([('scaler', scaler), ('elastic', elastic)])
    elastic_pipe.fit(X, y)
    
    return {
        'Lasso': X.columns[lasso_pipe.named_steps['lasso'].coef_ != 0],
        'ElasticNet': X.columns[elastic_pipe.named_steps['elastic'].coef_ != 0]
    }

# %%
from sklearn.model_selection import cross_val_score
import pandas as pd

# Prepare all datasets
datasets = {
    'ats_logik': (ats_feature_logik.drop(columns=['sprachliche logik']), 
                 ats_feature_logik['sprachliche logik']),
    'ats_eindeutig': (ats_feature_eindeutigkeit.drop(columns=['eindeutigkeit']), 
                      ats_feature_eindeutigkeit['eindeutigkeit']),
    'ats_vorherseh': (ats_feature_vorhersehbarkeit.drop(columns=['vorhersehbarkeit']),
                      ats_feature_vorhersehbarkeit['vorhersehbarkeit']),
        'ats_komplex': (ats_feature_komplexitaet.drop(columns=['komplexität']), 
                   ats_feature_komplexitaet['komplexität']),

    'mt_praezision': (mt_feature_praezision.drop(columns=['präzision']),
                     mt_feature_praezision['präzision']),
    'mt_grammatikal': (mt_feature_grammatikalitaet.drop(columns=['grammatikalität']),
                       mt_feature_grammatikalitaet['grammatikalität']),
    'mt_komplex': (mt_feature_komplexitaet.drop(columns=['komplexität']),
                   mt_feature_komplexitaet['komplexität']),
    'mt_transparenz': (mt_feature_transparenz.drop(columns=['transparenz']),
                       mt_feature_transparenz['transparenz']),
}

results = {}
for name, (X, y) in datasets.items():
    # Create stratified bins for continuous y
    cv_splits = list(skf.split(X, y))
    
    # Run all methods
    rfecv = RFECV(
        estimator=Pipeline([('scaler', scaler), ('lr', lr)]),
        step=0.1,
        cv=cv_splits,
        scoring='r2',
        n_jobs=-1,
        importance_getter='named_steps.lr.coef_'
    )
    rfecv.fit(X, y)
    
    # SFS Forward Pipeline
    sfs = SequentialFeatureSelector(
        Pipeline([('scaler', scaler), ('lr', lr)]),
        n_features_to_select='auto',
        direction='forward',
        cv=cv_splits,
        n_jobs=-1
    )
    sfs.fit(X, y)
    
    wrappers = {
        'RFECV': X.columns[rfecv.support_],
        'SFS': X.columns[sfs.get_support()] 
    }

    embedded = run_embedded(X, y, cv_splits)
    
    # Store results
    results[name] = {
        'RFECV': wrappers['RFECV'],
        'SFS': wrappers['SFS'],
        'Lasso': embedded['Lasso'],
        'ElasticNet': embedded['ElasticNet']
    }
    
    # Validate no data leakage
    assert not X.isna().any().any(), f"NaN values detected in {name}"

# %%
results

# %%
results["ats_eindeutigkeit"] = results["ats_eindeutig"]
results["ats_vorhersehbarkeit"] = results["ats_vorherseh"]
results["mt_grammatikalitaet"] = results["mt_grammatikal"]

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import cross_validate

# Initialize models and scaler
scaler = StandardScaler()
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Prepare all datasets (Pearson and MI filtered)
datasets = {
    'ats_logik': {
        'pearson': ats_logik_pearson,
        'mi': ats_logik_mi,
        'original': ats_feature_logik
    },
    'ats_komplex': {
        'pearson': ats_komplex_pearson,
        'mi': ats_komplex_mi,
        'original': ats_feature_komplexitaet
    },

    'ats_eindeutig': {
        'pearson': ats_eindeutig_pearson,
        'mi': ats_eindeutig_mi,
        'original': ats_feature_eindeutigkeit
    },
    'ats_vorhersehbarkeit': {
        'pearson': ats_vorherseh_pearson,
        'mi': ats_vorherseh_mi,
        'original': ats_feature_vorhersehbarkeit
    },

    'mt_komplex': {
        'pearson': mt_komplex_pearson,
        'mi': mt_komplex_mi,
        'original': mt_feature_komplexitaet
    },

    'mt_grammatikalitaet': {
        'pearson': mt_grammatikal_pearson,
        'mi': mt_grammatikal_mi,
        'original': mt_feature_grammatikalitaet
    },

    'mt_transparenz': {
        'pearson': mt_transparenz_pearson,
        'mi': mt_transparenz_mi,
        'original': mt_feature_transparenz
    },

    'mt_praezision': {
        'pearson': mt_praezision_pearson,
        'mi': mt_praezision_mi,
        'original': mt_feature_praezision
    },
}

methods = {
    'All Features': {
        'feature_fn': lambda df: df.drop(columns=df.columns[0]).columns.tolist(),
        'model': make_pipeline(scaler, lr)
    },
    'Pearson-filtered': {
        'feature_fn': lambda df: df.drop(columns=df.columns[0]).columns.tolist(),
        'model': make_pipeline(scaler, lr)
    },
    'MI-filtered': {
        'feature_fn': lambda df: df.drop(columns=df.columns[0]).columns.tolist(),
        'model': make_pipeline(scaler, lr)
    },
    'RFECV': {
        'feature_fn': lambda df: results[df.columns[1][:8]]['RFECV'],  # Use dataset name prefix
        'model': make_pipeline(scaler, lr)
    },
    'SFS': {
        'feature_fn': lambda df: results[df.columns[1][:8]]['SFS'],
        'model': make_pipeline(scaler, lr)
    },
    'Lasso': {
        'feature_fn': lambda df: results[df.columns[1][:8]]['Lasso'],
        'model': make_pipeline(scaler, lr)
    },
    'ElasticNet': {
        'feature_fn': lambda df: results[df.columns[1][:8]]['ElasticNet'],
        'model': make_pipeline(scaler, lr)
    },
    'RandomForest': {
        'feature_fn': lambda df: df.drop(columns=df.columns[0]).columns.tolist(),
        'model': rf
    }
}

# Main evaluation function
def evaluate_method(X, y, features, model, cv_splits):
    if len(features) == 0:
        return {'rmse': np.nan, 'n_features': 0}
    
    X_subset = X[features]
    
    cv_results = cross_validate(
        model,
        X_subset,
        y,
        cv=cv_splits,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    rmse_scores = -cv_results['test_score']
    return {
        'rmse': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'n_features': len(features)
    }

comparison_results = []

for dataset_name, data_dict in datasets.items():
    label_name = data_dict['original'].columns[0]
    
    # Create consistent CV splits
    X_original = data_dict['original'].drop(columns=label_name)
    y_original = data_dict['original'][label_name]
    cv_splits = list(skf.split(X_original, y_original))
    
    for method_name, method_config in methods.items():
        # Select appropriate dataframe
        if method_name == 'Pearson-filtered':
            df = data_dict['pearson']
        elif method_name == 'MI-filtered':
            df = data_dict['mi']
        else:
            df = data_dict['original']
        
        try:
            # Get features and model
            if method_name == 'RFECV':
                features = results[dataset_name]['RFECV']
            elif method_name == 'SFS':
                features = results[dataset_name]['SFS']
            elif method_name == 'Lasso':
                features = results[dataset_name]['Lasso']
            elif method_name == 'ElasticNet':
                features = results[dataset_name]['ElasticNet']
            else:
                features = method_config['feature_fn'](df)
            model = method_config['model']
            
            # Prepare X and y
            X = df[features]
            y = df[label_name]
            
            # Evaluate
            result = evaluate_method(X, y, features, model, cv_splits)
            
            comparison_results.append({
                'Dataset': dataset_name,
                'Method': method_name,
                'RMSE': f"{result['rmse']:.4f}", #± {result['rmse_std']:.4f}",
                'Num Features': result['n_features'],
                'features': str(features)
            })
        except KeyError as e:
            print(f"Skipping {method_name} for {dataset_name}: {str(e)}")
            continue

# Create results dataframe
results_df = pd.DataFrame(comparison_results)

# %%
results_df.to_csv("CSV/feature_selection_results.csv", index=False, sep=';')

# %%

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

# Configuration
MODEL_NAME = "deepset/gbert-large"
BATCH_SIZE = 16
MAX_LENGTH = 256
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming mt_df and ats_df are loaded with structure:
# First column: 'sentence' (string)
# Subsequent columns: target labels

def init_process_group():
    """
    Join the process group and return whether this is the rank 0 process,
    the CUDA device to use, and the total number of GPUs used for training.
    """
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    # dist.init_process_group('nccl')
    return rank == 0, torch.device(f'cuda:{local_rank}'), num_gpus

is_rank0, device, num_gpus = init_process_group()

torch.cuda.set_device(device)

class QualityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return {"rmse": rmse}

def train_and_evaluate(df, target_name, cv_splits):
    """Train and evaluate for one target dimension"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = df['sentence'].tolist()
    labels = df[target_name].values
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n=== Fold {fold+1} ===")
        
        # Create datasets
        train_dataset = QualityDataset(
            [texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            tokenizer
        )
        val_dataset = QualityDataset(
            [texts[i] for i in val_idx],
            [labels[i] for i in val_idx],
            tokenizer
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results_may/{target_name}_fold{fold}",
            evaluation_strategy="epoch",
            learning_rate=LR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            use_cpu=False
        )
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1
        ).to(DEVICE)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        fold_results.append(eval_results['eval_rmse'])
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return np.mean(fold_results), np.std(fold_results)

# Main execution
if __name__ == "__main__":
    
    # For MT dimensions
    mt_texts = mt_df['sentence'].values
    mt_labels = mt_df.drop(columns='sentence')
    mt_cv_splits = list(skf.split(mt_texts))
    
    mt_results = {}
    for target in ["präzision", "komplexität", "transparenz", "grammatikalität"]:
        print(f"\n\n=== Training MT: {target} ===")
        mean_rmse, std_rmse = train_and_evaluate(mt_df, target, mt_cv_splits)
        mt_results[target] = {'mean_rmse': mean_rmse, 'std_rmse': std_rmse}
    
    # For ATS dimensions
    ats_texts = ats_df['sentence'].values
    ats_labels = ats_df.drop(columns='sentence')
    ats_cv_splits = list(skf.split(ats_texts))
    
    ats_results = {}
    for target in ["sprachliche logik", "komplexität", "eindeutigkeit", "vorhersehbarkeit"]:
        print(f"\n\n=== Training ATS: {target} ===")
        mean_rmse, std_rmse = train_and_evaluate(ats_df, target, ats_cv_splits)
        ats_results[target] = {'mean_rmse': mean_rmse, 'std_rmse': std_rmse}
    
    # Combine and save results
    all_results = {
        'MT': pd.DataFrame.from_dict(mt_results, orient='index'),
        'ATS': pd.DataFrame.from_dict(ats_results, orient='index')
    }
        
    # Save to CSV
    all_results['MT'].to_csv("mt_gbert_results.csv")
    all_results['ATS'].to_csv("ats_gbert_results.csv")


