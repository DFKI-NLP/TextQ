import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from torch import nn
from sklearn.model_selection import KFold

# Device initialization
def init_process_group():
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    return rank == 0, torch.device(f'cuda:{local_rank}'), num_gpus

is_rank0, device, num_gpus = init_process_group()

# ===== Load ATS + MT datasets =====
ats = pd.read_csv("CSV/ats-new_features.csv", header=0, delimiter=';')
ats.drop(columns=ats.columns[2:6], inplace=True)
ats = ats.groupby('sentence').mean().reset_index()
ats["source"] = "ats"

mt = pd.read_csv("CSV/mt-new_features.csv", header=0, delimiter=';')
mt.drop(columns=mt.columns[2:6], inplace=True)
mt = mt.groupby('sentence').mean().reset_index()
mt["source"] = "mt"

df = pd.concat([ats, mt], ignore_index=True)
df = df.drop(columns=['dugast_uber_index', 'herdans_vm', 'yules_i'], errors='ignore')
df = df.sample(frac=1, random_state=23).reset_index(drop=True)

# ===== Settings =====
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=23)
target_column = "overall"
model_name = "deepset/gelectra-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

LEARNING_RATE = 2e-5
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 30

def create_training_args(fold):
    return TrainingArguments(
        output_dir=f"Outputs/overall-ats-mt-gelectra-large-with-features/fold-{fold}",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="rmse",
        greater_is_better=False,
        load_best_model_at_end=True,
        weight_decay=0.01,
        logging_dir=f"Logs/fold-{fold}"
    )

# ===== Model with handcrafted features =====
class ELECTRAWithFeaturesModel(nn.Module):
    def __init__(self, base_model, num_features):
        super().__init__()
        self.electra = base_model.electra
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(base_model.config.hidden_size + num_features, 1)
        
    def forward(self, input_ids=None, attention_mask=None, features=None, labels=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        pooled_output = outputs[0][:, 0, :]  # CLS token
        combined = torch.cat((pooled_output, features), dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        
        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(logits, labels.view(-1, 1))
        return {"loss": loss, "logits": logits}

# ===== Trainer that supports features =====
class RegressionTrainerWithFeatures(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        features = inputs.pop("features").to(device)
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        outputs = model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            features=features,
            labels=labels
        )
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']

# ===== Metrics =====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.reshape(-1)
    labels = labels.reshape(-1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, logits)
    acc = (abs(logits - labels) < 0.5).mean()

    epsilon = 0.25
    errors = logits - labels
    squared_errors = np.where(np.abs(errors) ** 2 > epsilon, (np.abs(errors) ** 2 - epsilon), 0)
    eise = squared_errors.mean()
    
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "seei": eise, "accuracy": acc}

# ===== Feature selection =====
def select_features(train_data, feature_columns, target_column, n_features=20):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[feature_columns].values.astype(float))
    y_train = train_data[target_column].values
    
    sfs = SequentialFeatureSelector(LinearRegression(),
                                    n_features_to_select=n_features,
                                    direction='forward',
                                    cv=5)
    sfs.fit(X_train_scaled, y_train)
    
    selected_indices = [i for i, selected in enumerate(sfs.get_support()) if selected]
    selected_features = [feature_columns[i] for i in selected_indices]
    
    selected_scaler = StandardScaler()
    selected_scaler.fit(train_data[selected_features].values.astype(float))
    return selected_features, selected_scaler

def preprocess_function_with_features(examples, tokenizer, target_column, selected_features, scaler):
    label = float(examples[target_column])
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=512)
    feature_values = examples[selected_features].values.astype(float)
    scaled_features = scaler.transform(feature_values.reshape(1, -1)).flatten()

    tokenized_inputs["labels"] = torch.tensor(label, dtype=torch.float)
    tokenized_inputs["features"] = torch.tensor(scaled_features, dtype=torch.float)
    tokenized_inputs["source"] = examples["source"]
    return tokenized_inputs

def make_dataset(df, tokenizer, target_column, selected_features, scaler):
    processed = df.apply(
        preprocess_function_with_features,
        tokenizer=tokenizer,
        target_column=target_column,
        selected_features=selected_features,
        scaler=scaler,
        axis=1
    )
    return processed.tolist()

# ===== Prepare model =====
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
base_model.config.problem_type = "regression"
model = ELECTRAWithFeaturesModel(base_model, num_features=20)
initial_state_dict = copy.deepcopy(model.state_dict())

# ===== Cross-validation =====
results = {"ats": [], "mt": []}
splits = list(kf.split(df))
feature_columns = [c for c in df.columns if c not in ["sentence", target_column, "source"]]

for fold, (train_idx, test_idx) in enumerate(splits):
    print(f"\n=== Fold {fold+1}/{k} ===")
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    val_data = df.iloc[splits[fold-1][1]] if fold > 0 else df.iloc[splits[-1][1]]
    train_data = train_data.drop(val_data.index)

    # Feature selection
    selected_features, selected_scaler = select_features(train_data, feature_columns, target_column, n_features=20)

    # Reset model
    model = ELECTRAWithFeaturesModel(base_model, num_features=20)
    model.load_state_dict(initial_state_dict)
    model.to(device)

    # Make datasets
    train_dataset = make_dataset(train_data, tokenizer, target_column, selected_features, selected_scaler)
    val_dataset = make_dataset(val_data, tokenizer, target_column, selected_features, selected_scaler)
    test_dataset = make_dataset(test_data, tokenizer, target_column, selected_features, selected_scaler)

    # Trainer
    trainer = RegressionTrainerWithFeatures(
        model=model,
        args=create_training_args(fold),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    torch.cuda.empty_cache()

    # Evaluate ATS/MT separately
    for subset in ["ats", "mt"]:
        subset_dataset = [x for x in test_dataset if x["source"] == subset]
        if subset_dataset:
            metrics = trainer.evaluate(eval_dataset=subset_dataset)
            print(f"  {subset.upper()} Test Results: {metrics}")
            results[subset].append(metrics)

# ===== Average results =====
avg_results = {}
for subset in results:
    if not results[subset]:  # skip empty
        continue
    avg_results[subset] = {
        k: np.mean([r[k] for r in results[subset]]) for k in results[subset][0].keys()
    }

print("\n=== Average Performance ===")
for subset, metrics in avg_results.items():
    print(f"{subset.upper()}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")