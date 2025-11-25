import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from torch import distributed as dist
from sklearn.model_selection import KFold
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from torch import nn

# Device initialization
def init_process_group():
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    return rank == 0, torch.device(f'cuda:{local_rank}'), num_gpus

is_rank0, device, num_gpus = init_process_group()

# Load dataset
train_data = pd.read_csv("CSV/"+"mt_features.csv", header=0, delimiter=';')
train_df = pd.DataFrame(train_data)

data = pd.read_csv("CSV/mt-new_features.csv", header=0, delimiter=';')
new_df = pd.DataFrame(data)
new_df.drop(columns=new_df.columns[1:2], inplace=True)
train_df = pd.concat([train_df, new_df], ignore_index=True)

# Remove the problematic feature completely, they contain some NaN or inf values
train_df = train_df.drop(columns=['dugast_uber_index', 'herdans_vm', 'yules_i'], errors='ignore')

# Groupby sentence and average the labels
train_df = train_df.groupby('sentence').mean().reset_index()

# Shuffle the dataset
shuffled_df = train_df.sample(frac=1, random_state=23).reset_index(drop=True)

# Initialize k-fold cross-validation
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=23)

# Define the target columns
target_columns = ["präzision", "komplexität", "transparenz", "grammatikalität"]

# Load the model and tokenizer
model_name = "deepset/gelectra-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Training arguments
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 30

def create_training_args(column, fold):
    return TrainingArguments(
        output_dir="Outputs/" + "mt-new" + "/gelectra-large-with-features" + f"/fold-{fold}" + f"/{column}",
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
    )

# Custom model class that concatenates ELECTRA outputs with selected features
class ELECTRAWithFeaturesModel(nn.Module):
    def __init__(self, base_model, num_features):
        super().__init__()
        self.electra = base_model.electra
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(base_model.config.hidden_size + num_features, 1)  # Single output
        
    def forward(self, input_ids=None, attention_mask=None, features=None, labels=None):
        outputs = self.electra(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        sequence_output = outputs[0]  # Shape: (batch_size, seq_length, hidden_size)
        pooled_output = sequence_output[:, 0, :]  # Take first token ([CLS])

        # Concatenate ELECTRA output with features
        combined = torch.cat((pooled_output, features), dim=1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)
        
        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(logits, labels.view(-1, 1))  # Single output
            
        return {'loss': loss, 'logits': logits}

# Custom trainer to handle features
class SingleRegressionTrainerWithFeatures(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        features = inputs.pop("features").to(device)
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        
        outputs = model(input_ids=inputs['input_ids'], 
                      attention_mask=inputs['attention_mask'],
                      features=features,
                      labels=labels)
        
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']

def compute_metrics_for_regression(eval_pred, target_column):
    logits, labels = eval_pred
    logits = logits.reshape(-1, 1)  # Reshape logits for single output
    labels = labels.reshape(-1, 1)  # Reshape labels for single output

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, logits)

    # Compute accuracy
    accuracy = (abs(logits - labels) < 0.5).mean()

    # Epsilon-insensitive squared error (ε=0.25)
    epsilon = 0.25
    errors = logits - labels
    squared_errors = np.where(np.abs(errors)**2 > epsilon, 
                            (np.abs(errors)**2 - epsilon), 
                            0)
    eise = squared_errors.mean()
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "seei": eise,
        "accuracy": accuracy
    }

# Get all feature columns (excluding text and labels)
feature_columns = [col for col in shuffled_df.columns 
                  if col not in ['sentence'] + target_columns]

# Precompute the splits for all folds
splits = list(kf.split(shuffled_df))

# Store performance metrics for each target column
column_results = []

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

def preprocess_function_with_features(examples, tokenizer, target_column, feature_columns, selected_feature_indices, scaler):
    # Extract the label
    label = float(examples[target_column])

    # Tokenize the input text
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True,
                               padding="max_length", max_length=512)

    # Get ONLY the selected features (not all feature_columns)
    selected_features_values = examples[feature_columns].values[selected_feature_indices].astype(float)
    
    # Scale the selected features (already filtered to 20)
    scaled_features = scaler.transform(selected_features_values.reshape(1, -1)).flatten()

    # Add the label and features to the tokenized inputs
    tokenized_inputs["labels"] = torch.tensor(label, dtype=torch.float)
    tokenized_inputs["features"] = torch.tensor(scaled_features, dtype=torch.float)

    return tokenized_inputs

def make_dataset(df, tokenizer, target_column, feature_columns, selected_feature_indices, scaler):
    processed = df.apply(
        preprocess_function_with_features,
        tokenizer=tokenizer,
        target_column=target_column,
        feature_columns=feature_columns,
        selected_feature_indices=selected_feature_indices,
        scaler=scaler,
        axis=1
    )
    return processed.tolist()

base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
base_model.config.problem_type = "regression"
model = ELECTRAWithFeaturesModel(base_model, num_features=20)
initial_state_dict = copy.deepcopy(model.state_dict())

# Perform k-fold cross-validation for each target column
for target_column in target_columns:
    print(f"Training model for target: {target_column}")

    # Store performance metrics for each fold
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {fold + 1}/{k}")
        
        # Split into training and test sets
        train_data = shuffled_df.iloc[train_idx]
        test_data = shuffled_df.iloc[test_idx]

        # Use the previous fold's test set as the validation set
        if fold == 0:
            val_data = shuffled_df.iloc[splits[-1][1]]  # Last fold's test set
        else:
            val_data = shuffled_df.iloc[splits[fold - 1][1]]  # Previous fold's test set
        
        # Remove validation data from training data
        train_data = train_data.drop(val_data.index)
        
        # Feature selection with SFS
        print("    Performing feature selection...")
        selected_features, selected_scaler = select_features(
            train_data, 
            feature_columns,
            target_column,
            n_features=20
        )
        selected_feature_indices = [feature_columns.index(f) for f in selected_features]

        # Initialize model for this fold
        model = ELECTRAWithFeaturesModel(base_model, num_features=20)
        model.load_state_dict(initial_state_dict)  # Reset model to initial state
        model.to(device)
        
        # Preprocess datasets
        train_dataset = make_dataset(train_data, tokenizer, target_column, feature_columns, selected_feature_indices, selected_scaler)
        val_dataset = make_dataset(val_data, tokenizer, target_column, feature_columns, selected_feature_indices, selected_scaler)
        test_dataset = make_dataset(test_data, tokenizer, target_column, feature_columns, selected_feature_indices, selected_scaler)

        # Initialize trainer
        trainer = SingleRegressionTrainerWithFeatures(
            model=model,
            args=create_training_args(target_column, fold),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda eval_pred: compute_metrics_for_regression(eval_pred, target_column)
        )
        
        # Train the model
        print("    Training model...")
        trainer.train()
        torch.cuda.empty_cache()
        
        # Evaluate on the test set
        print("    Evaluating model...")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        torch.cuda.empty_cache()
        
        # Store test results
        fold_scores.append(test_results)

    # Calculate average performance across all folds for this target column
    avg_metrics = {
        "target": target_column,
        "mse": np.mean([score["eval_mse"] for score in fold_scores]),
        "mae": np.mean([score["eval_mae"] for score in fold_scores]),
        "rmse": np.mean([score["eval_rmse"] for score in fold_scores]),
        "r2": np.mean([score["eval_r2"] for score in fold_scores]),
        "seei": np.mean([score["eval_seei"] for score in fold_scores]),
        "accuracy": np.mean([score["eval_accuracy"] for score in fold_scores])
    }

    # Store results for this target column
    column_results.append(avg_metrics)

# Print average performance for each target column
print("\nAverage performance across all target columns:")
for result in column_results:
    print(result)