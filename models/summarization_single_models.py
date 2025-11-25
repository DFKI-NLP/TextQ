# Import libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from torch import distributed as dist
from sklearn.model_selection import KFold
import numpy as np

# Device
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

# Pick dataset
size = "big"  # "small" or "big"
type = "averaged"  # "averaged" or "stratified"
dataset_name = "{}_data_{}_correct_format".format(size, type)

# Load dataset
train_data = pd.read_csv("CSV/" + dataset_name + "/train.csv", header=0, delimiter=';')
val_data = pd.read_csv("CSV/" + dataset_name + "/val.csv", header=0, delimiter=';')
test_data = pd.read_csv("CSV/" + dataset_name + "/test.csv", header=0, delimiter=';')

train_df, val_df, test_df = pd.DataFrame(train_data), pd.DataFrame(val_data), pd.DataFrame(test_data)

# Groupby sentence and average the labels
train_df = train_df.groupby('sentence').mean().reset_index()
val_df = val_df.groupby('sentence').mean().reset_index()
test_df = test_df.groupby('sentence').mean().reset_index()

# Combine all datasets
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Shuffle the combined dataset
shuffled_df = combined_df.sample(frac=1, random_state=23).reset_index(drop=True)

# Initialize k-fold cross-validation
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=23)

# Define the target columns
target_columns = ["sprachliche logik", "komplexität", "eindeutigkeit", "vorhersehbarkeit"]

# Load the model and tokenizer (only gbert-large)
model_name = "deepset/gbert-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function for a single target column
def preprocess_function(examples, tokenizer, target_column):
    # Extract the label for the target column
    label = float(examples[target_column])

    # Tokenize the input text
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True,
                                 padding="max_length", max_length=256)

    # Add the label to the tokenized inputs
    tokenized_inputs["labels"] = torch.tensor(label, dtype=torch.float)

    return tokenized_inputs

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
    eise = []
    
    errors = logits - labels
    squared_errors = np.where(np.abs(errors)**2 > epsilon, 
                            (np.abs(errors)**2 - epsilon), 
                            0)
    eise.append(squared_errors.mean())
    
    eise_mean = np.mean(eise)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "seei": eise_mean,
        "accuracy": accuracy
    }

# Training arguments
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 30

def create_training_args(column, fold):
    return TrainingArguments(
    output_dir="Outputs/" + dataset_name + "/gbert-large-single" + f"/fold-{fold}" + f"/{column}",
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


# Custom Trainer for regression
class SingleRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        for key in inputs:
            inputs[key] = inputs[key].to(device)  # Move input tensors to device
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.mse_loss(logits, labels.view(-1, 1))  # Single output
        return (loss, outputs) if return_outputs else loss

# Precompute the splits for all folds
splits = list(kf.split(shuffled_df))

# Store performance metrics for each target column
column_results = []

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

        # Further split training data into training and validation sets
        val_data = train_data.sample(frac=1 / 6, random_state=23)  # 1/6 of training data for validation
        train_data = train_data.drop(val_data.index)  # Remaining 5/6 for training

        # Preprocess the datasets for the current target column
        def preprocess_dataset(df, tokenizer, target_column):
            df = df.apply(preprocess_function, tokenizer=tokenizer, target_column=target_column, axis=1)
            df = df.drop(columns=target_columns)  # Drop all target columns
            df = df.reset_index(drop=True)
            return df

        train_dataset = preprocess_dataset(train_data, tokenizer, target_column)
        val_dataset = preprocess_dataset(val_data, tokenizer, target_column)
        test_dataset = preprocess_dataset(test_data, tokenizer, target_column)

        # Create datasets dictionary
        datasets = {"train": train_dataset, "validation": val_dataset, "test": test_dataset}

        # Load the model for the current target column
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        model.config.problem_type = "regression"

        # Initialize trainer
        trainer = SingleRegressionTrainer(
            model=model,
            args=create_training_args(target_column, fold),
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=lambda eval_pred: compute_metrics_for_regression(eval_pred, target_column)
        )

        # Train the model
        model.to(device)
        trainer.train()
        model.to("cpu")
        torch.cuda.empty_cache()

        # Evaluate on the test set
        model.to(device)
        test_results = trainer.evaluate(eval_dataset=datasets["test"])
        model.to("cpu")
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
print("Average performance across all target columns:")
for result in column_results:
    print(result)