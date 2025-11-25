#Import libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os
from torch import distributed as dist
from sklearn.model_selection import KFold
import numpy as np
import copy

# Device
def init_process_group():
    """
    Join the process group and return whether this is the rank 0 process,
    the CUDA device to use, and the total number of GPUs used for training.
    """
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    #dist.init_process_group('nccl')
    return rank == 0, torch.device(f'cuda:{local_rank}'), num_gpus

is_rank0, device, num_gpus = init_process_group()

#Pick dataset
size = "big" # "small" or "big"
type = "averaged" # "averaged" or "stratified"
dataset_name = "{}_data_{}_correct_format".format(size, type)

#Load dataset
data = pd.read_csv("CSV/ats-new_features.csv", header=0, delimiter=';')

train_df = pd.DataFrame(data)
train_df.drop(columns=train_df.columns[2:], inplace=True)

#Groupby sentence and average the labels 
train_df = train_df.groupby('sentence').mean().reset_index()

#Combine all datasets
combined_df = train_df

#Shuffle the combined dataset
shuffled_df = combined_df.sample(frac=1, random_state=23).reset_index(drop=True)

# Initialize k-fold cross-validation
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=23)

model_names = ["dbmdz/bert-base-german-uncased", "dbmdz/bert-base-german-cased", "deepset/gbert-base", "deepset/gbert-large", "deepset/gelectra-large"]

models = [] #list of lists with tokenizer and model 

#Load all pretrained models and their tokenizers
for model_name in model_names:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.config.problem_type = "regression"
    models.append([tokenizer, model])

datasets = [{"train": train_df} for _ in models]

# Preprocessing function
def preprocess_function(examples, tokenizer):
    # Extract the multi-labels as a list of floats
    labels = [
        float(examples["overall"])]

    # Tokenize the input text
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True,
                                 padding="max_length", max_length=512)

    # Add the labels to the tokenized inputs
    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float)

    return tokenized_inputs

for i in range(len(models)):
    tokenizer = models[i][0]
    ds = datasets[i]
    # Apply the preprocess function to the dataset splits
    for split in ds:
        # Apply preprocess function to each row in the DataFrame
        ds[split] = ds[split].apply(preprocess_function, tokenizer=tokenizer, axis=1)

        # Remove the columns that are no longer needed
        ds[split] = ds[split].drop(columns=["overall"])

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    logits = logits.reshape(-1, 1)
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits, multioutput='raw_values')
    mae = mean_absolute_error(labels, logits, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, logits, multioutput='raw_values')
    mse_mean = mse.mean()  # Mean MSE across outputs
    mae_mean = mae.mean()  # Mean MAE across outputs
    rmse_mean = rmse.mean()  # Mean RMSE across outputs
    r2_mean = r2.mean()  # Mean R2 across outputs

    # Compute accuracy
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    #accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)
    accuracy = [(abs(logits[:, i] - labels[:, i]) < 0.25).mean() for i in range(1)]
    accuracy_mean = sum(accuracy) / len(accuracy)

    # Epsilon-insensitive squared error (ε=0.25)
    epsilon = 0.25
    eise = []
    for i in range(1):
        errors = logits[:, i] - labels[:, i]
        squared_errors = np.where(np.abs(errors)**2 > epsilon, 
                                (np.abs(errors)**2 - epsilon), 
                                0)
        eise.append(squared_errors.mean())
    
    eise_mean = np.mean(eise)

    return {
        "mse": mse_mean,
        "mae": mae_mean,
        "rmse": rmse_mean,
        "r2": r2_mean,
        "epsilon_insensitive_squared_error": eise_mean,
        "accuracy": accuracy_mean
    }

#-----Training Arguments

training_args = []

LEARNING_RATE = 2e-5
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 30

def create_training_args(model_idx, fold):
    i = model_idx
    model = models[i]
    model_name = model_names[i].replace("/", "_")
    return TrainingArguments(
        output_dir="Outputs/"+ "ats-new-overall" +"/" + model_name + f"/fold_{fold}",
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

class MultiRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").to(device)
        for key in inputs:
            inputs[key] = inputs[key].to(device)  # Move input tensors to device
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.mse_loss(logits, labels.view(-1, 1))
        return (loss, outputs) if return_outputs else loss

# Precompute the splits for all folds
splits = list(kf.split(shuffled_df))

# Store performance metrics for each model
model_results = []

# Perform k-fold cross-validation for each model
for model_idx, (tokenizer, model) in enumerate(models):
    print(f"Training model: {model_names[model_idx]}")

    model_initial_state = copy.deepcopy(model.state_dict())  # Save the initial model state

    # Store performance metrics for each fold
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {fold + 1}/{k}")

        model.load_state_dict(model_initial_state)  # Reset model to initial state

        # Split into training and test sets
        train_data = shuffled_df.iloc[train_idx]
        test_data = shuffled_df.iloc[test_idx]

        # Use the previous fold's test set as the validation set
        if fold == 0:
        # For the first fold, use the last fold's test set as validation
            val_data = shuffled_df.iloc[splits[-1][1]]  # Last fold's test set
        else:
        # For other folds, use the previous fold's test set as validation
            val_data = shuffled_df.iloc[splits[fold - 1][1]]  # Previous fold's test set
        train_data = train_data.drop(val_data.index)  # Remaining 5/6 for training

        # Preprocess the datasets
        def preprocess_dataset(df, tokenizer):
            df = df.apply(preprocess_function, tokenizer=tokenizer, axis=1)
            df = df.drop(columns=["overall"])
            df = df.reset_index(drop=True)
            return df

        train_dataset = preprocess_dataset(train_data, tokenizer)
        val_dataset = preprocess_dataset(val_data, tokenizer)
        test_dataset = preprocess_dataset(test_data, tokenizer)

        # Create datasets dictionary
        datasets = {"train": train_dataset, "validation": val_dataset, "test": test_dataset}

        # Initialize trainer
        trainer = MultiRegressionTrainer(
            model=model,
            args=create_training_args(model_idx, fold),
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=compute_metrics_for_regression
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

    # Calculate average performance across all folds for this model
    avg_metrics = {
        "model": model_names[model_idx],
        "mse": np.mean([score["eval_mse"] for score in fold_scores]),
        "mae": np.mean([score["eval_mae"] for score in fold_scores]),
        "rmse": np.mean([score["eval_rmse"] for score in fold_scores]),
        "r2": np.mean([score["eval_r2"] for score in fold_scores]),
        "epsilon_insensitive_squared_error": np.mean([score["eval_epsilon_insensitive_squared_error"] for score in fold_scores]),
        "accuracy": np.mean([score["eval_accuracy"] for score in fold_scores])
    }

    # Store results for this model
    model_results.append(avg_metrics)

# Print average performance for each model
print("Average performance across all models:")
for result in model_results:
    print(result)