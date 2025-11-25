import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from torch import distributed as dist
from sklearn.model_selection import KFold
import numpy as np
import copy
import random
from torch import nn
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from transformers.trainer_utils import EvalLoopOutput

# Device setup
def init_process_group():
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    return rank == 0, torch.device(f'cuda:{local_rank}'), num_gpus

is_rank0, device, num_gpus = init_process_group()

# Load both datasets
data = pd.read_csv("CSV/ats-new_features.csv", header=0, delimiter=';')
train_df = pd.DataFrame(data)
train_df.drop(columns=train_df.columns[2:], inplace=True)

#Groupby sentence and average the labels 
train_df = train_df.groupby('sentence').mean().reset_index()

#Combine all datasets
combined_df = train_df

#Shuffle the combined dataset
shuffled_df1 = combined_df.sample(frac=1, random_state=23).reset_index(drop=True)

data = pd.read_csv("CSV/mt-new_features.csv", header=0, delimiter=';')

train_df = pd.DataFrame(data)
train_df.drop(columns=train_df.columns[2:], inplace=True)

#Groupby sentence and average the labels 
train_df = train_df.groupby('sentence').mean().reset_index()

#Combine all datasets
combined_df = train_df

#Shuffle the combined dataset
shuffled_df2 = combined_df.sample(frac=1, random_state=23).reset_index(drop=True)

# Initialize k-fold cross-validation (same splits for both datasets)
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=23)

# Precompute the splits for all folds
splits1 = list(kf.split(shuffled_df1))
splits2 = list(kf.split(shuffled_df2))

# Model names (we'll use only these two for multitask learning)
model_names = ["deepset/gbert-large", "deepset/gelectra-large"]

# Custom model for multitask learning
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_task1=1, num_labels_task2=1):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name).base_model
        self.task1_head = nn.Linear(self.backbone.config.hidden_size, num_labels_task1)
        self.task2_head = nn.Linear(self.backbone.config.hidden_size, num_labels_task2)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, task_id=0):
        if isinstance(task_id, torch.Tensor):
            task_id = task_id.item()

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        if task_id == 0:
            logits = self.task1_head(pooled_output)
        else:
            logits = self.task2_head(pooled_output)
            
        return logits

# Preprocessing functions
def preprocess_function_task1(examples, tokenizer):
    labels = [
        float(examples["overall"])    ]
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True,
                               padding="max_length", max_length=512)
    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float)
    tokenized_inputs["task_id"] = 0
    return tokenized_inputs

def preprocess_function_task2(examples, tokenizer):
    labels = [
        float(examples["overall"])    ]
    tokenized_inputs = tokenizer(examples["sentence"], truncation=True,
                               padding="max_length", max_length=512)
    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float)
    tokenized_inputs["task_id"] = 1
    return tokenized_inputs

# Custom dataset class
class MultiTaskDataset(Dataset):
    def __init__(self, dataset1, dataset2, tokenizer):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.tokenizer = tokenizer
        self.length = len(dataset1) + len(dataset2)  # Total length is sum of both

        # Create a fixed mapping of indices to dataset examples
        self.idx_mapping = []
        max_len = max(len(dataset1), len(dataset2))
        
        for i in range(max_len):
            if i < len(dataset1):
                self.idx_mapping.append(('dataset1', i))
            if i < len(dataset2):
                self.idx_mapping.append(('dataset2', i))
        
        # If one dataset is larger, fill remaining indices with the smaller dataset
        if len(dataset1) > len(dataset2):
            for i in range(len(dataset2), len(dataset1)):
                self.idx_mapping.append(('dataset1', i))
        elif len(dataset2) > len(dataset1):
            for i in range(len(dataset1), len(dataset2)):
                self.idx_mapping.append(('dataset2', i))
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        dataset_name, example_idx = self.idx_mapping[idx]
        
        if dataset_name == 'dataset1':
            item = self.dataset1.iloc[example_idx]
            processed = preprocess_function_task1(item, self.tokenizer)
        else:
            item = self.dataset2.iloc[example_idx]
            processed = preprocess_function_task2(item, self.tokenizer)
        
        processed['task_id'] = int(processed['task_id'])
        return processed
    

# Metrics computation
def compute_metrics_for_regression(eval_pred, task_id=0):
    logits, labels = eval_pred
    logits = logits.reshape(-1, 1)
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits, multioutput='raw_values')
    mae = mean_absolute_error(labels, logits, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, logits, multioutput='raw_values')
    
    metrics = {
        "mse": mse.mean(),
        "mae": mae.mean(),
        "rmse": rmse.mean(),
        "r2": r2.mean(),
    }
    
    # Add task-specific prefix
    return {f"task{task_id}_{k}": v for k, v in metrics.items()}

# Training arguments
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 30

def create_training_args(model_idx, fold):
    model_name = model_names[model_idx].replace("/", "_")
    return TrainingArguments(
        output_dir=f"Outputs/overall-multitask/{model_name}/fold_{fold}",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model="combined_rmse",  # Must match exactly
        greater_is_better=False,
        load_best_model_at_end=True,
        weight_decay=0.01,
    )

# Custom trainer for multitask learning
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get task IDs and labels
        task_ids = inputs.pop("task_id")  # Shape: [batch_size]
        labels = inputs.pop("labels")     # Shape: [batch_size, 1]
        
        # Move all inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
        labels = labels.to(device)
        task_ids = task_ids.to(device)
        
        # Forward pass for all examples
        outputs = []
        for i in range(len(task_ids)):
            # Get single example inputs
            single_input = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            # Forward pass with specific task head
            output = model(**single_input, task_id=task_ids[i].item())
            outputs.append(output)
        
        # Combine outputs
        outputs = torch.cat(outputs, dim=0)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
    self,
    dataloader,
    description,
    prediction_loss_only=None,
    ignore_keys=None,
    metric_key_prefix="eval"
):
        # Initialize variables
        task1_preds, task1_labels = [], []
        task2_preds, task2_labels = [], []
        total_loss = 0.0
        num_samples = 0
        
        # Get the model from the trainer instance
        model = self.model
        model.eval()
        
        for inputs in dataloader:
            task_ids = inputs.pop("task_id")
            labels = inputs.pop("labels")
            
            # Move everything to device
            inputs = {k: v.to(self.args.device) for k, v in inputs.items() if v is not None}
            labels = labels.to(self.args.device)
            task_ids = task_ids.to(self.args.device)
            
            with torch.no_grad():
                # Process each example separately
                batch_outputs = []
                for i in range(len(task_ids)):
                    single_input = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
                    output = model(**single_input, task_id=task_ids[i].item())
                    batch_outputs.append(output)
                
                outputs = torch.cat(batch_outputs, dim=0)
                
                # Compute loss for this batch
                loss = torch.nn.functional.mse_loss(outputs, labels)
                total_loss += loss.item() * len(task_ids)
                num_samples += len(task_ids)
            
            # Separate predictions by task
            for i, tid in enumerate(task_ids):
                if tid.item() == 0:
                    task1_preds.append(outputs[i].detach().cpu().numpy())
                    task1_labels.append(labels[i].detach().cpu().numpy())
                else:
                    task2_preds.append(outputs[i].detach().cpu().numpy())
                    task2_labels.append(labels[i].detach().cpu().numpy())
        
        # Compute metrics
        metrics = {}
        task0_rmse, task1_rmse = None, None
        
        if len(task1_preds) > 0:
            task1_preds = np.stack(task1_preds)
            task1_labels = np.stack(task1_labels)
            task1_metrics = compute_metrics_for_regression((task1_preds, task1_labels), task_id=0)
            task1_metrics = {k: float(v) for k, v in task1_metrics.items()}
            metrics.update({f"task0_{k}": v for k, v in task1_metrics.items()})
            task0_rmse = task1_metrics["task0_rmse"]
        
        if len(task2_preds) > 0:
            task2_preds = np.stack(task2_preds)
            task2_labels = np.stack(task2_labels)
            task2_metrics = compute_metrics_for_regression((task2_preds, task2_labels), task_id=1)
            task2_metrics = {k: float(v) for k, v in task2_metrics.items()}
            metrics.update({f"task1_{k}": v for k, v in task2_metrics.items()})
            task1_rmse = task2_metrics["task1_rmse"]
        
        # Calculate combined RMSE if both tasks were evaluated
        if task0_rmse is not None and task1_rmse is not None:
            combined_rmse = task0_rmse + task1_rmse
            metrics[f"eval_combined_rmse"] = combined_rmse
        
        # Calculate average loss
        metrics[f"{metric_key_prefix}_loss"] = total_loss / num_samples if num_samples > 0 else 0.0
        
        # Return as EvalLoopOutput object
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples
        )

# Store performance metrics for each model
model_results = []

# Perform k-fold cross-validation for each model
for model_idx, model_name in enumerate(model_names):
    print(f"Training model: {model_name}")
    
    # Initialize model
    model = MultiTaskModel(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    initialized_model_state_dict = copy.deepcopy(model.state_dict())  # Save the initial model state
    
    # Store performance metrics for each fold
    fold_scores = []
    
    for fold in range(k):
        print(f"  Fold {fold + 1}/{k}")

        # Reset model to initial state
        model.load_state_dict(initialized_model_state_dict)
        
        # Get the splits for this fold
        train_idx1, test_idx1 = splits1[fold]
        train_idx2, test_idx2 = splits2[fold]
        
        # For validation, use previous fold's test set (or last fold for first fold)
        if fold == 0:
            val_idx1 = splits1[-1][1]
            val_idx2 = splits2[-1][1]
        else:
            val_idx1 = splits1[fold-1][1]
            val_idx2 = splits2[fold-1][1]
        
        # Create datasets
        train_data1 = shuffled_df1.iloc[train_idx1]
        test_data1 = shuffled_df1.iloc[test_idx1]
        val_data1 = shuffled_df1.iloc[val_idx1]
        
        train_data2 = shuffled_df2.iloc[train_idx2]
        test_data2 = shuffled_df2.iloc[test_idx2]
        val_data2 = shuffled_df2.iloc[val_idx2]
        
        # Create combined datasets
        train_dataset = MultiTaskDataset(train_data1, train_data2, tokenizer)
        val_dataset = MultiTaskDataset(val_data1, val_data2, tokenizer)
        test_dataset = MultiTaskDataset(test_data1, test_data2, tokenizer)
        
        # Initialize trainer
        trainer = MultiTaskTrainer(
            model=model,
            args=create_training_args(model_idx, fold),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train the model
        model.to(device)
        trainer.train()
        
        # Evaluate on test set
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        model.to("cpu")
        torch.cuda.empty_cache()
        
        # Store test results
        fold_scores.append(test_results)
    
    # Calculate average performance across all folds for this model
    avg_metrics = {
        "model": model_name,
        "task0_mse": np.mean([score["task0_task0_mse"] for score in fold_scores]),
        "task0_mae": np.mean([score["task0_task0_mae"] for score in fold_scores]),
        "task0_rmse": np.mean([score["task0_task0_rmse"] for score in fold_scores]),
        "task0_r2": np.mean([score["task0_task0_r2"] for score in fold_scores]),
        "task1_mse": np.mean([score["task1_task1_mse"] for score in fold_scores]),
        "task1_mae": np.mean([score["task1_task1_mae"] for score in fold_scores]),
        "task1_rmse": np.mean([score["task1_task1_rmse"] for score in fold_scores]),
        "task1_r2": np.mean([score["task1_task1_r2"] for score in fold_scores]),
        "combined_rmse": np.mean([score["eval_combined_rmse"] for score in fold_scores]),
    }
    
    # Store results for this model
    model_results.append(avg_metrics)

# Print average performance for each model
print("Average performance across all models:")
for result in model_results:
    print(result)