import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

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

# Load dataset
train_data = pd.read_csv("CSV/crowdee-mt-ratings-for-model.csv", header=0, delimiter=';')
train_df = pd.DataFrame(train_data)

# Groupby sentence and average the labels
train_df = train_df.groupby('sentence').mean().reset_index()

# Shuffle the combined dataset
shuffled_df = train_df.sample(frac=1, random_state=23).reset_index(drop=True)

# Initialize k-fold cross-validation
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=23)

# Define the target columns
target_columns = ["präzision", "komplexität", "transparenz", "grammatikalität"]

# Load the model and tokenizer
model_name = "deepset/gelectra-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom Dataset class
class MultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sentence = row['sentence']
        labels = torch.tensor([row[col] for col in target_columns], dtype=torch.float)
        
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }

# Custom model for multi-task learning
class MultiTaskModel(torch.nn.Module):
    def __init__(self, model_name, num_tasks):
        super(MultiTaskModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.task_layers = torch.nn.ModuleList([
            torch.nn.Linear(self.bert.config.hidden_size, 1) for _ in range(num_tasks)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        logits = torch.cat([layer(pooled_output) for layer in self.task_layers], dim=1)
        return logits

# Training function
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask)
        loss = torch.nn.functional.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = torch.nn.functional.mse_loss(outputs, labels)
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    mse = mean_squared_error(all_labels, all_preds, multioutput='raw_values')
    mae = mean_absolute_error(all_labels, all_preds, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds, multioutput='raw_values')
    accuracy = [(abs(all_preds[:, i] - all_labels[:, i]) < 0.5).mean() for i in range(len(target_columns))]

    # Epsilon-insensitive squared error (ε=0.25)
    epsilon = 0.25
    eise = []
    for i in range(len(target_columns)):
        errors = all_preds[:, i] - all_labels[:, i]
        squared_errors = np.where(np.abs(errors)**2 > epsilon, 
                                (np.abs(errors)**2 - epsilon), 
                                0)
        eise.append(squared_errors.mean())
    
    eise_mean = np.mean(eise)
    
    return {
        "mse": np.mean(mse),
        "mae": np.mean(mae),
        "rmse": np.mean(rmse),
        "r2": np.mean(r2),
        "seei": eise_mean,
        "accuracy": np.mean(accuracy)
    }, total_loss / len(dataloader)

splits = list(kf.split(shuffled_df))

# Main training loop with k-fold cross-validation
column_results = []
for fold, (train_idx, test_idx) in enumerate(splits):
    print(f"Fold {fold + 1}/{k}")
    
    train_data = shuffled_df.iloc[train_idx]
    test_data = shuffled_df.iloc[test_idx]
    
    # Use the previous fold's test set as the validation set
    if fold == 0:
        # For the first fold, use the last fold's test set as validation
        val_data = shuffled_df.iloc[splits[-1][1]]  # Last fold's test set
    else:
        # For other folds, use the previous fold's test set as validation
        val_data = shuffled_df.iloc[splits[fold - 1][1]]  # Previous fold's test set

    train_data = train_data.drop(val_data.index)
    
    train_dataset = MultiTaskDataset(train_data, tokenizer)
    val_dataset = MultiTaskDataset(val_data, tokenizer)
    test_dataset = MultiTaskDataset(test_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    model = MultiTaskModel(model_name, num_tasks=len(target_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    best_val = float('inf')
    best_model = model.state_dict()
    for epoch in range(30):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_metrics, val_loss = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}")
        print(f"Validation Metrics: {val_metrics}")
        
        val_rmse = val_metrics["rmse"]
        if val_rmse < best_val:
            best_val = val_rmse
            best_model = model.state_dict()
    
    model.load_state_dict(best_model)
    test_metrics, _ = evaluate_model(model, test_loader, device)
    column_results.append(test_metrics)

    # Create the directory path if it doesn't exist
    output_dir = f"Outputs/summarization/gelectra-large-multitask/fold-{fold}/"
    os.makedirs(output_dir, exist_ok=True)

    # Now save the model
    torch.save(best_model, os.path.join(output_dir, "best_model.pth"))

# Calculate average performance across all folds
avg_metrics = {
    "mse": np.mean([score["mse"] for score in column_results]),
    "mae": np.mean([score["mae"] for score in column_results]),
    "rmse": np.mean([score["rmse"] for score in column_results]),
    "r2": np.mean([score["r2"] for score in column_results]),
    "seei": np.mean([score["seei"] for score in column_results]),
    "accuracy": np.mean([score["accuracy"] for score in column_results])
}

# Print average performance
print("Average performance across all folds:")
print(avg_metrics)