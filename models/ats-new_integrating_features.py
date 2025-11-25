import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from text_features import get_all_text_features  # Import the feature extraction function
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
features_df = pd.read_csv("CSV/ats_features.csv", header=0, delimiter=';') 
features_df = features_df.groupby('sentence').mean().reset_index()

data = pd.read_csv("CSV/ats-new_features.csv", header=0, delimiter=';')
new_df = pd.DataFrame(data)
new_df.drop(columns=new_df.columns[1:2], inplace=True)
new_df = new_df.groupby('sentence').mean().reset_index()
 
features_df = pd.concat([features_df, new_df], ignore_index=True)

# Remove problematic features
features_df = features_df.drop(columns=['dugast_uber_index', 'herdans_vm', 'yules_i'], errors='ignore')

# Shuffle the combined dataset
shuffled_df = features_df.sample(frac=1, random_state=23).reset_index(drop=True)

# Initialize k-fold cross-validation
k = 7
kf = KFold(n_splits=k, shuffle=True, random_state=23)

# Define the target columns
target_columns = ["sprachliche logik", "komplexität", "eindeutigkeit", "vorhersehbarkeit"]

# Extract textual features
feature_functions = get_all_text_features()
feature_names = list(feature_functions.keys())

# Compute features for the dataset
feature_data = []
for sentence in shuffled_df["sentence"]:
    feature_values = {name: func(sentence) for name, func in feature_functions.items()}
    feature_data.append(feature_values)

# Convert features to DataFrame
feature_df = pd.DataFrame(feature_data)
feature_df = feature_df.apply(pd.to_numeric, errors='coerce')

# Normalize textual features
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(feature_df)
normalized_feature_df = pd.DataFrame(normalized_features, columns=feature_names)

# Combine normalized features with the original dataset
shuffled_df = pd.concat([shuffled_df.reset_index(drop=True), normalized_feature_df], axis=1)

# DEBUG: Check for duplicate columns more thoroughly
print("=== DEBUG: Column Analysis ===")
print(f"Total columns: {len(shuffled_df.columns)}")
print(f"Unique columns: {len(set(shuffled_df.columns))}")

# Check for duplicates
duplicates = shuffled_df.columns[shuffled_df.columns.duplicated()]
print(f"Duplicate columns: {list(duplicates)}")

# Print all columns to see what's happening
print("All columns:")
for i, col in enumerate(shuffled_df.columns):
    print(f"{i}: {col}")

# Remove duplicate columns more aggressively
shuffled_df = shuffled_df.loc[:, ~shuffled_df.columns.duplicated()]

# Also check for duplicate column names case-insensitively
lowercase_columns = [col.lower() for col in shuffled_df.columns]
if len(lowercase_columns) != len(set(lowercase_columns)):
    print("Warning: Case-insensitive duplicates found!")
    # Create unique column names
    seen = {}
    new_columns = []
    for col in shuffled_df.columns:
        lower_col = col.lower()
        if lower_col in seen:
            seen[lower_col] += 1
            new_columns.append(f"{col}_{seen[lower_col]}")
        else:
            seen[lower_col] = 0
            new_columns.append(col)
    shuffled_df.columns = new_columns

print(f"Columns after cleanup: {len(shuffled_df.columns)}")

# Load the gbert-large model and tokenizer
model_name = "deepset/gbert-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get all feature columns (excluding text and labels)
feature_columns = [col for col in shuffled_df.columns 
                  if col not in ['sentence', 'sprachliche logik', 'komplexität', 
                                'eindeutigkeit', 'vorhersehbarkeit']]

print(f"Final feature columns count: {len(feature_columns)}")
print(f"First 10 feature columns: {feature_columns[:10]}")

# Custom Dataset class with textual features
class TextFeatureDataset(Dataset):
    def __init__(self, dataframe, tokenizer, feature_names, max_length=512, split="train", scaler=None):
        # Create a clean copy of the dataframe
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Filter feature names to only those that exist in the dataframe
        self.feature_names = [col for col in feature_names if col in self.dataframe.columns]
        print(f"Dataset: {split}, Available features: {len(self.feature_names)}")
        
        if split == "train":
            self.scaler = MinMaxScaler()
            # Use direct assignment instead of loc to avoid reindexing issues
            feature_values = self.dataframe[self.feature_names].values
            scaled_features = self.scaler.fit_transform(feature_values)
            for i, col in enumerate(self.feature_names):
                self.dataframe[col] = scaled_features[:, i]
        else:
            self.scaler = scaler
            if self.scaler is not None:
                feature_values = self.dataframe[self.feature_names].values
                scaled_features = self.scaler.transform(feature_values)
                for i, col in enumerate(self.feature_names):
                    self.dataframe[col] = scaled_features[:, i]
        
    def __len__(self):
        return len(self.dataframe)
    
    def getScaler(self):
        return self.scaler

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sentence = row['sentence']
        labels = torch.tensor([row[col] for col in target_columns], dtype=torch.float)
        
        # Get text features
        text_features = torch.tensor(row[self.feature_names].values.astype(np.float32), dtype=torch.float)
        
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
           
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text_features': text_features,
            'labels': labels
        }

# Custom model for combining BERT embeddings with textual features
class BertWithFeatures(torch.nn.Module):
    def __init__(self, model_name, num_tasks, feature_dim):
        super(BertWithFeatures, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.2)
        
        bert_hidden_size = self.bert.config.hidden_size
        combined_dim = bert_hidden_size + feature_dim
        print(f"Model: BERT hidden size: {bert_hidden_size}, Feature dim: {feature_dim}, Combined: {combined_dim}")
        
        self.classifier = torch.nn.Linear(combined_dim, num_tasks)

    def forward(self, input_ids, attention_mask, text_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        combined_features = torch.cat([pooled_output, text_features], dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        return logits

# Training function
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        text_features = batch['text_features'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask, text_features)
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
            text_features = batch['text_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, text_features)
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
    accuracy = [(abs(all_preds[:, i] - all_labels[:, i]) < 0.25).mean() for i in range(len(target_columns))]
    
    return {
        "mse": np.mean(mse),
        "mae": np.mean(mae),
        "rmse": np.mean(rmse),
        "r2": np.mean(r2),
        "accuracy": np.mean(accuracy)
    }, total_loss / len(dataloader)

# Main training loop
splits = list(kf.split(shuffled_df))
column_results = []

for fold, (train_idx, test_idx) in enumerate(splits):
    print(f"\n=== Fold {fold + 1}/{k} ===")
    
    train_data = shuffled_df.iloc[train_idx]
    test_data = shuffled_df.iloc[test_idx]
    
    # Validation set handling
    if fold == 0:
        val_data = shuffled_df.iloc[splits[-1][1]]
    else:
        val_data = shuffled_df.iloc[splits[fold - 1][1]]

    train_data = train_data[~train_data.index.isin(val_data.index)]

    print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    try:
        # Create datasets
        train_dataset = TextFeatureDataset(train_data, tokenizer, feature_names=feature_columns, split="train")
        
        # Get actual feature dimension
        sample_item = train_dataset[0]
        actual_feature_dim = sample_item['text_features'].shape[0]
        print(f"Actual feature dimension: {actual_feature_dim}")
        
        val_dataset = TextFeatureDataset(val_data, tokenizer, feature_names=feature_columns, 
                                       split="val", scaler=train_dataset.getScaler())
        test_dataset = TextFeatureDataset(test_data, tokenizer, feature_names=feature_columns, 
                                        split="test", scaler=train_dataset.getScaler())

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Initialize model
        model = BertWithFeatures(model_name, num_tasks=len(target_columns), 
                               feature_dim=actual_feature_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # Training loop
        best_val = float('inf')
        for epoch in range(30):
            train_loss = train_model(model, train_loader, optimizer, device)
            val_metrics, val_loss = evaluate_model(model, val_loader, device)
            
            if val_metrics["rmse"] < best_val:
                best_val = val_metrics["rmse"]
                torch.save(model.state_dict(), f"ats-new-hybrid_best_model_fold_{fold + 1}.pt")
            
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
        
        # Test best model
        model.load_state_dict(torch.load(f"ats-new-hybrid_best_model_fold_{fold + 1}.pt"))
        test_metrics, _ = evaluate_model(model, test_loader, device)
        column_results.append(test_metrics)
        print(f"Fold {fold + 1} completed successfully")
        
    except Exception as e:
        print(f"Error in fold {fold + 1}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Calculate average performance
if column_results:
    avg_metrics = {
        "mse": np.mean([score["mse"] for score in column_results]),
        "mae": np.mean([score["mae"] for score in column_results]),
        "rmse": np.mean([score["rmse"] for score in column_results]),
        "r2": np.mean([score["r2"] for score in column_results]),
        "accuracy": np.mean([score["accuracy"] for score in column_results])
    }
    print("\n=== Final Results ===")
    print(avg_metrics)
else:
    print("No successful folds completed.")