import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from text_features import get_all_text_features  # Import the feature extraction function

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load features
features_df = pd.read_csv("CSV/ats_features.csv", header=0, delimiter=';') 
features_df = features_df.groupby('sentence').mean().reset_index()  # If needed

# Remove the problematic feature completely, they contain some NaN or inf values
features_df = features_df.drop(columns=['dugast_uber_index'], errors='ignore')
features_df = features_df.drop(columns=['herdans_vm'], errors='ignore')
features_df = features_df.drop(columns=['yules_i'], errors='ignore')

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
feature_dim = len(feature_names)  # Number of textual features

# Compute features for the dataset
feature_data = []
for sentence in shuffled_df["sentence"]:
    feature_values = {name: func(sentence) for name, func in feature_functions.items()}
    feature_data.append(feature_values)

# Convert features to DataFrame
feature_df = pd.DataFrame(feature_data)

# Ensure all textual features are numeric
feature_df = feature_df.apply(pd.to_numeric, errors='coerce')

# Normalize textual features (using StandardScaler)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(feature_df)
normalized_feature_df = pd.DataFrame(normalized_features, columns=feature_names)

# Combine normalized features with the original dataset
shuffled_df = pd.concat([shuffled_df.reset_index(drop=True), normalized_feature_df], axis=1)

# Load the gbert-large model and tokenizer
model_name = "deepset/gbert-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom Dataset class with textual features
class TextFeatureDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, split="train", scaler=None, feature_names):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        if split=="train":
            self.scaler = MinMaxScaler()
            self.dataframe[feature_names] = self.scaler.fit_transform(self.dataframe[feature_names])
        else:
            self.scaler = scaler
            self.dataframe[feature_names] = self.scaler.transform(self.dataframe[feature_names])
        
    def __len__(self):
        return len(self.dataframe)
    
    def getScaler(self):
        return self.scaler

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sentence = row['sentence']
        labels = torch.tensor([row[col] for col in target_columns], dtype=torch.float)
        
        # Ensure text_features are numeric and convert to tensor
        text_features = torch.tensor(row[feature_names].values.astype(np.float32), dtype=torch.float)
        
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
            'text_features': text_features,
            'labels': labels
        }

# Custom model for combining BERT embeddings with textual features
class BertWithFeatures(torch.nn.Module):
    def __init__(self, model_name, num_tasks, feature_dim):
        super(BertWithFeatures, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size + feature_dim, num_tasks)

    def forward(self, input_ids, attention_mask, text_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        
        # Combine BERT embeddings with textual features
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

# Main training loop with k-fold cross-validation
splits = list(kf.split(shuffled_df))
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

    # Get all feature columns (excluding text and labels)
    feature_columns = [col for col in shuffled_df.columns 
                    if col not in ['sentence', 'sprachliche logik', 'komplexität', 
                                    'eindeutigkeit', 'vorhersehbarkeit']]

    train_dataset = TextFeatureDataset(train_data, tokenizer, split="train", feature_names=feature_columns)
    val_dataset = TextFeatureDataset(val_data, tokenizer, split="val", scaler=train_dataset.getScaler(), feature_names=feature_columns)
    test_dataset = TextFeatureDataset(test_data, tokenizer, split="test", scaler=train_dataset.getScaler(), feature_names=feature_columns)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    model = BertWithFeatures(model_name, num_tasks=len(target_columns), feature_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    best_val = float('inf')
    best_model = model.state_dict()
    for epoch in range(30):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_metrics, val_loss = evaluate_model(model, val_loader, device)
        val_rmse = val_metrics["rmse"]
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), f"best_model_fold_{fold + 1}.pt")

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}")
        print(f"Validation Metrics: {val_metrics}")
        

    
    model.load_state_dict(torch.load(f"best_model_fold_{fold + 1}.pt"))
    test_metrics, _ = evaluate_model(model, test_loader, device)
    column_results.append(test_metrics)

# Calculate average performance across all folds
avg_metrics = {
    "mse": np.mean([score["mse"] for score in column_results]),
    "mae": np.mean([score["mae"] for score in column_results]),
    "rmse": np.mean([score["rmse"] for score in column_results]),
    "r2": np.mean([score["r2"] for score in column_results]),
    "accuracy": np.mean([score["accuracy"] for score in column_results])
}

# Print average performance
print("Average performance across all folds:")
print(avg_metrics)