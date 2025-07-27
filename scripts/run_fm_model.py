#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def main():
    print('--- 1. Loading and Preprocessing Data ---')
    DATA_PATH = 'data/final_dataset_with_llm_keywords.parquet'

    try:
        df = pd.read_parquet(DATA_PATH)
        print(f"Real dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Warning: The data file was not found at {DATA_PATH}")
        print('Using a dummy dataframe for demonstration purposes.')
        df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            'movie_id': [101, 102, 101, 103, 102, 104, 103, 104, 105, 106, 105, 106, 107, 108, 107, 108],
            'rating': [5, 3, 4, 2, 5, 4, 3, 5, 2, 4, 3, 5, 4, 3, 2, 5],
            'human_keywords': ['action, thriller', 'comedy, romance', 'action, thriller', 'sci-fi, adventure', 'comedy, romance', 'drama', 'sci-fi, adventure', 'drama', 'mystery', 'fantasy', 'mystery', 'fantasy', 'crime', 'history', 'crime', 'history'],
            'llm_keywords': ['fast-paced, explosive', 'lighthearted, love', 'fast-paced, explosive', 'space, future', 'lighthearted, love', 'emotional, serious', 'space, future', 'emotional, serious', 'suspenseful, investigation', 'magical, mythical', 'suspenseful, investigation', 'magical, mythical', 'gritty, investigation', 'period piece, factual', 'gritty, investigation', 'period piece, factual']
        })

    print('\n--- 2. Performing Feature Engineering ---')
    df['user_id_cat'] = df['user_id'].astype('category').cat.codes
    df['movie_id_cat'] = df['movie_id'].astype('category').cat.codes

    def create_feature_matrix(df, keywords_column):
        df[keywords_column] = df[keywords_column].fillna('').str.replace(',', ' ')
        vectorizer = CountVectorizer()
        keyword_bow = vectorizer.fit_transform(df[keywords_column])
        user_ohe = pd.get_dummies(df['user_id_cat'], prefix='user', sparse=True)
        movie_ohe = pd.get_dummies(df['movie_id_cat'], prefix='movie', sparse=True)
        features_sparse = hstack([user_ohe, movie_ohe, keyword_bow], format='csr')
        return features_sparse

    X_human = create_feature_matrix(df.copy(), 'human_keywords')
    X_llm = create_feature_matrix(df.copy(), 'llm_keywords')
    y = df['rating'].values

    print(f"Control (Human) Feature Matrix Shape: {X_human.shape}")
    print(f"Experimental (LLM) Feature Matrix Shape: {X_llm.shape}")

    def create_dataloaders(X, y, test_size=0.2, batch_size=1024, random_state=42):
        indices = np.arange(X.shape[0])
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        X_train_tensor = torch.from_numpy(X_train.toarray()).float()
        X_test_tensor = torch.from_numpy(X_test.toarray()).float()
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
        y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    train_loader_human, test_loader_human = create_dataloaders(X_human, y)
    train_loader_llm, test_loader_llm = create_dataloaders(X_llm, y)
    print('\nDataLoaders created successfully.')

    class FactorizationMachine(nn.Module):
        def __init__(self, num_features, embedding_dim=10):
            super(FactorizationMachine, self).__init__()
            self.bias = nn.Parameter(torch.randn(1))
            self.weights = nn.Linear(num_features, 1, bias=False)
            self.embeddings = nn.Linear(num_features, embedding_dim, bias=False)
        def forward(self, x):
            linear_part = self.bias + self.weights(x)
            embedded_x = self.embeddings(x)
            sum_of_squares = embedded_x.pow(2).sum(1, keepdim=True)
            square_of_sum = self.embeddings(x.pow(2)).sum(1, keepdim=True)
            factorization_part = 0.5 * (sum_of_squares - square_of_sum)
            return linear_part + factorization_part

    def train_model(model, train_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        return total_loss / len(train_loader.dataset)

    def evaluate_model(model, test_loader, criterion, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features)
                loss = criterion(predictions, targets)
                total_loss += loss.item() * len(targets)
        mse = total_loss / len(test_loader.dataset)
        return np.sqrt(mse)

    def run_experiment(train_loader, test_loader, num_features, num_epochs=10, learning_rate=0.01, embedding_dim=50):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = FactorizationMachine(num_features=num_features, embedding_dim=embedding_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            train_loss = train_model(model, train_loader, optimizer, criterion, device)
            test_rmse = evaluate_model(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test RMSE: {test_rmse:.4f}")
        return evaluate_model(model, test_loader, criterion, device)

    print('\n--- 3. Starting Control Group Experiment (Human Keywords) ---')
    num_features_human = X_human.shape[1]
    rmse_human = run_experiment(train_loader_human, test_loader_human, num_features_human)
    print(f"\\nFinal RMSE for Control (Human) Model: {rmse_human:.4f}")

    print('\n--- 4. Starting Experimental Group Experiment (LLM Keywords) ---')
    num_features_llm = X_llm.shape[1]
    rmse_llm = run_experiment(train_loader_llm, test_loader_llm, num_features_llm)
    print(f"\\nFinal RMSE for Experimental (LLM) Model: {rmse_llm:.4f}")

    print('\n========== 5. EXPERIMENT RESULTS ==========')
    print(f"RMSE (Human Keywords): {rmse_human:.4f}")
    print(f"RMSE (LLM Keywords):   {rmse_llm:.4f}")
    print('========================================')

    improvement = rmse_human - rmse_llm
    if improvement > 0.0001:
        print(f"\\nHypothesis Confirmed: LLM-based model performed better by an RMSE of {improvement:.4f}.")
    elif improvement < -0.0001:
        print(f"\\nHypothesis Rejected: Human-based model performed better by an RMSE of {-improvement:.4f}.")
    else:
        print('\nResult: No significant difference in model performance.')

if __name__ == '__main__':
    main()
