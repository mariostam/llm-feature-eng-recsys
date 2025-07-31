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
import random
import os
import optuna

warnings.filterwarnings('ignore')

def set_seed(seed: int = 42):
    """Set the seed for reproducibility in python, numpy and torch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('\nDataLoaders created successfully.')

def create_feature_matrix(df, keyword_column):
    # Convert user_id and movie_id to one-hot encoded features
    user_features = pd.get_dummies(df['user_id_cat'], prefix='user')
    movie_features = pd.get_dummies(df['movie_id_cat'], prefix='movie')

    # Use CountVectorizer for keywords
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
    keyword_features = vectorizer.fit_transform(df[keyword_column])

    # Combine all features
    X = hstack([user_features, movie_features, keyword_features])
    return X

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

def bootstrap_rmse(model, test_dataset, criterion, device, n_bootstraps=10000):
    bootstrapped_rmses = []
    dataset_size = len(test_dataset)
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping RMSE"):
        # Resample with replacement
        indices = np.random.choice(dataset_size, dataset_size, replace=True)
        bootstrapped_subset = torch.utils.data.Subset(test_dataset, indices)
        bootstrapped_loader = DataLoader(bootstrapped_subset, batch_size=1024, shuffle=False)
        rmse = evaluate_model(model, bootstrapped_loader, criterion, device)
        bootstrapped_rmses.append(rmse)
    return np.array(bootstrapped_rmses)

def run_experiment(train_loader, test_loader, test_dataset, num_features, num_epochs, learning_rate=0.01, embedding_dim=50, weight_decay=0, optimizer_name='Adam', run_bootstrap=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = FactorizationMachine(num_features=num_features, embedding_dim=embedding_dim).to(device)
    criterion = nn.MSELoss()
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_model(model, train_loader, optimizer, criterion, device)
        train_rmse = evaluate_model(model, train_loader, criterion, device)
        test_rmse = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
    
    ci_95 = (0, 0)
    ci_99 = (0, 0)
    if run_bootstrap:
        # Perform bootstrapping for confidence interval
        bootstrapped_rmses = bootstrap_rmse(model, test_dataset, criterion, device)
        
        # 95% CI
        lower_bound_95 = np.percentile(bootstrapped_rmses, 2.5)
        upper_bound_95 = np.percentile(bootstrapped_rmses, 97.5)
        ci_95 = (lower_bound_95, upper_bound_95)

        # 99% CI
        lower_bound_99 = np.percentile(bootstrapped_rmses, 0.5)
        upper_bound_99 = np.percentile(bootstrapped_rmses, 99.5)
        ci_99 = (lower_bound_99, upper_bound_99)
    
    final_train_rmse = evaluate_model(model, train_loader, criterion, device)
    final_test_rmse = evaluate_model(model, test_loader, criterion, device)

    return final_train_rmse, final_test_rmse, ci_95, ci_99

def objective(trial, X, y, num_features):
    # Define hyperparameters to tune
    embedding_dim = trial.suggest_int('embedding_dim', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    num_epochs = trial.suggest_int('num_epochs', 5, 50) # Allow Optuna to tune the number of epochs
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    batch_size = trial.suggest_int('batch_size', 32, 2048, log=True)

    train_loader, test_loader, test_dataset = create_dataloaders(X, y, batch_size=batch_size)

    # Run the experiment with the suggested hyperparameters
    _, test_rmse, _, _ = run_experiment(
        train_loader,
        test_loader,
        test_dataset,
        num_features,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        embedding_dim=embedding_dim,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        run_bootstrap=False # Disable bootstrapping during tuning for speed
    )
    return test_rmse



def create_dataloaders(X, y, batch_size, test_size=0.2, random_state=42):
    indices = np.arange(X.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    X_train, X_test = X.tocsr()[train_indices], X.tocsr()[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    X_train_tensor = torch.from_numpy(X_train.toarray()).float()
    X_test_tensor = torch.from_numpy(X_test.toarray()).float()
    y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
    y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    set_seed(42)
    print('--- 1. Loading and Preprocessing Data ---')
    DATA_PATH = 'gs://llm-feature-engineering-thesis-bucket/final_llm_features_dataset.parquet'

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

    X_human = create_feature_matrix(df.copy(), 'human_keywords')
    X_llm = create_feature_matrix(df.copy(), 'llm_keywords')
    y = df['rating'].values

    print(f"Control (Human) Feature Matrix Shape: {X_human.shape}")
    print(f"Experimental (LLM) Feature Matrix Shape: {X_llm.shape}")

    # These will be created inside the objective function with the suggested batch size
    train_loader_human, test_loader_human, test_dataset_human = None, None, None
    train_loader_llm, test_loader_llm, test_dataset_llm = None, None, None
    print('\nDataLoaders created successfully.')

    print('\n--- 3. Starting Hyperparameter Tuning for Human Model ---')
    study_human = optuna.create_study(direction='minimize')
    study_human.optimize(lambda trial: objective(trial, X_human, y, X_human.shape[1]), n_trials=120)
    best_params_human = study_human.best_trial.params
    print("Best trial for Human Model:")
    print(f"  Value: {study_human.best_trial.value}")
    print("  Params: ")
    for key, value in best_params_human.items():
        print(f"    {key}: {value}")

    print('\n--- 4. Starting Hyperparameter Tuning for LLM Model ---')
    study_llm = optuna.create_study(direction='minimize')
    study_llm.optimize(lambda trial: objective(trial, X_llm, y, X_llm.shape[1]), n_trials=120)
    best_params_llm = study_llm.best_trial.params
    print("Best trial for LLM Model:")
    print(f"  Value: {study_llm.best_trial.value}")
    print("  Params: ")
    for key, value in best_params_llm.items():
        print(f"    {key}: {value}")

    print('\n--- 5. Starting Final Control Group Experiment (Human Keywords) with Best Hyperparameters ---')
    num_features_human = X_human.shape[1]
    train_loader_human, test_loader_human, test_dataset_human = create_dataloaders(X_human, y, batch_size=best_params_human['batch_size'])
    train_rmse_human, test_rmse_human, ci_95_human, ci_99_human = run_experiment(
        train_loader_human,         test_loader_human,         test_dataset_human,         num_features_human,         num_epochs=best_params_human['num_epochs'],
        learning_rate=best_params_human['learning_rate'],         embedding_dim=best_params_human['embedding_dim'],        weight_decay=best_params_human['weight_decay'],
        optimizer_name=best_params_human['optimizer']
    )
    print(f"\nFinal Train RMSE for Control (Human) Model: {train_rmse_human:.4f}")
    print(f"Final Test RMSE for Control (Human) Model: {test_rmse_human:.4f}")


    print('\n--- 6. Starting Final Experimental Group Experiment (LLM Keywords) with Best Hyperparameters ---')
    num_features_llm = X_llm.shape[1]
    train_loader_llm, test_loader_llm, test_dataset_llm = create_dataloaders(X_llm, y, batch_size=best_params_llm['batch_size'])
    train_rmse_llm, test_rmse_llm, ci_95_llm, ci_99_llm = run_experiment(
        train_loader_llm,         test_loader_llm,         test_dataset_llm,         num_features_llm,         num_epochs=best_params_llm['num_epochs'],
        learning_rate=best_params_llm['learning_rate'],         embedding_dim=best_params_llm['embedding_dim'],        weight_decay=best_params_llm['weight_decay'],
        optimizer_name=best_params_llm['optimizer']
    )

    print('\n========== 7. EXPERIMENT RESULTS ==========')
    print(f"Train RMSE (Human Keywords): {train_rmse_human:.4f}")
    print(f"Test RMSE (Human Keywords): {test_rmse_human:.4f} (95% CI: {ci_95_human[0]:.4f}-{ci_95_human[1]:.4f}) (99% CI: {ci_99_human[0]:.4f}-{ci_99_human[1]:.4f})")
    print(f"Train RMSE (LLM Keywords):   {train_rmse_llm:.4f}")
    print(f"Test RMSE (LLM Keywords):   {test_rmse_llm:.4f} (95% CI: {ci_95_llm[0]:.4f}-{ci_95_llm[1]:.4f}) (99% CI: {ci_99_llm[0]:.4f}-{ci_99_llm[1]:.4f})")
    print('========================================')

    
    if ci_95_llm[1] < ci_95_human[0]:
        print(f"\nHypothesis Confirmed (at 95% confidence): LLM-based model performed statistically significantly better (95% CI: {ci_95_llm[0]:.4f}-{ci_95_llm[1]:.4f} vs {ci_95_human[0]:.4f}-{ci_95_human[1]:.4f}).")
    elif ci_95_human[1] < ci_95_llm[0]:
        print(f"\nHypothesis Rejected (at 95% confidence): Human-based model performed statistically significantly better (95% CI: {ci_95_human[0]:.4f}-{ci_95_human[1]:.4f} vs {ci_95_llm[0]:.4f}-{ci_95_llm[1]:.4f}).")
    else:
        print('\nResult: No statistically significant difference in model performance (95% CIs overlap).')

    if ci_99_llm[1] < ci_99_human[0]:
        print(f"\nHypothesis Confirmed (at 99% confidence): LLM-based model performed statistically significantly better (99% CI: {ci_99_llm[0]:.4f}-{ci_99_llm[1]:.4f} vs {ci_99_human[0]:.4f}-{ci_99_human[1]:.4f}).")
    elif ci_99_human[1] < ci_99_llm[0]:
        print(f"\nHypothesis Rejected (at 99% confidence): Human-based model performed statistically significantly better (99% CI: {ci_99_human[0]:.4f}-{ci_99_human[1]:.4f} vs {ci_99_llm[0]:.4f}-{ci_99_llm[1]:.4f}).")
    else:
        print('\nResult: No statistically significant difference in model performance (99% CIs overlap).')


if __name__ == '__main__':
    main()
