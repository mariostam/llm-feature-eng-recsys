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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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
    return X, vectorizer

class FactorizationMachine(nn.Module):
    def __init__(self, num_features, embedding_dim=10):
        super(FactorizationMachine, self).__init__()
        self.bias = nn.Parameter(torch.randn(1))
        self.weights = nn.Linear(num_features, 1, bias=False)
        self.embeddings = nn.Linear(num_features, embedding_dim, bias=False)
    def forward(self, x, return_components=False):
        linear_part = self.bias + self.weights(x)
        embedded_x = self.embeddings(x)
        
        # More numerically stable way to compute the interaction term
        sum_of_squares = embedded_x.pow(2).sum(1, keepdim=True)
        square_of_sum = self.embeddings(x).sum(1, keepdim=True).pow(2)
        
        interaction_part = 0.5 * (square_of_sum - sum_of_squares)
        
        if return_components:
            return linear_part, interaction_part

        return linear_part + interaction_part

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    
    train_rmse_history = []
    test_rmse_history = []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_model(model, train_loader, optimizer, criterion, device)
        train_rmse = evaluate_model(model, train_loader, criterion, device)
        test_rmse = evaluate_model(model, test_loader, criterion, device)
        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)
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

    return final_train_rmse, final_test_rmse, ci_95, ci_99, model, train_rmse_history, test_rmse_history

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
    _, test_rmse, _, _, _, _, _ = run_experiment(
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

def inspect_linear_weights(model, df, vectorizer, model_type):
    print(f"\n--- Linear Weights Inspection for {model_type} Model ---")
    weights = model.weights.weight.detach().cpu().numpy().flatten()

    num_users = df['user_id_cat'].nunique()
    num_movies = df['movie_id_cat'].nunique()
    num_keywords = len(vectorizer.vocabulary_)

    # User weights
    user_weights = weights[0:num_users]
    avg_user_weight = np.mean(np.abs(user_weights))
    print(f"Average absolute user weight: {avg_user_weight:.4f}")

    # Movie weights
    movie_weights = weights[num_users:num_users + num_movies]
    avg_movie_weight = np.mean(np.abs(movie_weights))
    print(f"Average absolute movie weight: {avg_movie_weight:.4f}")

    # Keyword weights
    keyword_weights = weights[num_users + num_movies:num_users + num_movies + num_keywords]
    avg_keyword_weight = np.mean(np.abs(keyword_weights))
    print(f"Average absolute keyword weight: {avg_keyword_weight:.4f}")

    # Compare
    max_avg_weight = max(avg_user_weight, avg_movie_weight, avg_keyword_weight)
    if max_avg_weight == avg_user_weight:
        print("User features appear to have the highest average linear weight.")
    elif max_avg_weight == avg_movie_weight:
        print("Movie features appear to have the highest average linear weight.")
    else:
        print("Keyword features appear to have the highest average linear weight.")

    print("Note: This only considers the linear part of the FM. The factorization part (embeddings) also contributes significantly.")

def visualize_keyword_embeddings(model, vectorizer, model_type, df):
    print(f"\n--- Visualizing Keyword Embeddings for {model_type} Model ---")
    num_users = df['user_id_cat'].nunique()
    num_movies = df['movie_id_cat'].nunique()
    
    keyword_embeddings = model.embeddings.weight.detach().cpu().numpy()[num_users + num_movies:]
    n_samples = keyword_embeddings.shape[0]

    # Add a guard clause to prevent crashing if the vocabulary is too small
    if n_samples <= 1:
        print(f"Skipping t-SNE visualization for {model_type} model: not enough keywords ({n_samples}) to visualize.")
        return

    # Dynamically set perplexity to be less than n_samples
    perplexity_value = min(30, n_samples - 1)
    if perplexity_value <= 0:
        print(f"Skipping t-SNE visualization for {model_type} model: perplexity must be positive.")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, n_iter=300)
    keyword_tsne = tsne.fit_transform(keyword_embeddings)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(keyword_tsne[:, 0], keyword_tsne[:, 1], alpha=0.5)
    plt.title(f't-SNE Visualization of {model_type} Keyword Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Annotate some points
    vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    # Ensure we don't try to annotate more points than exist
    num_to_annotate = min(15, n_samples)
    indices_to_annotate = np.random.choice(len(vocab), num_to_annotate, replace=False)
    for i in indices_to_annotate:
        plt.annotate(vocab[i], (keyword_tsne[i, 0], keyword_tsne[i, 1]))
        
    filename = f'{model_type.lower()}_embeddings_tsne.png'
    plt.savefig(filename)
    print(f"Saved t-SNE plot to {filename}")


def analyze_prediction_contribution(model, test_loader, device):
    model.eval()
    total_linear_contribution = 0
    total_interaction_contribution = 0
    total_samples = 0

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            linear_part, interaction_part = model(features, return_components=True)
            
            total_linear_contribution += torch.abs(linear_part).sum().item()
            total_interaction_contribution += torch.abs(interaction_part).sum().item()
            total_samples += len(features)

    avg_linear_contribution = total_linear_contribution / total_samples
    avg_interaction_contribution = total_interaction_contribution / total_samples
    
    total_contribution = avg_linear_contribution + avg_interaction_contribution
    
    percent_linear = (avg_linear_contribution / total_contribution) * 100
    percent_interaction = (avg_interaction_contribution / total_contribution) * 100

    return percent_linear, percent_interaction

def plot_learning_curves(human_history, llm_history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(human_history[0], label='Train RMSE')
    plt.plot(human_history[1], label='Test RMSE')
    plt.title('Human Model Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(llm_history[0], label='Train RMSE')
    plt.plot(llm_history[1], label='Test RMSE')
    plt.title('LLM Model Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("\nSaved learning curves plot to learning_curves.png")

def plot_error_distribution(model_human, model_llm, test_loader_human, test_loader_llm, device):
    model_human.eval()
    model_llm.eval()
    errors_human = []
    errors_llm = []
    with torch.no_grad():
        for features, targets in test_loader_human:
            features, targets = features.to(device), targets.to(device)
            predictions = model_human(features)
            errors_human.extend((predictions - targets).cpu().numpy())
        for features, targets in test_loader_llm:
            features, targets = features.to(device), targets.to(device)
            predictions = model_llm(features)
            errors_llm.extend((predictions - targets).cpu().numpy())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(errors_human, bins=50, alpha=0.7)
    plt.title('Human Model Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(errors_llm, bins=50, alpha=0.7)
    plt.title('LLM Model Prediction Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    print("Saved error distribution plot to error_distribution.png")

def main():
    set_seed(42)
    print('Loading and Preprocessing Data...')
    # For reproducibility by other users, load the dataset directly from GitHub.
    # This avoids the need for Google Cloud Storage setup.
    DATA_PATH = 'https://github.com/mariostam/llm-feature-eng-recsys/raw/main/data/final_llm_features_dataset.parquet'

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

    print('\nPerforming Feature Engineering...')
    df['user_id_cat'] = df['user_id'].astype('category').cat.codes
    df['movie_id_cat'] = df['movie_id'].astype('category').cat.codes

    X_human, vectorizer_human = create_feature_matrix(df.copy(), 'human_keywords')
    X_llm, vectorizer_llm = create_feature_matrix(df.copy(), 'llm_keywords')
    y = df['rating'].values

    print(f"Control (Human) Feature Matrix Shape: {X_human.shape}")
    print(f"Experimental (LLM) Feature Matrix Shape: {X_llm.shape}")

    # These will be created inside the objective function with the suggested batch size
    train_loader_human, test_loader_human, test_dataset_human = None, None, None
    train_loader_llm, test_loader_llm, test_dataset_llm = None, None, None
    print('\nDataLoaders created successfully.')

    print('\nStarting Hyperparameter Tuning for Human Model...')
    study_human = optuna.create_study(direction='minimize')
    study_human.optimize(lambda trial: objective(trial, X_human, y, X_human.shape[1]), n_trials=120)
    best_params_human = study_human.best_trial.params
    print("Best trial for Human Model:")
    print(f"  Value: {study_human.best_trial.value}")
    print("  Params: ")
    for key, value in best_params_human.items():
        print(f"    {key}: {value}")

    print('\nStarting Hyperparameter Tuning for LLM Model...')
    study_llm = optuna.create_study(direction='minimize')
    study_llm.optimize(lambda trial: objective(trial, X_llm, y, X_llm.shape[1]), n_trials=120)
    best_params_llm = study_llm.best_trial.params
    print("Best trial for LLM Model:")
    print(f"  Value: {study_llm.best_trial.value}")
    print("  Params: ")
    for key, value in best_params_llm.items():
        print(f"    {key}: {value}")

    print('\nStarting Final Control Group Experiment (Human Keywords) with Best Hyperparameters...')
    num_features_human = X_human.shape[1]
    train_loader_human, test_loader_human, test_dataset_human = create_dataloaders(X_human, y, batch_size=best_params_human['batch_size'])
    train_rmse_human, test_rmse_human, ci_95_human, ci_99_human, model_human, train_rmse_history_human, test_rmse_history_human = run_experiment(
        train_loader_human,         test_loader_human,         test_dataset_human,         num_features_human,         num_epochs=best_params_human['num_epochs'],
        learning_rate=best_params_human['learning_rate'],         embedding_dim=best_params_human['embedding_dim'],        weight_decay=best_params_human['weight_decay'],
        optimizer_name=best_params_human['optimizer']
    )
    print(f"\nFinal Train RMSE for Control (Human) Model: {train_rmse_human:.4f}")
    print(f"Final Test RMSE for Control (Human) Model: {test_rmse_human:.4f}")

    print('\nStarting Final Experimental Group Experiment (LLM Keywords) with Best Hyperparameters...')
    num_features_llm = X_llm.shape[1]
    train_loader_llm, test_loader_llm, test_dataset_llm = create_dataloaders(X_llm, y, batch_size=best_params_llm['batch_size'])
    train_rmse_llm, test_rmse_llm, ci_95_llm, ci_99_llm, model_llm, train_rmse_history_llm, test_rmse_history_llm = run_experiment(
        train_loader_llm,         test_loader_llm,         test_dataset_llm,         num_features_llm,         num_epochs=best_params_llm['num_epochs'],
        learning_rate=best_params_llm['learning_rate'],         embedding_dim=best_params_llm['embedding_dim'],        weight_decay=best_params_llm['weight_decay'],
        optimizer_name=best_params_llm['optimizer']
    )

    print('\nInspecting Linear Weights...')
    inspect_linear_weights(model_human, df, vectorizer_human, "Human")
    inspect_linear_weights(model_llm, df, vectorizer_llm, "LLM")

    print('\nAnalyzing Prediction Contributions...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    linear_human, interaction_human = analyze_prediction_contribution(model_human, test_loader_human, device)
    linear_llm, interaction_llm = analyze_prediction_contribution(model_llm, test_loader_llm, device)

    print('\nVisualizing Keyword Embeddings...')
    visualize_keyword_embeddings(model_human, vectorizer_human, "Human", df)
    visualize_keyword_embeddings(model_llm, vectorizer_llm, "LLM", df)

    print('\nGenerating Additional Visualizations...')
    plot_learning_curves((train_rmse_history_human, test_rmse_history_human), (train_rmse_history_llm, test_rmse_history_llm))
    plot_error_distribution(model_human, model_llm, test_loader_human, test_loader_llm, device)

    print('\nEXPERIMENT RESULTS')
    print(f"Train RMSE (Human Keywords): {train_rmse_human:.4f}")
    print(f"Test RMSE (Human Keywords): {test_rmse_human:.4f} (95% CI: {ci_95_human[0]:.4f}-{ci_95_human[1]:.4f}) (99% CI: {ci_99_human[0]:.4f}-{ci_99_human[1]:.4f})")
    print(f"  Contribution -> Linear: {linear_human:.2f}%, Interaction: {interaction_human:.2f}%")
    print(f"Train RMSE (LLM Keywords):   {train_rmse_llm:.4f}")
    print(f"Test RMSE (LLM Keywords):   {test_rmse_llm:.4f} (95% CI: {ci_95_llm[0]:.4f}-{ci_95_llm[1]:.4f}) (99% CI: {ci_99_llm[0]:.4f}-{ci_99_llm[1]:.4f})")
    print(f"  Contribution -> Linear: {linear_llm:.2f}%, Interaction: {interaction_llm:.2f}%")

    
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