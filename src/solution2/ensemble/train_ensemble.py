"""
train_ensemble.py — Train neural network meta-learner ensemble.

Automatically loads predictions from models specified in models_config.py
and trains a neural network to optimally combine them.

Usage:
    python -m src.solution2.ensemble.train_ensemble
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.stats import entropy
from tqdm import tqdm

from . import config, models_config


class MetaLearner(nn.Module):
    """Neural network meta-learner for ensemble.
    
    Improved architecture with:
    - Deeper network (4 hidden layers instead of 2)
    - Batch normalization for stable training
    - Higher capacity (128 hidden units)
    - Better regularization via dropout
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Input → hidden_size
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2: hidden_size → hidden_size
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3: hidden_size → hidden_size//2 (compress)
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout - 0.1),  # Slightly less dropout in final layers
            
            # Layer 4: hidden_size//2 → hidden_size//4 (compress more)
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout - 0.1),
            
            # Output: hidden_size//4 → 1
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)


def load_model_predictions(model_probs_path: str, model_name: str) -> pd.DataFrame:
    """Load probability predictions from a single model."""
    df = pd.read_csv(model_probs_path)
    df = df.rename(columns={
        'probability': f'{model_name}_prob',
        'prediction': f'{model_name}_pred'
    })
    return df


def load_labels(csv_path: str) -> np.ndarray:
    """Load labels from training/dev CSV."""
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError(f"No 'label' column in {csv_path}")
    return df['label'].values


def create_feature_matrix(model_dfs_dict: dict) -> pd.DataFrame:
    """
    Create feature matrix from all models' predictions.
    
    Features per model:
    - {model}_prob, {model}_pred
    
    Aggregate features:
    - max_prob, min_prob, mean_prob, std_prob, prob_range, prob_variance
    - vote_pred, vote_certainty
    - agreement patterns and model disagreement metrics
    """
    features = pd.DataFrame()
    
    # Add individual model features
    for model_name, df in model_dfs_dict.items():
        features[f'{model_name}_prob'] = df[f'{model_name}_prob']
        features[f'{model_name}_pred'] = df[f'{model_name}_pred']
    
    # Aggregate features across all models
    model_names = list(model_dfs_dict.keys())
    probs = np.column_stack([model_dfs_dict[m][f'{m}_prob'] for m in model_names])
    preds = np.column_stack([model_dfs_dict[m][f'{m}_pred'] for m in model_names])
    
    # ── Basic aggregate features ────────────────────────────────────────
    features['max_prob'] = probs.max(axis=1)
    features['min_prob'] = probs.min(axis=1)
    features['mean_prob'] = probs.mean(axis=1)
    features['std_prob'] = probs.std(axis=1)
    
    # ── Enhanced features: probability distribution ────────────────────
    features['prob_range'] = features['max_prob'] - features['min_prob']
    features['prob_variance'] = probs.var(axis=1)
    
    # ── Voting features ────────────────────────────────────────────────
    num_models = len(model_names)
    vote_threshold = num_models / 2
    features['vote_pred'] = (preds.sum(axis=1) >= vote_threshold).astype(int)
    features['vote_certainty'] = np.abs(preds.sum(axis=1) - vote_threshold) / vote_threshold
    
    # ── Agreement patterns ─────────────────────────────────────────────
    # Perfect agreement: all models agree (all 0s or all 1s)
    features['perfect_agreement'] = ((preds.sum(axis=1) == 0) | (preds.sum(axis=1) == num_models)).astype(float)
    # Model disagreement entropy: higher = more disagreement
    model_agreement_ratios = preds.mean(axis=0)  # Per-model average prediction
    features['predictions_entropy'] = np.array([entropy([p, 1-p]) for p in preds.mean(axis=1)])
    
    # ── Model consensus distance ───────────────────────────────────────
    # How far is the consensus from 0.5 (confidence in ensemble direction)
    consensus_prob = features['mean_prob']
    features['consensus_confidence'] = np.abs(consensus_prob - 0.5) * 2  # 0 to 1 scale
    
    return features


def main():
    # Set random seed
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ensemble] Device: {device}")
    
    print(f"[ensemble] Loading {len(models_config.MODELS)} model predictions...")
    
    # Load training predictions and labels
    model_dfs_train = {}
    for model_name, paths in models_config.MODELS.items():
        print(f"  - Loading {model_name} train predictions...")
        model_dfs_train[model_name] = load_model_predictions(
            paths['train_probs'], model_name
        )
    y_train = load_labels(models_config.TRAIN_LABELS_FILE)
    
    # Load dev predictions and labels
    model_dfs_dev = {}
    for model_name, paths in models_config.MODELS.items():
        print(f"  - Loading {model_name} dev predictions...")
        model_dfs_dev[model_name] = load_model_predictions(
            paths['dev_probs'], model_name
        )
    y_dev = load_labels(models_config.DEV_LABELS_FILE)
    
    # Apply base model calibration if available
    print("[ensemble] Checking for base model calibrators...")
    calibrators_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "calibrators", "calibrators.pkl")
    if os.path.exists(calibrators_path):
        print("[ensemble] Loading base model calibrators...")
        with open(calibrators_path, 'rb') as f:
            calibrators = pickle.load(f)
        
        # Apply calibration to both train and dev
        for model_name in model_dfs_train.keys():
            if model_name in calibrators:
                iso_reg = calibrators[model_name]
                # Train set calibration
                train_raw = model_dfs_train[model_name][f'{model_name}_prob'].values
                train_calibrated = iso_reg.predict(train_raw)
                model_dfs_train[model_name][f'{model_name}_prob'] = train_calibrated
                # Dev set calibration
                dev_raw = model_dfs_dev[model_name][f'{model_name}_prob'].values
                dev_calibrated = iso_reg.predict(dev_raw)
                model_dfs_dev[model_name][f'{model_name}_prob'] = dev_calibrated
        
        print(f"[ensemble] Applied calibration to {len(calibrators)} models")
    else:
        print("[ensemble] No calibrators found, using raw model predictions")
    
    print(f"[ensemble] Train set: {len(y_train)} samples, {len(models_config.MODELS)} models")
    print(f"[ensemble] Dev set: {len(y_dev)} samples")
    
    # Create feature matrices
    print("[ensemble] Creating feature matrices...")
    X_train = create_feature_matrix(model_dfs_train)
    X_dev = create_feature_matrix(model_dfs_dev)
    
    print(f"[ensemble] Features: {list(X_train.columns)}")
    print(f"[ensemble] Feature count: {X_train.shape[1]}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_dev_tensor = torch.FloatTensor(X_dev.values).to(device)
    y_dev_tensor = torch.FloatTensor(y_dev).unsqueeze(1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    
    # Initialize model
    print("[ensemble] Initializing neural network meta-learner...")
    model = MetaLearner(
        input_size=X_train.shape[1],
        hidden_size=config.HIDDEN_SIZE,
        dropout=config.DROPOUT,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    # Use Focal Loss for better handling of hard examples
    # Focal Loss: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    # Helps focus learning on harder samples
    alpha = 1.0  # Balance factor (1.0 = no balancing)
    gamma = 2.0  # Focusing parameter (higher = focus more on hard negatives)
    criterion = nn.BCELoss()
    
    # Learning rate scheduler: cosine annealing (decays LR over epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    
    # Training loop
    print("[ensemble] Training...")
    best_f1 = 0
    patience_counter = 0
    
    pbar = tqdm(range(config.EPOCHS), desc="Training", unit="epoch")
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            # Label smoothing: smooth hard targets to prevent overconfidence
            y_batch_smooth = y_batch * 0.95 + 0.025
            # Apply focal loss weighting: down-weight easy examples
            ce_loss = criterion(logits, y_batch_smooth)
            pt = torch.where(y_batch == 1, logits, 1 - logits)  # Probability of correct class
            focal_weight = (1 - pt) ** gamma
            loss = alpha * focal_weight * ce_loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_train_probs = model(X_train_tensor).cpu().numpy().flatten()
            y_dev_probs = model(X_dev_tensor).cpu().numpy().flatten()
        
        y_train_preds = (y_train_probs > 0.5).astype(int)
        y_dev_preds = (y_dev_probs > 0.5).astype(int)
        
        train_f1 = f1_score(y_train, y_train_preds, average='macro')
        dev_f1 = f1_score(y_dev, y_dev_preds, average='macro')
        dev_acc = accuracy_score(y_dev, y_dev_preds)
        dev_auc = roc_auc_score(y_dev, y_dev_probs)
        
        # Update progress bar with metrics
        pbar.set_postfix({
            'Loss': f'{train_loss:.4f}',
            'Train_F1': f'{train_f1:.4f}',
            'Dev_F1': f'{dev_f1:.4f}',
            'Dev_Acc': f'{dev_acc:.4f}'
        })
        
        # Early stopping
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            # Save best model
            os.makedirs(config.ENSEMBLE_MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), 
                      os.path.join(config.ENSEMBLE_MODEL_DIR, "meta_learner.pt"))
            best_dev_f1 = dev_f1
            best_dev_acc = dev_acc
            best_dev_auc = dev_auc
            pbar.set_description(f"Training [Best F1: {best_f1:.4f}]")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                pbar.close()
                print(f"[ensemble] Early stopping at epoch {epoch+1}")
                break
        
        # Step learning rate scheduler
        scheduler.step()
    
    # Load best model and evaluate
    print("\n[ensemble] Loading best model and evaluating...")
    model.load_state_dict(
        torch.load(os.path.join(config.ENSEMBLE_MODEL_DIR, "meta_learner.pt"))
    )
    model.eval()
    
    with torch.no_grad():
        y_dev_probs = model(X_dev_tensor).cpu().numpy().flatten()
    
    # ── Threshold optimization ─────────────────────────────────────────
    # Find best threshold on dev set (instead of fixed 0.5)
    best_threshold = 0.5
    best_f1 = 0.0
    print("\n[ensemble] Optimizing decision threshold...")
    thresholds = np.arange(0.35, 0.66, 0.001)
    for threshold in tqdm(thresholds, desc="Threshold search", unit="threshold"):
        y_dev_preds_temp = (y_dev_probs > threshold).astype(int)
        f1_temp = f1_score(y_dev, y_dev_preds_temp, average='macro')
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold
    
    # ── Isotonic Regression Calibration ───────────────────────────────────────
    # Fit isotonic regressor to improve calibration
    print("\n[ensemble] Fitting isotonic regression for calibration...")
    
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_dev_probs, y_dev)
    y_dev_probs_calibrated = iso_reg.predict(y_dev_probs)
    
    # ── Calibration Analysis ──────────────────────────────────────────────────
    print("\n[ensemble] Analyzing prediction calibration...")
    prob_true, prob_pred = calibration_curve(y_dev, y_dev_probs_calibrated, n_bins=10, strategy='uniform')
    
    # Compute calibration metrics
    expected_calibration_error = np.mean(np.abs(prob_true - prob_pred))
    max_calibration_error = np.max(np.abs(prob_true - prob_pred))
    
    original_ece = np.mean(np.abs(calibration_curve(y_dev, y_dev_probs, n_bins=10, strategy='uniform')[0] - 
                                   calibration_curve(y_dev, y_dev_probs, n_bins=10, strategy='uniform')[1]))
    
    print(f"  Original ECE: {original_ece:.4f}")
    print(f"  Calibrated ECE: {expected_calibration_error:.4f}")
    print(f"  Max Calibration Error (MCE): {max_calibration_error:.4f}")
    
    if expected_calibration_error > 0.05:
        print("  ⚠ Calibration error still high (ECE > 0.05)")
    else:
        print("  ✓ Good calibration after isotonic regression")
    print(f"[ensemble] Calibration: ECE={expected_calibration_error:.4f}")
    
    y_dev_preds = (y_dev_probs_calibrated > best_threshold).astype(int)
    
    dev_f1 = f1_score(y_dev, y_dev_preds, average='macro')
    dev_acc = accuracy_score(y_dev, y_dev_preds)
    dev_auc = roc_auc_score(y_dev, y_dev_probs_calibrated)
    
    print(f"[ensemble] Final Results (Dev Set):")
    print(f"  F1 (macro): {dev_f1:.4f}")
    print(f"  Accuracy: {dev_acc:.4f}")
    print(f"  AUC-ROC: {dev_auc:.4f}")
    print(f"  Threshold: {best_threshold:.2f}")
    
    # Compute feature importance from first layer weights
    print("\n[ensemble] Feature Importance (how the network weights each feature):")
    print("-" * 70)
    first_layer_weights = model.net[0].weight.detach().cpu().numpy()  # (hidden_size, input_size)
    # Average absolute weights across hidden units
    feature_importance = np.abs(first_layer_weights).mean(axis=0)
    
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importance,
    }).sort_values('importance', ascending=False)
    
    # Normalize to percentages
    feature_importance_df['percentage'] = (
        feature_importance_df['importance'] / feature_importance_df['importance'].sum() * 100
    )
    
    # Print formatted output
    for idx, row in feature_importance_df.iterrows():
        pct = row['percentage']
        bar_length = int(pct / 2)  # Scale to fit on line
        bar = '█' * bar_length
        print(f"  {row['feature']:20s} | {bar:40s} | {pct:5.1f}%")
    
    print("-" * 70)
    
    # Break down by model (dynamically generated from models_config)
    print("\n[ensemble] Model Contribution Breakdown:")
    print("-" * 70)
    model_features = [
        (model_name, [f'{model_name}_prob', f'{model_name}_pred'])
        for model_name in models_config.MODELS.keys()
    ]
    
    for model_name, features in model_features:
        model_importance = feature_importance_df[
            feature_importance_df['feature'].isin(features)
        ]['percentage'].sum()
        pct = model_importance
        bar_length = int(pct / 2)
        bar = '█' * bar_length
        print(f"  {model_name:20s} | {bar:40s} | {pct:5.1f}%")
    
    print("-" * 70)
    
    # Save feature importance
    importance_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "feature_importance.csv")
    feature_importance_df.to_csv(importance_path, index=False)
    print(f"[ensemble] Feature importance saved → {importance_path}")
    
    # Save feature names for inference
    feature_names_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "feature_names.pkl")
    with open(feature_names_path, 'wb') as f:
        pickle.dump(list(X_train.columns), f)
    print(f"[ensemble] Feature names saved → {feature_names_path}")
    
    # Save optimal threshold for inference
    threshold_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "threshold.pkl")
    with open(threshold_path, 'wb') as f:
        pickle.dump(best_threshold, f)
    print(f"[ensemble] Optimal threshold saved → {threshold_path}")
    
    # Save isotonic regressor for calibration during inference
    iso_reg_path = os.path.join(config.ENSEMBLE_MODEL_DIR, "iso_regressor.pkl")
    with open(iso_reg_path, 'wb') as f:
        pickle.dump(iso_reg, f)
    print(f"[ensemble] Isotonic regressor saved → {iso_reg_path}")
    
    # Save dev predictions (with calibration applied)
    os.makedirs(config.ENSEMBLE_OUTPUT_DIR, exist_ok=True)
    dev_results = pd.DataFrame({
        'probability': np.round(y_dev_probs_calibrated, 6),
        'prediction': y_dev_preds,
    })
    dev_pred_path = os.path.join(config.ENSEMBLE_OUTPUT_DIR, "probs_dev.csv")
    dev_results.to_csv(dev_pred_path, index=False)
    print(f"[ensemble] Dev predictions saved → {dev_pred_path}")
    
    return model


if __name__ == "__main__":
    main()
