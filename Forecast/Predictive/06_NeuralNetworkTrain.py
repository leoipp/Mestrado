"""
06_NeuralNetworkTrain.py - Treinamento de Rede Neural MLP para Predicao de Volume

Este script treina um modelo MLP (Multi-Layer Perceptron) para predicao de
volume florestal (VTCC) utilizando metricas LiDAR e variaveis auxiliares.

Workflow:
    1. Carregamento e normalizacao dos dados
    2. Otimizacao de hiperparametros via Optuna
    3. Validacao cruzada (K-Fold) para avaliacao
    4. Calculo de metricas estatisticas (R2, RMSE, MAE, Bias)
    5. Analise de importancia das variaveis (via gradientes)
    6. Geracao de graficos diagnosticos
    7. Exportacao do modelo treinado (.pt)

Variaveis selecionadas:
    - Elev P90: Percentil 90 de altura
    - Elev P60: Percentil 60 de altura
    - Elev maximum: Altura maxima
    - ROTACAO: Numero da rotacao florestal
    - REGIONAL: Codigo da regional
    - Idade (meses): Idade do plantio

Dependencias:
    - torch (PyTorch)
    - optuna (otimizacao de hiperparametros)
    - scikit-learn
    - pandas, numpy, matplotlib

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predicao de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    raise ImportError("PyTorch e necessario. Execute: pip install torch")

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna nao encontrado. Otimizacao de hiperparametros desabilitada.")


# =============================================================================
# CONFIGURACOES GLOBAIS
# =============================================================================

# Caminhos
INPUT_FILE = r"G:\PycharmProjects\Mestrado\Data\DataFrames\IFC_LiDAR_Plots_RTK_Cleaned_v02.xlsx"
OUTPUT_DIR = Path(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results")
MODEL_DIR = Path(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Models")

# Variaveis do modelo
FEATURE_NAMES = [
    'Elev P90',
    'Elev P60',
    'Elev maximum',
    'ROTACAO',
    'REGIONAL',
    'Idade (meses)'
]

# Variavel alvo
TARGET_COLUMN = 'VTCC(m³/ha)'

# Parametros de treinamento
CV_FOLDS = 10
RANDOM_STATE = 42
N_TRIALS_OPTUNA = 100
EPOCHS_DEFAULT = 500
EARLY_STOPPING_PATIENCE = 50

# Device (GPU se disponivel)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuracao de graficos
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

COLOR_PRIMARY = '#1f77b4'
COLOR_SECONDARY = '#ff7f0e'


# =============================================================================
# ARQUITETURA MLP
# =============================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron para regressao.

    Parameters
    ----------
    input_size : int
        Numero de features de entrada.
    hidden_sizes : list
        Lista com tamanho de cada camada oculta.
    dropout_rate : float
        Taxa de dropout (0 a 1).
    activation : str
        Funcao de ativacao ('relu', 'leaky_relu', 'elu', 'tanh', 'selu').
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        activation: str = 'relu'
    ):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        # Funcao de ativacao
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'selu': nn.SELU()
        }
        self.activation = activations.get(activation, nn.ReLU())

        # Constroi camadas
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Camada de saida
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def set_seed(seed: int = 42):
    """Define semente para reproducibilidade."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula metricas de avaliacao do modelo.

    Parameters
    ----------
    y_true : array-like
        Valores observados.
    y_pred : array-like
        Valores preditos.

    Returns
    -------
    dict
        Dicionario com as metricas calculadas.
    """
    y_mean = np.mean(y_true)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    rmse_pct = (rmse / y_mean) * 100
    mae_pct = (mae / y_mean) * 100
    bias_pct = (bias / y_mean) * 100

    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Bias': bias,
        'RMSE_pct': rmse_pct,
        'MAE_pct': mae_pct,
        'Bias_pct': bias_pct
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Treina o modelo por uma epoca."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Avalia o modelo no conjunto de validacao."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_targets)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    patience: int = 50,
    verbose: bool = False
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Treina o modelo com early stopping.

    Parameters
    ----------
    model : nn.Module
        Modelo a ser treinado.
    train_loader : DataLoader
        DataLoader de treinamento.
    val_loader : DataLoader
        DataLoader de validacao.
    epochs : int
        Numero maximo de epocas.
    learning_rate : float
        Taxa de aprendizado.
    weight_decay : float
        Regularizacao L2.
    device : torch.device
        Dispositivo (CPU/GPU).
    patience : int
        Paciencia para early stopping.
    verbose : bool
        Se True, imprime progresso.

    Returns
    -------
    tuple
        (modelo treinado, historico de treinamento)
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"    Early stopping na epoca {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


# =============================================================================
# OTIMIZACAO DE HIPERPARAMETROS
# =============================================================================

def create_optuna_objective(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    device: torch.device
):
    """
    Cria funcao objetivo para otimizacao Optuna.

    Parameters
    ----------
    X : np.ndarray
        Features normalizadas.
    y : np.ndarray
        Target.
    cv_folds : int
        Numero de folds para CV.
    device : torch.device
        Dispositivo.

    Returns
    -------
    callable
        Funcao objetivo para Optuna.
    """
    def objective(trial: optuna.Trial) -> float:
        # Sugere hiperparametros
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_sizes = []
        for i in range(n_layers):
            hidden_sizes.append(trial.suggest_int(f'hidden_size_{i}', 16, 256))

        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'selu'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        # Validacao cruzada
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = MLP(
                input_size=X.shape[1],
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate,
                activation=activation
            )

            model, _ = train_model(
                model, train_loader, val_loader,
                epochs=200,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                patience=30,
                verbose=False
            )

            _, preds, targets = evaluate(model, val_loader, nn.MSELoss(), device)
            fold_rmse = np.sqrt(mean_squared_error(targets, preds))
            fold_scores.append(fold_rmse)

            trial.report(np.mean(fold_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    return objective


# =============================================================================
# IMPORTANCIA DAS VARIAVEIS
# =============================================================================

def compute_feature_importance(
    model: nn.Module,
    X: np.ndarray,
    feature_names: List[str],
    device: torch.device
) -> pd.DataFrame:
    """
    Calcula importancia das variaveis usando gradientes.

    Parameters
    ----------
    model : nn.Module
        Modelo treinado.
    X : np.ndarray
        Dados de entrada (normalizados).
    feature_names : list
        Nomes das features.
    device : torch.device
        Dispositivo.

    Returns
    -------
    pd.DataFrame
        DataFrame com importancia das variaveis.
    """
    model.eval()

    X_tensor = torch.FloatTensor(X).to(device)
    X_tensor.requires_grad = True

    outputs = model(X_tensor)
    outputs.sum().backward()

    gradients = X_tensor.grad.abs().mean(dim=0).cpu().numpy()
    importances = gradients / gradients.sum()

    indices = np.argsort(importances)[::-1]

    importance_df = pd.DataFrame({
        'Variable': [feature_names[i] for i in indices],
        'Importance': importances[indices],
        'Importance_pct': importances[indices] * 100
    })

    return importance_df


# =============================================================================
# VISUALIZACAO
# =============================================================================

def plot_training_history(history: Dict[str, List[float]], output_path: Optional[Path] = None):
    """Plota historico de treinamento."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history['train_loss'], label='Treino', color=COLOR_PRIMARY)
    ax.plot(history['val_loss'], label='Validacao', color=COLOR_SECONDARY)

    ax.set_xlabel('Epoca')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Historico de Treinamento - MLP')
    ax.legend()
    ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Grafico salvo: {output_path}")

    plt.show()
    return fig


def plot_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    output_path: Optional[Path] = None
):
    """Gera graficos diagnosticos do modelo."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    residuals_pct = (y_true - y_pred) / y_true

    # (a) Observado vs Predito
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=30, c=COLOR_PRIMARY,
                edgecolors='white', linewidth=0.5)

    lim_min = 0
    lim_max = max(max(y_true), max(y_pred)) * 1.05
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5)

    textstr = f"R2 = {metrics['R2']:.4f}\nRMSE = {metrics['RMSE']:.2f} m3/ha\nBias = {metrics['Bias']:.2f} m3/ha"
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('VTCC Observado (m3/ha)')
    ax1.set_ylabel('VTCC Predito (m3/ha)')
    ax1.set_xlim(lim_min, lim_max)
    ax1.set_ylim(lim_min, lim_max)
    ax1.set_aspect('equal')
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_title('(a) Observado vs Predito')

    # (b) Residuos vs Predito
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals_pct, alpha=0.6, s=30, c=COLOR_PRIMARY,
                edgecolors='white', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax2.axhline(y=0.2, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.axhline(y=-0.2, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax2.set_xlabel('VTCC Predito (m3/ha)')
    ax2.set_ylabel('Residuos Relativos')
    ax2.set_ylim(-1, 1)
    ax2.grid(linestyle='--', alpha=0.3)
    ax2.set_title('(b) Residuos vs Predito')

    # (c) Histograma dos Residuos
    ax3 = axes[2]

    bin_edges = np.arange(-1.05, 1.15, 0.1)
    bin_centers = np.arange(-1.0, 1.1, 0.1)
    counts, _ = np.histogram(residuals_pct, bins=bin_edges)
    percentages = counts / len(residuals_pct) * 100

    ax3.bar(bin_centers, percentages, width=0.08, alpha=0.7,
            color=COLOR_PRIMARY, edgecolor=COLOR_PRIMARY, linewidth=0.8)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5)

    ax3.set_xlabel('Residuos Relativos')
    ax3.set_ylabel('Frequencia (%)')
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_xticks(np.arange(-1.0, 1.1, 0.2))
    ax3.grid(linestyle='--', alpha=0.3)
    ax3.set_title('(c) Distribuicao dos Residuos')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Grafico salvo: {output_path}")

    plt.show()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Optional[Path] = None
):
    """Gera grafico de importancia das variaveis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_features = len(importance_df)
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_features))

    bars = ax.barh(range(n_features), importance_df['Importance_pct'],
                   color=colors[::-1], edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(importance_df['Variable'])
    ax.set_xlabel('Importancia (%)')
    ax.set_title('Importancia das Variaveis - MLP (Gradientes)')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, importance_df['Importance_pct'])):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Grafico salvo: {output_path}")

    plt.show()
    return fig


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def train_mlp(
    input_file: str = INPUT_FILE,
    feature_names: List[str] = FEATURE_NAMES,
    target_column: str = TARGET_COLUMN,
    cv_folds: int = CV_FOLDS,
    n_trials: int = N_TRIALS_OPTUNA,
    epochs: int = EPOCHS_DEFAULT,
    random_state: int = RANDOM_STATE,
    optimize_hyperparams: bool = True,
    save_model: bool = True,
    save_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Pipeline completo de treinamento do MLP.

    Parameters
    ----------
    input_file : str
        Caminho do arquivo Excel com os dados.
    feature_names : list
        Lista com nomes das variaveis preditoras.
    target_column : str
        Nome da coluna alvo.
    cv_folds : int
        Numero de folds para validacao cruzada.
    n_trials : int
        Numero de trials para otimizacao Optuna.
    epochs : int
        Numero de epocas para treinamento final.
    random_state : int
        Semente para reproducibilidade.
    optimize_hyperparams : bool
        Se True, otimiza hiperparametros com Optuna.
    save_model : bool
        Se True, salva o modelo treinado.
    save_results : bool
        Se True, salva metricas e importancias.
    verbose : bool
        Se True, imprime progresso.

    Returns
    -------
    dict
        Dicionario com modelo, metricas, hiperparametros e predicoes.
    """
    print("=" * 70)
    print("TREINAMENTO DO MODELO MLP (Multi-Layer Perceptron)")
    print("=" * 70)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print()

    set_seed(random_state)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. CARREGAMENTO E PREPROCESSAMENTO
    # -------------------------------------------------------------------------
    print("[1/7] Carregando e normalizando dados...")

    df = pd.read_excel(input_file)
    X = df[feature_names].values.astype(np.float32)
    y = df[target_column].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Amostras: {len(y)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Variaveis: {feature_names}")
    print()

    # -------------------------------------------------------------------------
    # 2. OTIMIZACAO DE HIPERPARAMETROS
    # -------------------------------------------------------------------------
    best_params = {}

    if optimize_hyperparams and HAS_OPTUNA:
        print(f"[2/7] Otimizando hiperparametros (Optuna - {n_trials} trials)...")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
        )

        objective = create_optuna_objective(X_scaled, y, cv_folds, DEVICE)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        print(f"\n  Melhores hiperparametros encontrados:")
        for param, value in best_params.items():
            print(f"    - {param}: {value}")
        print()
    else:
        print("[2/7] Usando hiperparametros padrao...")
        best_params = {
            'n_layers': 2,
            'hidden_size_0': 64,
            'hidden_size_1': 32,
            'dropout_rate': 0.2,
            'activation': 'relu',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 32
        }
        print()

    # -------------------------------------------------------------------------
    # 3. VALIDACAO CRUZADA
    # -------------------------------------------------------------------------
    print(f"[3/7] Executando validacao cruzada ({cv_folds}-fold)...")

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    all_preds = np.zeros_like(y)
    all_histories = []

    n_layers = best_params.get('n_layers', 2)
    hidden_sizes = [best_params.get(f'hidden_size_{i}', 64) for i in range(n_layers)]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
        if verbose:
            print(f"  Fold {fold + 1}/{cv_folds}...")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        batch_size = best_params.get('batch_size', 32)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = MLP(
            input_size=X_scaled.shape[1],
            hidden_sizes=hidden_sizes,
            dropout_rate=best_params.get('dropout_rate', 0.2),
            activation=best_params.get('activation', 'relu')
        )

        model, history = train_model(
            model, train_loader, val_loader,
            epochs=epochs,
            learning_rate=best_params.get('learning_rate', 1e-3),
            weight_decay=best_params.get('weight_decay', 1e-4),
            device=DEVICE,
            patience=EARLY_STOPPING_PATIENCE,
            verbose=False
        )

        all_histories.append(history)

        _, preds, _ = evaluate(model, val_loader, nn.MSELoss(), DEVICE)
        all_preds[val_idx] = preds

    metrics = calculate_metrics(y, all_preds)

    print(f"\n  Metricas de Validacao Cruzada ({cv_folds}-fold):")
    print(f"    R2:    {metrics['R2']:.4f} ({metrics['R2']*100:.2f}%)")
    print(f"    RMSE:  {metrics['RMSE']:.2f} m3/ha ({metrics['RMSE_pct']:.2f}%)")
    print(f"    MAE:   {metrics['MAE']:.2f} m3/ha ({metrics['MAE_pct']:.2f}%)")
    print(f"    Bias:  {metrics['Bias']:.4f} m3/ha ({metrics['Bias_pct']:.2f}%)")
    print()

    # -------------------------------------------------------------------------
    # 4. TREINAMENTO DO MODELO FINAL
    # -------------------------------------------------------------------------
    print("[4/7] Treinando modelo final com todos os dados...")

    full_dataset = TensorDataset(torch.FloatTensor(X_scaled), torch.FloatTensor(y))
    full_loader = DataLoader(full_dataset, batch_size=best_params.get('batch_size', 32), shuffle=True)

    final_model = MLP(
        input_size=X_scaled.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rate=best_params.get('dropout_rate', 0.2),
        activation=best_params.get('activation', 'relu')
    )

    final_model, final_history = train_model(
        final_model, full_loader, full_loader,
        epochs=epochs,
        learning_rate=best_params.get('learning_rate', 1e-3),
        weight_decay=best_params.get('weight_decay', 1e-4),
        device=DEVICE,
        patience=EARLY_STOPPING_PATIENCE,
        verbose=verbose
    )

    print("  Modelo final treinado!")
    print()

    # -------------------------------------------------------------------------
    # 5. IMPORTANCIA DAS VARIAVEIS
    # -------------------------------------------------------------------------
    print("[5/7] Calculando importancia das variaveis...")

    importance_df = compute_feature_importance(final_model, X_scaled, feature_names, DEVICE)

    if save_results:
        plot_feature_importance(
            importance_df,
            output_path=OUTPUT_DIR / 'MLP_Feature_Importance.png'
        )
    print()

    # -------------------------------------------------------------------------
    # 6. GRAFICOS DIAGNOSTICOS
    # -------------------------------------------------------------------------
    print("[6/7] Gerando graficos diagnosticos...")

    if save_results:
        plot_diagnostics(
            y, all_preds, metrics,
            output_path=OUTPUT_DIR / 'MLP_Diagnostics.png'
        )

        # Usa o historico do fold mais longo para o grafico
        longest_history = max(all_histories, key=lambda h: len(h['train_loss']))
        plot_training_history(
            longest_history,
            output_path=OUTPUT_DIR / 'MLP_Training_History.png'
        )
    print()

    # -------------------------------------------------------------------------
    # 7. EXPORTACAO
    # -------------------------------------------------------------------------
    print("[7/7] Exportando resultados...")

    if save_model:
        model_path = MODEL_DIR / 'MLP_Regressor.pt'
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'model_config': {
                'input_size': X_scaled.shape[1],
                'hidden_sizes': hidden_sizes,
                'dropout_rate': best_params.get('dropout_rate', 0.2),
                'activation': best_params.get('activation', 'relu')
            },
            'scaler': scaler,
            'feature_names': feature_names,
            'best_params': best_params,
            'metrics': metrics
        }, model_path)
        print(f"  Modelo salvo: {model_path}")

    if save_results:
        metrics_df = pd.DataFrame([{
            'Data_Treinamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Modelo': 'MLP (Multi-Layer Perceptron)',
            'N_Amostras': len(y),
            'N_Features': len(feature_names),
            'Features': str(feature_names),
            'CV_Folds': cv_folds,
            'Device': str(DEVICE),
            **metrics,
            'Best_Params': str(best_params)
        }])

        metrics_path = OUTPUT_DIR / 'MLP_Training_Metrics.xlsx'
        with pd.ExcelWriter(metrics_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

            pred_df = pd.DataFrame({
                'Observado': y,
                'Predito_CV': all_preds,
                'Residuo': y - all_preds,
                'Residuo_pct': (y - all_preds) / y * 100
            })
            pred_df.to_excel(writer, sheet_name='Predictions', index=False)

        print(f"  Metricas salvas: {metrics_path}")

    print()
    print("=" * 70)
    print("TREINAMENTO CONCLUIDO")
    print(f"Termino: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return {
        'model': final_model,
        'scaler': scaler,
        'metrics': metrics,
        'best_params': best_params,
        'feature_importance': importance_df,
        'predictions': all_preds,
        'history': final_history
    }


def load_trained_model(model_path: str) -> Tuple[nn.Module, StandardScaler, Dict]:
    """
    Carrega modelo MLP treinado do disco.

    Parameters
    ----------
    model_path : str
        Caminho do arquivo .pt.

    Returns
    -------
    tuple
        (modelo, scaler, config)
    """
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint['model_config']

    model = MLP(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate'],
        activation=config['activation']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    return model, checkpoint['scaler'], checkpoint


# =============================================================================
# EXECUCAO
# =============================================================================

if __name__ == '__main__':
    results = train_mlp()
