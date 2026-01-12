"""
03_FeatureSelection.py - Busca Exaustiva de Combinações de Variáveis com Random Forest

Este script realiza uma busca exaustiva testando todas as combinações possíveis de
variáveis LiDAR para encontrar o melhor conjunto de features para predição de volume
florestal (VTCC) usando Random Forest Regressor.

Metodologia:
    1. Gera todas as combinações possíveis das variáveis candidatas
    2. Para cada combinação (em paralelo):
       a. Otimiza hiperparâmetros via RandomizedSearchCV (10-fold CV)
       b. Avalia o modelo otimizado via cross_val_predict (10-fold CV)
       c. Calcula métricas: R², RMSE, MAE, Bias
    3. Exporta resultados ordenados por performance
    4. Gera gráficos apenas para os top resultados

VERSÃO OTIMIZADA:
    - Processamento paralelo com joblib
    - Checkpoints automáticos a cada 100 combinações
    - Gráficos apenas para top N resultados
    - Barra de progresso com tqdm

Saídas:
    - RandomForest_Comb_Results_CV_KFold10_TuningCV10_RMSE_NEW.xlsx
    - Gráficos de diagnóstico para top combinações

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from itertools import combinations
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminhos dos arquivos
INPUT_FILE = r"G:\PycharmProjects\Mestrado\Data\DataFrames\IFC_LiDAR_Plots_RTK_Cleaned_v02.xlsx"
OUTPUT_FILE = "../../Data/DataFrames/RandomForest_Comb_Results_CV_KFold10_TuningCV10_RMSE_2026.xlsx"
CHECKPOINT_FILE = "../../Data/DataFrames/RandomForest_Checkpoint.pkl"

# Variável alvo
TARGET_COLUMN = 'VTCC(m³/ha)'

# Parâmetros de validação cruzada
CV_FOLDS = 5                    # Número de folds para avaliação e tuning
RANDOM_STATE = 42                # Semente para reprodutibilidade
N_ITER_SEARCH = 20              # Iterações do RandomizedSearchCV (reduzido de 500)

# Parâmetros de paralelização
N_JOBS_OUTER = -1                # Número de jobs para loop principal (-1 = todos os cores)
N_JOBS_INNER = 1                 # Jobs internos (1 quando outer é paralelo)
CHECKPOINT_INTERVAL = 500        # Salvar checkpoint a cada N combinações
TOP_N_PLOTS = 10                 # Gerar gráficos apenas para top N resultados

# Parâmetros de One-Hot Encoding
APPLY_ONE_HOT = True             # True = aplica OHE em variáveis categóricas
DROP_FIRST_OHE = False           # True se quiser evitar colinearidade (não necessário em RF)
FORCE_OHE_COLUMNS = []           # Colunas a forçar OHE mesmo se numéricas

# Limite de features por combinação
MAX_FEATURES = 6                 # Máximo de features por combinação (None = sem limite)

# Features candidatas (LiDAR + variáveis de inventário)
FEATURE_NAMES = [
    'Elev P90', 'Elev P80', 'Elev P95', 'Elev P99', 'Elev P75', 'Elev CURT mean CUBE',
    'Elev maximum', 'Elev P70', 'Elev SQRT mean SQ', 'Elev stddev', 'Elev P60', 'Elev L2',
    'Regime', 'REGIONAL', 'Idade (meses)'
]

# Configuração de estilo dos gráficos
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

# Cores padrão
COLOR_PRIMARY = '#1f77b4'

# =============================================================================
# GRID DE HIPERPARÂMETROS DO RANDOM FOREST
# =============================================================================

HYPERPARAM_GRID = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=50, num=20)],
    'max_features': ['sqrt', 'log2', 1, 0.2, 0.3, 0.4, 0.5, 0.8],
    'max_depth': [int(x) for x in np.linspace(10, 150, num=10)] + [None],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'bootstrap': [True, False]
}


# =============================================================================
# FUNÇÃO PARA APLICAR ONE-HOT ENCODING
# =============================================================================

def apply_one_hot_encoding(df, feature_names, drop_first=False, force_ohe_columns=None):
    """
    Aplica One-Hot Encoding em variáveis categóricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    feature_names : list
        Lista de nomes das features a serem processadas.
    drop_first : bool
        Se True, remove a primeira categoria para evitar colinearidade.
    force_ohe_columns : list
        Lista de colunas para forçar OHE mesmo se numéricas.

    Returns
    -------
    tuple
        (X_values, feature_names_out, X_df, categorical_cols)
    """
    X_df = df[feature_names].copy()

    force_ohe_columns = force_ohe_columns or []

    # Detecta colunas não numéricas
    categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Adiciona colunas a forçar OHE (mesmo se forem numéricas)
    for col in force_ohe_columns:
        if col in X_df.columns and col not in categorical_cols:
            categorical_cols.append(col)

    if categorical_cols:
        X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=drop_first)

    return X_df.values.astype(float), list(X_df.columns), X_df, categorical_cols


# =============================================================================
# FUNÇÃO PARA AVALIAR UMA COMBINAÇÃO (PARALELIZÁVEL)
# =============================================================================

def evaluate_combination(combo, df, target_column, hyperparam_grid, cv_folds,
                         n_iter_search, random_state, n_jobs_inner,
                         apply_ohe=True, drop_first_ohe=False, force_ohe_columns=None):
    """
    Avalia uma única combinação de features.

    Args:
        combo: Tupla com nomes das features
        df: DataFrame com os dados
        target_column: Nome da coluna alvo
        hyperparam_grid: Grid de hiperparâmetros
        cv_folds: Número de folds para CV
        n_iter_search: Iterações do RandomizedSearchCV
        random_state: Semente aleatória
        n_jobs_inner: Número de jobs para operações internas
        apply_ohe: Se True, aplica One-Hot Encoding em variáveis categóricas
        drop_first_ohe: Se True, remove a primeira categoria (evita colinearidade)
        force_ohe_columns: Lista de colunas para forçar OHE

    Returns:
        dict: Resultados da avaliação
    """
    try:
        # Prepara dados
        y = df[target_column].values.astype(float)

        # Aplica One-Hot Encoding se configurado
        if apply_ohe:
            X, feature_names_ohe, _, categorical_cols = apply_one_hot_encoding(
                df, list(combo), drop_first=drop_first_ohe, force_ohe_columns=force_ohe_columns
            )
        else:
            X = df[list(combo)].values.astype(float)
            feature_names_ohe = list(combo)
            categorical_cols = []

        # KFolds
        cv_tuning = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        kf_evaluate = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Otimização de hiperparâmetros
        rf = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs_inner)

        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=hyperparam_grid,
            n_iter=n_iter_search,
            cv=cv_tuning,
            verbose=0,
            random_state=random_state,
            n_jobs=n_jobs_inner,
            scoring='neg_root_mean_squared_error'
        )

        rf_random.fit(X, y)

        # Avaliação com modelo otimizado
        rf_optimized = RandomForestRegressor(
            **rf_random.best_params_,
            random_state=random_state,
            n_jobs=n_jobs_inner
        )

        y_pred_cv = cross_val_predict(rf_optimized, X, y, cv=kf_evaluate, n_jobs=n_jobs_inner)

        # Métricas
        r2_cv = r2_score(y, y_pred_cv)
        rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
        mae_cv = np.mean(np.abs(y - y_pred_cv))
        bias_cv = np.mean(y_pred_cv - y)

        # Resíduos percentuais
        residuals_cv_percent = np.zeros_like(y, dtype=float)
        non_zero_mask = y != 0
        residuals_cv_percent[non_zero_mask] = (y[non_zero_mask] - y_pred_cv[non_zero_mask]) / y[non_zero_mask]

        return {
            'Features': combo,
            'Categorical_Cols': categorical_cols if categorical_cols else None,
            'N_Features': len(combo),
            'N_Features_Model': len(feature_names_ohe),
            'CV R²': r2_cv,
            'CV RMSE': rmse_cv,
            'CV MAE': mae_cv,
            'CV Residual Mean (%)': np.mean(residuals_cv_percent) * 100,
            'CV Bias': bias_cv,
            'Best Params': rf_random.best_params_,
            'y_true': y,
            'y_pred': y_pred_cv
        }

    except Exception as e:
        return {
            'Features': combo,
            'Categorical_Cols': None,
            'N_Features': len(combo),
            'N_Features_Model': np.nan,
            'CV R²': np.nan,
            'CV RMSE': np.nan,
            'CV MAE': np.nan,
            'CV Residual Mean (%)': np.nan,
            'CV Bias': np.nan,
            'Best Params': None,
            'Error': str(e)
        }

def load_checkpoint(checkpoint_file):
    """Carrega checkpoint se existir."""
    if os.path.exists(checkpoint_file):
        try:
            return pd.read_pickle(checkpoint_file)
        except:
            return None
    return None


def save_checkpoint(results, checkpoint_file):
    """Salva checkpoint."""
    pd.to_pickle(results, checkpoint_file)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BUSCA EXAUSTIVA DE COMBINAÇÕES DE VARIÁVEIS")
    print("VERSÃO OTIMIZADA COM PROCESSAMENTO PARALELO")
    print("=" * 60)

    # Carrega dados
    df = pd.read_excel(INPUT_FILE)
    print(f"Dados carregados: {df.shape[0]} amostras")

    # Gera todas as combinações (limitado por MAX_FEATURES se definido)
    all_combinations = []
    max_r = MAX_FEATURES if MAX_FEATURES else len(FEATURE_NAMES)
    for r in range(1, max_r + 1):
        all_combinations += list(combinations(FEATURE_NAMES, r))

    print(f"Features candidatas: {len(FEATURE_NAMES)}")
    print(f"Máximo de features por combinação: {max_r}")
    print(f"Total de combinações a testar: {len(all_combinations):,}")

    # Verifica checkpoint
    checkpoint_data = load_checkpoint(CHECKPOINT_FILE)
    if checkpoint_data is not None:
        completed_combos = set([tuple(r['Features']) for r in checkpoint_data])
        remaining_combinations = [c for c in all_combinations if c not in completed_combos]
        results = checkpoint_data
        print(f"Checkpoint encontrado: {len(completed_combos):,} combinações já processadas")
        print(f"Combinações restantes: {len(remaining_combinations):,}")
    else:
        remaining_combinations = all_combinations
        results = []

    if len(remaining_combinations) == 0:
        print("Todas as combinações já foram processadas!")
    else:
        # Detecta número de cores
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        print(f"\nUsando {n_cores} cores para processamento paralelo")
        print(f"N_ITER_SEARCH: {N_ITER_SEARCH}")
        print(f"Checkpoint a cada {CHECKPOINT_INTERVAL} combinações")
        print(f"One-Hot Encoding: {'Ativado' if APPLY_ONE_HOT else 'Desativado'}")

        print("\n" + "=" * 60)
        print("INICIANDO AVALIAÇÃO DAS COMBINAÇÕES")
        print("=" * 60)

        # Processa em batches para permitir checkpoints
        batch_size = CHECKPOINT_INTERVAL
        n_batches = (len(remaining_combinations) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(remaining_combinations))
            batch = remaining_combinations[start_idx:end_idx]

            print(f"\nBatch {batch_idx + 1}/{n_batches} ({len(batch)} combinações)")

            # Processamento paralelo do batch
            batch_results = Parallel(n_jobs=N_JOBS_OUTER, verbose=1)(
                delayed(evaluate_combination)(
                    combo, df, TARGET_COLUMN, HYPERPARAM_GRID,
                    CV_FOLDS, N_ITER_SEARCH, RANDOM_STATE, N_JOBS_INNER,
                    APPLY_ONE_HOT, DROP_FIRST_OHE, FORCE_OHE_COLUMNS
                )
                for combo in tqdm(batch, desc=f"Batch {batch_idx + 1}")
            )

            results.extend(batch_results)

            # Salva checkpoint
            save_checkpoint(results, CHECKPOINT_FILE)
            print(f"Checkpoint salvo: {len(results):,} combinações processadas")

    # =============================================================================
    # RESULTADOS FINAIS
    # =============================================================================
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)

    # Prepara DataFrame de resultados (sem y_true e y_pred para exportação)
    results_for_export = []
    for r in results:
        export_row = {k: v for k, v in r.items() if k not in ['y_true', 'y_pred', 'Error']}
        results_for_export.append(export_row)

    results_df = pd.DataFrame(results_for_export)
    results_df = results_df.sort_values(by='CV R²', ascending=False)

    # Top 10 por R²
    print(f"\nTop {TOP_N_PLOTS} combinações por R²:")
    top_n = results_df.head(TOP_N_PLOTS)
    for idx, (_, row) in enumerate(top_n.iterrows(), 1):
        print(f"  {idx}. R²={row['CV R²']:.4f} | RMSE={row['CV RMSE']:.2f} | Features: {row['N_Features']}")
        print(f"     {row['Features']}")

    # Exporta para Excel
    results_df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nResultados exportados para:\n  {OUTPUT_FILE}")

    # Remove checkpoint após conclusão bem-sucedida
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint removido (processamento concluído)")

    print("\nProcessamento concluído!")