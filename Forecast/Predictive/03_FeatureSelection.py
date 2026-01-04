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
OUTPUT_FILE = "../../Data/DataFrames/RandomForest_Comb_Results_CV_KFold10_TuningCV10_RMSE_NEW.xlsx"
CHECKPOINT_FILE = "../../Data/DataFrames/RandomForest_Checkpoint.pkl"

# Variável alvo
TARGET_COLUMN = 'VTCC(m³/ha)'

# Parâmetros de validação cruzada
CV_FOLDS = 10                    # Número de folds para avaliação e tuning
RANDOM_STATE = 42                # Semente para reprodutibilidade
N_ITER_SEARCH = 100              # Iterações do RandomizedSearchCV (reduzido de 500)

# Parâmetros de paralelização
N_JOBS_OUTER = -1                # Número de jobs para loop principal (-1 = todos os cores)
N_JOBS_INNER = 1                 # Jobs internos (1 quando outer é paralelo)
CHECKPOINT_INTERVAL = 100        # Salvar checkpoint a cada N combinações
TOP_N_PLOTS = 10                 # Gerar gráficos apenas para top N resultados

# Features candidatas (LiDAR + variáveis de inventário)
FEATURE_NAMES = [
    # Métricas LiDAR de altura (percentis)
    'Elev P90', 'Elev P80', 'Elev P95', 'Elev P99', 'Elev P75', 'Elev P70', 'Elev P60',
    # Métricas LiDAR derivadas
    'Elev CURT mean CUBE', 'Elev maximum', 'Elev SQRT mean SQ',
    'Elev variance', 'Elev L1', 'Elev stddev',
    # Variáveis de inventário
    'ROTACAO', 'REGIONAL', 'Idade (meses)'
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
# FUNÇÃO PARA AVALIAR UMA COMBINAÇÃO (PARALELIZÁVEL)
# =============================================================================

def evaluate_combination(combo, df, target_column, hyperparam_grid, cv_folds,
                         n_iter_search, random_state, n_jobs_inner):
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

    Returns:
        dict: Resultados da avaliação
    """
    try:
        # Prepara dados
        X = df[list(combo)].values.astype(float)
        y = df[target_column].values.astype(float)

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
            'N_Features': len(combo),
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
            'N_Features': len(combo),
            'CV R²': np.nan,
            'CV RMSE': np.nan,
            'CV MAE': np.nan,
            'CV Residual Mean (%)': np.nan,
            'CV Bias': np.nan,
            'Best Params': None,
            'Error': str(e)
        }


def generate_diagnostic_plots(result, output_dir="../../Data/Plots"):
    """
    Gera gráficos de diagnóstico para um resultado.
    """
    if 'y_true' not in result or 'y_pred' not in result:
        return

    os.makedirs(output_dir, exist_ok=True)

    y = result['y_true']
    y_pred_cv = result['y_pred']
    r2_cv = result['CV R²']
    rmse_cv = result['CV RMSE']
    n_features = result['N_Features']

    fig, axis = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Random Forest - Validação Cruzada - {n_features} Features (R²={r2_cv:.4f})", fontsize=12)

    # Gráfico 1: Observado vs Predito
    axis[0].scatter(y, y_pred_cv, alpha=0.7, color=COLOR_PRIMARY)
    axis[0].set_xlabel("VTCC Observado (m³/ha)")
    axis[0].set_ylabel("VTCC Estimado (m³/ha)")
    axis[0].plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='1:1')
    axis[0].grid(linestyle='--', color='gray', alpha=0.2)
    min_val = min(min(y), min(y_pred_cv)) * 0.95
    max_val = max(max(y), max(y_pred_cv)) * 1.05
    axis[0].set_xlim(min_val, max_val)
    axis[0].set_ylim(min_val, max_val)
    axis[0].set_title(f"Observado vs. Estimado\nR²={r2_cv:.3f}, RMSE={rmse_cv:.2f}")

    # Gráfico 2: Resíduos vs Predito
    residuals_raw_cv = (y - y_pred_cv) / y
    axis[1].scatter(y_pred_cv, residuals_raw_cv, alpha=0.7, color=COLOR_PRIMARY)
    axis[1].set_xlabel("VTCC Estimado (m³/ha)")
    axis[1].set_ylabel("Resíduos Relativos")
    axis[1].set_ylim(-1, 1)
    axis[1].axhline(y=0, color='red', linestyle='--')
    axis[1].grid(linestyle='--', color='gray', alpha=0.2)
    axis[1].set_title("Resíduos vs. Estimado")

    # Gráfico 3: Histograma de resíduos
    weights = np.ones_like(residuals_raw_cv) / len(residuals_raw_cv) * 100
    axis[2].hist(residuals_raw_cv, bins=30, weights=weights, alpha=0.7,
                 color=COLOR_PRIMARY, edgecolor=COLOR_PRIMARY, linewidth=0.8)
    axis[2].set_title("Distribuição dos Resíduos")
    axis[2].set_xlabel("Resíduos Relativos")
    axis[2].set_ylabel("Frequência (%)")
    axis[2].set_xlim(-1, 1)
    axis[2].axvline(x=0, color='red', linestyle='--')
    axis[2].grid(linestyle='--', alpha=0.2, color='grey')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Salva figura
    feature_str = "_".join([f[:6] for f in result['Features'][:3]])
    filename = f"RF_R2_{r2_cv:.4f}_{n_features}feat_{feature_str}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


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

    # Gera todas as combinações
    all_combinations = []
    for r in range(1, len(FEATURE_NAMES) + 1):
        all_combinations += list(combinations(FEATURE_NAMES, r))

    print(f"Features candidatas: {len(FEATURE_NAMES)}")
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
                    CV_FOLDS, N_ITER_SEARCH, RANDOM_STATE, N_JOBS_INNER
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

    # Gera gráficos apenas para top N
    print(f"\nGerando gráficos para top {TOP_N_PLOTS} combinações...")
    top_results = sorted(results, key=lambda x: x.get('CV R²', 0), reverse=True)[:TOP_N_PLOTS]
    for r in tqdm(top_results, desc="Gerando gráficos"):
        generate_diagnostic_plots(r)

    # Remove checkpoint após conclusão bem-sucedida
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint removido (processamento concluído)")

    print("\nProcessamento concluído!")