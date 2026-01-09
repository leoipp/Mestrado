"""
04_RandomForestTrain.py - Treinamento do Modelo Random Forest para Predição de Volume

Este script treina um modelo Random Forest Regressor para predição de volume florestal
(VTCC - Volume Total Com Casca) utilizando métricas LiDAR e variáveis auxiliares.

Workflow:
    1. Carregamento dos dados limpos (saída do 01_DataConsistency.py)
    2. Otimização de hiperparâmetros via RandomizedSearchCV
    3. Validação cruzada (K-Fold) para avaliação de desempenho
    4. Cálculo de métricas estatísticas (R², RMSE, MAE, Bias)
    5. Análise de importância das variáveis
    6. Geração de gráficos diagnósticos
    7. Exportação do modelo treinado (.pkl)

Variáveis selecionadas (saída do 02_VariablesCorrelation.py / 03_FeatureSelection.py)

Saídas:
    - Models/RF_Regressor_MAX_CUB_STD.pkl: Modelo treinado serializado
    - Results/RF_Training_Metrics.xlsx: Métricas de validação cruzada
    - Results/RF_Feature_Importance.png: Gráfico de importância das variáveis
    - Results/RF_Diagnostics.png: Gráficos diagnósticos (Obs vs Pred, Resíduos)

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict


# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminhos
INPUT_FILE = r"G:\PycharmProjects\Mestrado\Data\DataFrames\IFC_LiDAR_Plots_RTK_Cleaned_v02.xlsx"
OUTPUT_DIR = Path(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results")
MODEL_DIR = Path(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Models")

# Variáveis do modelo (definidas na etapa de seleção de features)
FEATURE_NAMES = [
    'Elev P90',        # Percentil 90 - estrutura dominante
    'Elev CURT mean CUBE',        # Percentil 60 - estrutura média do dossel
    'ROTACAO',         # Rotação florestal
    'REGIONAL',        # Regional
    'Idade (meses)'    # Idade do plantio
]

# Variável alvo
TARGET_COLUMN = 'VTCC(m³/ha)'

# Parâmetros de validação cruzada
CV_FOLDS_TUNING = 5      # Folds para otimização de hiperparâmetros
CV_FOLDS_EVALUATION = 5  # Folds para avaliação final
RANDOM_STATE = 42         # Semente para reprodutibilidade
N_ITER_SEARCH = 1000      # Iterações do RandomizedSearchCV
APPLY_ONE_HOT = True   # True = aplica OHE em variáveis categóricas
DROP_FIRST_OHE = False  # True se quiser evitar colinearidade (não necessário em RF)
FORCE_OHE_COLUMNS = [] # Colunas a forçar OHE mesmo se numéricas

# Feature Engineering (Interações)
CREATE_INTERACTIONS = True  # Se True, cria variáveis de interação
# Quais variáveis (após OHE) você quer multiplicar (ex.: LiDAR contínuas)
INTERACTION_BASE_FEATURES = ['Elev P90', 'Elev CURT mean CUBE']
# Prefixos/colunas que serão tratadas como "indicadores" (ex.: OHE de REGIONAL/ROTACAO)
# Você pode colocar prefixos, e o código pega todas as colunas que começam com isso.
INTERACTION_INDICATOR_PREFIXES = ['REGIONAL_', 'ROTACAO_', 'Idade (meses)']
# Se True, remove as colunas originais usadas nas interações (geralmente deixe False)
DROP_ORIGINAL_AFTER_INTERACTIONS = False


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

COLOR_PRIMARY = '#1f77b4'
COLOR_SECONDARY = '#ff7f0e'


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================
def create_interaction_features(X_df,
                                base_features,
                                indicator_prefixes,
                                drop_original=False):
    """
    Cria variáveis de interação: indicador(0/1) * feature_contínua.

    Ex.: REGIONAL_GN * Elev P90  ->  REGIONAL_GN__x__Elev P90

    Parameters
    ----------
    X_df : pd.DataFrame
        DataFrame com as features já numéricas (idealmente após OHE).
    base_features : list
        Lista de colunas contínuas para multiplicar (P90, P60, etc).
    indicator_prefixes : list
        Prefixos para selecionar colunas indicadoras (REGIONAL_, ROTACAO_, etc).
    drop_original : bool
        Se True, remove base_features e indicadores originais após criar interações.

    Returns
    -------
    pd.DataFrame
        DataFrame com interações adicionadas.
    """
    df_out = X_df.copy()

    # Seleciona colunas indicadoras por prefixo
    indicator_cols = []
    for pref in indicator_prefixes:
        indicator_cols.extend([c for c in df_out.columns if c.startswith(pref)])

    indicator_cols = sorted(set(indicator_cols))

    # Filtra base_features existentes
    base_cols = [c for c in base_features if c in df_out.columns]

    if not indicator_cols:
        print("  [Interações] Nenhuma coluna indicadora encontrada (prefixos não bateram).")
        return df_out

    if not base_cols:
        print("  [Interações] Nenhuma base_feature encontrada para interação.")
        return df_out

    print(f"  [Interações] Indicadores: {indicator_cols}")
    print(f"  [Interações] Bases: {base_cols}")

    # Cria interações
    for ind in indicator_cols:
        for base in base_cols:
            new_name = f"{ind}__x__{base}"
            df_out[new_name] = df_out[ind].astype(float) * df_out[base].astype(float)

    if drop_original:
        df_out = df_out.drop(columns=list(set(indicator_cols + base_cols)))

    return df_out

def apply_one_hot_encoding(df, feature_names, drop_first=False, force_ohe_columns=None):
    X_df = df[feature_names].copy()

    force_ohe_columns = force_ohe_columns or []

    # Detecta colunas não numéricas
    categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Adiciona colunas a forçar OHE (mesmo se forem numéricas)
    for col in force_ohe_columns:
        if col in X_df.columns and col not in categorical_cols:
            categorical_cols.append(col)

    if categorical_cols:
        print(f"  Aplicando One-Hot Encoding em: {categorical_cols}")
        X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=drop_first)
    else:
        print("  Nenhuma variável categórica detectada para One-Hot Encoding.")

    return X_df.values.astype(float), list(X_df.columns), X_df

def create_hyperparameter_grid():
    """
    Cria o grid de hiperparâmetros para busca randomizada.

    Returns
    -------
    dict
        Dicionário com distribuições de hiperparâmetros para RandomizedSearchCV.

    Notes
    -----
    Hiperparâmetros otimizados:
        - n_estimators: Número de árvores na floresta
        - max_features: Número de features consideradas em cada split
        - max_depth: Profundidade máxima das árvores
        - min_samples_split: Mínimo de amostras para dividir um nó
        - min_samples_leaf: Mínimo de amostras em uma folha
        - bootstrap: Se usa amostragem bootstrap
    """
    return {
        'n_estimators': [int(x) for x in np.linspace(10, 50, num=20)],
        'max_features': ['sqrt', 'log2', 1, 0.2, 0.3, 0.4, 0.5, 0.8],
        'max_depth': [int(x) for x in np.linspace(10, 150, num=10)] + [None],
        'min_samples_split': list(range(2, 13)),
        'min_samples_leaf': list(range(1, 11)),
        'bootstrap': [True, False]
    }


def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de avaliação do modelo.

    Parameters
    ----------
    y_true : array-like
        Valores observados.
    y_pred : array-like
        Valores preditos.

    Returns
    -------
    dict
        Dicionário com as métricas calculadas:
        - R²: Coeficiente de determinação
        - RMSE: Raiz do erro quadrático médio (m³/ha)
        - MAE: Erro absoluto médio (m³/ha)
        - Bias: Viés médio (m³/ha)
        - RMSE%: RMSE relativo (%)
        - MAE%: MAE relativo (%)
    """
    y_mean = np.mean(y_true)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    # Métricas relativas
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


def plot_diagnostics(y_true, y_pred, metrics, output_path=None):
    """
    Gera gráficos diagnósticos do modelo.

    Parameters
    ----------
    y_true : array-like
        Valores observados.
    y_pred : array-like
        Valores preditos.
    metrics : dict
        Dicionário com métricas do modelo.
    output_path : str or Path, optional
        Caminho para salvar a figura. Se None, apenas exibe.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com os gráficos diagnósticos.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Resíduos relativos
    residuals_pct = (y_pred - y_true) / y_true

    # --- (a) Observado vs Predito ---
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=30, c=COLOR_PRIMARY, edgecolors='white', linewidth=0.5)

    # Linha 1:1
    lim_min = 0
    lim_max = max(max(y_true), max(y_pred)) * 1.05
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label='Linha 1:1')

    # Anotação com métricas
    textstr = f"R² = {metrics['R2']:.4f}\nRMSE = {metrics['RMSE']:.2f} m³/ha\nBias = {metrics['Bias']:.2f} m³/ha"
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('VTCC Observado (m³/ha)')
    ax1.set_ylabel('VTCC Predito (m³/ha)')
    ax1.set_xlim(lim_min, lim_max)
    ax1.set_ylim(lim_min, lim_max)
    ax1.set_aspect('equal')
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_title('(a) Observado vs Predito')

    # --- (b) Resíduos vs Predito ---
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals_pct, alpha=0.6, s=30, c=COLOR_PRIMARY, edgecolors='white', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax2.axhline(y=0.2, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.axhline(y=-0.2, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax2.set_xlabel('VTCC Predito (m³/ha)')
    ax2.set_ylabel('Resíduos Relativos')
    ax2.set_ylim(-1, 1)
    ax2.grid(linestyle='--', alpha=0.3)
    ax2.set_title('(b) Resíduos vs Predito')

    # --- (c) Histograma dos Resíduos ---
    ax3 = axes[2]

    # Calcula bins centralizados
    bin_edges = np.arange(-1.05, 1.15, 0.1)
    bin_centers = np.arange(-1.0, 1.1, 0.1)
    counts, _ = np.histogram(residuals_pct, bins=bin_edges)
    percentages = counts / len(residuals_pct) * 100

    ax3.bar(bin_centers, percentages, width=0.08, alpha=0.7,
            color=COLOR_PRIMARY, edgecolor=COLOR_PRIMARY, linewidth=0.8)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5)

    ax3.set_xlabel('Resíduos Relativos')
    ax3.set_ylabel('Frequência (%)')
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_xticks(np.arange(-1.0, 1.1, 0.2))
    ax3.grid(linestyle='--', alpha=0.3)
    ax3.set_title('(c) Distribuição dos Resíduos')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Gráfico salvo: {output_path}")

    plt.show()
    return fig


def plot_feature_importance(model, feature_names, output_path=None):
    """
    Gera gráfico de importância das variáveis.

    Parameters
    ----------
    model : RandomForestRegressor
        Modelo treinado.
    feature_names : list
        Lista com nomes das variáveis.
    output_path : str or Path, optional
        Caminho para salvar a figura.

    Returns
    -------
    pd.DataFrame
        DataFrame com importância das variáveis ordenada.
    """
    # Extrai importâncias
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Cria DataFrame
    importance_df = pd.DataFrame({
        'Variable': [feature_names[i] for i in indices],
        'Importance': importances[indices],
        'Importance_pct': importances[indices] * 100
    })

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(feature_names)))
    bars = ax.barh(range(len(feature_names)), importance_df['Importance_pct'],
                   color=colors[::-1], edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(importance_df['Variable'])
    ax.set_xlabel('Importância (%)')
    ax.set_title('Importância das Variáveis - Random Forest')
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Adiciona valores nas barras
    for i, (bar, val) in enumerate(zip(bars, importance_df['Importance_pct'])):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Gráfico salvo: {output_path}")

    plt.show()
    return importance_df


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def train_random_forest(
    input_file=INPUT_FILE,
    feature_names=FEATURE_NAMES,
    target_column=TARGET_COLUMN,
    cv_folds_tuning=CV_FOLDS_TUNING,
    cv_folds_evaluation=CV_FOLDS_EVALUATION,
    n_iter_search=N_ITER_SEARCH,
    random_state=RANDOM_STATE,
    save_model=True,
    save_results=True
):
    """
    Pipeline completo de treinamento do Random Forest.

    Parameters
    ----------
    input_file : str
        Caminho do arquivo Excel com os dados.
    feature_names : list
        Lista com nomes das variáveis preditoras.
    target_column : str
        Nome da coluna alvo.
    cv_folds_tuning : int
        Número de folds para otimização de hiperparâmetros.
    cv_folds_evaluation : int
        Número de folds para avaliação final.
    n_iter_search : int
        Número de iterações do RandomizedSearchCV.
    random_state : int
        Semente para reprodutibilidade.
    save_model : bool
        Se True, salva o modelo treinado.
    save_results : bool
        Se True, salva métricas e importâncias em Excel.

    Returns
    -------
    dict
        Dicionário contendo:
        - model: Modelo treinado
        - metrics: Métricas de validação cruzada
        - best_params: Melhores hiperparâmetros
        - feature_importance: DataFrame com importância das variáveis
        - predictions: Predições da validação cruzada
    """
    print("=" * 70)
    print("TREINAMENTO DO MODELO RANDOM FOREST")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Cria diretórios de saída
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. CARREGAMENTO DOS DADOS
    # -------------------------------------------------------------------------
    print("[1/6] Carregando dados...")
    df = pd.read_excel(input_file)

    y = df[target_column].values.astype(float)

    # ---- One-Hot (se ligado) ----
    if APPLY_ONE_HOT:
        X, feature_names, X_df = apply_one_hot_encoding(
            df,
            feature_names,
            drop_first=DROP_FIRST_OHE,
            force_ohe_columns=FORCE_OHE_COLUMNS
        )
    else:
        X_df = df[feature_names].copy()
        X = X_df.values.astype(float)

    # ---- Interações (se ligado) ----
    if CREATE_INTERACTIONS:
        X_df = create_interaction_features(
            X_df,
            base_features=INTERACTION_BASE_FEATURES,
            indicator_prefixes=INTERACTION_INDICATOR_PREFIXES,
            drop_original=DROP_ORIGINAL_AFTER_INTERACTIONS
        )
        X = X_df.values.astype(float)
        feature_names = list(X_df.columns)

    print(f"  Amostras: {len(y)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Variáveis: {feature_names}")
    print()

    # -------------------------------------------------------------------------
    # 2. OTIMIZAÇÃO DE HIPERPARÂMETROS
    # -------------------------------------------------------------------------
    print("[2/6] Otimizando hiperparâmetros (RandomizedSearchCV)...")
    print(f"  Iterações: {n_iter_search}")
    print(f"  CV Folds (tuning): {cv_folds_tuning}")

    # Configuração do modelo base
    rf_base = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    # Grid de hiperparâmetros
    param_grid = create_hyperparameter_grid()

    # KFold para tuning
    cv_tuning = KFold(n_splits=cv_folds_tuning, shuffle=True, random_state=random_state)

    # RandomizedSearchCV
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        cv=cv_tuning,
        scoring='neg_root_mean_squared_error',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    rf_search.fit(X, y)
    best_params = rf_search.best_params_

    print(f"\n  Melhores hiperparâmetros encontrados:")
    for param, value in best_params.items():
        print(f"    - {param}: {value}")
    print()

    # -------------------------------------------------------------------------
    # 3. TREINAMENTO DO MODELO OTIMIZADO
    # -------------------------------------------------------------------------
    print("[3/6] Treinando modelo com hiperparâmetros otimizados...")

    rf_optimized = RandomForestRegressor(**best_params, random_state=random_state, n_jobs=-1)
    rf_optimized.fit(X, y)

    # Armazena nomes das features no modelo
    rf_optimized.feature_names_in_ = np.array(feature_names)

    print("  Modelo treinado com sucesso!")
    print()

    # -------------------------------------------------------------------------
    # 4. VALIDAÇÃO CRUZADA
    # -------------------------------------------------------------------------
    print("[4/6] Executando validação cruzada...")
    print(f"  CV Folds (avaliação): {cv_folds_evaluation}")

    cv_evaluate = KFold(n_splits=cv_folds_evaluation, shuffle=True, random_state=random_state)
    y_pred_cv = cross_val_predict(rf_optimized, X, y, cv=cv_evaluate, n_jobs=-1)

    # Calcula métricas
    metrics = calculate_metrics(y, y_pred_cv)

    print(f"\n  Métricas de Validação Cruzada ({cv_folds_evaluation}-fold):")
    print(f"    R²:    {metrics['R2']:.4f} ({metrics['R2']*100:.2f}%)")
    print(f"    RMSE:  {metrics['RMSE']:.2f} m³/ha ({metrics['RMSE_pct']:.2f}%)")
    print(f"    MAE:   {metrics['MAE']:.2f} m³/ha ({metrics['MAE_pct']:.2f}%)")
    print(f"    Bias:  {metrics['Bias']:.4f} m³/ha ({metrics['Bias_pct']:.2f}%)")
    print()

    # -------------------------------------------------------------------------
    # 5. IMPORTÂNCIA DAS VARIÁVEIS
    # -------------------------------------------------------------------------
    print("[5/7] Calculando importância das variáveis...")

    importance_df = plot_feature_importance(
        rf_optimized,
        feature_names,
        output_path=OUTPUT_DIR / 'RF_Feature_Importance.png' if save_results else None
    )

    print()

    # -------------------------------------------------------------------------
    # 6. GRÁFICOS DIAGNÓSTICOS
    # -------------------------------------------------------------------------
    print("[6/7] Gerando gráficos diagnósticos...")

    if save_results:
        plot_diagnostics(
            y, y_pred_cv, metrics,
            output_path=OUTPUT_DIR / 'RF_Diagnostics.png'
        )
    print()

    # -------------------------------------------------------------------------
    # 7. EXPORTAÇÃO DOS RESULTADOS
    # -------------------------------------------------------------------------
    print("[7/7] Exportando resultados...")

    if save_model:
        model_path = MODEL_DIR / 'RF_Regressor_P90_CUB.pkl'
        joblib.dump(rf_optimized, model_path)
        print(f"  Modelo salvo: {model_path}")

    if save_results:
        metrics_df = pd.DataFrame([{
            'Data_Treinamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Modelo': 'Random Forest Regressor',
            'N_Amostras': len(y),
            'N_Features': len(feature_names),
            'Features': str(feature_names),
            'CV_Folds': cv_folds_evaluation,
            **metrics,
            'Best_Params': str(best_params)
        }])

        metrics_path = OUTPUT_DIR / 'RF_Training_Metrics.xlsx'
        with pd.ExcelWriter(metrics_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

            pred_df = pd.DataFrame({
                'Observado': y,
                'Predito_CV': y_pred_cv,
                'Residuo': y - y_pred_cv,
                'Residuo_pct': (y - y_pred_cv) / y * 100
            })
            pred_df.to_excel(writer, sheet_name='Predictions', index=False)

        print(f"  Metricas salvas: {metrics_path}")

    print()
    print("=" * 70)
    print("TREINAMENTO CONCLUÍDO")
    print(f"Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return {
        'model': rf_optimized,
        'metrics': metrics,
        'best_params': best_params,
        'feature_importance': importance_df,
        'predictions': y_pred_cv
    }


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    results = train_random_forest()
