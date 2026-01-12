"""
07_SHAPAnalysis.py - Análise SHAP para Interpretabilidade do Modelo

Este script realiza uma análise crítica do modelo Random Forest utilizando
SHAP (SHapley Additive exPlanations) para interpretabilidade e explicabilidade.

SHAP é baseado na teoria dos jogos cooperativos (valores de Shapley) e fornece
uma explicação consistente e localmente precisa para cada predição.

Workflow:
    1. Carregamento do modelo treinado e dados
    2. Cálculo dos valores SHAP (TreeExplainer para RF)
    3. Análise de importância global das features
    4. Análise de efeitos direcionais (positivo/negativo)
    5. Análise de interações entre variáveis
    6. Análise de dependência parcial
    7. Explicações locais (casos individuais)
    8. Identificação de padrões e anomalias
    9. Exportação de resultados e visualizações

Interpretação dos valores SHAP:
    - Valor SHAP positivo: feature aumenta a predição em relação à média
    - Valor SHAP negativo: feature diminui a predição em relação à média
    - Magnitude: quanto maior |SHAP|, maior a contribuição da feature

Saídas:
    - Results/SHAP_Summary.png: Resumo da importância das features
    - Results/SHAP_Beeswarm.png: Distribuição dos valores SHAP
    - Results/SHAP_Dependence_*.png: Gráficos de dependência parcial
    - Results/SHAP_Interactions.png: Análise de interações
    - Results/SHAP_Waterfall_*.png: Explicações individuais
    - Results/SHAP_Analysis_Report.xlsx: Relatório completo

Autor: Leonardo Ippolito Rodrigues
Data: 2026
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminhos
INPUT_FILE = r"G:\PycharmProjects\Mestrado\Data\DataFrames\IFC_LiDAR_Plots_RTK_Cleaned_v02.xlsx"
MODEL_FILE = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Models\RF_Regressor_P90_CUB.pkl"
OUTPUT_DIR = Path(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results")

# Variáveis (devem corresponder às usadas no treinamento)
FEATURE_NAMES = [
    'Elev P90',
    'Elev CURT mean CUBE',        # Percentil 60 - estrutura média do dossel
    'Elev stddev',
    'ROTACAO',
    'REGIONAL',
    'Idade (meses)'
]

TARGET_COLUMN = 'VTCC(m³/ha)'

# Configurações de análise
APPLY_ONE_HOT = True
DROP_FIRST_OHE = False
FORCE_OHE_COLUMNS = []

# Feature Engineering (mesmo do treinamento)
CREATE_INTERACTIONS = True
INTERACTION_BASE_FEATURES = ['Elev P90', 'Elev P60', 'Elev maximum']
INTERACTION_INDICATOR_PREFIXES = ['REGIONAL_', 'ROTACAO_', 'Idade (meses)']
DROP_ORIGINAL_AFTER_INTERACTIONS = False

# Número de amostras para análise SHAP (None = todas)
SHAP_SAMPLE_SIZE = None

# Número de features principais para plots detalhados
TOP_FEATURES = 10

# Casos individuais para análise (índices ou None para automático)
INDIVIDUAL_CASES = None  # Se None, seleciona automaticamente

# Configuração de estilo dos gráficos
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "figure.facecolor": "white"
})

# Cores
COLOR_PRIMARY = '#1f77b4'
COLOR_SECONDARY = '#ff7f0e'
COLOR_POSITIVE = '#d62728'
COLOR_NEGATIVE = '#2ca02c'


# =============================================================================
# FUNÇÕES AUXILIARES (do treinamento)
# =============================================================================

def apply_one_hot_encoding(df, feature_names, drop_first=False, force_ohe_columns=None):
    """Aplica One-Hot Encoding nas variáveis categóricas."""
    X_df = df[feature_names].copy()
    force_ohe_columns = force_ohe_columns or []

    categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in force_ohe_columns:
        if col in X_df.columns and col not in categorical_cols:
            categorical_cols.append(col)

    if categorical_cols:
        X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=drop_first)

    return X_df.values.astype(float), list(X_df.columns), X_df


def create_interaction_features(X_df, base_features, indicator_prefixes, drop_original=False):
    """Cria variáveis de interação."""
    df_out = X_df.copy()

    indicator_cols = []
    for pref in indicator_prefixes:
        indicator_cols.extend([c for c in df_out.columns if c.startswith(pref)])

    indicator_cols = sorted(set(indicator_cols))
    base_cols = [c for c in base_features if c in df_out.columns]

    if not indicator_cols or not base_cols:
        return df_out

    for ind in indicator_cols:
        for base in base_cols:
            new_name = f"{ind}__x__{base}"
            df_out[new_name] = df_out[ind].astype(float) * df_out[base].astype(float)

    if drop_original:
        df_out = df_out.drop(columns=list(set(indicator_cols + base_cols)))

    return df_out


# =============================================================================
# FUNÇÕES DE ANÁLISE SHAP
# =============================================================================

def calculate_shap_values(model, X: np.ndarray, feature_names: List[str],
                          sample_size: int = None) -> Tuple[shap.Explainer, np.ndarray]:
    """
    Calcula os valores SHAP usando TreeExplainer.

    Parameters
    ----------
    model : RandomForestRegressor
        Modelo treinado.
    X : np.ndarray
        Matriz de features.
    feature_names : list
        Nomes das features.
    sample_size : int, optional
        Número de amostras para análise (None = todas).

    Returns
    -------
    tuple
        (explainer, shap_values)
    """
    print("Calculando valores SHAP...")

    if sample_size and sample_size < len(X):
        np.random.seed(42)
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
        indices = np.arange(len(X))

    # TreeExplainer é otimizado para modelos baseados em árvores
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    print(f"  Amostras analisadas: {len(X_sample)}")
    print(f"  Features: {len(feature_names)}")

    return explainer, shap_values, indices


def plot_summary_bar(shap_values: np.ndarray, feature_names: List[str],
                     output_path: Path = None, top_n: int = 15,
                     group_others: bool = True, show_pct: bool = True):
    """
    Gera gráfico de barras com importância média |SHAP|.

    Parameters
    ----------
    shap_values : np.ndarray
        Valores SHAP calculados.
    feature_names : list
        Nomes das features.
    output_path : Path, optional
        Caminho para salvar a figura.
    top_n : int
        Número de features principais a mostrar.
    group_others : bool
        Se True, agrupa features menos importantes em "Outras".
    show_pct : bool
        Se True, mostra porcentagem além do valor absoluto.
    """
    # Calcula importância média absoluta
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_importance = mean_abs_shap.sum()

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP': mean_abs_shap
    }).sort_values('Mean_SHAP', ascending=False)

    # Adiciona porcentagem
    importance_df['Pct'] = importance_df['Mean_SHAP'] / total_importance * 100
    importance_df['Cumulative_Pct'] = importance_df['Pct'].cumsum()

    # Agrupa "Outras" se necessário
    if group_others and len(importance_df) > top_n:
        top_df = importance_df.head(top_n).copy()
        others_df = importance_df.tail(len(importance_df) - top_n)

        others_row = pd.DataFrame([{
            'Feature': f'Outras ({len(others_df)} variáveis)',
            'Mean_SHAP': others_df['Mean_SHAP'].sum(),
            'Pct': others_df['Pct'].sum(),
            'Cumulative_Pct': 100.0
        }])

        plot_df = pd.concat([top_df, others_row], ignore_index=True)
        plot_df = plot_df.sort_values('Mean_SHAP', ascending=True)
    else:
        plot_df = importance_df.head(top_n).sort_values('Mean_SHAP', ascending=True)

    # Configura figura
    n_bars = len(plot_df)
    fig_height = max(6, n_bars * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Cores - destaca "Outras" em cinza
    colors = []
    for fname in plot_df['Feature']:
        if fname.startswith('Outras'):
            colors.append('#808080')  # Cinza para "Outras"
        else:
            colors.append(plt.cm.Blues(0.3 + 0.6 * (list(plot_df['Feature']).index(fname) / n_bars)))

    bars = ax.barh(range(n_bars), plot_df['Mean_SHAP'],
                   color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(n_bars))
    ax.set_yticklabels(plot_df['Feature'])
    ax.set_xlabel('Importância SHAP Média (|SHAP|)')
    ax.set_title('Importância Global das Variáveis - SHAP')
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Adiciona valores e porcentagens
    max_val = plot_df['Mean_SHAP'].max()
    for i, (bar, row) in enumerate(zip(bars, plot_df.itertuples())):
        if show_pct:
            label = f'{row.Mean_SHAP:.2f} ({row.Pct:.1f}%)'
        else:
            label = f'{row.Mean_SHAP:.2f}'
        ax.text(row.Mean_SHAP + 0.01 * max_val,
                bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Gráfico salvo: {output_path}")

    plt.show()

    # Retorna DataFrame completo (não agrupado) para análise
    return importance_df


def plot_beeswarm(shap_values: np.ndarray, X: np.ndarray, feature_names: List[str],
                  output_path: Path = None, top_n: int = 10):
    """
    Gera beeswarm plot mostrando distribuição dos valores SHAP.

    Este gráfico mostra:
    - Eixo X: valor SHAP (impacto na predição)
    - Eixo Y: features ordenadas por importância (apenas top_n)
    - Cor: valor da feature (azul=baixo, vermelho=alto)

    Parameters
    ----------
    shap_values : np.ndarray
        Valores SHAP calculados.
    X : np.ndarray
        Matriz de features.
    feature_names : list
        Nomes das features.
    output_path : Path, optional
        Caminho para salvar a figura.
    top_n : int
        Número de features a mostrar.
    """
    # Calcula importância e seleciona top N
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    # Filtra para top N features apenas
    shap_values_filtered = shap_values[:, top_indices]
    X_filtered = X[:, top_indices]
    feature_names_filtered = [feature_names[i] for i in top_indices]

    # Altura dinâmica
    fig_height = max(6, top_n * 0.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Cria objeto Explanation com dados filtrados
    explanation = shap.Explanation(
        values=shap_values_filtered,
        base_values=np.zeros(len(shap_values_filtered)),
        data=X_filtered,
        feature_names=feature_names_filtered
    )

    shap.plots.beeswarm(explanation, max_display=top_n, show=False)

    plt.title(f'Distribuição dos Valores SHAP (Top {top_n} Variáveis)')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Gráfico salvo: {output_path}")

    plt.show()


def plot_dependence(shap_values: np.ndarray, X: np.ndarray, feature_names: List[str],
                    feature_idx: int, interaction_idx: int = None,
                    output_path: Path = None):
    """
    Gera gráfico de dependência parcial SHAP.

    Mostra como o valor SHAP de uma feature varia com seu valor,
    opcionalmente colorido por uma feature de interação.
    """
    feature_name = feature_names[feature_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    if interaction_idx is not None:
        interaction_name = feature_names[interaction_idx]
        scatter = ax.scatter(X[:, feature_idx], shap_values[:, feature_idx],
                             c=X[:, interaction_idx], cmap='coolwarm',
                             alpha=0.6, s=20)
        plt.colorbar(scatter, label=interaction_name)
        title = f'Dependência SHAP: {feature_name}\n(cor: {interaction_name})'
    else:
        ax.scatter(X[:, feature_idx], shap_values[:, feature_idx],
                   alpha=0.6, s=20, c=COLOR_PRIMARY)
        title = f'Dependência SHAP: {feature_name}'

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(f'Valor de {feature_name}')
    ax.set_ylabel(f'Valor SHAP para {feature_name}')
    ax.set_title(title)
    ax.grid(linestyle='--', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Gráfico salvo: {output_path}")

    plt.show()


def plot_waterfall(explainer, shap_values: np.ndarray, X: np.ndarray,
                   feature_names: List[str], sample_idx: int,
                   y_true: float = None, y_pred: float = None,
                   output_path: Path = None):
    """
    Gera waterfall plot para explicação individual.

    Mostra como cada feature contribui para a predição de uma amostra específica.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Trata expected_value como array ou escalar
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value.mean() if len(base_value) > 1 else base_value[0]

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=base_value,
        data=X[sample_idx],
        feature_names=feature_names
    )

    shap.plots.waterfall(explanation, max_display=15, show=False)

    title = f'Explicação Individual - Amostra {sample_idx}'
    if y_true is not None and y_pred is not None:
        title += f'\nObservado: {y_true:.2f} m³/ha | Predito: {y_pred:.2f} m³/ha'

    plt.title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Gráfico salvo: {output_path}")

    plt.show()


def plot_force(explainer, shap_values: np.ndarray, X: np.ndarray,
               feature_names: List[str], sample_idx: int,
               output_path: Path = None):
    """
    Gera force plot para explicação individual.
    """
    # Trata expected_value como array ou escalar
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value.mean() if len(base_value) > 1 else base_value[0]

    # Force plot em matplotlib
    shap.force_plot(
        base_value,
        shap_values[sample_idx],
        X[sample_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )

    plt.title(f'Force Plot - Amostra {sample_idx}')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Gráfico salvo: {output_path}")

    plt.show()


def analyze_feature_effects(shap_values: np.ndarray, X: np.ndarray,
                            feature_names: List[str]) -> pd.DataFrame:
    """
    Analisa os efeitos direcionais das features.

    Returns
    -------
    pd.DataFrame
        DataFrame com estatísticas dos efeitos SHAP por feature.
    """
    results = []

    for i, fname in enumerate(feature_names):
        shap_col = shap_values[:, i]
        feat_col = X[:, i]

        # Estatísticas básicas
        mean_shap = np.mean(shap_col)
        std_shap = np.std(shap_col)
        mean_abs_shap = np.mean(np.abs(shap_col))

        # Proporção de efeitos positivos/negativos
        pct_positive = np.mean(shap_col > 0) * 100
        pct_negative = np.mean(shap_col < 0) * 100

        # Correlação entre valor da feature e SHAP
        if np.std(feat_col) > 0:
            correlation = np.corrcoef(feat_col, shap_col)[0, 1]
        else:
            correlation = 0

        # Range dos efeitos
        min_shap = np.min(shap_col)
        max_shap = np.max(shap_col)

        results.append({
            'Feature': fname,
            'Mean_SHAP': mean_shap,
            'Std_SHAP': std_shap,
            'Mean_Abs_SHAP': mean_abs_shap,
            'Min_SHAP': min_shap,
            'Max_SHAP': max_shap,
            'Pct_Positive': pct_positive,
            'Pct_Negative': pct_negative,
            'Correlation': correlation,
            'Direction': 'Positivo' if correlation > 0.3 else ('Negativo' if correlation < -0.3 else 'Misto')
        })

    df = pd.DataFrame(results)
    df = df.sort_values('Mean_Abs_SHAP', ascending=False)

    return df


def analyze_interactions(shap_values: np.ndarray, X: np.ndarray,
                         feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Analisa interações entre features baseado em SHAP.

    Usa a variância dos valores SHAP condicionada aos valores de outras features.
    """
    n_features = len(feature_names)
    interactions = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Calcula correlação entre SHAP de i e valor de j
            if np.std(X[:, j]) > 0 and np.std(shap_values[:, i]) > 0:
                corr_ij = np.abs(np.corrcoef(shap_values[:, i], X[:, j])[0, 1])
            else:
                corr_ij = 0

            if np.std(X[:, i]) > 0 and np.std(shap_values[:, j]) > 0:
                corr_ji = np.abs(np.corrcoef(shap_values[:, j], X[:, i])[0, 1])
            else:
                corr_ji = 0

            interaction_strength = (corr_ij + corr_ji) / 2

            interactions.append({
                'Feature_1': feature_names[i],
                'Feature_2': feature_names[j],
                'Interaction_Strength': interaction_strength
            })

    df = pd.DataFrame(interactions)
    df = df.sort_values('Interaction_Strength', ascending=False).head(top_n)

    return df


def plot_interaction_heatmap(shap_values: np.ndarray, X: np.ndarray,
                             feature_names: List[str], output_path: Path = None,
                             top_n: int = 15):
    """
    Gera heatmap de interações entre features.
    """
    # Seleciona top features por importância
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:]

    n_top = len(top_indices)
    interaction_matrix = np.zeros((n_top, n_top))

    for ii, i in enumerate(top_indices):
        for jj, j in enumerate(top_indices):
            if i != j:
                if np.std(X[:, j]) > 0 and np.std(shap_values[:, i]) > 0:
                    interaction_matrix[ii, jj] = np.abs(
                        np.corrcoef(shap_values[:, i], X[:, j])[0, 1]
                    )

    top_names = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.eye(n_top, dtype=bool)
    sns.heatmap(interaction_matrix, mask=mask, annot=True, fmt='.2f',
                xticklabels=top_names, yticklabels=top_names,
                cmap='YlOrRd', ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'Força da Interação'})

    ax.set_title('Matriz de Interações SHAP entre Features')
    ax.set_xlabel('Feature que Modifica')
    ax.set_ylabel('Feature Modificada')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Gráfico salvo: {output_path}")

    plt.show()


def identify_anomalies(shap_values: np.ndarray, y_true: np.ndarray,
                       y_pred: np.ndarray, feature_names: List[str],
                       threshold_percentile: float = 95) -> pd.DataFrame:
    """
    Identifica amostras com comportamento anômalo baseado em SHAP.

    Procura por:
    - Amostras com valores SHAP extremos
    - Grandes erros de predição
    - Contribuições inesperadas
    """
    # Soma absoluta dos SHAP por amostra (complexidade da explicação)
    total_abs_shap = np.sum(np.abs(shap_values), axis=1)

    # Erro de predição
    errors = np.abs(y_pred - y_true)
    errors_pct = errors / y_true * 100

    # Threshold para anomalias
    shap_threshold = np.percentile(total_abs_shap, threshold_percentile)
    error_threshold = np.percentile(errors_pct, threshold_percentile)

    anomalies = []

    for i in range(len(y_true)):
        is_shap_anomaly = total_abs_shap[i] > shap_threshold
        is_error_anomaly = errors_pct[i] > error_threshold

        if is_shap_anomaly or is_error_anomaly:
            # Encontra feature com maior contribuição
            max_shap_idx = np.argmax(np.abs(shap_values[i]))
            max_shap_feature = feature_names[max_shap_idx]
            max_shap_value = shap_values[i, max_shap_idx]

            anomalies.append({
                'Sample_Index': i,
                'Y_True': y_true[i],
                'Y_Pred': y_pred[i],
                'Error_Abs': errors[i],
                'Error_Pct': errors_pct[i],
                'Total_Abs_SHAP': total_abs_shap[i],
                'Max_SHAP_Feature': max_shap_feature,
                'Max_SHAP_Value': max_shap_value,
                'Is_SHAP_Anomaly': is_shap_anomaly,
                'Is_Error_Anomaly': is_error_anomaly
            })

    return pd.DataFrame(anomalies)


def generate_critical_analysis(effects_df: pd.DataFrame, interactions_df: pd.DataFrame,
                               anomalies_df: pd.DataFrame, model, feature_names: List[str]) -> str:
    """
    Gera análise crítica textual dos resultados SHAP.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ANÁLISE CRÍTICA DO MODELO - INTERPRETABILIDADE SHAP")
    lines.append("=" * 70)
    lines.append("")

    # 1. Importância das features
    lines.append("1. IMPORTÂNCIA DAS VARIÁVEIS")
    lines.append("-" * 40)
    top5 = effects_df.head(5)
    for _, row in top5.iterrows():
        lines.append(f"   {row['Feature']}: {row['Mean_Abs_SHAP']:.3f} ({row['Direction']})")
    lines.append("")

    # 2. Análise de direção dos efeitos
    lines.append("2. DIREÇÃO DOS EFEITOS")
    lines.append("-" * 40)
    for _, row in effects_df.head(10).iterrows():
        direction_text = "aumenta" if row['Correlation'] > 0 else "diminui"
        if abs(row['Correlation']) < 0.3:
            direction_text = "tem efeito misto sobre"
        lines.append(f"   {row['Feature']}: {direction_text} a predição")
        lines.append(f"      Correlação: {row['Correlation']:.3f}, "
                     f"Positivo: {row['Pct_Positive']:.1f}%, Negativo: {row['Pct_Negative']:.1f}%")
    lines.append("")

    # 3. Interações importantes
    lines.append("3. INTERAÇÕES ENTRE VARIÁVEIS")
    lines.append("-" * 40)
    if len(interactions_df) > 0:
        for _, row in interactions_df.head(5).iterrows():
            lines.append(f"   {row['Feature_1']} x {row['Feature_2']}: "
                         f"força = {row['Interaction_Strength']:.3f}")
    else:
        lines.append("   Nenhuma interação significativa detectada.")
    lines.append("")

    # 4. Anomalias
    lines.append("4. AMOSTRAS ANÔMALAS")
    lines.append("-" * 40)
    if len(anomalies_df) > 0:
        n_shap_anomalies = anomalies_df['Is_SHAP_Anomaly'].sum()
        n_error_anomalies = anomalies_df['Is_Error_Anomaly'].sum()
        lines.append(f"   Anomalias SHAP: {n_shap_anomalies}")
        lines.append(f"   Anomalias de erro: {n_error_anomalies}")
        lines.append(f"   Total de casos atípicos: {len(anomalies_df)}")

        # Piores casos
        worst = anomalies_df.nlargest(3, 'Error_Pct')
        lines.append("\n   Casos com maiores erros:")
        for _, row in worst.iterrows():
            lines.append(f"      Amostra {row['Sample_Index']}: erro = {row['Error_Pct']:.1f}%")
            lines.append(f"         Principal contribuição: {row['Max_SHAP_Feature']} "
                         f"(SHAP = {row['Max_SHAP_Value']:.2f})")
    else:
        lines.append("   Nenhuma anomalia detectada.")
    lines.append("")

    # 5. Considerações críticas
    lines.append("5. CONSIDERAÇÕES CRÍTICAS")
    lines.append("-" * 40)

    # Verifica dominância de features
    top1_importance = effects_df.iloc[0]['Mean_Abs_SHAP']
    total_importance = effects_df['Mean_Abs_SHAP'].sum()
    top1_pct = top1_importance / total_importance * 100

    if top1_pct > 50:
        lines.append(f"   ATENÇÃO: A variável '{effects_df.iloc[0]['Feature']}' domina "
                     f"o modelo ({top1_pct:.1f}% da importância total).")
        lines.append("   Isso pode indicar overfitting ou redundância nas outras features.")

    # Verifica features com baixa importância
    low_importance = effects_df[effects_df['Mean_Abs_SHAP'] < 0.1 * top1_importance]
    if len(low_importance) > 0:
        lines.append(f"\n   {len(low_importance)} variáveis têm importância < 10% da principal:")
        for fname in low_importance['Feature'].head(5).values:
            lines.append(f"      - {fname}")
        lines.append("   Considere remover essas variáveis para simplificar o modelo.")

    # Verifica interações fortes
    if len(interactions_df) > 0:
        strong_interactions = interactions_df[interactions_df['Interaction_Strength'] > 0.5]
        if len(strong_interactions) > 0:
            lines.append(f"\n   {len(strong_interactions)} interações fortes detectadas.")
            lines.append("   O modelo pode estar capturando relações não-lineares importantes.")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_shap_analysis(
        input_file: str = INPUT_FILE,
        model_file: str = MODEL_FILE,
        feature_names: List[str] = FEATURE_NAMES,
        target_column: str = TARGET_COLUMN,
        output_dir: Path = OUTPUT_DIR,
        sample_size: int = SHAP_SAMPLE_SIZE,
        top_features: int = TOP_FEATURES
) -> Dict:
    """
    Executa pipeline completo de análise SHAP.

    Returns
    -------
    dict
        Dicionário com todos os resultados da análise.
    """
    print("=" * 70)
    print("ANÁLISE SHAP - INTERPRETABILIDADE DO MODELO")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. CARREGAMENTO DOS DADOS E MODELO
    # -------------------------------------------------------------------------
    print("[1/8] Carregando dados e modelo...")

    df = pd.read_excel(input_file)
    model = joblib.load(model_file)

    y = df[target_column].values.astype(float)

    # Processa features (mesmo pipeline do treinamento)
    if APPLY_ONE_HOT:
        X, processed_features, X_df = apply_one_hot_encoding(
            df, feature_names, drop_first=DROP_FIRST_OHE,
            force_ohe_columns=FORCE_OHE_COLUMNS
        )
    else:
        X_df = df[feature_names].copy()
        X = X_df.values.astype(float)
        processed_features = feature_names

    if CREATE_INTERACTIONS:
        X_df = create_interaction_features(
            X_df,
            base_features=INTERACTION_BASE_FEATURES,
            indicator_prefixes=INTERACTION_INDICATOR_PREFIXES,
            drop_original=DROP_ORIGINAL_AFTER_INTERACTIONS
        )
        X = X_df.values.astype(float)
        processed_features = list(X_df.columns)

    print(f"  Amostras: {len(y)}")
    print(f"  Features processadas: {len(processed_features)}")
    print()

    # Predições do modelo
    y_pred = model.predict(X)

    # -------------------------------------------------------------------------
    # 2. CÁLCULO DOS VALORES SHAP
    # -------------------------------------------------------------------------
    print("[2/8] Calculando valores SHAP...")

    explainer, shap_values, sample_indices = calculate_shap_values(
        model, X, processed_features, sample_size=sample_size
    )

    X_sample = X[sample_indices]
    y_sample = y[sample_indices]
    y_pred_sample = y_pred[sample_indices]

    expected_val = explainer.expected_value
    if isinstance(expected_val, np.ndarray):
        expected_val = expected_val.mean() if len(expected_val) > 1 else expected_val[0]
    print(f"  Valor base (expected value): {expected_val:.2f}")
    print()

    # -------------------------------------------------------------------------
    # 3. IMPORTÂNCIA GLOBAL
    # -------------------------------------------------------------------------
    print("[3/8] Analisando importância global...")

    importance_df = plot_summary_bar(
        shap_values, processed_features,
        output_path=output_dir / 'SHAP_Summary.png',
        top_n=top_features,
        group_others=True,
        show_pct=True
    )
    print()

    # -------------------------------------------------------------------------
    # 4. BEESWARM PLOT
    # -------------------------------------------------------------------------
    print("[4/8] Gerando beeswarm plot...")

    plot_beeswarm(
        shap_values, X_sample, processed_features,
        output_path=output_dir / 'SHAP_Beeswarm.png',
        top_n=top_features
    )
    print()

    # -------------------------------------------------------------------------
    # 5. ANÁLISE DE EFEITOS
    # -------------------------------------------------------------------------
    print("[5/8] Analisando efeitos direcionais...")

    effects_df = analyze_feature_effects(shap_values, X_sample, processed_features)
    print(effects_df[['Feature', 'Mean_Abs_SHAP', 'Direction', 'Correlation']].head(10).to_string())
    print()

    # -------------------------------------------------------------------------
    # 6. GRÁFICOS DE DEPENDÊNCIA
    # -------------------------------------------------------------------------
    print("[6/8] Gerando gráficos de dependência...")

    # Top 3 features mais importantes
    top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[-3:][::-1]

    for idx in top_indices:
        fname = processed_features[idx]
        safe_fname = fname.replace(' ', '_').replace('/', '_')

        plot_dependence(
            shap_values, X_sample, processed_features,
            feature_idx=idx,
            output_path=output_dir / f'SHAP_Dependence_{safe_fname}.png'
        )
    print()

    # -------------------------------------------------------------------------
    # 7. ANÁLISE DE INTERAÇÕES
    # -------------------------------------------------------------------------
    print("[7/8] Analisando interações...")

    interactions_df = analyze_interactions(shap_values, X_sample, processed_features)
    print("Top 5 interações:")
    print(interactions_df.head().to_string())
    print()

    plot_interaction_heatmap(
        shap_values, X_sample, processed_features,
        output_path=output_dir / 'SHAP_Interactions.png',
        top_n=min(15, len(processed_features))
    )

    # -------------------------------------------------------------------------
    # 8. EXPLICAÇÕES INDIVIDUAIS E ANOMALIAS
    # -------------------------------------------------------------------------
    print("[8/8] Gerando explicações individuais...")

    # Identifica anomalias
    anomalies_df = identify_anomalies(
        shap_values, y_sample, y_pred_sample, processed_features
    )

    # Seleciona casos para waterfall
    if INDIVIDUAL_CASES:
        cases_to_explain = INDIVIDUAL_CASES
    else:
        # Seleciona: melhor predição, pior predição, caso típico
        errors = np.abs(y_sample - y_pred_sample)
        best_idx = np.argmin(errors)
        worst_idx = np.argmax(errors)
        median_error = np.median(errors)
        typical_idx = np.argmin(np.abs(errors - median_error))

        cases_to_explain = [best_idx, typical_idx, worst_idx]
        case_names = ['Best', 'Typical', 'Worst']

    for i, idx in enumerate(cases_to_explain):
        case_name = case_names[i] if not INDIVIDUAL_CASES else f'Sample_{idx}'

        plot_waterfall(
            explainer, shap_values, X_sample, processed_features,
            sample_idx=idx,
            y_true=y_sample[idx],
            y_pred=y_pred_sample[idx],
            output_path=output_dir / f'SHAP_Waterfall_{case_name}.png'
        )

    # -------------------------------------------------------------------------
    # ANÁLISE CRÍTICA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    critical_analysis = generate_critical_analysis(
        effects_df, interactions_df, anomalies_df, model, processed_features
    )
    print(critical_analysis)

    # -------------------------------------------------------------------------
    # EXPORTAÇÃO DOS RESULTADOS
    # -------------------------------------------------------------------------
    print("\nExportando resultados...")

    report_path = output_dir / 'SHAP_Analysis_Report.xlsx'
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        # Metadados
        meta_df = pd.DataFrame([{
            'Data_Analise': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Modelo': model_file,
            'N_Amostras': len(X_sample),
            'N_Features': len(processed_features),
            'Expected_Value': expected_val
        }])
        meta_df.to_excel(writer, sheet_name='Metadata', index=False)

        # Importância
        effects_df.to_excel(writer, sheet_name='Feature_Effects', index=False)

        # Interações
        interactions_df.to_excel(writer, sheet_name='Interactions', index=False)

        # Anomalias
        if len(anomalies_df) > 0:
            anomalies_df.to_excel(writer, sheet_name='Anomalies', index=False)

        # SHAP values (amostra)
        shap_df = pd.DataFrame(shap_values, columns=processed_features)
        shap_df['Y_True'] = y_sample
        shap_df['Y_Pred'] = y_pred_sample
        shap_df.to_excel(writer, sheet_name='SHAP_Values', index=False)

    print(f"  Relatório salvo: {report_path}")

    # Salva análise crítica em texto
    analysis_path = output_dir / 'SHAP_Critical_Analysis.txt'
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(critical_analysis)
    print(f"  Análise crítica salva: {analysis_path}")

    print()
    print("=" * 70)
    print("ANÁLISE SHAP CONCLUÍDA")
    print(f"Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'effects_df': effects_df,
        'interactions_df': interactions_df,
        'anomalies_df': anomalies_df,
        'feature_names': processed_features,
        'critical_analysis': critical_analysis
    }


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    results = run_shap_analysis()
