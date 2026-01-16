#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
Feature Combination Analysis - Model Comparison Visualization
================================================================================

Script para análise e visualização de resultados de busca de combinações de
variáveis (feature subsets) comparando três grupos de modelos:
- Cadastro: variáveis de inventário/cadastro florestal
- LiDAR: métricas derivadas de LiDAR
- Cadastro+LiDAR: combinação de ambos

Gera um pacote completo de gráficos para tese acadêmica com estética adequada.
Estilo: articletables (sem frame direito/superior, grid horizontal sutil).

Autor: Data Science Pipeline
Data: 2026-01-12
================================================================================
"""

import os
import ast
import warnings
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES DO USUÁRIO - EDITAR AQUI
# =============================================================================

# Caminho do arquivo de entrada (Excel ou CSV)
RESULTS_FILE = r"G:\PycharmProjects\Mestrado\Data\DataFrames\RandomForest_Comb_Results_CV_KFold10_TuningCV10_RMSE_2026.xlsx"

# Pasta de saída para figuras (será criada se não existir)
FIGURES_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results\Figures"

# Mapeamento de colunas (ajustar conforme necessidade)
# Chave: nome esperado interno -> Valor: nome real no arquivo
COLUMN_MAPPING = {
    'group': 'Group',           # Coluna com tipo do modelo (pode ser None se não existir)
    'n_features': 'N_Features', # Número de features na combinação
    'r2': 'CV_R2',              # R² em cross-validation
    'rmse': 'CV_RMSE',          # RMSE em cross-validation
    'mae': 'CV_MAE',            # MAE em cross-validation (opcional)
    'features': 'Features',     # String/lista com nomes das features
}

# Definição de features por grupo (para inferência automática do grupo)
# IMPORTANTE: Ajustar conforme as variáveis do seu dataset
CADASTRO_FEATURES = [
    'Idade (meses)',
    'Idade_meses',
    'Regime',
    'REGIONAL',
    'Regional',
    'ROTACAO',
    'Rotacao',
    'ESPECIE',
    'Especie',
    'CLONE',
    'Clone',
    'ESPACAMENTO',
    'Espacamento',
    'AREA',
    'Area',
    'DAP',
    'dap',
    'HT',
    'ht',
    'Altitude',
    'ALTITUDE',
]

# Features LiDAR (métricas de elevação, intensidade, etc.)
# Padrões comuns: Elev_*, Int_*, Dens_*, etc.
LIDAR_FEATURES_PATTERNS = [
    'Elev_',
    'Int_',
    'Dens_',
    'Canopy_',
    'CHM_',
    'LAI_',
    'Cover_',
    'Gap_',
    'Perc_',
    'P10', 'P20', 'P25', 'P30', 'P40', 'P50', 'P60', 'P70', 'P75', 'P80', 'P90', 'P95', 'P99',
    'maximum', 'minimum', 'mean', 'std', 'skewness', 'kurtosis', 'cv',
]

# Cores para cada grupo (paleta YlGnBu)
# Obtidas do colormap YlGnBu em posições espaçadas
_cmap = plt.cm.YlGnBu
GROUP_COLORS = {
    'Cadastro': _cmap(0.3),        # Amarelo-verde claro
    'LiDAR': _cmap(0.55),          # Verde-azul médio
    'Cadastro+LiDAR': _cmap(0.85), # Azul escuro
}

# Ordem dos grupos para plotagem consistente
GROUP_ORDER = ['Cadastro', 'LiDAR', 'Cadastro+LiDAR']

# =============================================================================
# CONFIGURAÇÕES DE ESTILO PARA TESE (articletables style)
# =============================================================================
from matplotlib.ticker import FuncFormatter

def pt_br_formatter(x, pos):
    """
    Formata número com vírgula decimal (pt-BR).
    Ex: 1.23 -> 1,23
    """
    return f"{x:.1f}".replace(".", ",")

def pt_br_smart(x, pos):
    if abs(x - int(x)) < 1e-6:
        return f"{int(x)}"
    return f"{x:.1f}".replace(".", ",")

def setup_thesis_style():
    """Configura estilo de gráficos para tese acadêmica (articletables style)."""
    plt.rcParams.update({
        # Fonte
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,

        # Eixos
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,     # Sem frame superior
        'axes.spines.right': False,   # Sem frame direito
        'axes.spines.left': True,
        'axes.spines.bottom': True,

        # Ticks
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Legenda
        'legend.fontsize': 10,
        'legend.frameon': False,
        'legend.framealpha': 0.9,

        # Figura
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Grid (horizontal sutil)
        'axes.grid': False,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'grid.color': '#cccccc',
    })


def apply_articletables_style(ax):
    """
    Aplica estilo articletables a um eixo específico.
    - Remove spines direito e superior
    - Adiciona grid horizontal sutil
    """
    # Remover spines direito e superior
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ajustar spines restantes
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')

    # Grid horizontal sutil
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#cccccc', alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Ticks apenas para fora
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.8)


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def create_output_directory(path: str) -> str:
    """Cria diretório de saída se não existir."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Diretório criado: {path}")
    return path


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega arquivo Excel ou CSV.

    Parameters
    ----------
    filepath : str
        Caminho do arquivo de entrada

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados carregados
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    elif ext == '.csv':
        # Tenta detectar separador
        df = pd.read_csv(filepath, sep=None, engine='python')
    else:
        raise ValueError(f"Formato não suportado: {ext}")

    print(f"[INFO] Arquivo carregado: {filepath}")
    print(f"[INFO] Shape: {df.shape}")

    return df


def get_column(df: pd.DataFrame, internal_name: str, required: bool = True) -> Optional[str]:
    """
    Retorna o nome real da coluna no DataFrame baseado no mapeamento.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada
    internal_name : str
        Nome interno esperado (chave do COLUMN_MAPPING)
    required : bool
        Se True, levanta erro se coluna não existir

    Returns
    -------
    str or None
        Nome real da coluna ou None se não existir
    """
    mapped_name = COLUMN_MAPPING.get(internal_name)

    if mapped_name and mapped_name in df.columns:
        return mapped_name

    # Tenta encontrar variações comuns
    variations = {
        'r2': ['CV_R2', 'CV R²', 'R2', 'R²', 'r2_cv', 'test_r2'],
        'rmse': ['CV_RMSE', 'CV RMSE', 'RMSE', 'rmse_cv', 'test_rmse'],
        'mae': ['CV_MAE', 'CV MAE', 'MAE', 'mae_cv', 'test_mae'],
        'n_features': ['N_Features', 'n_features', 'num_features', 'Num_Features', 'n_vars'],
        'features': ['Features', 'features', 'Feature_Names', 'Variables', 'vars'],
        'group': ['Group', 'group', 'Model_Type', 'Type', 'Tipo'],
    }

    for var in variations.get(internal_name, []):
        if var in df.columns:
            return var

    if required:
        raise KeyError(f"Coluna '{internal_name}' não encontrada. "
                      f"Mapeamento atual: {mapped_name}. "
                      f"Colunas disponíveis: {list(df.columns)}")

    return None


def parse_features(feature_str) -> List[str]:
    """
    Converte string de features em lista.

    Handles:
    - String com vírgulas: "A, B, C"
    - String de lista Python: "['A', 'B', 'C']"
    - Lista já formatada
    - NaN/None
    """
    if pd.isna(feature_str):
        return []

    if isinstance(feature_str, list):
        return feature_str

    feature_str = str(feature_str).strip()

    # Tenta interpretar como literal Python (lista)
    if feature_str.startswith('['):
        try:
            return ast.literal_eval(feature_str)
        except (ValueError, SyntaxError):
            pass

    # Split por vírgula
    features = [f.strip().strip("'\"") for f in feature_str.split(',')]
    return [f for f in features if f]


def is_lidar_feature(feature_name: str) -> bool:
    """Verifica se uma feature é do tipo LiDAR baseado nos padrões."""
    feature_upper = feature_name.upper()
    for pattern in LIDAR_FEATURES_PATTERNS:
        if pattern.upper() in feature_upper:
            return True
    return False


def is_cadastro_feature(feature_name: str) -> bool:
    """Verifica se uma feature é do tipo Cadastro."""
    for cadastro_feat in CADASTRO_FEATURES:
        if cadastro_feat.lower() in feature_name.lower():
            return True
    return False


def infer_group(features: List[str]) -> str:
    """
    Infere o grupo (Cadastro, LiDAR, Cadastro+LiDAR) baseado nas features.

    Parameters
    ----------
    features : List[str]
        Lista de nomes de features

    Returns
    -------
    str
        Nome do grupo inferido
    """
    has_cadastro = False
    has_lidar = False

    for feat in features:
        if is_cadastro_feature(feat):
            has_cadastro = True
        elif is_lidar_feature(feat):
            has_lidar = True
        else:
            # Feature desconhecida - tentar classificar por padrão
            if any(p in feat for p in ['Elev', 'Int', 'P90', 'P95', 'P50']):
                has_lidar = True

    if has_cadastro and has_lidar:
        return 'Cadastro+LiDAR'
    elif has_cadastro:
        return 'Cadastro'
    elif has_lidar:
        return 'LiDAR'
    else:
        # Default: assumir LiDAR se não conseguir classificar
        return 'LiDAR'


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara o DataFrame: renomeia colunas, infere grupos se necessário.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original

    Returns
    -------
    pd.DataFrame
        DataFrame preparado com colunas padronizadas
    """
    df = df.copy()

    # Mapear colunas para nomes padronizados internos
    col_r2 = get_column(df, 'r2')
    col_rmse = get_column(df, 'rmse')
    col_n_features = get_column(df, 'n_features')
    col_features = get_column(df, 'features')
    col_mae = get_column(df, 'mae', required=False)
    col_group = get_column(df, 'group', required=False)

    # Criar DataFrame padronizado
    df_std = pd.DataFrame()
    df_std['R2'] = df[col_r2]
    df_std['RMSE'] = df[col_rmse]
    df_std['N_Features'] = df[col_n_features]
    df_std['Features_Raw'] = df[col_features]

    if col_mae:
        df_std['MAE'] = df[col_mae]

    # Parsear features
    df_std['Features'] = df_std['Features_Raw'].apply(parse_features)

    # Inferir ou usar grupo existente
    if col_group and col_group in df.columns:
        df_std['Group'] = df[col_group]
        print("[INFO] Usando coluna de grupo existente.")
    else:
        print("[INFO] Inferindo grupos a partir das features...")
        df_std['Group'] = df_std['Features'].apply(infer_group)

    # Padronizar nomes de grupos
    group_mapping = {
        'cadastro': 'Cadastro',
        'lidar': 'LiDAR',
        'cadastro+lidar': 'Cadastro+LiDAR',
        'cadastro_lidar': 'Cadastro+LiDAR',
        'mixed': 'Cadastro+LiDAR',
    }
    df_std['Group'] = df_std['Group'].str.strip().str.lower().map(
        lambda x: group_mapping.get(x, x.title() if x else 'Unknown')
    )

    # Garantir que grupos estão corretos
    valid_groups = set(GROUP_ORDER)
    df_std.loc[~df_std['Group'].isin(valid_groups), 'Group'] = 'Unknown'

    return df_std


def save_figure(fig, filename: str, output_dir: str):
    """
    Salva figura em PNG e PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura a salvar
    filename : str
        Nome base do arquivo (sem extensão)
    output_dir : str
        Diretório de saída
    """
    base_path = os.path.join(output_dir, filename)

    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f"{base_path}.pdf", bbox_inches='tight', facecolor='white')

    print(f"  -> Salvo: {filename}.png/.pdf")
    plt.close(fig)


def add_jitter(values: np.ndarray, amount: float = 0.1) -> np.ndarray:
    """Adiciona jitter (ruído) para dispersão de pontos."""
    return values + np.random.normal(0, amount, size=len(values))


# =============================================================================
# FUNÇÕES DE PLOTAGEM
# =============================================================================

def _draw_violin_horizontal(ax, df: pd.DataFrame, metric: str, xlabel: str, show_ylabel: bool = True):
    """
    Desenha violin plot horizontal em um eixo específico (grupos no eixo Y).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eixo onde desenhar
    df : pd.DataFrame
        DataFrame preparado
    metric : str
        Nome da coluna de métrica ('R2' ou 'RMSE')
    xlabel : str
        Label do eixo X
    show_ylabel : bool
        Se True, mostra labels dos grupos no eixo Y
    """
    # Filtrar dados válidos
    df_valid = df.dropna(subset=[metric, 'Group'])

    # Preparar dados por grupo
    data_by_group = []
    positions = []
    labels = []
    colors = []

    for i, group in enumerate(GROUP_ORDER):
        group_data = df_valid[df_valid['Group'] == group][metric].values
        if len(group_data) > 0:
            data_by_group.append(group_data)
            positions.append(i + 1)
            labels.append(group)
            colors.append(GROUP_COLORS.get(group, '#888888'))

    if not data_by_group:
        return

    # Violin plot horizontal (vert=False)
    vp = ax.violinplot(data_by_group, positions=positions, widths=0.7,
                       showmeans=False, showmedians=False, showextrema=False,
                       vert=False)

    # Colorir violinos
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor(colors[i])
        body.set_alpha(0.5)
        body.set_linewidth(1.5)

    # Aplicar estilo articletables
    apply_articletables_style(ax)

    # Formatação - grupos no eixo Y
    ax.set_yticks(positions)
    if show_ylabel:
        # Labels dos grupos: Cadastro, LiDAR, Cadastro + LiDAR
        ax.set_yticklabels(labels)
    else:
        # Remove labels e ticks do eixo Y
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)
        ax.spines['left'].set_visible(False)

    ax.set_xlabel(xlabel)

    # Ajustar limites
    ax.margins(y=0.15)

    ax.xaxis.set_major_formatter(FuncFormatter(pt_br_smart))

    ax.grid(linestyle="--", alpha=0.35)

    # Inverter eixo Y para Cadastro ficar no topo
    ax.invert_yaxis()


def plot_violin_subplot_r2_rmse(df: pd.DataFrame, filename: str, output_dir: str):
    """
    Cria subplot com violin plots horizontais de R² e RMSE lado a lado.
    Grupos no eixo Y (apenas no painel esquerdo). Estilo articletables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    filename : str
        Nome do arquivo de saída
    output_dir : str
        Diretório de saída
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Subplot esquerdo: R² (com labels dos grupos)
    _draw_violin_horizontal(axes[0], df, 'R2', 'Coeficiente de Determinação (R²)', show_ylabel=True)
    axes[0].set_title('(a)', fontsize=12, loc='left')
    axes[0].set_xlim(left=0, right=1)

    # Subplot direito: RMSE (sem labels nem ticks no eixo Y)
    _draw_violin_horizontal(axes[1], df, 'RMSE', 'Raiz do Erro Quadrático Médio (RMSE)', show_ylabel=False)
    axes[1].set_title('(b)', fontsize=12, loc='left')
    axes[1].set_xlim(left=0, right=100)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)


def plot_violin_with_strip(df: pd.DataFrame, metric: str, ylabel: str,
                           filename: str, output_dir: str):
    """
    Cria violin plot com pontos (strip) sobrepostos (figura individual).
    Estilo articletables: sem frame direito/superior, grid horizontal sutil.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    metric : str
        Nome da coluna de métrica ('R2' ou 'RMSE')
    ylabel : str
        Label do eixo Y
    filename : str
        Nome do arquivo de saída
    output_dir : str
        Diretório de saída
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filtrar dados válidos
    df_valid = df.dropna(subset=[metric, 'Group'])

    # Preparar dados por grupo
    data_by_group = []
    positions = []
    labels = []
    colors = []

    for i, group in enumerate(GROUP_ORDER):
        group_data = df_valid[df_valid['Group'] == group][metric].values
        if len(group_data) > 0:
            data_by_group.append(group_data)
            positions.append(i + 1)
            labels.append(group)
            colors.append(GROUP_COLORS.get(group, '#888888'))

    if not data_by_group:
        print(f"  [WARN] Sem dados válidos para {metric}")
        plt.close(fig)
        return

    # Violin plot
    vp = ax.violinplot(data_by_group, positions=positions, widths=0.7,
                       showmeans=True, showmedians=True, showextrema=True)

    # Colorir violinos
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor(colors[i])
        body.set_alpha(0.3)
        body.set_linewidth(1.5)

    # Aplicar estilo articletables
    apply_articletables_style(ax)

    # Formatação
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Grupo de Modelo')
    ax.grid(linestyle="--", alpha=0.20)

    # Ajustar limites
    ax.margins(x=0.15)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)


def _draw_scatter_vertical(ax, df: pd.DataFrame, metric: str, ylabel: str, show_xlabel: bool = True):
    """
    Desenha scatter plot vertical (k no eixo X, métrica no eixo Y).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eixo onde desenhar
    df : pd.DataFrame
        DataFrame preparado
    metric : str
        Nome da coluna de métrica ('R2' ou 'RMSE')
    ylabel : str
        Label do eixo Y
    show_xlabel : bool
        Se True, mostra label do eixo X
    """
    # Filtrar dados válidos
    df_valid = df.dropna(subset=[metric, 'N_Features'])

    # Plotar todos os pontos com uma única cor
    ax.scatter(df_valid['N_Features'], df_valid[metric],
               alpha=0.3, s=15, color='#555555', edgecolors='none')

    # Aplicar estilo articletables
    apply_articletables_style(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(pt_br_smart))
    ax.yaxis.set_major_formatter(FuncFormatter(pt_br_smart))

    # Formatação
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel('Número de Variáveis (k)')

    # Limites
    ax.set_xlim(left=0)
    ax.grid(linestyle="--", alpha=0.35)


def plot_scatter_subplot_r2_rmse(df: pd.DataFrame, filename: str, output_dir: str):
    """
    Cria subplot com scatter plots de R² e RMSE lado a lado.
    k no eixo X compartilhado. Estilo articletables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    filename : str
        Nome do arquivo de saída
    output_dir : str
        Diretório de saída
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Subplot esquerdo: R² (sem label no eixo X)
    _draw_scatter_vertical(axes[0], df, 'R2', 'Coeficiente de Determinação (R²)', show_xlabel=False)
    axes[0].set_title('(a)', fontsize=12, loc='left')
    axes[0].set_ylim(0, 1)

    # Subplot direito: RMSE (sem label no eixo X)
    _draw_scatter_vertical(axes[1], df, 'RMSE', 'Raiz do Erro Quadrático Médio (RMSE)', show_xlabel=False)
    axes[1].set_title('(b)', fontsize=12, loc='left')
    axes[1].set_ylim(0, 100)

    # Label do eixo X centralizado
    fig.text(0.5, 0.02, 'Número de Variáveis (k)', ha='center', fontsize=12)

    # Legenda compartilhada
    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g, alpha=0.6)
               for g in GROUP_ORDER if g in GROUP_COLORS]

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.12)
    save_figure(fig, filename, output_dir)


def plot_envelope_best_k(df: pd.DataFrame, metric: str, ylabel: str,
                         filename: str, output_dir: str, maximize: bool = True):
    """
    Curva envelope: melhor valor da métrica para cada k (número de features).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    metric : str
        Nome da coluna de métrica
    ylabel : str
        Label do eixo Y
    filename : str
        Nome do arquivo de saída
    output_dir : str
        Diretório de saída
    maximize : bool
        Se True, busca máximo; se False, busca mínimo
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Filtrar dados válidos
    df_valid = df.dropna(subset=[metric, 'N_Features', 'Group'])

    # Determinar range de k
    k_min = int(df_valid['N_Features'].min())
    k_max = int(df_valid['N_Features'].max())
    k_range = range(k_min, k_max + 1)

    legend_handles = []

    for group in GROUP_ORDER:
        group_data = df_valid[df_valid['Group'] == group]
        if len(group_data) == 0:
            continue

        best_values = []
        k_values = []

        for k in k_range:
            k_data = group_data[group_data['N_Features'] == k][metric]
            if len(k_data) > 0:
                if maximize:
                    best_values.append(k_data.max())
                else:
                    best_values.append(k_data.min())
                k_values.append(k)

        if len(k_values) > 0:
            color = GROUP_COLORS.get(group, '#888888')
            ax.plot(k_values, best_values, '-o', color=color,
                   linewidth=2, markersize=5, label=group)
            legend_handles.append(mpatches.Patch(color=color, label=group))

    # Aplicar estilo articletables
    apply_articletables_style(ax)

    # Formatação
    ax.set_xlabel('Número de Variáveis (k)')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')

    # Limites
    ax.set_xlim(left=max(0, k_min - 0.5))

    fig.tight_layout()
    save_figure(fig, filename, output_dir)


def plot_ecdf(df: pd.DataFrame, metric: str, xlabel: str,
              filename: str, output_dir: str):
    """
    Curva ECDF (Empirical Cumulative Distribution Function) por grupo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    metric : str
        Nome da coluna de métrica
    xlabel : str
        Label do eixo X
    filename : str
        Nome do arquivo de saída
    output_dir : str
        Diretório de saída
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filtrar dados válidos
    df_valid = df.dropna(subset=[metric, 'Group'])

    for group in GROUP_ORDER:
        group_data = df_valid[df_valid['Group'] == group][metric].values
        if len(group_data) == 0:
            continue

        # Calcular ECDF
        sorted_data = np.sort(group_data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        color = GROUP_COLORS.get(group, '#888888')
        ax.plot(sorted_data, ecdf, '-', color=color, linewidth=2,
               label=f"{group} (n={len(group_data)})")

    # Aplicar estilo articletables
    apply_articletables_style(ax)

    # Formatação
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probabilidade Acumulada')
    ax.legend(loc='best')

    # Limites do eixo y
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)


def plot_heatmap_freq_vars(df: pd.DataFrame, group: str,
                           filename: str, output_dir: str,
                           percentiles: List[int] = [5, 10, 20]):
    """
    Heatmap de frequência de variáveis nos Top p% por R².

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    group : str
        Nome do grupo
    filename : str
        Nome do arquivo de saída
    output_dir : str
        Diretório de saída
    percentiles : List[int]
        Percentis para análise
    """
    # Filtrar dados válidos do grupo
    df_group = df[(df['Group'] == group) & df['R2'].notna()].copy()

    if len(df_group) == 0:
        print(f"  [WARN] Sem dados para grupo {group}")
        return

    # Coletar todas as variáveis únicas
    all_vars = set()
    for features in df_group['Features']:
        all_vars.update(features)

    if len(all_vars) == 0:
        print(f"  [WARN] Sem variáveis identificadas para grupo {group}")
        return

    # Calcular frequências para cada percentil
    freq_data = {}

    for p in percentiles:
        threshold = df_group['R2'].quantile(1 - p/100)
        top_df = df_group[df_group['R2'] >= threshold]
        n_top = len(top_df)

        # Contar ocorrências
        var_counts = {v: 0 for v in all_vars}
        for features in top_df['Features']:
            for feat in features:
                if feat in var_counts:
                    var_counts[feat] += 1

        # Converter para frequência relativa (%)
        freq_data[f'Top {p}%'] = {v: (c / n_top * 100) if n_top > 0 else 0
                                   for v, c in var_counts.items()}

    # Ordenar variáveis por frequência no Top 5% (descendente)
    sorted_vars = sorted(all_vars,
                        key=lambda v: freq_data['Top 5%'].get(v, 0),
                        reverse=True)

    # Limitar a no máximo 30 variáveis mais frequentes
    max_vars = 30
    if len(sorted_vars) > max_vars:
        sorted_vars = sorted_vars[:max_vars]

    # Criar matriz para heatmap
    col_names = [f'Top {p}%' for p in percentiles]
    matrix = np.zeros((len(sorted_vars), len(col_names)))

    for i, var in enumerate(sorted_vars):
        for j, col in enumerate(col_names):
            matrix[i, j] = freq_data[col].get(var, 0)

    # Criar figura
    fig_height = max(6, len(sorted_vars) * 0.35)
    fig, ax = plt.subplots(figsize=(6, fig_height))

    # Criar colormap customizado (branco -> azul)
    cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#f0f0f0', '#a0c4e8', '#2166ac'])

    # Plotar heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    # Adicionar colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label='Frequência (%)')

    # Configurar ticks
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names)
    ax.set_yticks(range(len(sorted_vars)))
    ax.set_yticklabels(sorted_vars, fontsize=9)

    # Adicionar valores nas células
    for i in range(len(sorted_vars)):
        for j in range(len(col_names)):
            val = matrix[i, j]
            text_color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                   fontsize=8, color=text_color)

    # Título e formatação
    ax.set_title(f'Frequência de Variáveis - {group}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Seleção por R²')

    # Remover apenas spines superior e direito para heatmap
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)


def compute_pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcula a fronteira de Pareto (minimização de ambos os eixos).

    Parameters
    ----------
    x : np.ndarray
        Valores do eixo X
    y : np.ndarray
        Valores do eixo Y

    Returns
    -------
    np.ndarray
        Índices dos pontos não-dominados
    """
    n = len(x)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # j domina i se j é melhor ou igual em ambos e estritamente melhor em pelo menos um
                if (x[j] <= x[i] and y[j] <= y[i]) and (x[j] < x[i] or y[j] < y[i]):
                    pareto_mask[i] = False
                    break

    return np.where(pareto_mask)[0]


def plot_pareto_front(df: pd.DataFrame, filename: str, output_dir: str):
    """
    Pareto front: RMSE vs N_Features (minimizar ambos).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    filename : str
        Nome do arquivo de saída
    output_dir : str
        Diretório de saída
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Filtrar dados válidos
    df_valid = df.dropna(subset=['RMSE', 'N_Features', 'Group'])

    for group in GROUP_ORDER:
        group_data = df_valid[df_valid['Group'] == group]
        if len(group_data) == 0:
            continue

        x = group_data['N_Features'].values
        y = group_data['RMSE'].values

        color = GROUP_COLORS.get(group, '#888888')

        # Plotar todos os pontos (alpha baixo)
        ax.scatter(x, y, alpha=0.25, s=20, c=color, edgecolors='none')

        # Calcular e destacar fronteira de Pareto
        pareto_idx = compute_pareto_front(x, y)

        if len(pareto_idx) > 0:
            # Ordenar pontos de Pareto por x para conectar com linha
            pareto_points = list(zip(x[pareto_idx], y[pareto_idx]))
            pareto_points.sort(key=lambda p: p[0])
            px, py = zip(*pareto_points)

            ax.scatter(px, py, s=80, c=color, edgecolors='black',
                      linewidth=1, zorder=5, label=f'{group} (Pareto)')
            ax.plot(px, py, '-', color=color, linewidth=1.5, alpha=0.7, zorder=4)

    # Aplicar estilo articletables
    apply_articletables_style(ax)

    # Formatação
    ax.set_xlabel('Número de Variáveis (k)')
    ax.set_ylabel('RMSE')
    ax.legend(loc='best')

    # Limites
    ax.set_xlim(left=0)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)


def lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.6, num_points: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementação simplificada de LOWESS (Locally Weighted Scatterplot Smoothing).

    Parameters
    ----------
    x : np.ndarray
        Valores do eixo X
    y : np.ndarray
        Valores do eixo Y
    frac : float
        Fração dos dados a usar para cada ajuste local (0 < frac <= 1)
    num_points : int
        Número de pontos na curva suavizada

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays (x_smooth, y_smooth) com a curva suavizada
    """
    # Ordenar por x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    n = len(x_sorted)
    k = int(np.ceil(frac * n))  # Número de vizinhos

    # Pontos onde calcular a suavização
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
    y_smooth = np.zeros(num_points)

    for i, x0 in enumerate(x_smooth):
        # Calcular distâncias
        distances = np.abs(x_sorted - x0)

        # Encontrar os k vizinhos mais próximos
        idx_neighbors = np.argsort(distances)[:k]
        max_dist = distances[idx_neighbors].max()

        if max_dist == 0:
            max_dist = 1e-10

        # Pesos tricúbicos
        u = distances[idx_neighbors] / max_dist
        weights = (1 - u**3)**3
        weights = np.clip(weights, 0, None)

        # Regressão linear ponderada local
        x_local = x_sorted[idx_neighbors]
        y_local = y_sorted[idx_neighbors]

        # Weighted least squares: y = a + b*x
        sum_w = np.sum(weights)
        sum_wx = np.sum(weights * x_local)
        sum_wy = np.sum(weights * y_local)
        sum_wxx = np.sum(weights * x_local**2)
        sum_wxy = np.sum(weights * x_local * y_local)

        denom = sum_w * sum_wxx - sum_wx**2
        if abs(denom) < 1e-10:
            y_smooth[i] = np.average(y_local, weights=weights)
        else:
            b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
            a = (sum_wy - b * sum_wx) / sum_w
            y_smooth[i] = a + b * x0

    return x_smooth, y_smooth


def plot_lowess_complexity_rmse(df: pd.DataFrame, filename: str, output_dir: str,
                                 frac: float = 0.4):
    """
    Gráfico LOWESS: Complexidade (k) vs RMSE por grupo (apenas curvas).
    Layout horizontal: k no eixo Y, RMSE no eixo X.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    filename : str
        Nome do arquivo de saída
    output_dir : str
        Diretório de saída
    frac : float
        Fração dos dados para suavização LOWESS
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Filtrar dados válidos
    df_valid = df.dropna(subset=['RMSE', 'N_Features', 'Group'])

    for group in GROUP_ORDER:
        group_data = df_valid[df_valid['Group'] == group]
        if len(group_data) < 5:  # Mínimo de pontos para LOWESS
            continue

        # k no eixo Y, RMSE no eixo X
        x = group_data['N_Features'].values.astype(float)
        y = group_data['RMSE'].values.astype(float)

        color = GROUP_COLORS.get(group, '#888888')

        # Calcular LOWESS (k vs RMSE) e plotar invertido (RMSE no X, k no Y)
        try:
            k_smooth, rmse_smooth = lowess(x, y, frac=frac, num_points=200)
            # Plotar com RMSE no eixo X e k no eixo Y
            ax.plot(rmse_smooth, k_smooth, '-', color=color, linewidth=2.5,
                   label=f'{group}')
        except Exception as e:
            print(f"  [WARN] LOWESS falhou para {group}: {e}")
            continue

    # Aplicar estilo articletables
    apply_articletables_style(ax)

    # Formatação - RMSE no X, k no Y
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Número de Variáveis (k)')
    ax.legend(loc='best')

    # Limites
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Inverter eixo Y para menor k ficar no topo (como no violin)
    ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, filename, output_dir)


def plot_best_by_group(df: pd.DataFrame, metric: str, ylabel: str,
                       filename: str, output_dir: str, maximize: bool = True):
    """
    Bar plot comparando o melhor modelo de cada grupo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    metric : str
        Nome da coluna de métrica
    ylabel : str
        Label do eixo Y
    filename : str
        Nome do arquivo de saída
    output_dir : str
        Diretório de saída
    maximize : bool
        Se True, busca máximo; se False, busca mínimo
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Filtrar dados válidos
    df_valid = df.dropna(subset=[metric, 'Group'])

    best_values = []
    colors = []
    labels = []

    for group in GROUP_ORDER:
        group_data = df_valid[df_valid['Group'] == group][metric]
        if len(group_data) == 0:
            continue

        if maximize:
            best_val = group_data.max()
        else:
            best_val = group_data.min()

        best_values.append(best_val)
        colors.append(GROUP_COLORS.get(group, '#888888'))
        labels.append(group)

    if not best_values:
        print(f"  [WARN] Sem dados para barplot de {metric}")
        plt.close(fig)
        return

    # Criar barras
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, best_values, color=colors, edgecolor='black', linewidth=0.8)

    # Adicionar valores nas barras
    for bar, val in zip(bars, best_values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    # Aplicar estilo articletables
    apply_articletables_style(ax)

    # Formatação
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Grupo de Modelo')

    # Ajustar limites y para acomodar anotações
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.08)

    fig.tight_layout()
    save_figure(fig, filename, output_dir)


# =============================================================================
# FUNÇÕES DE RESUMO
# =============================================================================

def print_summary(df: pd.DataFrame):
    """
    Imprime resumo dos dados no console.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame preparado
    """
    print("\n" + "=" * 70)
    print("RESUMO DOS DADOS")
    print("=" * 70)

    # Total de combinações
    print(f"\nTotal de combinações: {len(df)}")

    # Por grupo
    print("\nCombinações por grupo:")
    for group in GROUP_ORDER:
        n = len(df[df['Group'] == group])
        print(f"  - {group}: {n}")

    # Filtrar dados válidos
    df_r2 = df.dropna(subset=['R2'])
    df_rmse = df.dropna(subset=['RMSE'])

    # Top 10 por R²
    print("\n" + "-" * 70)
    print("TOP 10 COMBINAÇÕES POR R² (decrescente)")
    print("-" * 70)

    top_r2 = df_r2.nlargest(10, 'R2')[['Group', 'N_Features', 'R2', 'RMSE', 'Features_Raw']]
    for i, (_, row) in enumerate(top_r2.iterrows(), 1):
        features_str = str(row['Features_Raw'])[:60] + '...' if len(str(row['Features_Raw'])) > 60 else str(row['Features_Raw'])
        print(f"{i:2}. [{row['Group']:15}] R²={row['R2']:.4f} | RMSE={row['RMSE']:.4f} | k={int(row['N_Features']):2} | {features_str}")

    # Top 10 por RMSE (menor é melhor)
    print("\n" + "-" * 70)
    print("TOP 10 COMBINAÇÕES POR RMSE (crescente)")
    print("-" * 70)

    top_rmse = df_rmse.nsmallest(10, 'RMSE')[['Group', 'N_Features', 'R2', 'RMSE', 'Features_Raw']]
    for i, (_, row) in enumerate(top_rmse.iterrows(), 1):
        features_str = str(row['Features_Raw'])[:60] + '...' if len(str(row['Features_Raw'])) > 60 else str(row['Features_Raw'])
        print(f"{i:2}. [{row['Group']:15}] RMSE={row['RMSE']:.4f} | R²={row['R2']:.4f} | k={int(row['N_Features']):2} | {features_str}")

    # Melhor por grupo
    print("\n" + "-" * 70)
    print("MELHOR MODELO POR GRUPO")
    print("-" * 70)

    for group in GROUP_ORDER:
        group_data = df[(df['Group'] == group) & df['R2'].notna() & df['RMSE'].notna()]
        if len(group_data) == 0:
            print(f"\n{group}: Sem dados válidos")
            continue

        # Melhor R²
        best_r2_idx = group_data['R2'].idxmax()
        best_r2_row = group_data.loc[best_r2_idx]

        # Melhor RMSE
        best_rmse_idx = group_data['RMSE'].idxmin()
        best_rmse_row = group_data.loc[best_rmse_idx]

        print(f"\n{group}:")
        print(f"  Melhor R²:   {best_r2_row['R2']:.4f} (k={int(best_r2_row['N_Features'])}, RMSE={best_r2_row['RMSE']:.4f})")
        print(f"    Features: {best_r2_row['Features_Raw']}")
        print(f"  Melhor RMSE: {best_rmse_row['RMSE']:.4f} (k={int(best_rmse_row['N_Features'])}, R²={best_rmse_row['R2']:.4f})")
        print(f"    Features: {best_rmse_row['Features_Raw']}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal do script."""

    print("\n" + "=" * 70)
    print("ANÁLISE DE COMBINAÇÕES DE VARIÁVEIS - COMPARAÇÃO DE MODELOS")
    print("=" * 70)

    # Configurar estilo
    setup_thesis_style()

    # Criar diretório de saída
    output_dir = create_output_directory(FIGURES_DIR)
    print(f"[INFO] Figuras serão salvas em: {output_dir}")

    # Carregar dados
    print(f"\n[INFO] Carregando dados de: {RESULTS_FILE}")
    df_raw = load_data(RESULTS_FILE)

    # Preparar dados
    print("\n[INFO] Preparando dados...")
    df = prepare_data(df_raw)

    # Imprimir resumo
    print_summary(df)

    # Gerar gráficos
    print("\n" + "=" * 70)
    print("GERANDO GRÁFICOS")
    print("=" * 70 + "\n")

    # 1. Violin plot R² e RMSE por grupo (subplot lado a lado)
    print("[1/11] Violin Plot R² e RMSE por Grupo (subplot)...")
    plot_violin_subplot_r2_rmse(df, 'Fig01_Violin_R2_RMSE_byGroup', output_dir)

    # 2. Scatter: k vs R² e RMSE (subplot lado a lado)
    print("[2/11] Scatter Complexidade vs R² e RMSE (subplot)...")
    plot_scatter_subplot_r2_rmse(df, 'Fig02_Scatter_Complexity_R2_RMSE', output_dir)

    # 3. Envelope Best R² by k
    print("[3/11] Curva Envelope - Melhor R² por k...")
    plot_envelope_best_k(df, 'R2', 'Melhor R²',
                        'Fig03_Envelope_BestR2_byK', output_dir, maximize=True)

    # 4. Envelope Best RMSE by k
    print("[4/11] Curva Envelope - Melhor RMSE por k...")
    plot_envelope_best_k(df, 'RMSE', 'Melhor RMSE',
                        'Fig04_Envelope_BestRMSE_byK', output_dir, maximize=False)

    # 5. ECDF R²
    print("[5/11] ECDF R²...")
    plot_ecdf(df, 'R2', 'R² (Validação Cruzada)',
             'Fig05_ECDF_R2', output_dir)

    # 6. ECDF RMSE
    print("[6/11] ECDF RMSE...")
    plot_ecdf(df, 'RMSE', 'RMSE (Validação Cruzada)',
             'Fig06_ECDF_RMSE', output_dir)

    # 7. Heatmaps de frequência por grupo
    print("[7/11] Heatmaps de Frequência de Variáveis...")
    for group in GROUP_ORDER:
        group_safe = group.replace('+', '_').replace(' ', '_')
        plot_heatmap_freq_vars(df, group,
                              f'Fig07_Heatmap_FreqVars_TopPct_{group_safe}',
                              output_dir)

    # 8. Pareto Front
    print("[8/11] Fronteira de Pareto (RMSE vs Complexidade)...")
    plot_pareto_front(df, 'Fig08_Pareto_RMSE_vs_Complexity', output_dir)

    # 9. LOWESS: Complexidade vs RMSE
    print("[9/11] LOWESS Complexidade vs RMSE...")
    plot_lowess_complexity_rmse(df, 'Fig09_LOWESS_Complexity_vs_RMSE', output_dir)

    # 10. Bar plot melhor R² por grupo
    print("[10/11] Melhor R² por Grupo...")
    plot_best_by_group(df, 'R2', 'Melhor R²',
                      'Fig10_BestByGroup_R2', output_dir, maximize=True)

    # 11. Bar plot melhor RMSE por grupo
    print("[11/11] Melhor RMSE por Grupo...")
    plot_best_by_group(df, 'RMSE', 'Melhor RMSE',
                      'Fig11_BestByGroup_RMSE', output_dir, maximize=False)

    # Finalização
    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)
    print(f"\nTodos os gráficos foram salvos em:")
    print(f"  {output_dir}")
    print(f"\nFormatos: PNG (300 dpi) e PDF")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
