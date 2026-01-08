"""
07_Validation.py - Validação do Modelo por Talhão

Este script realiza a validação do modelo de predição de volume florestal
comparando os resultados preditos (média por talhão) com os valores de referência (IFPc).

Workflow:
    1. Carregamento dos dados de validação (Excel com resultados por talhão)
    2. Cálculo de métricas estatísticas (R², RMSE, MAE, Bias, MdAPE)
    3. Geração de gráficos diagnósticos:
        - Observado vs Predito
        - Análise de Resíduos
        - Bland-Altman Plot
        - Distribuição por Classe de Idade
        - Análise por Área
        - Mapa de Calor de Erros
    4. Exportação de relatório de validação

Colunas esperadas no arquivo de entrada:
    - TALHAO: Identificador do talhão
    - VTCC_IFPC: Volume observado pelo IFPc (m³/ha)
    - VTCC_PRED: Volume predito pelo modelo (m³/ha)
    - AREA_CADASTRO: Área do cadastro florestal (ha)
    - AREA_RASTER: Área calculada pelo raster/LiDAR (ha)
    - IDADE_MESES: Idade do plantio em meses
    - REGIONAL: Regional (opcional)
    - ROTACAO: Número da rotação (opcional)

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminhos - ALTERE CONFORME NECESSÁRIO
INPUT_FILE = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results\raster_stats_consolidado.xlsx"
SHEET_NAME = "CONSISTIDO"
OUTPUT_DIR = Path(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results")

# Nomes das colunas no arquivo de entrada (adapte conforme seu arquivo)
COL_TALHAO = 'TALHAO'
COL_OBSERVADO = 'Prod_ifpc'        # Valor de referência (IFPc)
COL_PREDITO = 'Mean'          # Valor predito pelo modelo
COL_AREA_CADASTRO = 'Area_ifpc'
COL_AREA_RASTER = 'Area_ha'
COL_IDADE = 'Idade'
COL_REGIONAL = 'REGIONAL'
COL_ROTACAO = 'Regime'

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
COLOR_ACCENT = '#2ca02c'
COLOR_ERROR = '#d62728'


# =============================================================================
# FUNÇÕES DE MÉTRICAS
# =============================================================================

def calculate_validation_metrics(y_obs, y_pred):
    """
    Calcula métricas de validação completas.

    Parameters
    ----------
    y_obs : array-like
        Valores observados (referência IFPc).
    y_pred : array-like
        Valores preditos pelo modelo.

    Returns
    -------
    dict
        Dicionário com métricas calculadas.
    """
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    # Métricas básicas
    n = len(y_obs)
    y_mean = np.mean(y_obs)
    residuals = y_obs - y_pred

    # R²
    r2 = r2_score(y_obs, y_pred)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    rmse_pct = (rmse / y_mean) * 100

    # MAE
    mae = mean_absolute_error(y_obs, y_pred)
    mae_pct = (mae / y_mean) * 100

    # Bias (viés médio)
    bias = np.mean(y_pred - y_obs)
    bias_pct = (bias / y_mean) * 100

    # MdAPE (Median Absolute Percentage Error)
    ape = np.abs((y_obs - y_pred) / y_obs) * 100
    mdape = np.median(ape)

    # MAPE
    mape = np.mean(ape)

    # Correlação de Pearson
    pearson_r, pearson_p = stats.pearsonr(y_obs, y_pred)

    # Correlação de Spearman
    spearman_r, spearman_p = stats.spearmanr(y_obs, y_pred)

    # Índice de concordância de Willmott (d)
    ss_pred = np.sum((y_pred - y_mean) ** 2)
    ss_obs = np.sum((y_obs - y_mean) ** 2)
    willmott_d = 1 - (np.sum(residuals ** 2) /
                      np.sum((np.abs(y_pred - y_mean) + np.abs(y_obs - y_mean)) ** 2))

    # Coeficiente de Eficiência de Nash-Sutcliffe (NSE)
    nse = 1 - (np.sum(residuals ** 2) / np.sum((y_obs - y_mean) ** 2))

    return {
        'N': n,
        'Media_Obs': y_mean,
        'Media_Pred': np.mean(y_pred),
        'R2': r2,
        'RMSE': rmse,
        'RMSE_pct': rmse_pct,
        'MAE': mae,
        'MAE_pct': mae_pct,
        'Bias': bias,
        'Bias_pct': bias_pct,
        'MAPE': mape,
        'MdAPE': mdape,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        'Willmott_d': willmott_d,
        'NSE': nse
    }


# =============================================================================
# FUNÇÕES DE GRÁFICOS
# =============================================================================

def plot_observed_vs_predicted(y_obs, y_pred, metrics, ax=None, title=''):
    """
    Gráfico de dispersão: Observado vs Predito.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_obs, y_pred, alpha=0.6, s=40, c=COLOR_PRIMARY,
               edgecolors='white', linewidth=0.5)

    # Linha 1:1
    lim_min = 0
    lim_max = max(max(y_obs), max(y_pred)) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label='Linha 1:1')

    # Linha de regressão
    z = np.polyfit(y_obs, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(y_obs), max(y_obs), 100)
    ax.plot(x_line, p(x_line), '--', color=COLOR_SECONDARY, linewidth=1.5,
            label=f'Regressão (y = {z[0]:.2f}x + {z[1]:.2f})')

    # Anotação com métricas
    textstr = (f"N = {metrics['N']}\n"
               f"R² = {metrics['R2']:.4f}\n"
               f"RMSE = {metrics['RMSE']:.2f} m³/ha ({metrics['RMSE_pct']:.1f}%)\n"
               f"Bias = {metrics['Bias']:.2f} m³/ha ({metrics['Bias_pct']:.1f}%)")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('VTCC Observado - IFPc (m³/ha)')
    ax.set_ylabel('VTCC Predito - LiDAR (m³/ha)')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.grid(linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    if title:
        ax.set_title(title)

    return ax


def plot_residuals_vs_predicted(y_obs, y_pred, ax=None, relative=True):
    """
    Gráfico de resíduos vs valores preditos.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if relative:
        residuals = ((y_pred - y_obs) / y_obs) * 100
        ylabel = 'Resíduos Relativos (%)'
        ylim = (-100, 100)
    else:
        residuals = y_pred - y_obs
        ylabel = 'Resíduos (m³/ha)'
        ylim = None

    ax.scatter(y_pred, residuals, alpha=0.6, s=40, c=COLOR_PRIMARY,
               edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

    if relative:
        ax.axhline(y=20, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axhline(y=-20, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.fill_between([min(y_pred)*0.95, max(y_pred)*1.05], -20, 20,
                        alpha=0.1, color='green')

    ax.set_xlabel('VTCC Predito (m³/ha)')
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(linestyle='--', alpha=0.3)

    return ax


def plot_residuals_histogram(y_obs, y_pred, ax=None, relative=True):
    """
    Histograma de distribuição dos resíduos.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if relative:
        residuals = ((y_obs - y_pred) / y_obs) * 100
        xlabel = 'Resíduos Relativos (%)'
        bins = np.arange(-100, 105, 10)
    else:
        residuals = y_obs - y_pred
        xlabel = 'Resíduos (m³/ha)'
        bins = 20

    ax.hist(residuals, bins=bins, alpha=0.7, color=COLOR_PRIMARY,
            edgecolor='white', linewidth=0.8, density=True)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.axvline(x=np.mean(residuals), color=COLOR_SECONDARY, linestyle='-',
               linewidth=1.5, label=f'Média: {np.mean(residuals):.1f}')
    ax.axvline(x=np.median(residuals), color=COLOR_ACCENT, linestyle='-.',
               linewidth=1.5, label=f'Mediana: {np.median(residuals):.1f}')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Densidade')
    ax.legend(fontsize=8)
    ax.grid(linestyle='--', alpha=0.3)

    return ax


def plot_bland_altman(y_obs, y_pred, ax=None):
    """
    Gráfico de Bland-Altman (diferença vs média).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    mean_values = (y_obs + y_pred) / 2
    diff_values = y_obs - y_pred

    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)

    ax.scatter(mean_values, diff_values, alpha=0.6, s=40, c=COLOR_PRIMARY,
               edgecolors='white', linewidth=0.5)

    # Linhas de referência
    ax.axhline(y=mean_diff, color='red', linestyle='-', linewidth=1.5,
               label=f'Bias: {mean_diff:.2f}')
    ax.axhline(y=mean_diff + 1.96*std_diff, color='gray', linestyle='--',
               linewidth=1.5, label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
    ax.axhline(y=mean_diff - 1.96*std_diff, color='gray', linestyle='--',
               linewidth=1.5, label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')

    # Área de concordância
    ax.fill_between([min(mean_values)*0.95, max(mean_values)*1.05],
                    mean_diff - 1.96*std_diff, mean_diff + 1.96*std_diff,
                    alpha=0.1, color='gray')

    ax.set_xlabel('Média (Obs + Pred) / 2 (m³/ha)')
    ax.set_ylabel('Diferença (Obs - Pred) (m³/ha)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_title('Bland-Altman Plot')

    return ax


def plot_by_age_class(df, col_obs, col_pred, col_idade, ax=None):
    """
    Boxplot de erro por classe de idade.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    df = df.copy()
    df['erro_pct'] = ((df[col_obs] - df[col_pred]) / df[col_obs]) * 100

    # Criar classes de idade (em anos)
    df['idade_anos'] = df[col_idade] / 12
    bins = [0, 2, 3, 4, 5, 6, 7, 8, 100]
    labels = ['<2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '>8']
    df['classe_idade'] = pd.cut(df['idade_anos'], bins=bins, labels=labels, right=False)

    # Boxplot
    order = [l for l in labels if l in df['classe_idade'].unique()]
    sns.boxplot(data=df, x='classe_idade', y='erro_pct', ax=ax,
                color=COLOR_PRIMARY, order=order)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(y=20, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=-20, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax.set_xlabel('Classe de Idade (anos)')
    ax.set_ylabel('Erro Relativo (%)')
    ax.set_title('Distribuição do Erro por Classe de Idade')
    ax.grid(linestyle='--', alpha=0.3, axis='y')

    return ax


def plot_by_area_scatter(df, col_obs, col_pred, col_area, ax=None, area_type='Cadastro'):
    """
    Gráfico de dispersão do erro por área do talhão.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    df = df.copy()
    df['erro_pct'] = ((df[col_obs] - df[col_pred]) / df[col_obs]) * 100

    scatter = ax.scatter(df[col_area], df['erro_pct'], alpha=0.6, s=40,
                         c=COLOR_PRIMARY, edgecolors='white', linewidth=0.5)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(y=20, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=-20, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    # Adicionar linha de tendência
    z = np.polyfit(df[col_area], df['erro_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col_area].min(), df[col_area].max(), 100)
    ax.plot(x_line, p(x_line), '--', color=COLOR_SECONDARY, linewidth=1.5)

    ax.set_xlabel(f'Área {area_type} (ha)')
    ax.set_ylabel('Erro Relativo (%)')
    ax.set_title(f'Erro vs Área do Talhão ({area_type})')
    ax.grid(linestyle='--', alpha=0.3)

    return ax


def plot_area_comparison(df, col_area_cad, col_area_raster, ax=None):
    """
    Comparação entre área do cadastro e área do raster.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(df[col_area_cad], df[col_area_raster], alpha=0.6, s=40,
               c=COLOR_PRIMARY, edgecolors='white', linewidth=0.5)

    # Linha 1:1
    lim_min = 0
    lim_max = max(df[col_area_cad].max(), df[col_area_raster].max()) * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label='Linha 1:1')

    # Métricas
    r2 = r2_score(df[col_area_cad], df[col_area_raster])
    diff_pct = ((df[col_area_raster] - df[col_area_cad]) / df[col_area_cad] * 100).mean()

    textstr = f"R² = {r2:.4f}\nDif. Média = {diff_pct:.1f}%"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Área Cadastro (ha)')
    ax.set_ylabel('Área Raster (ha)')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('Comparação de Áreas')
    ax.grid(linestyle='--', alpha=0.3)

    return ax


def plot_error_heatmap(df, col_obs, col_pred, col_idade, col_area, ax=None):
    """
    Mapa de calor do erro por idade e área.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    df = df.copy()
    df['erro_abs_pct'] = np.abs((df[col_obs] - df[col_pred]) / df[col_obs]) * 100

    # Criar classes
    df['idade_anos'] = df[col_idade] / 12
    df['classe_idade'] = pd.cut(df['idade_anos'],
                                 bins=[0, 3, 5, 7, 100],
                                 labels=['<3', '3-5', '5-7', '>7'])
    df['classe_area'] = pd.cut(df[col_area],
                                bins=[0, 5, 10, 20, 1000],
                                labels=['<5', '5-10', '10-20', '>20'])

    # Pivot table
    pivot = df.pivot_table(values='erro_abs_pct',
                           index='classe_idade',
                           columns='classe_area',
                           aggfunc='mean')

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'Erro Abs. Médio (%)'})

    ax.set_xlabel('Classe de Área (ha)')
    ax.set_ylabel('Classe de Idade (anos)')
    ax.set_title('Erro Absoluto Médio (%) por Idade e Área')

    return ax


def plot_qq_residuals(y_obs, y_pred, ax=None):
    """
    Q-Q Plot dos resíduos.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    residuals = y_obs - y_pred
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot dos Resíduos')
    ax.grid(linestyle='--', alpha=0.3)

    return ax


def plot_cumulative_distribution(y_obs, y_pred, ax=None):
    """
    Distribuição cumulativa dos erros absolutos.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    error_pct = np.abs((y_obs - y_pred) / y_obs) * 100
    error_sorted = np.sort(error_pct)
    cdf = np.arange(1, len(error_sorted) + 1) / len(error_sorted) * 100

    ax.plot(error_sorted, cdf, color=COLOR_PRIMARY, linewidth=2)
    ax.axvline(x=10, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='10%')
    ax.axvline(x=20, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='20%')
    ax.axvline(x=30, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='30%')

    # Calcular percentuais
    pct_10 = np.sum(error_pct <= 10) / len(error_pct) * 100
    pct_20 = np.sum(error_pct <= 20) / len(error_pct) * 100
    pct_30 = np.sum(error_pct <= 30) / len(error_pct) * 100

    textstr = f"{pct_10:.1f}% < 10%\n{pct_20:.1f}% < 20%\n{pct_30:.1f}% < 30%"
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Erro Absoluto (%)')
    ax.set_ylabel('Frequência Acumulada (%)')
    ax.set_title('Distribuição Cumulativa dos Erros')
    ax.set_xlim(0, max(error_sorted) * 1.05)
    ax.set_ylim(0, 105)
    ax.legend(loc='center right', fontsize=8)
    ax.grid(linestyle='--', alpha=0.3)

    return ax


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_validation(
    input_file=INPUT_FILE,
    worksheet=SHEET_NAME,
    output_dir=OUTPUT_DIR,
    col_talhao=COL_TALHAO,
    col_observado=COL_OBSERVADO,
    col_predito=COL_PREDITO,
    col_area_cadastro=COL_AREA_CADASTRO,
    col_area_raster=COL_AREA_RASTER,
    col_idade=COL_IDADE,
    col_regional=COL_REGIONAL,
    col_rotacao=COL_ROTACAO,
    save_figures=True,
    save_report=True
):
    """
    Pipeline completo de validação por talhão.

    Parameters
    ----------
    input_file : str
        Caminho do arquivo Excel com dados de validação.
    output_dir : Path
        Diretório para salvar os resultados.
    col_* : str
        Nomes das colunas no arquivo de entrada.
    save_figures : bool
        Se True, salva as figuras geradas.
    save_report : bool
        Se True, salva relatório Excel com métricas.

    Returns
    -------
    dict
        Dicionário com métricas e dados de validação.
    """
    print("=" * 70)
    print("VALIDAÇÃO DO MODELO POR TALHÃO")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Criar diretório de saída
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. CARREGAMENTO DOS DADOS
    # -------------------------------------------------------------------------
    print("[1/5] Carregando dados de validação...")
    df = pd.read_excel(input_file, worksheet)

    print(f"  Arquivo: {input_file}")
    print(f"  Talhões: {len(df)}")
    print(f"  Colunas: {df.columns.tolist()}")
    print()

    # Verificar colunas obrigatórias
    required_cols = [col_observado, col_predito]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias não encontradas: {missing_cols}")

    # Extrair dados
    y_obs = df[col_observado].values
    y_pred = df[col_predito].values

    # -------------------------------------------------------------------------
    # 2. CÁLCULO DAS MÉTRICAS
    # -------------------------------------------------------------------------
    print("[2/5] Calculando métricas de validação...")
    metrics = calculate_validation_metrics(y_obs, y_pred)

    print(f"\n  Métricas de Validação:")
    print(f"    N:        {metrics['N']}")
    print(f"    R²:       {metrics['R2']:.4f}")
    print(f"    RMSE:     {metrics['RMSE']:.2f} m³/ha ({metrics['RMSE_pct']:.1f}%)")
    print(f"    MAE:      {metrics['MAE']:.2f} m³/ha ({metrics['MAE_pct']:.1f}%)")
    print(f"    Bias:     {metrics['Bias']:.2f} m³/ha ({metrics['Bias_pct']:.1f}%)")
    print(f"    MdAPE:    {metrics['MdAPE']:.1f}%")
    print(f"    MAPE:     {metrics['MAPE']:.1f}%")
    print(f"    Pearson:  {metrics['Pearson_r']:.4f} (p={metrics['Pearson_p']:.2e})")
    print(f"    Willmott: {metrics['Willmott_d']:.4f}")
    print(f"    NSE:      {metrics['NSE']:.4f}")
    print()

    # -------------------------------------------------------------------------
    # 3. GRÁFICOS DIAGNÓSTICOS PRINCIPAIS
    # -------------------------------------------------------------------------
    print("[3/5] Gerando gráficos diagnósticos principais...")

    # Figura 1: Gráficos principais (2x2)
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))

    plot_observed_vs_predicted(y_obs, y_pred, metrics, ax=axes1[0, 0],
                               title='(a) Observado vs Predito')
    plot_residuals_vs_predicted(y_obs, y_pred, ax=axes1[0, 1], relative=True)
    axes1[0, 1].set_title('(b) Resíduos vs Predito')
    plot_residuals_histogram(y_obs, y_pred, ax=axes1[1, 0], relative=True)
    axes1[1, 0].set_title('(c) Distribuição dos Resíduos')
    plot_bland_altman(y_obs, y_pred, ax=axes1[1, 1])
    axes1[1, 1].set_title('(d) Bland-Altman')

    plt.tight_layout()

    if save_figures:
        fig1.savefig(output_dir / 'Validation_Main_Diagnostics.png',
                     dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'Validation_Main_Diagnostics.png'}")

    plt.show()

    # -------------------------------------------------------------------------
    # 4. GRÁFICOS ADICIONAIS
    # -------------------------------------------------------------------------
    print("[4/5] Gerando gráficos adicionais...")

    # Figura 2: Análises por idade e área
    has_idade = col_idade in df.columns
    has_area_cad = col_area_cadastro in df.columns
    has_area_raster = col_area_raster in df.columns

    n_plots = sum([has_idade, has_area_cad, has_area_raster and has_area_cad,
                   has_idade and has_area_cad])

    if n_plots > 0:
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        axes2 = axes2.flatten()

        plot_idx = 0

        if has_idade:
            plot_by_age_class(df, col_observado, col_predito, col_idade, ax=axes2[plot_idx])
            plot_idx += 1

        if has_area_cad:
            plot_by_area_scatter(df, col_observado, col_predito, col_area_cadastro,
                                 ax=axes2[plot_idx], area_type='Cadastro')
            plot_idx += 1

        if has_area_raster and has_area_cad:
            plot_area_comparison(df, col_area_cadastro, col_area_raster, ax=axes2[plot_idx])
            plot_idx += 1

        if has_idade and has_area_cad:
            plot_error_heatmap(df, col_observado, col_predito, col_idade,
                               col_area_cadastro, ax=axes2[plot_idx])
            plot_idx += 1

        # Remover eixos não utilizados
        for i in range(plot_idx, 4):
            axes2[i].set_visible(False)

        plt.tight_layout()

        if save_figures:
            fig2.savefig(output_dir / 'Validation_Age_Area_Analysis.png',
                         dpi=300, bbox_inches='tight')
            print(f"  Salvo: {output_dir / 'Validation_Age_Area_Analysis.png'}")

        plt.show()

    # Figura 3: Q-Q Plot e Distribuição Cumulativa
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4.5))

    plot_qq_residuals(y_obs, y_pred, ax=axes3[0])
    plot_cumulative_distribution(y_obs, y_pred, ax=axes3[1])

    plt.tight_layout()

    if save_figures:
        fig3.savefig(output_dir / 'Validation_QQ_CDF.png',
                     dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'Validation_QQ_CDF.png'}")

    plt.show()

    # -------------------------------------------------------------------------
    # 5. EXPORTAÇÃO DO RELATÓRIO
    # -------------------------------------------------------------------------
    print("[5/5] Exportando relatório...")

    if save_report:
        report_path = output_dir / 'Validation_Report.xlsx'

        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Métricas gerais
            metrics_df = pd.DataFrame([{
                'Data_Validacao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                **metrics
            }])
            metrics_df.to_excel(writer, sheet_name='Metricas', index=False)

            # Dados por talhão
            df_results = df.copy()
            df_results['Residuo'] = y_obs - y_pred
            df_results['Residuo_pct'] = ((y_obs - y_pred) / y_obs) * 100
            df_results['Erro_Abs_pct'] = np.abs(df_results['Residuo_pct'])
            df_results.to_excel(writer, sheet_name='Dados_Talhao', index=False)

            # Estatísticas por classe de idade (se disponível)
            if has_idade:
                df_age = df_results.copy()
                df_age['idade_anos'] = df_age[col_idade] / 12
                df_age['classe_idade'] = pd.cut(df_age['idade_anos'],
                                                 bins=[0, 2, 3, 4, 5, 6, 7, 8, 100],
                                                 labels=['<2', '2-3', '3-4', '4-5',
                                                        '5-6', '6-7', '7-8', '>8'])
                stats_age = df_age.groupby('classe_idade').agg({
                    col_observado: ['count', 'mean', 'std'],
                    col_predito: ['mean', 'std'],
                    'Residuo_pct': ['mean', 'std', 'median'],
                    'Erro_Abs_pct': ['mean', 'median']
                }).round(2)
                stats_age.columns = ['_'.join(col).strip() for col in stats_age.columns]
                stats_age.to_excel(writer, sheet_name='Stats_por_Idade')

            # Estatísticas por classe de área (se disponível)
            if has_area_cad:
                df_area = df_results.copy()
                df_area['classe_area'] = pd.cut(df_area[col_area_cadastro],
                                                 bins=[0, 5, 10, 20, 50, 1000],
                                                 labels=['<5', '5-10', '10-20',
                                                        '20-50', '>50'])
                stats_area = df_area.groupby('classe_area').agg({
                    col_observado: ['count', 'mean', 'std'],
                    col_predito: ['mean', 'std'],
                    'Residuo_pct': ['mean', 'std', 'median'],
                    'Erro_Abs_pct': ['mean', 'median']
                }).round(2)
                stats_area.columns = ['_'.join(col).strip() for col in stats_area.columns]
                stats_area.to_excel(writer, sheet_name='Stats_por_Area')

        print(f"  Relatório salvo: {report_path}")

    print()
    print("=" * 70)
    print("VALIDAÇÃO CONCLUÍDA")
    print(f"Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return {
        'metrics': metrics,
        'data': df,
        'predictions': {
            'observed': y_obs,
            'predicted': y_pred
        }
    }


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    # Exemplo de uso:
    # Ajuste os caminhos e nomes das colunas conforme seu arquivo

    print("\n" + "=" * 70)
    print("CONFIGURAÇÃO")
    print("=" * 70)
    print(f"Arquivo de entrada: {INPUT_FILE}")
    print(f"Diretório de saída: {OUTPUT_DIR}")
    print()
    print("Colunas esperadas:")
    print(f"  - Talhão:        {COL_TALHAO}")
    print(f"  - VTCC Obs:      {COL_OBSERVADO}")
    print(f"  - VTCC Pred:     {COL_PREDITO}")
    print(f"  - Área Cadastro: {COL_AREA_CADASTRO}")
    print(f"  - Área Raster:   {COL_AREA_RASTER}")
    print(f"  - Idade (meses): {COL_IDADE}")
    print()

    # Verificar se o arquivo existe
    if not Path(INPUT_FILE).exists():
        print(f"AVISO: Arquivo de entrada não encontrado: {INPUT_FILE}")
        print("\nPara executar a validação, crie um arquivo Excel com as colunas:")
        print("  TALHAO, VTCC_IFPC, VTCC_PRED, AREA_CADASTRO, AREA_RASTER, IDADE_MESES")
        print("\nOu ajuste as constantes no início do script para corresponder")
        print("aos nomes das colunas do seu arquivo.")
    else:
        results = run_validation()
