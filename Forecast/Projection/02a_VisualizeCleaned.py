"""
02a_VisualizeCleaned.py - Visualização dos Dados Limpos por Estrato

Este script gera visualizações dos dados limpos (saída do 01_DataConsistency.py)
com scatter plots de Idade vs Variável, coloridos por regime de manejo (Alto fuste
vs Talhadia) e divididos por regional.

Layout: 3 linhas (regionais) x 3 colunas (variáveis)
- Cores por regime: Verde = Alto fuste, Vermelho = Talhadia
- Estilo articletables para publicação acadêmica

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configurar encoding para Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Caminhos dos arquivos de entrada (dados limpos)
INPUT_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\cleaned"
INPUT_FILE_1 = rf"{INPUT_DIR}\cubmean_comp_cleaned.xlsx"
INPUT_FILE_2 = rf"{INPUT_DIR}\p90_comp_cleaned.xlsx"
INPUT_FILE_3 = rf"{INPUT_DIR}\stddev_comp_cleaned.xlsx"

# Nomes das variáveis
VAR1 = "Z Kurt"
VAR2 = "Z P90"
VAR3 = "Z \u03c3"  # Z sigma - usando unicode para evitar problemas de encoding

# Sheet com dados long
SHEET_LONG = "long"

# Diretório de saída
OUTPUT_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cores para visualização por regime (usando ciclo padrão do matplotlib para consistência com ArticleTables)
# As cores serão definidas dinamicamente em setup_thesis_style()
COLORS_REGIME = {}

# Ordem dos regimes para legenda
REGIME_ORDER = ['Alto fuste', 'Talhadia']


def get_regime_colors():
    """Obtém cores do regime usando o ciclo padrão do matplotlib."""
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4', '#ff7f0e'])
    return {
        'Alto fuste': color_cycle[0],
        'Talhadia': color_cycle[1]
    }


# =============================================================================
# FUNÇÕES DE ESTRATIFICAÇÃO
# =============================================================================

def extrair_regional(ref_id):
    """Extrai os primeiros 2 caracteres do REF_ID como regional."""
    if pd.isna(ref_id) or len(str(ref_id)) < 2:
        return 'XX'
    return str(ref_id)[:2].upper()


def extrair_regime(ref_id):
    """
    Extrai o regime de manejo do REF_ID (caracteres 11:13).

    Retorna:
        'Talhadia' se começar com 'R'
        'Alto fuste' caso contrário
    """
    if pd.isna(ref_id) or len(str(ref_id)) < 13:
        return 'Alto fuste'

    codigo = str(ref_id)[11:13].upper()

    if codigo.startswith('R'):
        return 'Talhadia'
    else:
        return 'Alto fuste'


def preparar_dados_estrato(df):
    """Prepara o DataFrame adicionando colunas regional, regime e grupo."""
    df = df.copy()
    df['regional'] = df['REF_ID'].apply(extrair_regional)
    df['regime'] = df['REF_ID'].apply(extrair_regime)
    df['grupo'] = df['regional'] + '_' + df['regime']
    return df


# =============================================================================
# CONFIGURAÇÃO DE ESTILO (articletables)
# =============================================================================

def setup_thesis_style():
    """Configura estilo de gráficos para tese acadêmica."""
    plt.rcParams.update({
        # Fonte
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,

        # Eixos
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
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

        # Grid
        'axes.grid': False,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'grid.color': '#cccccc',
    })


def apply_articletables_style(ax):
    """Aplica estilo articletables a um eixo específico (consistente com ArticleTables.py)."""
    # Remove moldura superior e direita (estilo ArticleTables)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    # Grid mais sutil (alpha=0.20 como no ArticleTables)
    ax.grid(linestyle='--', alpha=0.20)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.8)


def add_panel_label(ax, label):
    """Adiciona label do painel (a.), (b.), etc. no canto superior esquerdo."""
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=11,
        fontweight='bold'
    )


# =============================================================================
# FUNÇÕES DE PLOTAGEM
# =============================================================================

def plot_scatter_by_regional_regime(dados_dict, regionais, output_path):
    """
    Plota scatter plots em grid (variáveis x regionais).

    Layout:
        - Linhas = variáveis (Z Kurt, Z P90, Z σ)
        - Colunas = regionais (GN, NE, RD)
        - Legenda apenas no primeiro gráfico (a.1)

    Args:
        dados_dict: dict com {var_name: df} para cada variável
        regionais: lista de regionais a plotar
        output_path: caminho para salvar a figura
    """
    n_regionais = len(regionais)
    n_vars = len(dados_dict)

    # Obter cores do regime (ciclo padrão matplotlib)
    regime_colors = get_regime_colors()

    # Grid: linhas = variáveis, colunas = regionais
    fig, axes = plt.subplots(n_vars, n_regionais, figsize=(4.5 * n_regionais, 4 * n_vars))

    # Garantir que axes seja 2D
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    if n_regionais == 1:
        axes = axes.reshape(-1, 1)

    var_names = list(dados_dict.keys())

    # Labels de painel: (a.1), (a.2), (a.3), (b.1), (b.2), (b.3), etc.
    panel_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    for i, var_name in enumerate(var_names):
        df = dados_dict[var_name]

        for j, regional in enumerate(regionais):
            ax = axes[i, j]

            # Filtrar por regional
            df_regional = df[df['regional'] == regional]

            # Plotar cada regime com sua cor
            for regime in REGIME_ORDER:
                df_regime = df_regional[df_regional['regime'] == regime]
                if len(df_regime) > 0:
                    cor = regime_colors.get(regime, '#888888')
                    ax.scatter(
                        df_regime['IDADE'],
                        df_regime[var_name],
                        alpha=0.75,
                        s=16,
                        c=cor,
                        edgecolors='none',
                        label=regime
                    )

            # Aplicar estilo
            apply_articletables_style(ax)

            # Adicionar label do painel (a.1), (a.2), etc.
            panel_label = f"({panel_letters[i]}.{j+1})"
            add_panel_label(ax, panel_label)

            # Labels dos eixos
            if i == n_vars - 1:  # Última linha
                ax.set_xlabel('Idade (meses)')
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

            # Ylabel apenas na primeira coluna
            if j == 0:
                ax.set_ylabel(var_name)
            else:
                ax.set_ylabel('')


            # Contagem total no canto inferior direito
            """n_total = len(df_regional)
            ax.text(0.98, 0.02, f'n={n_total}', transform=ax.transAxes,
                   fontsize=9, va='bottom', ha='right', color='gray')"""

            # Legenda apenas no primeiro gráfico (a.1)
            if i == 0 and j == 0:
                ax.legend(loc='lower right', fontsize=9, frameon=True)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.15, hspace=0.08)

    # Salvar
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Figura salva: {output_path}")

    # Salvar também em PDF
    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  Figura salva: {pdf_path}")

    plt.close(fig)


def plot_scatter_all_combined(dados_dict, output_path):
    """
    Plota scatter plots 1x3 (todas as regionais juntas, uma coluna por variável).

    Args:
        dados_dict: dict com {var_name: df} para cada variável
        output_path: caminho para salvar a figura
    """
    n_vars = len(dados_dict)

    # Obter cores do regime (ciclo padrão matplotlib)
    regime_colors = get_regime_colors()

    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 5))

    if n_vars == 1:
        axes = [axes]

    var_names = list(dados_dict.keys())

    for j, var_name in enumerate(var_names):
        ax = axes[j]
        df = dados_dict[var_name]

        # Plotar cada regime com sua cor
        for regime in REGIME_ORDER:
            df_regime = df[df['regime'] == regime]
            if len(df_regime) > 0:
                cor = regime_colors.get(regime, '#888888')
                ax.scatter(
                    df_regime['IDADE'],
                    df_regime[var_name],
                    alpha=0.75,
                    s=16,
                    c=cor,
                    edgecolors='none',
                    label=f'{regime} (n={len(df_regime)})'
                )

        # Aplicar estilo
        apply_articletables_style(ax)

        # Adicionar label do painel (a.), (b.), (c.)
        panel_label = f"({chr(97+j)}.)"
        add_panel_label(ax, panel_label)

        # Labels dos eixos
        ax.set_xlabel('Idade (meses)')
        ax.set_ylabel(var_name)

        # Legenda individual (canto inferior direito, sem frame)
        ax.legend(loc='lower right', fontsize=9, frameon=True)

    plt.tight_layout()

    # Salvar
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Figura salva: {output_path}")

    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  Figura salva: {pdf_path}")

    plt.close(fig)


def plot_histograms_by_regime(dados_dict, output_path):
    """
    Plota histogramas das variáveis separados por regime.

    Args:
        dados_dict: dict com {var_name: df} para cada variável
        output_path: caminho para salvar a figura
    """
    n_vars = len(dados_dict)

    # Obter cores do regime (ciclo padrão matplotlib)
    regime_colors = get_regime_colors()

    fig, axes = plt.subplots(2, n_vars, figsize=(4.5 * n_vars, 8))

    var_names = list(dados_dict.keys())

    # Labels de painel
    panel_idx = 0

    for j, var_name in enumerate(var_names):
        df = dados_dict[var_name]

        for i, regime in enumerate(REGIME_ORDER):
            ax = axes[i, j]
            df_regime = df[df['regime'] == regime]

            if len(df_regime) > 0:
                cor = regime_colors.get(regime, '#888888')

                # Histograma com frequência relativa
                weights = np.ones(len(df_regime)) / len(df_regime) * 100
                ax.hist(df_regime[var_name], bins=30, weights=weights,
                       alpha=0.75, color=cor, edgecolor='white', linewidth=0.5)

            # Aplicar estilo
            apply_articletables_style(ax)

            # Adicionar label do painel (a.), (b.), (c.), etc.
            panel_letter = chr(97 + i * n_vars + j)  # a, b, c, d, e, f...
            add_panel_label(ax, f"({panel_letter}.)")

            # Labels
            if i == 1:  # Última linha
                ax.set_xlabel(var_name)
            else:
                ax.set_xlabel('')
            ax.set_ylabel('Frequência (%)')

            # Título
            if i == 0:
                ax.set_title(f'{var_name}', fontsize=11, fontweight='bold')

            # Info do regime no canto inferior direito
            ax.text(0.98, 0.98, f'{regime}\nn={len(df_regime)}',
                   transform=ax.transAxes, fontsize=9,
                   va='top', ha='right', color='gray')

    plt.tight_layout()

    # Salvar
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Figura salva: {output_path}")

    plt.close(fig)


def print_summary(dados_dict):
    """Imprime resumo dos dados."""
    print("\n" + "=" * 70)
    print("RESUMO DOS DADOS LIMPOS")
    print("=" * 70)

    for var_name, df in dados_dict.items():
        print(f"\n{var_name}:")
        print(f"  Total: {len(df)} registros")
        print(f"  Por Regional: {dict(df['regional'].value_counts())}")
        print(f"  Por Regime: {dict(df['regime'].value_counts())}")
        print(f"  Idade: {df['IDADE'].min():.0f} - {df['IDADE'].max():.0f} meses")
        print(f"  {var_name}: {df[var_name].min():.3f} - {df[var_name].max():.3f}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal do script."""

    print("\n" + "=" * 70)
    print("VISUALIZAÇÃO DOS DADOS LIMPOS POR ESTRATO")
    print("=" * 70)

    # Configurar estilo
    setup_thesis_style()

    # Carregar dados
    print("\n[1/4] Carregando dados limpos...")

    df_var1 = pd.read_excel(INPUT_FILE_1, sheet_name=SHEET_LONG)
    print(f"  {VAR1}: {len(df_var1)} registros")

    df_var2 = pd.read_excel(INPUT_FILE_2, sheet_name=SHEET_LONG)
    print(f"  {VAR2}: {len(df_var2)} registros")

    df_var3 = pd.read_excel(INPUT_FILE_3, sheet_name=SHEET_LONG)
    print(f"  {VAR3}: {len(df_var3)} registros")

    # Preparar dados (adicionar estratos)
    print("\n[2/4] Preparando estratos...")

    df_var1 = preparar_dados_estrato(df_var1)
    df_var2 = preparar_dados_estrato(df_var2)
    df_var3 = preparar_dados_estrato(df_var3)

    # Dicionário com todos os dados
    dados_dict = {
        VAR1: df_var1,
        VAR2: df_var2,
        VAR3: df_var3
    }

    # Imprimir resumo
    print_summary(dados_dict)

    # Identificar regionais únicas (usar a união de todas as variáveis)
    regionais = sorted(set(df_var1['regional'].unique()) |
                       set(df_var2['regional'].unique()) |
                       set(df_var3['regional'].unique()))

    print(f"\n  Regionais encontradas: {regionais}")

    # Gerar gráficos
    print("\n[3/4] Gerando gráficos...")

    # Plot 1: Grid 3x3 (regionais x variáveis)
    print("\n  Scatter plots por regional (grid)...")
    plot_scatter_by_regional_regime(
        dados_dict,
        regionais,
        rf"{OUTPUT_DIR}\scatter_idade_var_by_regional.png"
    )

    # Plot 2: Todas as regionais juntas
    print("\n  Scatter plots combinados (todas as regionais)...")
    plot_scatter_all_combined(
        dados_dict,
        rf"{OUTPUT_DIR}\scatter_idade_var_combined.png"
    )

    # Plot 3: Histogramas por regime
    print("\n  Histogramas por regime...")
    plot_histograms_by_regime(
        dados_dict,
        rf"{OUTPUT_DIR}\histograms_by_regime.png"
    )

    # Finalização
    print("\n[4/4] Concluído!")
    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)
    print(f"\nFiguras salvas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
