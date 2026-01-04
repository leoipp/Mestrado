"""
01_DataConsistency.py - Análise de Consistência de Dados para Modelagem Florestal

Este script realiza a limpeza e validação de dados de inventário florestal combinados
com métricas LiDAR para uso em modelos de predição de volume (VTCC - Volume Total Com Casca).

Etapas de processamento:
    1. Carregamento e visualização exploratória dos dados
    2. Verificação de valores ausentes e duplicados
    3. Remoção de valores negativos (VTCC e Idade)
    4. Filtragem por faixa de idade (36-200 meses)
    5. Filtragem por sobrevivência de fustes (>50%)
    6. Detecção de outliers via IQR (não aplicado) e Z-score
    7. Remoção de valores extremos de volume (<100 m³/ha)
    8. Validação de métricas LiDAR (remoção de ruídos)
    9. Exportação do DataFrame limpo

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminhos dos arquivos
INPUT_FILE = r"G:\PycharmProjects\Mestrado\Data\DataFrames\IFC_LiDAR_Plots_RTK_v02.xlsx"
OUTPUT_FILE = r"G:\PycharmProjects\Mestrado\Data\DataFrames\IFC_LiDAR_Plots_RTK_Cleaned_v02.xlsx"
SHEET_NAME = "DADOS"

# Parâmetros de filtragem
MIN_AGE_MONTHS = 36          # Idade mínima em meses (plantios muito jovens são excluídos)
MAX_AGE_MONTHS = 200         # Idade máxima em meses (plantios muito antigos são excluídos)
MIN_STEM_SURVIVAL = 0.50     # Sobrevivência mínima de fustes (50%)
MIN_VTCC_THRESHOLD = 100     # Volume mínimo aceitável (m³/ha)
ZSCORE_THRESHOLD = 3         # Limiar para detecção de outliers via Z-score
MAX_TREE_HEIGHT = 50         # Altura máxima esperada para eucalipto (m)

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

# Cores padrão para visualização
COLOR_PRIMARY = '#1f77b4'
COLOR_OUTLIER = 'red'


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def create_histogram(ax, data, xlabel, color=COLOR_PRIMARY):
    """
    Cria um histograma padronizado com frequência relativa (%).

    Args:
        ax: Eixo do matplotlib para plotagem
        data: Série de dados para o histograma
        xlabel: Rótulo do eixo X
        color: Cor do histograma
    """
    # weights converte contagem absoluta para frequência relativa (%)
    weights = np.ones_like(data) / len(data) * 100
    ax.hist(data, bins=30, weights=weights, alpha=0.7, color=color,
            edgecolor=color, linewidth=0.8, zorder=2)
    ax.grid(linestyle='--', color='gray', alpha=0.5, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequência (%)')


def create_scatter_with_outliers(df_clean, df_outliers, xlabel, ylabel, outlier_label, xlim=None):
    """
    Cria um scatter plot destacando outliers.

    Args:
        df_clean: DataFrame com dados válidos
        df_outliers: DataFrame com outliers identificados
        xlabel: Rótulo do eixo X
        ylabel: Rótulo do eixo Y
        outlier_label: Descrição dos outliers para a legenda
        xlim: Limite opcional do eixo X (tuple)
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df_clean['Idade (meses)'], df_clean['VTCC(m³/ha)'], alpha=0.7)
    plt.scatter(df_outliers['Idade (meses)'], df_outliers['VTCC(m³/ha)'],
                color=COLOR_OUTLIER, label=outlier_label, alpha=0.7)
    plt.grid(linestyle='--', color='gray', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    plt.legend()
    plt.show()


def print_removal_summary(name, count, removed=True):
    """
    Imprime um resumo padronizado de remoção de dados.

    Args:
        name: Nome do tipo de dado removido
        count: Quantidade de registros
        removed: Se os dados foram removidos (True) ou apenas identificados (False)
    """
    if count > 0:
        action = "removidos" if removed else "identificados"
        print(f"  → {count} registros {action}: {name}")


# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================
#%% Dataframe reader
print("=" * 60)
print("CARREGAMENTO DOS DADOS")
print("=" * 60)
df = pd.ExcelFile(INPUT_FILE).parse(SHEET_NAME)
print(f"Dimensões iniciais: {df.shape[0]} linhas × {df.shape[1]} colunas")

# =============================================================================
# VISUALIZAÇÃO EXPLORATÓRIA INICIAL
# =============================================================================
#%% Scatter plot: VTCC vs Idade e Fustes
print("\n" + "=" * 60)
print("VISUALIZAÇÃO EXPLORATÓRIA")
print("=" * 60)

fig, axis = plt.subplots(1, 2, figsize=(10, 6))

# Scatter plot: VTCC vs Idade
axis[0].scatter(df['Idade (meses)'], df['VTCC(m³/ha)'], alpha=0.7)
axis[0].grid(linestyle='--', color='gray', alpha=0.5)
axis[0].set_xlabel('Idade (meses)')
axis[0].set_xlim(0, MAX_AGE_MONTHS)
axis[0].set_ylabel('Volume total com casca (m³/ha)')

# Scatter plot: VTCC vs Fustes
axis[1].scatter(df['Fustes (n)'], df['VTCC(m³/ha)'], alpha=0.7)
axis[1].grid(linestyle='--', color='gray', alpha=0.5)
axis[1].set_xlabel('Fustes (n)')
axis[1].set_xlim(0, 60)
axis[1].set_ylabel('Volume total com casca (m³/ha)')

plt.tight_layout()
plt.show()

#%% Histogramas das variáveis principais
fig, axis = plt.subplots(1, 4, figsize=(14, 6))

create_histogram(axis[0], df['VTCC(m³/ha)'], 'Volume total com casca (m³/ha)')
create_histogram(axis[1], df['Idade (meses)'], 'Idade (meses)')
create_histogram(axis[2], df['Fustes (n)'], 'Fustes (n)')
create_histogram(axis[3], df['VTCC(m³/Fuste)'], 'Volume médio individual (m³/n)')

plt.tight_layout()
plt.show()

# =============================================================================
# VERIFICAÇÃO DE QUALIDADE DOS DADOS
# =============================================================================
#%% Verificação de valores ausentes
print("\n" + "=" * 60)
print("VERIFICAÇÃO DE QUALIDADE")
print("=" * 60)

missing_values = df.isnull().sum()
missing_with_values = missing_values[missing_values > 0]
if len(missing_with_values) > 0:
    print("\nValores ausentes por coluna:")
    print(missing_with_values)
else:
    print("\n✓ Nenhum valor ausente encontrado")

#%% Verificação e remoção de duplicatas
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✓ {duplicates} linhas duplicadas removidas")
else:
    print("✓ Nenhuma linha duplicada encontrada")

# =============================================================================
# REMOÇÃO DE VALORES INVÁLIDOS
# =============================================================================
#%% Verificação de valores negativos em VTCC
print("\n" + "=" * 60)
print("REMOÇÃO DE VALORES INVÁLIDOS")
print("=" * 60)

negative_vtcc = df[df['VTCC(m³/ha)'] < 0]
if not negative_vtcc.empty:
    df = df[df['VTCC(m³/ha)'] >= 0]
    print(f"✓ {len(negative_vtcc)} registros com VTCC negativo removidos")

#%% Verificação de valores negativos em Idade
negative_age = df[df['Idade (meses)'] < 0]
if not negative_age.empty:
    df = df[df['Idade (meses)'] >= 0]
    print(f"✓ {len(negative_age)} registros com idade negativa removidos")

# =============================================================================
# FILTRAGEM POR CRITÉRIOS DE NEGÓCIO
# =============================================================================
#%% Filtragem por faixa de idade (plantios fora do ciclo produtivo típico)
print("\n" + "=" * 60)
print("FILTRAGEM POR CRITÉRIOS DE NEGÓCIO")
print("=" * 60)

age_outliers = df[(df['Idade (meses)'] < MIN_AGE_MONTHS) | (df['Idade (meses)'] > MAX_AGE_MONTHS)]
if not age_outliers.empty:
    df = df[(df['Idade (meses)'] >= MIN_AGE_MONTHS) & (df['Idade (meses)'] <= MAX_AGE_MONTHS)]
    print(f"✓ {len(age_outliers)} registros removidos: idade fora da faixa {MIN_AGE_MONTHS}-{MAX_AGE_MONTHS} meses")

# Visualização dos outliers de idade
create_scatter_with_outliers(
    df, age_outliers,
    xlabel='Idade (meses)',
    ylabel='Volume total com casca (m³/ha)',
    outlier_label=f'Outliers (idade): idade < {MIN_AGE_MONTHS} | idade > {MAX_AGE_MONTHS}',
    xlim=(0, 500)
)

#%% Filtragem por sobrevivência de fustes
# Cálculo da área ocupada por árvore baseado no espaçamento
dim_split = df['ESPACAMENTO'].str.replace(',', '.').str.split('x', expand=True)
df['ESP_AB'] = dim_split[0].astype(float) * dim_split[1].astype(float)

# Cálculo do percentual de sobrevivência
# Fustes esperados = Área corrigida / Área por árvore
df['Fustes (%)'] = df['Fustes (n)'] / (df['Área.corrigida (m²)'] / df['ESP_AB'])

stem_outliers = df[df['Fustes (%)'] < MIN_STEM_SURVIVAL]
if not stem_outliers.empty:
    df = df[df['Fustes (%)'] >= MIN_STEM_SURVIVAL]
    print(f"✓ {len(stem_outliers)} registros removidos: sobrevivência < {MIN_STEM_SURVIVAL*100:.0f}%")

# Visualização dos outliers de fustes
create_scatter_with_outliers(
    df, stem_outliers,
    xlabel='Idade (meses)',
    ylabel='Volume total com casca (m³/ha)',
    outlier_label=f'Outliers (fustes): sobrevivência < {MIN_STEM_SURVIVAL*100:.0f}%'
)


#%% Detecção e remoção de outliers via Z-score (MÉTODO ADOTADO)
# O Z-score identifica valores extremos que se desviam significativamente
# da média (> 3 desvios padrão), sendo mais adequado para dados florestais
z_scores = stats.zscore(df['VTCC(m³/ha)'])
abs_z_scores = np.abs(z_scores)

outliers_zscore = df[abs_z_scores > ZSCORE_THRESHOLD]
print(f"\nMétodo Z-score (threshold={ZSCORE_THRESHOLD}):")
print(f"  {len(outliers_zscore)} outliers identificados e removidos")

if not outliers_zscore.empty:
    df = df[~df.index.isin(outliers_zscore.index)]

# Visualização dos outliers Z-score
create_scatter_with_outliers(
    df, outliers_zscore,
    xlabel='Idade (meses)',
    ylabel='Volume total com casca (m³/ha)',
    outlier_label=f'Outliers Z-score (|z| > {ZSCORE_THRESHOLD})'
)

#%% Remoção de valores extremamente baixos (não detectados pelo Z-score)
# Volumes muito baixos (<100 m³/ha) em plantios maduros indicam problemas
# como falhas severas no plantio, pragas, ou erros de medição
extreme_lower = df[df['VTCC(m³/ha)'] < MIN_VTCC_THRESHOLD]
print(f"\nValores extremamente baixos (VTCC < {MIN_VTCC_THRESHOLD} m³/ha):")
print(f"  {len(extreme_lower)} registros identificados e removidos")

if not extreme_lower.empty:
    df = df[~df.index.isin(extreme_lower.index)]

# Visualização dos valores extremamente baixos
create_scatter_with_outliers(
    df, extreme_lower,
    xlabel='Idade (meses)',
    ylabel='Volume total com casca (m³/ha)',
    outlier_label=f'Outliers estruturais: VTCC < {MIN_VTCC_THRESHOLD} m³/ha'
)

# =============================================================================
# VALIDAÇÃO DE MÉTRICAS LIDAR
# =============================================================================
#%% Verificação de inconsistências nas métricas LiDAR
# Critérios de validação:
# - Elev P90 <= 0: Percentil 90 de altura não pode ser zero ou negativo
# - Elev variance < 0: Variância não pode ser negativa (erro matemático)
# - Elev CURT mean CUBE < 0: Métrica derivada não pode ser negativa
# - Elev maximum > MAX_TREE_HEIGHT: Altura máxima excede limite biológico
print("\n" + "=" * 60)
print("VALIDAÇÃO DE MÉTRICAS LIDAR")
print("=" * 60)

lidar_inconsistencies = df[
    (df['Elev P90'] <= 0) |
    (df['Elev variance'] < 0) |
    (df['Elev CURT mean CUBE'] < 0) |
    (df['Elev maximum'] > MAX_TREE_HEIGHT)
]

print(f"\nInconsistências LiDAR detectadas:")
print(f"  {len(lidar_inconsistencies)} registros com ruídos ou valores impossíveis")

if not lidar_inconsistencies.empty:
    df = df[~df.index.isin(lidar_inconsistencies.index)]
    print("  ✓ Registros inconsistentes removidos")

# Visualização das inconsistências LiDAR
fig, axis = plt.subplots(1, 2, figsize=(10, 6))

# Gráfico 1: Altura máxima vs Percentil 99
axis[0].scatter(df['Elev maximum'], df['Elev P99'], alpha=0.7, label='Dados válidos')
if not lidar_inconsistencies.empty:
    axis[0].scatter(lidar_inconsistencies['Elev maximum'], lidar_inconsistencies['Elev P99'],
                    color=COLOR_OUTLIER, label='Ruídos LiDAR', alpha=0.7)
axis[0].grid(linestyle='--', color='gray', alpha=0.5)
axis[0].set_xlabel('Altura máxima (m)')
axis[0].set_ylabel('Altura percentil 99 (m)')
axis[0].legend()

# Gráfico 2: Altura máxima vs Percentil 50
axis[1].scatter(df['Elev maximum'], df['Elev P50'], alpha=0.7, label='Dados válidos')
if not lidar_inconsistencies.empty:
    axis[1].scatter(lidar_inconsistencies['Elev maximum'], lidar_inconsistencies['Elev P50'],
                    color=COLOR_OUTLIER, label='Ruídos LiDAR', alpha=0.7)
axis[1].grid(linestyle='--', color='gray', alpha=0.5)
axis[1].set_xlabel('Altura máxima (m)')
axis[1].set_ylabel('Altura percentil 50 (m)')
axis[1].legend()

plt.tight_layout()
plt.show()

# =============================================================================
# EXPORTAÇÃO DO DATAFRAME LIMPO
# =============================================================================
#%% Resumo final e exportação
print("\n" + "=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f"\nDimensões finais: {df.shape[0]} linhas × {df.shape[1]} colunas")

# Exportação para Excel
df.to_excel(OUTPUT_FILE, index=False)
print(f"\n✓ DataFrame limpo exportado para:\n  {OUTPUT_FILE}")
