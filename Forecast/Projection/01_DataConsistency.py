"""
01_DataConsistency.py - Análise de Consistência de Dados para Modelagem Florestal

Este script realiza a limpeza e validação de dados de métricas LiDAR para uso em modelos de projeção.
Trabalha com 3 dataframes distintos, cada um contendo uma métrica LiDAR diferente.

Etapas de processamento:
    1. Carregamento dos 3 dataframes (Z Kurt, Z P90, Z σ)
    2. Visualização exploratória inicial
    3. Verificação de valores ausentes e duplicados
    4. Remoção de valores negativos
    5. Filtragem por faixa de idade
    6. Detecção de outliers via Z-score
    7. Exportação dos DataFrames limpos

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import os
import numpy as np
import pandas as pd

# Forçar backend interativo ANTES de importar pyplot
import matplotlib
# Tentar backends interativos em ordem de preferência
for backend in ['TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg']:
    try:
        matplotlib.use(backend)
        print(f"[INFO] Usando backend: {backend}")
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from scipy import stats

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminhos dos arquivos de entrada
INPUT_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\stack_por_variavel_union"

INPUT_FILE_1 = rf"{INPUT_DIR}\cubmean_comp.xlsx"
INPUT_FILE_2 = rf"{INPUT_DIR}\p90_comp.xlsx"
INPUT_FILE_3 = rf"{INPUT_DIR}\stddev_comp.xlsx"

# Nomes das variáveis em cada arquivo
VAR1 = "Z Kurt"
VAR2 = "Z P90"
VAR3 = "Z σ"

# Sheet name (todos usam a mesma)
SHEET_NAME = "long"

# Caminhos dos arquivos de saída
OUTPUT_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\cleaned"
OUTPUT_FILE_1 = rf"{OUTPUT_DIR}\cubmean_comp_cleaned.xlsx"
OUTPUT_FILE_2 = rf"{OUTPUT_DIR}\p90_comp_cleaned.xlsx"
OUTPUT_FILE_3 = rf"{OUTPUT_DIR}\stddev_comp_cleaned.xlsx"

# Parâmetros de filtragem
MIN_AGE_MONTHS = 36          # Idade mínima em meses
MAX_AGE_MONTHS = 120        # Idade máxima em meses
ZSCORE_THRESHOLD = 3         # Limiar para detecção de outliers via Z-score

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
def lasso_filter_dataframe(df, x_col, y_col, title):
    """
    Permite selecionar pontos com lasso e removê-los do DataFrame.
    - Desenhe com o mouse para selecionar pontos (pode fazer múltiplas seleções)
    - Pressione ENTER para confirmar e fechar
    - Pressione ESC para cancelar (não remove nada)

    Retorna df_filtrado e df_removidos.
    """
    # Resetar índice para garantir alinhamento
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    points_xy = np.column_stack((df[x_col].values, df[y_col].values))

    # Scatter plot inicial
    scatter = ax.scatter(points_xy[:, 0], points_xy[:, 1], s=12, alpha=0.6, c='blue')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{title}\n[Desenhe para selecionar | ENTER=confirmar | ESC=cancelar]")
    ax.grid(ls="--", alpha=0.5)

    # Usar set para acumular índices selecionados (múltiplas seleções)
    selected_indices = set()
    highlight_scatter = None

    def update_highlight():
        """Atualiza a visualização dos pontos selecionados."""
        nonlocal highlight_scatter
        if highlight_scatter is not None:
            highlight_scatter.remove()
            highlight_scatter = None

        if selected_indices:
            sel_list = list(selected_indices)
            highlight_scatter = ax.scatter(
                df.iloc[sel_list][x_col],
                df.iloc[sel_list][y_col],
                facecolors='none',
                edgecolors='red',
                s=50,
                linewidths=1.5,
                label=f'Selecionados: {len(selected_indices)}'
            )
            ax.legend(loc='upper right')
        fig.canvas.draw_idle()

    def onselect(verts):
        """Callback chamado quando o lasso é completado."""
        if len(verts) < 3:
            return
        path = Path(verts)
        mask = path.contains_points(points_xy)
        new_selected = set(np.where(mask)[0])
        selected_indices.update(new_selected)  # Acumula seleções
        update_highlight()
        print(f"  Selecionados até agora: {len(selected_indices)} pontos")

    def on_key(event):
        """Callback para teclas."""
        if event.key == 'enter':
            plt.close(fig)
        elif event.key == 'escape':
            selected_indices.clear()
            plt.close(fig)

    # Criar o LassoSelector e manter referência
    lasso = LassoSelector(ax, onselect, useblit=True)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Manter referência ao lasso (evita garbage collection)
    fig._lasso = lasso

    plt.tight_layout()
    plt.show(block=True)

    # Processar resultado
    if not selected_indices:
        print("Nenhum ponto selecionado.")
        return df, pd.DataFrame()

    sel_list = list(selected_indices)
    df_removed = df.iloc[sel_list].copy()
    df_cleaned = df.drop(index=sel_list).reset_index(drop=True)

    print(f"Pontos removidos manualmente: {len(df_removed)}")

    return df_cleaned, df_removed

def create_histogram(ax, data, xlabel, color=COLOR_PRIMARY):
    """
    Cria um histograma padronizado com frequência relativa (%).
    """
    weights = np.ones_like(data) / len(data) * 100
    ax.hist(data, bins=30, weights=weights, alpha=0.7, color=color,
            edgecolor=color, linewidth=0.8, zorder=2)
    ax.grid(linestyle='--', color='gray', alpha=0.5, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequência (%)')


def create_scatter_with_outliers(df_clean, df_outliers, var_name, outlier_label, xlim=None):
    """
    Cria um scatter plot destacando outliers.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df_clean['IDADE'], df_clean[var_name], alpha=0.7, label='Dados válidos')
    if not df_outliers.empty:
        plt.scatter(df_outliers['IDADE'], df_outliers[var_name],
                    color=COLOR_OUTLIER, label=outlier_label, alpha=0.7)
    plt.grid(linestyle='--', color='gray', alpha=0.5)
    plt.xlabel('Idade (meses)')
    plt.ylabel(var_name)
    if xlim:
        plt.xlim(xlim)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_section_header(title):
    """Imprime cabeçalho de seção formatado."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def apply_consistency_filters(df, var_name, df_name):
    """
    Aplica filtros de consistência em um dataframe.

    Args:
        df: DataFrame a ser filtrado
        var_name: Nome da coluna da variável principal
        df_name: Nome do dataframe para logs

    Returns:
        DataFrame filtrado
    """
    initial_count = len(df)
    print(f"\n--- Processando: {df_name} ({var_name}) ---")
    print(f"Registros iniciais: {initial_count}")

    # 1. Remover valores ausentes na variável principal e IDADE
    df = df.dropna(subset=[var_name, 'IDADE'])
    removed = initial_count - len(df)
    if removed > 0:
        print(f"  → {removed} registros removidos: valores ausentes")

    # 2. Remover valores negativos na variável principal (se aplicável)
    if var_name in ['Z P90', 'Z σ']:  # Z Kurt pode ser negativo (curtose)
        negative_mask = df[var_name] < 0
        negative_count = negative_mask.sum()
        if negative_count > 0:
            df = df[~negative_mask]
            print(f"  → {negative_count} registros removidos: {var_name} negativo")

    # 3. Remover valores com idade negativa ou zero
    invalid_age = df[df['IDADE'] <= 0]
    if not invalid_age.empty:
        df = df[df['IDADE'] > 0]
        print(f"  → {len(invalid_age)} registros removidos: idade inválida (<=0)")

    # 4. Filtrar por faixa de idade
    age_outliers = df[(df['IDADE'] < MIN_AGE_MONTHS) | (df['IDADE'] > MAX_AGE_MONTHS)]
    if not age_outliers.empty:
        df = df[(df['IDADE'] >= MIN_AGE_MONTHS) & (df['IDADE'] <= MAX_AGE_MONTHS)]
        print(f"  → {len(age_outliers)} registros removidos: idade fora da faixa [{MIN_AGE_MONTHS}, {MAX_AGE_MONTHS}] meses")

    # 5. Detecção e remoção de outliers via Z-score na variável principal
    z_scores = np.abs(stats.zscore(df[var_name].dropna()))
    outliers_mask = z_scores > ZSCORE_THRESHOLD
    outliers_count = outliers_mask.sum()
    if outliers_count > 0:
        df = df[~outliers_mask]
        print(f"  → {outliers_count} registros removidos: outliers Z-score (|z| > {ZSCORE_THRESHOLD})")

    final_count = len(df)
    total_removed = initial_count - final_count
    print(f"Registros finais: {final_count} ({total_removed} removidos, {100*total_removed/initial_count:.1f}%)")

    return df


# =============================================================================
# CRIAÇÃO DO DIRETÓRIO DE SAÍDA
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================
#%% Dataframe reader
print_section_header("CARREGAMENTO DOS DADOS")

print(f"\nCarregando {VAR1} de: {INPUT_FILE_1}")
df_var1 = pd.read_excel(INPUT_FILE_1, sheet_name=SHEET_NAME)
print(f"  Dimensões: {df_var1.shape[0]} linhas × {df_var1.shape[1]} colunas")

print(f"\nCarregando {VAR2} de: {INPUT_FILE_2}")
df_var2 = pd.read_excel(INPUT_FILE_2, sheet_name=SHEET_NAME)
print(f"  Dimensões: {df_var2.shape[0]} linhas × {df_var2.shape[1]} colunas")

print(f"\nCarregando {VAR3} de: {INPUT_FILE_3}")
df_var3 = pd.read_excel(INPUT_FILE_3, sheet_name=SHEET_NAME)
print(f"  Dimensões: {df_var3.shape[0]} linhas × {df_var3.shape[1]} colunas")


# =============================================================================
# VISUALIZAÇÃO EXPLORATÓRIA INICIAL
# =============================================================================
#%% Scatter plots: Variável vs Idade
print_section_header("VISUALIZAÇÃO EXPLORATÓRIA")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Scatter plot: VAR1 vs Idade
axes[0].scatter(df_var1['IDADE'], df_var1[VAR1], alpha=0.5, s=10)
axes[0].grid(linestyle='--', color='gray', alpha=0.5)
axes[0].set_xlabel('Idade (meses)')
axes[0].set_ylabel(VAR1)
axes[0].set_title(f'{VAR1} vs Idade')

# Scatter plot: VAR2 vs Idade
axes[1].scatter(df_var2['IDADE'], df_var2[VAR2], alpha=0.5, s=10)
axes[1].grid(linestyle='--', color='gray', alpha=0.5)
axes[1].set_xlabel('Idade (meses)')
axes[1].set_ylabel(VAR2)
axes[1].set_title(f'{VAR2} vs Idade')

# Scatter plot: VAR3 vs Idade
axes[2].scatter(df_var3['IDADE'], df_var3[VAR3], alpha=0.5, s=10)
axes[2].grid(linestyle='--', color='gray', alpha=0.5)
axes[2].set_xlabel('Idade (meses)')
axes[2].set_ylabel(VAR3)
axes[2].set_title(f'{VAR3} vs Idade')

plt.tight_layout()
plt.savefig(rf"{OUTPUT_DIR}\scatter_exploratorio_inicial.png", dpi=300, bbox_inches='tight')
plt.show()

#%% Histogramas das variáveis principais
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

create_histogram(axes[0], df_var1[VAR1].dropna(), VAR1)
axes[0].set_title(f'Distribuição de {VAR1}')

create_histogram(axes[1], df_var2[VAR2].dropna(), VAR2)
axes[1].set_title(f'Distribuição de {VAR2}')

create_histogram(axes[2], df_var3[VAR3].dropna(), VAR3)
axes[2].set_title(f'Distribuição de {VAR3}')

plt.tight_layout()
plt.savefig(rf"{OUTPUT_DIR}\histogramas_inicial.png", dpi=300, bbox_inches='tight')
plt.show()


# =============================================================================
# VERIFICAÇÃO DE QUALIDADE DOS DADOS
# =============================================================================
#%% Verificação de valores ausentes
print_section_header("VERIFICAÇÃO DE QUALIDADE")

for df, var, name in [(df_var1, VAR1, "df_var1"), (df_var2, VAR2, "df_var2"), (df_var3, VAR3, "df_var3")]:
    print(f"\n{name} ({var}):")
    missing = df[[var, 'IDADE']].isnull().sum()
    print(f"  Valores ausentes em {var}: {missing[var]}")
    print(f"  Valores ausentes em IDADE: {missing['IDADE']}")

    # Estatísticas descritivas
    print(f"  Estatísticas de {var}:")
    print(f"    Min: {df[var].min():.3f}")
    print(f"    Max: {df[var].max():.3f}")
    print(f"    Média: {df[var].mean():.3f}")
    print(f"    Mediana: {df[var].median():.3f}")
    print(f"    Desvio Padrão: {df[var].std():.3f}")


# =============================================================================
# APLICAÇÃO DOS FILTROS DE CONSISTÊNCIA
# =============================================================================
#%% Aplicar filtros
print_section_header("APLICAÇÃO DOS FILTROS DE CONSISTÊNCIA")

df_var1_clean = apply_consistency_filters(df_var1.copy(), VAR1, "cubmean_comp")
df_var2_clean = apply_consistency_filters(df_var2.copy(), VAR2, "p90_comp")
df_var3_clean = apply_consistency_filters(df_var3.copy(), VAR3, "stddev_comp")


# =============================================================================
# VISUALIZAÇÃO PÓS-FILTRAGEM
# =============================================================================
#%% Scatter plots pós-filtragem
print_section_header("VISUALIZAÇÃO PÓS-FILTRAGEM")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].scatter(df_var1_clean['IDADE'], df_var1_clean[VAR1], alpha=0.5, s=10)
axes[0].grid(linestyle='--', color='gray', alpha=0.5)
axes[0].set_xlabel('Idade (meses)')
axes[0].set_ylabel(VAR1)
axes[0].set_title(f'{VAR1} vs Idade (Filtrado)')

axes[1].scatter(df_var2_clean['IDADE'], df_var2_clean[VAR2], alpha=0.5, s=10)
axes[1].grid(linestyle='--', color='gray', alpha=0.5)
axes[1].set_xlabel('Idade (meses)')
axes[1].set_ylabel(VAR2)
axes[1].set_title(f'{VAR2} vs Idade (Filtrado)')

axes[2].scatter(df_var3_clean['IDADE'], df_var3_clean[VAR3], alpha=0.5, s=10)
axes[2].grid(linestyle='--', color='gray', alpha=0.5)
axes[2].set_xlabel('Idade (meses)')
axes[2].set_ylabel(VAR3)
axes[2].set_title(f'{VAR3} vs Idade (Filtrado)')

plt.tight_layout()
plt.savefig(rf"{OUTPUT_DIR}\scatter_pos_filtragem.png", dpi=300, bbox_inches='tight')
plt.show()

#%% Histogramas pós-filtragem
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

create_histogram(axes[0], df_var1_clean[VAR1].dropna(), VAR1)
axes[0].set_title(f'Distribuição de {VAR1} (Filtrado)')

create_histogram(axes[1], df_var2_clean[VAR2].dropna(), VAR2)
axes[1].set_title(f'Distribuição de {VAR2} (Filtrado)')

create_histogram(axes[2], df_var3_clean[VAR3].dropna(), VAR3)
axes[2].set_title(f'Distribuição de {VAR3} (Filtrado)')

plt.tight_layout()
plt.savefig(rf"{OUTPUT_DIR}\histogramas_pos_filtragem.png", dpi=300, bbox_inches='tight')
plt.show()



# =============================================================================
# SELEÇÃO MANUAL DE OUTLIERS (LASSO)
# =============================================================================
#%% Seleção manual com lasso para cada variável
print_section_header("SELEÇÃO MANUAL DE OUTLIERS")

# Desativar modo interativo para garantir que plt.show() bloqueie
plt.ioff()

print("\n>>> Selecione outliers para Z Kurt (VAR1)")
print("    Desenhe com o mouse | ENTER=confirmar | ESC=cancelar")
df_var1_final, df_var1_removed = lasso_filter_dataframe(
    df_var1_clean,
    x_col="IDADE",
    y_col=VAR1,
    title="Seleção manual de outliers – Z Kurt"
)

print("\n>>> Selecione outliers para Z P90 (VAR2)")
print("    Desenhe com o mouse | ENTER=confirmar | ESC=cancelar")
df_var2_final, df_var2_removed = lasso_filter_dataframe(
    df_var2_clean,
    x_col="IDADE",
    y_col=VAR2,
    title="Seleção manual de outliers – Z P90"
)

print("\n>>> Selecione outliers para Z σ (VAR3)")
print("    Desenhe com o mouse | ENTER=confirmar | ESC=cancelar")
df_var3_final, df_var3_removed = lasso_filter_dataframe(
    df_var3_clean,
    x_col="IDADE",
    y_col=VAR3,
    title="Seleção manual de outliers – Z σ"
)


# =============================================================================
# VISUALIZAÇÃO FINAL (PÓS-FILTRAGEM MANUAL)
# =============================================================================
#%% Scatter plots finais
print_section_header("VISUALIZAÇÃO FINAL (PÓS-FILTRAGEM MANUAL)")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].scatter(df_var1_final['IDADE'], df_var1_final[VAR1], alpha=0.5, s=10, c='green')
axes[0].grid(linestyle='--', color='gray', alpha=0.5)
axes[0].set_xlabel('Idade (meses)')
axes[0].set_ylabel(VAR1)
axes[0].set_title(f'{VAR1} vs Idade (Final)')

axes[1].scatter(df_var2_final['IDADE'], df_var2_final[VAR2], alpha=0.5, s=10, c='green')
axes[1].grid(linestyle='--', color='gray', alpha=0.5)
axes[1].set_xlabel('Idade (meses)')
axes[1].set_ylabel(VAR2)
axes[1].set_title(f'{VAR2} vs Idade (Final)')

axes[2].scatter(df_var3_final['IDADE'], df_var3_final[VAR3], alpha=0.5, s=10, c='green')
axes[2].grid(linestyle='--', color='gray', alpha=0.5)
axes[2].set_xlabel('Idade (meses)')
axes[2].set_ylabel(VAR3)
axes[2].set_title(f'{VAR3} vs Idade (Final)')

plt.tight_layout()
plt.savefig(rf"{OUTPUT_DIR}\scatter_final.png", dpi=300, bbox_inches='tight')
plt.show()

#%% Histogramas finais
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

create_histogram(axes[0], df_var1_final[VAR1].dropna(), VAR1, color='green')
axes[0].set_title(f'Distribuição de {VAR1} (Final)')

create_histogram(axes[1], df_var2_final[VAR2].dropna(), VAR2, color='green')
axes[1].set_title(f'Distribuição de {VAR2} (Final)')

create_histogram(axes[2], df_var3_final[VAR3].dropna(), VAR3, color='green')
axes[2].set_title(f'Distribuição de {VAR3} (Final)')

plt.tight_layout()
plt.savefig(rf"{OUTPUT_DIR}\histogramas_final.png", dpi=300, bbox_inches='tight')
plt.show()


# =============================================================================
# EXPORTAÇÃO DOS DATAFRAMES LIMPOS
# =============================================================================
#%% Resumo final e exportação
print_section_header("RESUMO FINAL E EXPORTAÇÃO")

print("\nResumo dos DataFrames após filtragem manual:")
print(f"  {VAR1}: {len(df_var1_final)} registros (removidos manualmente: {len(df_var1_removed)})")
print(f"  {VAR2}: {len(df_var2_final)} registros (removidos manualmente: {len(df_var2_removed)})")
print(f"  {VAR3}: {len(df_var3_final)} registros (removidos manualmente: {len(df_var3_removed)})")

# Exportação para Excel
df_var1_final.to_excel(OUTPUT_FILE_1, index=False)
print(f"\n✓ {VAR1} exportado para:\n  {OUTPUT_FILE_1}")

df_var2_final.to_excel(OUTPUT_FILE_2, index=False)
print(f"\n✓ {VAR2} exportado para:\n  {OUTPUT_FILE_2}")

df_var3_final.to_excel(OUTPUT_FILE_3, index=False)
print(f"\n✓ {VAR3} exportado para:\n  {OUTPUT_FILE_3}")

print("\n" + "=" * 60)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 60)
