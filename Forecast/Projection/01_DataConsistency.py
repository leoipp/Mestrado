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
    7. Seleção manual de outliers com lasso (dados brutos)
    8. Sincronização wide <- long
    9. Transformação para pares consecutivos (i1, i2, v1, v2)
    10. Cálculo de métricas de delta:
        - dv: v2 - v1 (com filtro para valores < 0)
        - di: i2 - i1
        - delta: dv / di (taxa de variação)
    11. Seleção manual de outliers com lasso (dados de delta)
    12. Exportação dos DataFrames limpos com 3 sheets:
        - stack_enriquecido (wide)
        - long
        - pares_delta (com dv, di, delta)

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
from matplotlib.widgets import LassoSelector, Button
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

# Sheet names
SHEET_WIDE = "stack_enriquecido"
SHEET_LONG = "long"

# Chaves para sincronização entre sheets
SYNC_KEYS = ['x', 'y', 'REF_ID']

# Caminhos dos arquivos de saída
OUTPUT_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\cleaned"
OUTPUT_FILE_1 = rf"{OUTPUT_DIR}\cubmean_comp_cleaned.xlsx"
OUTPUT_FILE_2 = rf"{OUTPUT_DIR}\p90_comp_cleaned.xlsx"
OUTPUT_FILE_3 = rf"{OUTPUT_DIR}\stddev_comp_cleaned.xlsx"

# Parâmetros de filtragem
MIN_AGE_MONTHS = 36          # Idade mínima em meses
MAX_AGE_MONTHS = 120        # Idade máxima em meses
ZSCORE_THRESHOLD = 3         # Limiar para detecção de outliers via Z-score

# =============================================================================
# CONFIGURAÇÃO DOS PARES PARA TRANSFORMAÇÃO
# =============================================================================
# Definição de TODAS as combinações onde i1 < i2
PARES = [
    {
        'nome': '2019_2022',
        'i1': 'IDADE_19_meses',
        'i2': 'IDADE_22_meses',
        'v1': 'v_2019',
        'v2': 'v_2022'
    },
    {
        'nome': '2019_2024',
        'i1': 'IDADE_19_meses',
        'i2': 'IDADE_24_meses',
        'v1': 'v_2019',
        'v2': 'v_2024'
    },
    {
        'nome': '2019_2025',
        'i1': 'IDADE_19_meses',
        'i2': 'IDADE_25_meses',
        'v1': 'v_2019',
        'v2': 'v_2025'
    },
    {
        'nome': '2022_2024',
        'i1': 'IDADE_22_meses',
        'i2': 'IDADE_24_meses',
        'v1': 'v_2022',
        'v2': 'v_2024'
    },
    {
        'nome': '2022_2025',
        'i1': 'IDADE_22_meses',
        'i2': 'IDADE_25_meses',
        'v1': 'v_2022',
        'v2': 'v_2025'
    },
    {
        'nome': '2024_2025',
        'i1': 'IDADE_24_meses',
        'i2': 'IDADE_25_meses',
        'v1': 'v_2024',
        'v2': 'v_2025'
    }
]

# Colunas de identificação que serão mantidas na transformação
COLS_ID_TRANSFORM = ['x', 'y', 'variable', 'projeto', 'REF_ID', 'DATA_PLANTIO']

# Nome da terceira sheet para dados transformados
SHEET_TRANSFORM = "pares_delta"

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

# Cores para visualização por regime
COLORS_REGIME = {
    'Talhadia': '#d62728',
    'Alto fuste': '#2ca02c'
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

    # R ou R_ indica Talhadia (reforma)
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
# FUNÇÕES AUXILIARES
# =============================================================================
def lasso_filter_dataframe(df, x_col, y_col, title):
    """
    Permite selecionar pontos com lasso e removê-los do DataFrame.
    - Desenhe com o mouse para selecionar pontos (pode fazer múltiplas seleções)
    - Clique no botão "Aplicar" para remover os pontos selecionados e atualizar o gráfico
    - Pressione ENTER para confirmar e fechar
    - Pressione ESC para cancelar (não remove nada)

    Retorna df_filtrado e df_removidos.
    """
    # Resetar índice para garantir alinhamento
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Ajustar layout para dar espaço ao botão
    fig.subplots_adjust(bottom=0.15)

    # Estado mutável para rastrear pontos
    state = {
        'points_xy': np.column_stack((df[x_col].values, df[y_col].values)),
        'visible_mask': np.ones(len(df), dtype=bool),  # True = visível
        'selected_indices': set(),  # Índices selecionados na seleção atual
        'all_removed_indices': set(),  # Todos os índices já removidos
        'highlight_scatter': None,
        'main_scatter': None
    }

    # Scatter plot inicial
    state['main_scatter'] = ax.scatter(
        state['points_xy'][:, 0],
        state['points_xy'][:, 1],
        s=12, alpha=0.6, c='blue'
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{title}\n[Desenhe para selecionar | Aplicar=remover | ENTER=confirmar | ESC=cancelar]")
    ax.grid(ls="--", alpha=0.5)

    def update_scatter():
        """Atualiza o scatter principal mostrando apenas pontos visíveis."""
        if state['main_scatter'] is not None:
            state['main_scatter'].remove()

        visible_idx = np.where(state['visible_mask'])[0]
        if len(visible_idx) > 0:
            visible_x = state['points_xy'][visible_idx, 0]
            visible_y = state['points_xy'][visible_idx, 1]

            state['main_scatter'] = ax.scatter(
                visible_x, visible_y,
                s=12, alpha=0.6, c='blue'
            )

            # Recalcular limites dos eixos com margem de 5%
            x_min, x_max = visible_x.min(), visible_x.max()
            y_min, y_max = visible_y.min(), visible_y.max()
            x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 0.1
            y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        else:
            state['main_scatter'] = None

        # Atualizar título com contagem
        n_removed = len(state['all_removed_indices'])
        ax.set_title(f"{title}\n[Removidos: {n_removed} | ENTER=confirmar | ESC=cancelar]")
        fig.canvas.draw_idle()

    def update_highlight():
        """Atualiza a visualização dos pontos selecionados."""
        if state['highlight_scatter'] is not None:
            state['highlight_scatter'].remove()
            state['highlight_scatter'] = None

        if state['selected_indices']:
            sel_list = list(state['selected_indices'])
            state['highlight_scatter'] = ax.scatter(
                state['points_xy'][sel_list, 0],
                state['points_xy'][sel_list, 1],
                facecolors='none',
                edgecolors='red',
                s=50,
                linewidths=1.5,
                label=f'Selecionados: {len(state["selected_indices"])}'
            )
            ax.legend(loc='upper right')
        else:
            ax.legend().remove() if ax.get_legend() else None
        fig.canvas.draw_idle()

    def onselect(verts):
        """Callback chamado quando o lasso é completado."""
        if len(verts) < 3:
            return
        path = Path(verts)
        # Só considerar pontos que ainda estão visíveis
        visible_idx = np.where(state['visible_mask'])[0]
        visible_points = state['points_xy'][visible_idx]
        mask = path.contains_points(visible_points)
        new_selected = set(visible_idx[mask])
        state['selected_indices'].update(new_selected)
        update_highlight()
        print(f"  Selecionados até agora: {len(state['selected_indices'])} pontos")

    def on_apply(event):
        """Callback do botão Aplicar - remove pontos selecionados e atualiza gráfico."""
        if not state['selected_indices']:
            print("  Nenhum ponto selecionado para aplicar.")
            return

        # Mover selecionados para removidos
        state['all_removed_indices'].update(state['selected_indices'])

        # Atualizar máscara de visibilidade
        for idx in state['selected_indices']:
            state['visible_mask'][idx] = False

        n_applied = len(state['selected_indices'])
        state['selected_indices'].clear()

        # Remover highlight
        if state['highlight_scatter'] is not None:
            state['highlight_scatter'].remove()
            state['highlight_scatter'] = None

        # Atualizar scatter
        update_scatter()

        print(f"  Aplicado: {n_applied} pontos removidos. Total removidos: {len(state['all_removed_indices'])}")

    def on_key(event):
        """Callback para teclas."""
        if event.key == 'enter':
            # Incluir seleção atual nos removidos antes de fechar
            state['all_removed_indices'].update(state['selected_indices'])
            plt.close(fig)
        elif event.key == 'escape':
            state['all_removed_indices'].clear()
            plt.close(fig)

    # Criar o LassoSelector e manter referência
    lasso = LassoSelector(ax, onselect, useblit=True)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Criar botão "Aplicar"
    ax_button = fig.add_axes([0.4, 0.02, 0.2, 0.05])
    btn_apply = Button(ax_button, 'Aplicar (Remover Seleção)')
    btn_apply.on_clicked(on_apply)

    # Manter referências (evita garbage collection)
    fig._lasso = lasso
    fig._btn_apply = btn_apply

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show(block=True)

    # Processar resultado
    if not state['all_removed_indices']:
        print("Nenhum ponto selecionado.")
        return df, pd.DataFrame()

    sel_list = list(state['all_removed_indices'])
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


def sync_wide_from_long(df_wide, df_long, chaves=SYNC_KEYS):
    """
    Sincroniza sheet wide com base na sheet long limpa.
    Remove da wide as linhas cujas chaves (x, y, REF_ID) não existem mais na long.

    Args:
        df_wide: DataFrame da sheet wide original
        df_long: DataFrame da sheet long após limpeza
        chaves: lista de colunas para identificar linhas únicas

    Returns:
        DataFrame wide filtrado
    """
    # Verificar se as chaves existem em ambos os dataframes
    for chave in chaves:
        if chave not in df_wide.columns:
            print(f"  AVISO: Coluna '{chave}' não encontrada na sheet wide")
            return df_wide
        if chave not in df_long.columns:
            print(f"  AVISO: Coluna '{chave}' não encontrada na sheet long")
            return df_wide

    # Obter combinações únicas de chaves no long limpo
    chaves_long = df_long[chaves].drop_duplicates()

    # Criar coluna temporária para merge
    df_wide = df_wide.copy()
    df_wide['_merge_key'] = df_wide[chaves].astype(str).agg('|'.join, axis=1)
    chaves_long['_merge_key'] = chaves_long[chaves].astype(str).agg('|'.join, axis=1)

    # Filtrar wide mantendo apenas linhas que existem no long
    chaves_validas = set(chaves_long['_merge_key'])
    mask = df_wide['_merge_key'].isin(chaves_validas)

    df_wide_filtrado = df_wide[mask].drop(columns=['_merge_key']).copy()

    linhas_removidas = len(df_wide) - len(df_wide_filtrado)
    print(f"  Wide sincronizado: {len(df_wide_filtrado)} linhas (removidas: {linhas_removidas})")

    return df_wide_filtrado


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


def transformar_para_pares(df):
    """
    Transforma DataFrame de formato wide para pares consecutivos.

    Cada linha original pode gerar até 6 linhas (pares):
    - Par 2019->2022, 2019->2024, 2019->2025
    - Par 2022->2024, 2022->2025, 2024->2025

    Só gera o par se ambos v1 e v2 forem válidos (não NaN e não zero).

    Args:
        df: DataFrame original com colunas v_2019, v_2022, etc.

    Returns:
        DataFrame com colunas i1, i2, v1, v2 e identificação do par
    """
    linhas_resultado = []

    for idx, row in df.iterrows():
        for par in PARES:
            # Verificar se ambos os valores são válidos
            v1 = row.get(par['v1'])
            v2 = row.get(par['v2'])
            i1 = row.get(par['i1'])
            i2 = row.get(par['i2'])

            # Pular se v1 ou v2 for NaN ou zero
            if pd.isna(v1) or pd.isna(v2) or v1 == 0 or v2 == 0:
                continue

            # Pular se i1 ou i2 for NaN
            if pd.isna(i1) or pd.isna(i2):
                continue

            # Criar linha do par
            linha_par = {
                'i1': i1,
                'i2': i2,
                'v1': v1,
                'v2': v2,
                'par': par['nome']
            }

            # Adicionar colunas de identificação
            for col in COLS_ID_TRANSFORM:
                if col in row.index:
                    linha_par[col] = row[col]

            linhas_resultado.append(linha_par)

    # Criar DataFrame resultado
    df_resultado = pd.DataFrame(linhas_resultado)

    # Reordenar colunas
    cols_ordem = COLS_ID_TRANSFORM + ['par', 'i1', 'i2', 'v1', 'v2']
    cols_existentes = [c for c in cols_ordem if c in df_resultado.columns]
    df_resultado = df_resultado[cols_existentes]

    return df_resultado


def calcular_delta_metrics(df):
    """
    Calcula as métricas de delta a partir do DataFrame de pares.

    Calcula:
    - dv: v2 - v1 (diferença de valores)
    - di: i2 - i1 (diferença de idades)
    - delta: dv / di (taxa de variação)

    Aplica filtro para remover valores onde dv < 0.

    Args:
        df: DataFrame com colunas i1, i2, v1, v2

    Returns:
        DataFrame com colunas dv, di, delta adicionadas (filtrado)
    """
    df = df.copy()

    # Calcular diferenças
    df['dv'] = df['v2'] - df['v1']
    df['di'] = df['i2'] - df['i1']

    # Registrar quantidade antes do filtro
    n_antes = len(df)

    # Filtrar valores onde dv < 0
    df = df[df['dv'] >= 0].copy()

    n_apos = len(df)
    n_removidos = n_antes - n_apos
    if n_removidos > 0:
        print(f"  → {n_removidos} registros removidos: dv (v2-v1) < 0")

    # Calcular delta (evitar divisão por zero)
    df['delta'] = np.where(df['di'] != 0, (df['dv'] / df['v1']) / df['di'], np.nan)

    # Remover linhas onde delta é NaN (di era zero)
    n_antes_nan = len(df)
    df = df.dropna(subset=['delta'])
    n_nan_removidos = n_antes_nan - len(df)
    if n_nan_removidos > 0:
        print(f"  → {n_nan_removidos} registros removidos: di = 0 (divisão impossível)")

    return df


def lasso_filter_delta(df, x_col, y_col, title):
    """
    Permite selecionar pontos com lasso e removê-los do DataFrame de delta.
    Versão específica para a worksheet de delta/pares.

    - Desenhe com o mouse para selecionar pontos (pode fazer múltiplas seleções)
    - Clique no botão "Aplicar" para remover os pontos selecionados e atualizar o gráfico
    - Pressione ENTER para confirmar e fechar
    - Pressione ESC para cancelar (não remove nada)

    Retorna df_filtrado e df_removidos.
    """
    # Resetar índice para garantir alinhamento
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Ajustar layout para dar espaço ao botão
    fig.subplots_adjust(bottom=0.15)

    # Estado mutável para rastrear pontos
    state = {
        'points_xy': np.column_stack((df[x_col].values, df[y_col].values)),
        'visible_mask': np.ones(len(df), dtype=bool),  # True = visível
        'selected_indices': set(),  # Índices selecionados na seleção atual
        'all_removed_indices': set(),  # Todos os índices já removidos
        'highlight_scatter': None,
        'main_scatter': None
    }

    # Scatter plot inicial
    state['main_scatter'] = ax.scatter(
        state['points_xy'][:, 0],
        state['points_xy'][:, 1],
        s=12, alpha=0.6, c='purple'
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{title}\n[Desenhe para selecionar | Aplicar=remover | ENTER=confirmar | ESC=cancelar]")
    ax.grid(ls="--", alpha=0.5)

    def update_scatter():
        """Atualiza o scatter principal mostrando apenas pontos visíveis."""
        if state['main_scatter'] is not None:
            state['main_scatter'].remove()

        visible_idx = np.where(state['visible_mask'])[0]
        if len(visible_idx) > 0:
            visible_x = state['points_xy'][visible_idx, 0]
            visible_y = state['points_xy'][visible_idx, 1]

            state['main_scatter'] = ax.scatter(
                visible_x, visible_y,
                s=12, alpha=0.6, c='purple'
            )

            # Recalcular limites dos eixos com margem de 5%
            x_min, x_max = visible_x.min(), visible_x.max()
            y_min, y_max = visible_y.min(), visible_y.max()
            x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 0.1
            y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        else:
            state['main_scatter'] = None

        # Atualizar título com contagem
        n_removed = len(state['all_removed_indices'])
        ax.set_title(f"{title}\n[Removidos: {n_removed} | ENTER=confirmar | ESC=cancelar]")
        fig.canvas.draw_idle()

    def update_highlight():
        """Atualiza a visualização dos pontos selecionados."""
        if state['highlight_scatter'] is not None:
            state['highlight_scatter'].remove()
            state['highlight_scatter'] = None

        if state['selected_indices']:
            sel_list = list(state['selected_indices'])
            state['highlight_scatter'] = ax.scatter(
                state['points_xy'][sel_list, 0],
                state['points_xy'][sel_list, 1],
                facecolors='none',
                edgecolors='red',
                s=50,
                linewidths=1.5,
                label=f'Selecionados: {len(state["selected_indices"])}'
            )
            ax.legend(loc='upper right')
        else:
            ax.legend().remove() if ax.get_legend() else None
        fig.canvas.draw_idle()

    def onselect(verts):
        """Callback chamado quando o lasso é completado."""
        if len(verts) < 3:
            return
        path = Path(verts)
        # Só considerar pontos que ainda estão visíveis
        visible_idx = np.where(state['visible_mask'])[0]
        visible_points = state['points_xy'][visible_idx]
        mask = path.contains_points(visible_points)
        new_selected = set(visible_idx[mask])
        state['selected_indices'].update(new_selected)
        update_highlight()
        print(f"  Selecionados até agora: {len(state['selected_indices'])} pontos")

    def on_apply(event):
        """Callback do botão Aplicar - remove pontos selecionados e atualiza gráfico."""
        if not state['selected_indices']:
            print("  Nenhum ponto selecionado para aplicar.")
            return

        # Mover selecionados para removidos
        state['all_removed_indices'].update(state['selected_indices'])

        # Atualizar máscara de visibilidade
        for idx in state['selected_indices']:
            state['visible_mask'][idx] = False

        n_applied = len(state['selected_indices'])
        state['selected_indices'].clear()

        # Remover highlight
        if state['highlight_scatter'] is not None:
            state['highlight_scatter'].remove()
            state['highlight_scatter'] = None

        # Atualizar scatter
        update_scatter()

        print(f"  Aplicado: {n_applied} pontos removidos. Total removidos: {len(state['all_removed_indices'])}")

    def on_key(event):
        """Callback para teclas."""
        if event.key == 'enter':
            # Incluir seleção atual nos removidos antes de fechar
            state['all_removed_indices'].update(state['selected_indices'])
            plt.close(fig)
        elif event.key == 'escape':
            state['all_removed_indices'].clear()
            plt.close(fig)

    # Criar o LassoSelector e manter referência
    lasso = LassoSelector(ax, onselect, useblit=True)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Criar botão "Aplicar"
    ax_button = fig.add_axes([0.4, 0.02, 0.2, 0.05])
    btn_apply = Button(ax_button, 'Aplicar (Remover Seleção)')
    btn_apply.on_clicked(on_apply)

    # Manter referências (evita garbage collection)
    fig._lasso = lasso
    fig._btn_apply = btn_apply

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show(block=True)

    # Processar resultado
    if not state['all_removed_indices']:
        print("Nenhum ponto selecionado para remoção no delta.")
        return df, pd.DataFrame()

    sel_list = list(state['all_removed_indices'])
    df_removed = df.iloc[sel_list].copy()
    df_cleaned = df.drop(index=sel_list).reset_index(drop=True)

    print(f"Pontos removidos manualmente (delta): {len(df_removed)}")

    return df_cleaned, df_removed


def lasso_filter_por_estrato(df, x_col, y_col, var_name, color_col='regime'):
    """
    Aplica filtro lasso por estrato (regional + regime).

    Para cada grupo, abre uma janela de seleção manual.
    Retorna o DataFrame consolidado após todas as filtragens.

    Args:
        df: DataFrame com colunas 'regional', 'regime', 'grupo'
        x_col: Nome da coluna para eixo X
        y_col: Nome da coluna para eixo Y
        var_name: Nome da variável para títulos
        color_col: Coluna para colorir pontos ('regime' ou None)

    Returns:
        df_final: DataFrame consolidado após filtragens
        df_removed_total: DataFrame com todos os pontos removidos
    """
    # Garantir que temos as colunas de estrato
    if 'grupo' not in df.columns:
        df = preparar_dados_estrato(df)

    grupos = df['grupo'].unique()
    grupos = sorted(grupos)

    print(f"\n  Grupos encontrados: {len(grupos)}")
    for g in grupos:
        n = len(df[df['grupo'] == g])
        print(f"    {g}: {n} registros")

    dfs_limpos = []
    dfs_removidos = []

    for grupo in grupos:
        grupo_df = df[df['grupo'] == grupo].copy()
        n_grupo = len(grupo_df)

        if n_grupo == 0:
            continue

        # Extrair regime para cor
        regime = grupo_df['regime'].iloc[0] if 'regime' in grupo_df.columns else 'Alto fuste'
        cor = COLORS_REGIME.get(regime, COLOR_PRIMARY)

        print(f"\n>>> [{grupo}] n={n_grupo} - Selecione outliers")
        print("    Desenhe com o mouse | ENTER=confirmar | ESC=cancelar")

        # Usar lasso filter com cor do regime
        df_limpo, df_removido = lasso_filter_dataframe_com_cor(
            grupo_df,
            x_col=x_col,
            y_col=y_col,
            title=f"Seleção manual - {var_name}\nGrupo: {grupo} (n={n_grupo})",
            cor=cor
        )

        dfs_limpos.append(df_limpo)
        if not df_removido.empty:
            dfs_removidos.append(df_removido)

        print(f"    [{grupo}] Removidos: {len(df_removido)} | Restantes: {len(df_limpo)}")

    # Consolidar
    df_final = pd.concat(dfs_limpos, ignore_index=True) if dfs_limpos else pd.DataFrame()
    df_removed_total = pd.concat(dfs_removidos, ignore_index=True) if dfs_removidos else pd.DataFrame()

    print(f"\n  TOTAL: {len(df_final)} registros (removidos: {len(df_removed_total)})")

    return df_final, df_removed_total


def lasso_filter_dataframe_com_cor(df, x_col, y_col, title, cor='blue'):
    """
    Versão do lasso_filter_dataframe com cor customizável.
    """
    # Resetar índice para garantir alinhamento
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(bottom=0.15)

    state = {
        'points_xy': np.column_stack((df[x_col].values, df[y_col].values)),
        'visible_mask': np.ones(len(df), dtype=bool),
        'selected_indices': set(),
        'all_removed_indices': set(),
        'highlight_scatter': None,
        'main_scatter': None
    }

    state['main_scatter'] = ax.scatter(
        state['points_xy'][:, 0],
        state['points_xy'][:, 1],
        s=12, alpha=0.6, c=cor
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{title}\n[Desenhe para selecionar | Aplicar=remover | ENTER=confirmar | ESC=cancelar]")
    ax.grid(ls="--", alpha=0.5)

    def update_scatter():
        if state['main_scatter'] is not None:
            state['main_scatter'].remove()

        visible_idx = np.where(state['visible_mask'])[0]
        if len(visible_idx) > 0:
            visible_x = state['points_xy'][visible_idx, 0]
            visible_y = state['points_xy'][visible_idx, 1]

            state['main_scatter'] = ax.scatter(
                visible_x, visible_y,
                s=12, alpha=0.6, c=cor
            )

            x_min, x_max = visible_x.min(), visible_x.max()
            y_min, y_max = visible_y.min(), visible_y.max()
            x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 0.1
            y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        else:
            state['main_scatter'] = None

        n_removed = len(state['all_removed_indices'])
        ax.set_title(f"{title}\n[Removidos: {n_removed} | ENTER=confirmar | ESC=cancelar]")
        fig.canvas.draw_idle()

    def update_highlight():
        if state['highlight_scatter'] is not None:
            state['highlight_scatter'].remove()
            state['highlight_scatter'] = None

        if state['selected_indices']:
            sel_list = list(state['selected_indices'])
            state['highlight_scatter'] = ax.scatter(
                state['points_xy'][sel_list, 0],
                state['points_xy'][sel_list, 1],
                facecolors='none',
                edgecolors='red',
                s=50,
                linewidths=1.5,
                label=f'Selecionados: {len(state["selected_indices"])}'
            )
            ax.legend(loc='upper right')
        else:
            ax.legend().remove() if ax.get_legend() else None
        fig.canvas.draw_idle()

    def onselect(verts):
        if len(verts) < 3:
            return
        path = Path(verts)
        visible_idx = np.where(state['visible_mask'])[0]
        visible_points = state['points_xy'][visible_idx]
        mask = path.contains_points(visible_points)
        new_selected = set(visible_idx[mask])
        state['selected_indices'].update(new_selected)
        update_highlight()
        print(f"  Selecionados até agora: {len(state['selected_indices'])} pontos")

    def on_apply(event):
        if not state['selected_indices']:
            print("  Nenhum ponto selecionado para aplicar.")
            return

        state['all_removed_indices'].update(state['selected_indices'])

        for idx in state['selected_indices']:
            state['visible_mask'][idx] = False

        n_applied = len(state['selected_indices'])
        state['selected_indices'].clear()

        if state['highlight_scatter'] is not None:
            state['highlight_scatter'].remove()
            state['highlight_scatter'] = None

        update_scatter()

        print(f"  Aplicado: {n_applied} pontos removidos. Total removidos: {len(state['all_removed_indices'])}")

    def on_key(event):
        if event.key == 'enter':
            state['all_removed_indices'].update(state['selected_indices'])
            plt.close(fig)
        elif event.key == 'escape':
            state['all_removed_indices'].clear()
            plt.close(fig)

    lasso = LassoSelector(ax, onselect, useblit=True)
    fig.canvas.mpl_connect('key_press_event', on_key)

    ax_button = fig.add_axes([0.4, 0.02, 0.2, 0.05])
    btn_apply = Button(ax_button, 'Aplicar (Remover Seleção)')
    btn_apply.on_clicked(on_apply)

    fig._lasso = lasso
    fig._btn_apply = btn_apply

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show(block=True)

    if not state['all_removed_indices']:
        return df, pd.DataFrame()

    sel_list = list(state['all_removed_indices'])
    df_removed = df.iloc[sel_list].copy()
    df_cleaned = df.drop(index=sel_list).reset_index(drop=True)

    return df_cleaned, df_removed


def lasso_filter_delta_por_estrato(df, x_col, y_col, var_name):
    """
    Aplica filtro lasso por estrato para dados de delta.

    Args:
        df: DataFrame de delta com colunas REF_ID
        x_col: Nome da coluna para eixo X
        y_col: Nome da coluna para eixo Y
        var_name: Nome da variável para títulos

    Returns:
        df_final: DataFrame consolidado após filtragens
        df_removed_total: DataFrame com todos os pontos removidos
    """
    # Garantir que temos as colunas de estrato
    if 'grupo' not in df.columns:
        df = preparar_dados_estrato(df)

    grupos = df['grupo'].unique()
    grupos = sorted(grupos)

    print(f"\n  Grupos encontrados: {len(grupos)}")
    for g in grupos:
        n = len(df[df['grupo'] == g])
        print(f"    {g}: {n} registros")

    dfs_limpos = []
    dfs_removidos = []

    for grupo in grupos:
        grupo_df = df[df['grupo'] == grupo].copy()
        n_grupo = len(grupo_df)

        if n_grupo == 0:
            continue

        regime = grupo_df['regime'].iloc[0] if 'regime' in grupo_df.columns else 'Alto fuste'
        cor = COLORS_REGIME.get(regime, 'purple')

        print(f"\n>>> [{grupo}] n={n_grupo} - Selecione outliers no Delta")
        print("    Desenhe com o mouse | ENTER=confirmar | ESC=cancelar")

        df_limpo, df_removido = lasso_filter_dataframe_com_cor(
            grupo_df,
            x_col=x_col,
            y_col=y_col,
            title=f"Delta {var_name} - Grupo: {grupo}\n(v1 x delta, n={n_grupo})",
            cor=cor
        )

        dfs_limpos.append(df_limpo)
        if not df_removido.empty:
            dfs_removidos.append(df_removido)

        print(f"    [{grupo}] Removidos: {len(df_removido)} | Restantes: {len(df_limpo)}")

    # Consolidar
    df_final = pd.concat(dfs_limpos, ignore_index=True) if dfs_limpos else pd.DataFrame()
    df_removed_total = pd.concat(dfs_removidos, ignore_index=True) if dfs_removidos else pd.DataFrame()

    print(f"\n  TOTAL: {len(df_final)} registros (removidos: {len(df_removed_total)})")

    return df_final, df_removed_total


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
df_var1_wide = pd.read_excel(INPUT_FILE_1, sheet_name=SHEET_WIDE)
df_var1 = pd.read_excel(INPUT_FILE_1, sheet_name=SHEET_LONG)
print(f"  Sheet '{SHEET_WIDE}': {df_var1_wide.shape[0]} linhas × {df_var1_wide.shape[1]} colunas")
print(f"  Sheet '{SHEET_LONG}': {df_var1.shape[0]} linhas × {df_var1.shape[1]} colunas")

print(f"\nCarregando {VAR2} de: {INPUT_FILE_2}")
df_var2_wide = pd.read_excel(INPUT_FILE_2, sheet_name=SHEET_WIDE)
df_var2 = pd.read_excel(INPUT_FILE_2, sheet_name=SHEET_LONG)
print(f"  Sheet '{SHEET_WIDE}': {df_var2_wide.shape[0]} linhas × {df_var2_wide.shape[1]} colunas")
print(f"  Sheet '{SHEET_LONG}': {df_var2.shape[0]} linhas × {df_var2.shape[1]} colunas")

print(f"\nCarregando {VAR3} de: {INPUT_FILE_3}")
df_var3_wide = pd.read_excel(INPUT_FILE_3, sheet_name=SHEET_WIDE)
df_var3 = pd.read_excel(INPUT_FILE_3, sheet_name=SHEET_LONG)
print(f"  Sheet '{SHEET_WIDE}': {df_var3_wide.shape[0]} linhas × {df_var3_wide.shape[1]} colunas")
print(f"  Sheet '{SHEET_LONG}': {df_var3.shape[0]} linhas × {df_var3.shape[1]} colunas")


# =============================================================================
# VISUALIZAÇÃO EXPLORATÓRIA INICIAL
# =============================================================================
#%% Scatter plots: Variável vs Idade
print_section_header("VISUALIZAÇÃO EXPLORATÓRIA")

# Dicionário para armazenar figuras (serão salvas no final)
figuras_para_salvar = {}

fig_scatter_inicial, axes = plt.subplots(1, 3, figsize=(14, 5))

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
figuras_para_salvar['scatter_exploratorio_inicial'] = fig_scatter_inicial
plt.show()

#%% Histogramas das variáveis principais
fig_hist_inicial, axes = plt.subplots(1, 3, figsize=(14, 5))

create_histogram(axes[0], df_var1[VAR1].dropna(), VAR1)
axes[0].set_title(f'Distribuição de {VAR1}')

create_histogram(axes[1], df_var2[VAR2].dropna(), VAR2)
axes[1].set_title(f'Distribuição de {VAR2}')

create_histogram(axes[2], df_var3[VAR3].dropna(), VAR3)
axes[2].set_title(f'Distribuição de {VAR3}')

plt.tight_layout()
figuras_para_salvar['histogramas_inicial'] = fig_hist_inicial
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
# VISUALIZAÇÃO PÓS-FILTRAGEM (AUTOMÁTICA)
# =============================================================================
#%% Scatter plots pós-filtragem automática (apenas visualização, salvamento no final)
print_section_header("VISUALIZAÇÃO PÓS-FILTRAGEM (AUTOMÁTICA)")

fig_scatter_pos_auto, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].scatter(df_var1_clean['IDADE'], df_var1_clean[VAR1], alpha=0.5, s=10)
axes[0].grid(linestyle='--', color='gray', alpha=0.5)
axes[0].set_xlabel('Idade (meses)')
axes[0].set_ylabel(VAR1)
axes[0].set_title(f'{VAR1} vs Idade (Filtrado Auto)')

axes[1].scatter(df_var2_clean['IDADE'], df_var2_clean[VAR2], alpha=0.5, s=10)
axes[1].grid(linestyle='--', color='gray', alpha=0.5)
axes[1].set_xlabel('Idade (meses)')
axes[1].set_ylabel(VAR2)
axes[1].set_title(f'{VAR2} vs Idade (Filtrado Auto)')

axes[2].scatter(df_var3_clean['IDADE'], df_var3_clean[VAR3], alpha=0.5, s=10)
axes[2].grid(linestyle='--', color='gray', alpha=0.5)
axes[2].set_xlabel('Idade (meses)')
axes[2].set_ylabel(VAR3)
axes[2].set_title(f'{VAR3} vs Idade (Filtrado Auto)')

plt.tight_layout()
# Figura será salva no final após todas as filtragens
plt.show()
plt.close(fig_scatter_pos_auto)

#%% Histogramas pós-filtragem automática (apenas visualização, salvamento no final)
fig_hist_pos_auto, axes = plt.subplots(1, 3, figsize=(14, 5))

create_histogram(axes[0], df_var1_clean[VAR1].dropna(), VAR1)
axes[0].set_title(f'Distribuição de {VAR1} (Filtrado Auto)')

create_histogram(axes[1], df_var2_clean[VAR2].dropna(), VAR2)
axes[1].set_title(f'Distribuição de {VAR2} (Filtrado Auto)')

create_histogram(axes[2], df_var3_clean[VAR3].dropna(), VAR3)
axes[2].set_title(f'Distribuição de {VAR3} (Filtrado Auto)')

plt.tight_layout()
# Figura será salva no final após todas as filtragens
plt.show()
plt.close(fig_hist_pos_auto)



# =============================================================================
# PREPARAÇÃO DOS ESTRATOS (REGIONAL + REGIME)
# =============================================================================
print_section_header("PREPARAÇÃO DOS ESTRATOS")

print("\nAdicionando colunas de estratificação (regional, regime, grupo)...")
df_var1_clean = preparar_dados_estrato(df_var1_clean)
df_var2_clean = preparar_dados_estrato(df_var2_clean)
df_var3_clean = preparar_dados_estrato(df_var3_clean)

# Mostrar distribuição por grupo
for df, var in [(df_var1_clean, VAR1), (df_var2_clean, VAR2), (df_var3_clean, VAR3)]:
    print(f"\n{var}:")
    print(f"  Por Regional: {dict(df['regional'].value_counts())}")
    print(f"  Por Regime: {dict(df['regime'].value_counts())}")


# =============================================================================
# SELEÇÃO MANUAL DE OUTLIERS (LASSO) - POR ESTRATO
# =============================================================================
#%% Seleção manual com lasso para cada variável, por grupo (regional + regime)
print_section_header("SELEÇÃO MANUAL DE OUTLIERS (POR ESTRATO)")

# Desativar modo interativo para garantir que plt.show() bloqueie
plt.ioff()

print("\n" + "="*50)
print(f">>> {VAR1} - Filtragem por estrato")
print("="*50)
df_var1_final, df_var1_removed = lasso_filter_por_estrato(
    df_var1_clean,
    x_col="IDADE",
    y_col=VAR1,
    var_name=VAR1
)

print("\n" + "="*50)
print(f">>> {VAR2} - Filtragem por estrato")
print("="*50)
df_var2_final, df_var2_removed = lasso_filter_por_estrato(
    df_var2_clean,
    x_col="IDADE",
    y_col=VAR2,
    var_name=VAR2
)

print("\n" + "="*50)
print(f">>> {VAR3} - Filtragem por estrato")
print("="*50)
df_var3_final, df_var3_removed = lasso_filter_por_estrato(
    df_var3_clean,
    x_col="IDADE",
    y_col=VAR3,
    var_name=VAR3
)


# =============================================================================
# VISUALIZAÇÃO FINAL (PÓS-FILTRAGEM MANUAL)
# =============================================================================
#%% Scatter plots finais (apenas visualização, salvamento no final)
print_section_header("VISUALIZAÇÃO FINAL (PÓS-FILTRAGEM MANUAL)")

fig_scatter_final, axes = plt.subplots(1, 3, figsize=(14, 5))

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
figuras_para_salvar['scatter_final'] = fig_scatter_final
plt.show()

#%% Histogramas finais (apenas visualização, salvamento no final)
fig_hist_final, axes = plt.subplots(1, 3, figsize=(14, 5))

create_histogram(axes[0], df_var1_final[VAR1].dropna(), VAR1, color='green')
axes[0].set_title(f'Distribuição de {VAR1} (Final)')

create_histogram(axes[1], df_var2_final[VAR2].dropna(), VAR2, color='green')
axes[1].set_title(f'Distribuição de {VAR2} (Final)')

create_histogram(axes[2], df_var3_final[VAR3].dropna(), VAR3, color='green')
axes[2].set_title(f'Distribuição de {VAR3} (Final)')

plt.tight_layout()
figuras_para_salvar['histogramas_final'] = fig_hist_final
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

# =============================================================================
# SINCRONIZAÇÃO WIDE <- LONG
# =============================================================================
print_section_header("SINCRONIZAÇÃO WIDE <- LONG")

print(f"\nSincronizando {VAR1}...")
df_var1_wide_sync = sync_wide_from_long(df_var1_wide, df_var1_final)

print(f"\nSincronizando {VAR2}...")
df_var2_wide_sync = sync_wide_from_long(df_var2_wide, df_var2_final)

print(f"\nSincronizando {VAR3}...")
df_var3_wide_sync = sync_wide_from_long(df_var3_wide, df_var3_final)

# =============================================================================
# TRANSFORMAÇÃO PARA PARES E CÁLCULO DE DELTA
# =============================================================================
#%% Transformação e cálculo de delta
print_section_header("TRANSFORMAÇÃO PARA PARES E CÁLCULO DE DELTA")

# Processar VAR1 (Z Kurt)
print(f"\n--- Processando {VAR1} ---")
print("  [1] Transformando para pares...")
df_var1_pares = transformar_para_pares(df_var1_wide_sync)
print(f"      Pares gerados: {len(df_var1_pares)}")

print("  [2] Calculando dv, di e delta...")
df_var1_delta = calcular_delta_metrics(df_var1_pares)
print(f"      Registros após filtro dv >= 0: {len(df_var1_delta)}")

print("  [3] Adicionando estratos...")
df_var1_delta = preparar_dados_estrato(df_var1_delta)

# Processar VAR2 (Z P90)
print(f"\n--- Processando {VAR2} ---")
print("  [1] Transformando para pares...")
df_var2_pares = transformar_para_pares(df_var2_wide_sync)
print(f"      Pares gerados: {len(df_var2_pares)}")

print("  [2] Calculando dv, di e delta...")
df_var2_delta = calcular_delta_metrics(df_var2_pares)
print(f"      Registros após filtro dv >= 0: {len(df_var2_delta)}")

print("  [3] Adicionando estratos...")
df_var2_delta = preparar_dados_estrato(df_var2_delta)

# Processar VAR3 (Z σ)
print(f"\n--- Processando {VAR3} ---")
print("  [1] Transformando para pares...")
df_var3_pares = transformar_para_pares(df_var3_wide_sync)
print(f"      Pares gerados: {len(df_var3_pares)}")

print("  [2] Calculando dv, di e delta...")
df_var3_delta = calcular_delta_metrics(df_var3_pares)
print(f"      Registros após filtro dv >= 0: {len(df_var3_delta)}")

print("  [3] Adicionando estratos...")
df_var3_delta = preparar_dados_estrato(df_var3_delta)


# =============================================================================
# VISUALIZAÇÃO DO DELTA (PRÉ-FILTRAGEM MANUAL)
# =============================================================================
#%% Scatter plots do delta (apenas visualização, salvamento no final)
print_section_header("VISUALIZAÇÃO DO DELTA")

fig_delta_pre, axes = plt.subplots(1, 3, figsize=(14, 5))

# Delta VAR1 vs v1 (valor inicial)
axes[0].scatter(df_var1_delta['v1'], df_var1_delta['delta'], alpha=0.5, s=10, c='purple')
axes[0].grid(linestyle='--', color='gray', alpha=0.5)
axes[0].set_xlabel('Valor Inicial (v1)')
axes[0].set_ylabel('Delta (dv/di)')
axes[0].set_title(f'Delta {VAR1} vs v1')

# Delta VAR2 vs v1
axes[1].scatter(df_var2_delta['v1'], df_var2_delta['delta'], alpha=0.5, s=10, c='purple')
axes[1].grid(linestyle='--', color='gray', alpha=0.5)
axes[1].set_xlabel('Valor Inicial (v1)')
axes[1].set_ylabel('Delta (dv/di)')
axes[1].set_title(f'Delta {VAR2} vs v1')

# Delta VAR3 vs v1
axes[2].scatter(df_var3_delta['v1'], df_var3_delta['delta'], alpha=0.5, s=10, c='purple')
axes[2].grid(linestyle='--', color='gray', alpha=0.5)
axes[2].set_xlabel('Valor Inicial (v1)')
axes[2].set_ylabel('Delta (dv/di)')
axes[2].set_title(f'Delta {VAR3} vs v1')

plt.tight_layout()
# Figura será salva no final após todas as filtragens
plt.show()
plt.close(fig_delta_pre)


# =============================================================================
# SELEÇÃO MANUAL DE OUTLIERS NO DELTA (LASSO) - POR ESTRATO
# =============================================================================
#%% Seleção manual com lasso para delta de cada variável, por grupo
print_section_header("SELEÇÃO MANUAL DE OUTLIERS NO DELTA (POR ESTRATO)")

print("\n" + "="*50)
print(f">>> Delta {VAR1} - Filtragem por estrato")
print("="*50)
df_var1_delta_final, df_var1_delta_removed = lasso_filter_delta_por_estrato(
    df_var1_delta,
    x_col="v1",
    y_col="delta",
    var_name=VAR1
)

print("\n" + "="*50)
print(f">>> Delta {VAR2} - Filtragem por estrato")
print("="*50)
df_var2_delta_final, df_var2_delta_removed = lasso_filter_delta_por_estrato(
    df_var2_delta,
    x_col="v1",
    y_col="delta",
    var_name=VAR2
)

print("\n" + "="*50)
print(f">>> Delta {VAR3} - Filtragem por estrato")
print("="*50)
df_var3_delta_final, df_var3_delta_removed = lasso_filter_delta_por_estrato(
    df_var3_delta,
    x_col="v1",
    y_col="delta",
    var_name=VAR3
)


# =============================================================================
# VISUALIZAÇÃO FINAL DO DELTA (PÓS-FILTRAGEM MANUAL)
# =============================================================================
#%% Scatter plots finais do delta (apenas visualização, salvamento no final)
print_section_header("VISUALIZAÇÃO FINAL DO DELTA (PÓS-FILTRAGEM MANUAL)")

fig_delta_final, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].scatter(df_var1_delta_final['v1'], df_var1_delta_final['delta'], alpha=0.5, s=10, c='green')
axes[0].grid(linestyle='--', color='gray', alpha=0.5)
axes[0].set_xlabel('Valor Inicial (v1)')
axes[0].set_ylabel('Delta (dv/di)')
axes[0].set_title(f'Delta {VAR1} vs v1 (Final)')

axes[1].scatter(df_var2_delta_final['v1'], df_var2_delta_final['delta'], alpha=0.5, s=10, c='green')
axes[1].grid(linestyle='--', color='gray', alpha=0.5)
axes[1].set_xlabel('Valor Inicial (v1)')
axes[1].set_ylabel('Delta (dv/di)')
axes[1].set_title(f'Delta {VAR2} vs v1 (Final)')

axes[2].scatter(df_var3_delta_final['v1'], df_var3_delta_final['delta'], alpha=0.5, s=10, c='green')
axes[2].grid(linestyle='--', color='gray', alpha=0.5)
axes[2].set_xlabel('Valor Inicial (v1)')
axes[2].set_ylabel('Delta (dv/di)')
axes[2].set_title(f'Delta {VAR3} vs v1 (Final)')

plt.tight_layout()
figuras_para_salvar['scatter_delta_final'] = fig_delta_final
plt.show()


# =============================================================================
# RESUMO DO DELTA
# =============================================================================
print_section_header("RESUMO DOS DADOS DE DELTA")

print("\nResumo dos DataFrames de delta após filtragem manual:")
print(f"  {VAR1}: {len(df_var1_delta_final)} registros (removidos manualmente: {len(df_var1_delta_removed)})")
print(f"  {VAR2}: {len(df_var2_delta_final)} registros (removidos manualmente: {len(df_var2_delta_removed)})")
print(f"  {VAR3}: {len(df_var3_delta_final)} registros (removidos manualmente: {len(df_var3_delta_removed)})")


# =============================================================================
# GERAÇÃO E SALVAMENTO DE TODAS AS FIGURAS
# =============================================================================
print_section_header("SALVAMENTO DE TODAS AS FIGURAS")

print("\nGerando scatter plot pós-filtragem (dados finais)...")
fig_scatter_pos, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].scatter(df_var1_final['IDADE'], df_var1_final[VAR1], alpha=0.5, s=10)
axes[0].grid(linestyle='--', color='gray', alpha=0.5)
axes[0].set_xlabel('Idade (meses)')
axes[0].set_ylabel(VAR1)
axes[0].set_title(f'{VAR1} vs Idade (Filtrado)')

axes[1].scatter(df_var2_final['IDADE'], df_var2_final[VAR2], alpha=0.5, s=10)
axes[1].grid(linestyle='--', color='gray', alpha=0.5)
axes[1].set_xlabel('Idade (meses)')
axes[1].set_ylabel(VAR2)
axes[1].set_title(f'{VAR2} vs Idade (Filtrado)')

axes[2].scatter(df_var3_final['IDADE'], df_var3_final[VAR3], alpha=0.5, s=10)
axes[2].grid(linestyle='--', color='gray', alpha=0.5)
axes[2].set_xlabel('Idade (meses)')
axes[2].set_ylabel(VAR3)
axes[2].set_title(f'{VAR3} vs Idade (Filtrado)')

plt.tight_layout()
figuras_para_salvar['scatter_pos_filtragem'] = fig_scatter_pos

print("Gerando histogramas pós-filtragem (dados finais)...")
fig_hist_pos, axes = plt.subplots(1, 3, figsize=(14, 5))

create_histogram(axes[0], df_var1_final[VAR1].dropna(), VAR1)
axes[0].set_title(f'Distribuição de {VAR1} (Filtrado)')

create_histogram(axes[1], df_var2_final[VAR2].dropna(), VAR2)
axes[1].set_title(f'Distribuição de {VAR2} (Filtrado)')

create_histogram(axes[2], df_var3_final[VAR3].dropna(), VAR3)
axes[2].set_title(f'Distribuição de {VAR3} (Filtrado)')

plt.tight_layout()
figuras_para_salvar['histogramas_pos_filtragem'] = fig_hist_pos

print("Gerando scatter plot delta pré-filtragem (dados finais)...")
fig_delta_pre_final, axes = plt.subplots(1, 3, figsize=(14, 5))

axes[0].scatter(df_var1_delta_final['v1'], df_var1_delta_final['delta'], alpha=0.5, s=10, c='purple')
axes[0].grid(linestyle='--', color='gray', alpha=0.5)
axes[0].set_xlabel('Valor Inicial (v1)')
axes[0].set_ylabel('Delta (dv/di)')
axes[0].set_title(f'Delta {VAR1} vs v1')

axes[1].scatter(df_var2_delta_final['v1'], df_var2_delta_final['delta'], alpha=0.5, s=10, c='purple')
axes[1].grid(linestyle='--', color='gray', alpha=0.5)
axes[1].set_xlabel('Valor Inicial (v1)')
axes[1].set_ylabel('Delta (dv/di)')
axes[1].set_title(f'Delta {VAR2} vs v1')

axes[2].scatter(df_var3_delta_final['v1'], df_var3_delta_final['delta'], alpha=0.5, s=10, c='purple')
axes[2].grid(linestyle='--', color='gray', alpha=0.5)
axes[2].set_xlabel('Valor Inicial (v1)')
axes[2].set_ylabel('Delta (dv/di)')
axes[2].set_title(f'Delta {VAR3} vs v1')

plt.tight_layout()
figuras_para_salvar['scatter_delta_pre_filtragem'] = fig_delta_pre_final

# Salvar todas as figuras
print("\nSalvando todas as figuras...")
for nome, fig in figuras_para_salvar.items():
    caminho = rf"{OUTPUT_DIR}\{nome}.png"
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    print(f"  Salvo: {nome}.png")
    plt.close(fig)

print(f"\nTotal de figuras salvas: {len(figuras_para_salvar)}")


# =============================================================================
# EXPORTAÇÃO COM TRÊS SHEETS
# =============================================================================
print_section_header("EXPORTAÇÃO DOS ARQUIVOS LIMPOS")

# Exportação VAR1 com três sheets
with pd.ExcelWriter(OUTPUT_FILE_1, engine='openpyxl') as writer:
    df_var1_wide_sync.to_excel(writer, sheet_name=SHEET_WIDE, index=False)
    df_var1_final.to_excel(writer, sheet_name=SHEET_LONG, index=False)
    df_var1_delta_final.to_excel(writer, sheet_name=SHEET_TRANSFORM, index=False)
print(f"\n✓ {VAR1} exportado para:\n  {OUTPUT_FILE_1}")
print(f"    - Sheet '{SHEET_WIDE}': {len(df_var1_wide_sync)} linhas")
print(f"    - Sheet '{SHEET_LONG}': {len(df_var1_final)} linhas")
print(f"    - Sheet '{SHEET_TRANSFORM}': {len(df_var1_delta_final)} linhas (com dv, di, delta)")

# Exportação VAR2 com três sheets
with pd.ExcelWriter(OUTPUT_FILE_2, engine='openpyxl') as writer:
    df_var2_wide_sync.to_excel(writer, sheet_name=SHEET_WIDE, index=False)
    df_var2_final.to_excel(writer, sheet_name=SHEET_LONG, index=False)
    df_var2_delta_final.to_excel(writer, sheet_name=SHEET_TRANSFORM, index=False)
print(f"\n✓ {VAR2} exportado para:\n  {OUTPUT_FILE_2}")
print(f"    - Sheet '{SHEET_WIDE}': {len(df_var2_wide_sync)} linhas")
print(f"    - Sheet '{SHEET_LONG}': {len(df_var2_final)} linhas")
print(f"    - Sheet '{SHEET_TRANSFORM}': {len(df_var2_delta_final)} linhas (com dv, di, delta)")

# Exportação VAR3 com três sheets
with pd.ExcelWriter(OUTPUT_FILE_3, engine='openpyxl') as writer:
    df_var3_wide_sync.to_excel(writer, sheet_name=SHEET_WIDE, index=False)
    df_var3_final.to_excel(writer, sheet_name=SHEET_LONG, index=False)
    df_var3_delta_final.to_excel(writer, sheet_name=SHEET_TRANSFORM, index=False)
print(f"\n✓ {VAR3} exportado para:\n  {OUTPUT_FILE_3}")
print(f"    - Sheet '{SHEET_WIDE}': {len(df_var3_wide_sync)} linhas")
print(f"    - Sheet '{SHEET_LONG}': {len(df_var3_final)} linhas")
print(f"    - Sheet '{SHEET_TRANSFORM}': {len(df_var3_delta_final)} linhas (com dv, di, delta)")

print("\n" + "=" * 60)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 60)
