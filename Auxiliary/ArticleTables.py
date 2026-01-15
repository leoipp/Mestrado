"""
ArticleTables.py - Gerador de Tabelas para Artigos Cientificos

Gera tabelas formatadas para a secao de Materiais e Metodos de artigos
cientificos sobre predicao de volume florestal com LiDAR.

Formatos de saida:
    - Excel (.xlsx)
    - LaTeX (.tex)
    - Word (.docx)
    - HTML (para visualizacao)

Tabelas geradas:
    1. Caracterizacao da area de estudo
    2. Estatisticas descritivas das variaveis de campo
    3. Metricas LiDAR utilizadas
    4. Parametros do modelo preditivo
    5. Resultados por regional/regime

Autor: Leonardo Ippolito Rodrigues
Ano: 2026
Projeto: Mestrado - Predicao de Volume Florestal com LiDAR
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from great_tables import GT, md, html, style, loc
    HAS_GT = True
except ImportError:
    HAS_GT = False
    print("Aviso: great_tables nao instalado. Use: pip install great-tables")


# =============================================================================
# CONFIGURACAO
# =============================================================================

class TableConfig:
    """Configuracao para geracao de tabelas."""

    # Mapeamento de regioes
    REGION_MAP = {
        'GN': 'Regiao 01',
        'NE': 'Regiao 02',
        'RD': 'Regiao 03'
    }

    # Mapeamento de regime
    REGIME_MAP = {
        'P': 'Alto Fuste',
        'T': 'Talhadia'
    }

    # Nomes das colunas para exibicao
    COLUMN_LABELS = {
        'Regional': 'Regional',
        'Regime_': 'Regime',
        'Classe_idade_m': 'Classe de Idade (meses)',
        'n': 'n',
        'Area': 'Area (m²)',
        'DAP': 'DAP (cm)',
        'HT': 'Ht (m)',
        'VTCC': 'V (m³/ha)',
        'Fustes': 'N (arv/ha)',
        'Idade': 'Idade (meses)'
    }

    # Metricas LiDAR
    LIDAR_METRICS = {
        'max': ('Altura maxima', 'Zmax', 'm'),
        'p90': ('Percentil 90', 'P90', 'm'),
        'p60': ('Percentil 60', 'P60', 'm'),
        'kur': ('Curtose', 'Kurt', '-'),
        'mean': ('Altura media', 'Zmean', 'm'),
        'std': ('Desvio padrao', 'Zsd', 'm'),
        'cv': ('Coef. variacao', 'Zcv', '%'),
        'skew': ('Assimetria', 'Zskew', '-'),
    }


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega dados do inventario florestal.

    Parameters
    ----------
    filepath : str
        Caminho para o arquivo Excel ou CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados carregados.
    """
    path = Path(filepath)

    if path.suffix == '.xlsx':
        df = pd.read_excel(filepath)
    elif path.suffix == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Formato nao suportado: {path.suffix}")

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados para geracao das tabelas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto do inventario.

    Returns
    -------
    pd.DataFrame
        DataFrame preparado com colunas auxiliares.
    """
    df = df.copy()

    # Regional (primeiros 2 caracteres do codigo do lote)
    if 'LOTE_CODIGO' in df.columns:
        df['Regional'] = df['LOTE_CODIGO'].str[:2].map(TableConfig.REGION_MAP)

        # Regime (caracter na posicao 11)
        df['Regime'] = df['LOTE_CODIGO'].str[11].map(
            lambda x: 'Alto Fuste' if x == 'P' else 'Talhadia'
        )

    # Classe de idade (intervalos de 2 anos)
    if 'Idade (meses)' in df.columns:
        df['Classe_Idade'] = (df['Idade (meses)'] // 2) * 2 + 1

    return df


def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    """Formata valor como media ± desvio padrao."""
    if pd.isna(std) or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_range(min_val: float, max_val: float, decimals: int = 2) -> str:
    """Formata valor como intervalo (min - max)."""
    return f"{min_val:.{decimals}f} - {max_val:.{decimals}f}"

# =============================================================================
# FIGURAS: ANALISE EXPLORATORIA (EDA) / PRE-SELECAO
# =============================================================================

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Encontra a primeira coluna existente em df que case com algum candidato.
    Faz match por igualdade (case-insensitive) e por 'contains' simples.
    """
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # 1) match exato (case-insensitive)
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    # 2) contains (case-insensitive)
    for cand in candidates:
        cl = cand.lower()
        for c in cols:
            if cl in c.lower():
                return c

    return None


def _ensure_regional_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas Regional e Regime (se possível a partir de LOTE_CODIGO)."""
    df = prepare_data(df)
    # Se já existirem mas vazias, tenta recompor
    if 'Regional' not in df.columns and 'LOTE_CODIGO' in df.columns:
        df['Regional'] = df['LOTE_CODIGO'].str[:2].map(TableConfig.REGION_MAP)
    if 'Regime' not in df.columns and 'LOTE_CODIGO' in df.columns:
        df['Regime'] = df['LOTE_CODIGO'].str[11].map(lambda x: 'Alto Fuste' if x == 'P' else 'Talhadia')
    return df


def _scatter_by_groups(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    alpha: float = 0.75
) -> None:
    """
    Scatter plot com cores por Regional e marcadores por Regime.
    Salva PNG em alta resolução.
    """
    df = df[[x_col, y_col, 'Regional', 'Regime']].dropna().copy()
    if df.empty:
        return

    # Mapas (cores por regional; marcadores por regime)
    region_order = [r for r in df['Regional'].dropna().unique()]
    regime_order = [r for r in df['Regime'].dropna().unique()]

    # Paleta simples baseada no ciclo padrão do matplotlib (não fixa cor manual)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    region_colors = {reg: color_cycle[i % len(color_cycle)] for i, reg in enumerate(region_order)} if color_cycle else {}
    regime_markers = {
        'Alto Fuste': 'o',
        'Talhadia': '^'
    }
    # fallback para regimes não previstos
    for i, reg in enumerate(regime_order):
        regime_markers.setdefault(reg, ['o', '^', 's', 'D', 'P', 'X'][i % 6])

    fig = plt.figure(figsize=(7.5, 5.5), dpi=300)
    ax = plt.gca()

    # Plota por combinações para legenda clara
    for reg in region_order:
        for rgm in regime_order:
            sub = df[(df['Regional'] == reg) & (df['Regime'] == rgm)]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col], sub[y_col],
                s=26,
                alpha=alpha,
                marker=regime_markers[rgm],
                c=region_colors.get(reg, None),
                edgecolors='none',
                label=f"{reg} | {rgm}"
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='--', alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc='best', ncol=1)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_eda_grid_3x3(
    df: pd.DataFrame,
    output_path: Path,
    y_vtcc: str,
    x_dap: Optional[str],
    x_ht: Optional[str],
    x_idade: Optional[str],
    x_p90: Optional[str],
    x_cub: Optional[str],
    x_std: Optional[str]
) -> None:
    """
    Gera um grid 3x3 de analise exploratoria (EDA) com padrao de artigo.
    """

    df = _ensure_regional_regime(df)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()

    def draw(ax, x_col, y_col, title, xlabel, ylabel):
        if x_col is None or y_col is None:
            ax.axis("off")
            return

        for reg in df['Regional'].dropna().unique():
            for rgm in df['Regime'].dropna().unique():
                sub = df[(df['Regional'] == reg) & (df['Regime'] == rgm)]
                if sub.empty:
                    continue

                marker = 'o' if rgm == 'Alto Fuste' else '^'
                ax.scatter(
                    sub[x_col], sub[y_col],
                    s=22, alpha=0.7,
                    marker=marker,
                    label=f"{reg} | {rgm}"
                )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(linestyle='--', alpha=0.25)

    # ===== Linha 1 – Cadastro =====
    draw(axes[0], y_vtcc, x_dap,   "DAP vs VTCC",   "VTCC (m³/ha)", x_dap)
    draw(axes[1], y_vtcc, x_ht,    "HT vs VTCC",    "VTCC (m³/ha)", x_ht)

    # EXCEÇÃO: IDADE (VTCC no eixo Y)
    draw(axes[2], x_idade, y_vtcc, "VTCC vs Idade", x_idade, "VTCC (m³/ha)")

    # ===== Linha 2 – LiDAR =====
    draw(axes[3], y_vtcc, x_p90,   "P90 vs VTCC",   "VTCC (m³/ha)", x_p90)
    draw(axes[4], y_vtcc, x_cub,   "CUB vs VTCC",   "VTCC (m³/ha)", x_cub)
    draw(axes[5], y_vtcc, x_std,   "StdDev vs VTCC","VTCC (m³/ha)", x_std)

    # Legenda única
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def generate_exploratory_figures(df: pd.DataFrame, output_dir: Union[str, Path]) -> Dict[str, str]:
    """
    Gera figuras de análise exploratória e pré-seleção de variáveis.

    Figuras:
      - VTCC x DAP
      - VTCC x HT
      - VTCC x IDADE
      - VTCC x P90
      - VTCC x CUB (se existir)
      - VTCC x STDDEV (se existir)

    Retorna dict {nome_figura: caminho_arquivo}.
    """
    out = Path(output_dir) / "figures_eda"
    out.mkdir(parents=True, exist_ok=True)

    df = _ensure_regional_regime(df)

    # Coluna alvo (tolerante)
    y = _find_column(df, ['VTCC', 'VTCC(m³/ha)', 'V (m³/ha)', 'V (m3/ha)'])
    if y is None:
        print("EDA: coluna VTCC não encontrada. Figuras não geradas.")
        return {}

    # Campo
    x_dap = _find_column(df, ['Dap.médio (cm)', 'DAP', 'DAP (cm)', 'DBH'])
    x_ht  = _find_column(df, ['HT.média (m)', 'HT', 'Ht', 'Ht (m)', 'Height'])
    x_id  = _find_column(df, ['Idade (meses)', 'IDADE', 'Idade', 'Age'])

    # LiDAR (tolerante)
    x_p90 = _find_column(df, ['p90', 'P90', 'Elev_P90', 'p90_cub', 'p90_cub_std'])
    x_cub = _find_column(df, ['cub', 'CUB', 'prod_ifpc', 'CUBAGEM', 'CUB_STD', 'curt_mean_cube'])
    x_sd  = _find_column(df, ['std', 'stddev', 'stdev', 'desvio', 'Zsd', 'std_cub', 'p90_cub_std', 'max_cub_std'])

    figures = {}

    # =========================
    # CADASTRO / CAMPO (VTCC no eixo X)
    # =========================

    if x_dap:
        p = out / "Fig_EDA_VTCC_x_DAP.png"
        _scatter_by_groups(
            df,
            x_col=y,  # VTCC no X
            y_col=x_dap,  # DAP no Y
            title="DAP vs VTCC",
            xlabel=y,
            ylabel=x_dap,
            outpath=p
        )
        figures["VTCC_x_DAP"] = str(p)

    if x_ht:
        p = out / "Fig_EDA_VTCC_x_HT.png"
        _scatter_by_groups(
            df,
            x_col=y,  # VTCC no X
            y_col=x_ht,  # HT no Y
            title="Altura (HT) vs VTCC",
            xlabel=y,
            ylabel=x_ht,
            outpath=p
        )
        figures["VTCC_x_HT"] = str(p)

    if x_id:
        p = out / "Fig_EDA_IDADE_x_VTCC.png"
        _scatter_by_groups(
            df,
            x_col=x_id,  # IDADE no eixo X
            y_col=y,  # VTCC no eixo Y
            title="VTCC vs Idade",
            xlabel=x_id,
            ylabel=y,
            outpath=p
        )

        figures["VTCC_x_IDADE"] = str(p)

    # =========================
    # LiDAR (VTCC no eixo X)
    # =========================

    if x_p90:
        p = out / "Fig_EDA_VTCC_x_P90.png"
        _scatter_by_groups(
            df,
            x_col=y,  # VTCC no X
            y_col=x_p90,  # P90 no Y
            title="P90 vs VTCC (LiDAR)",
            xlabel=y,
            ylabel=x_p90,
            outpath=p
        )
        figures["VTCC_x_P90"] = str(p)

    if x_cub:
        p = out / "Fig_EDA_VTCC_x_CUB.png"
        _scatter_by_groups(
            df,
            x_col=y,  # VTCC no X
            y_col=x_cub,  # CUB no Y
            title="CUB vs VTCC",
            xlabel=y,
            ylabel=x_cub,
            outpath=p
        )
        figures["VTCC_x_CUB"] = str(p)

    if x_sd:
        p = out / "Fig_EDA_VTCC_x_STDDEV.png"
        _scatter_by_groups(
            df,
            x_col=y,  # VTCC no X
            y_col=x_sd,  # StdDev no Y
            title="StdDev vs VTCC",
            xlabel=y,
            ylabel=x_sd,
            outpath=p
        )
        figures["VTCC_x_STDDEV"] = str(p)

    print(f"EDA: {len(figures)} figuras geradas em: {out}")

    # ============================================================
    # GRID 3x3 (FIGURA ÚNICA PARA ARTIGO)
    # ============================================================
    # ============================================================
    # FIGURAS "PAPER-READY": 2 GRIDS (CAMPO e LiDAR) POR REGIONAL
    #   - linhas = regional (1,2,3)
    #   - colunas = variáveis
    #   - cores = regime (2 cores)
    #   - labels (a1), (b1)... no canto sup/esq (número = regional)
    #   - eixos compartilhados por coluna
    #   - sem frame superior e direito
    # ============================================================

    df_plot = _ensure_regional_regime(df).copy()

    # --- ordem fixa de regionais (se existirem) ---
    regional_order = ["Regiao 01", "Regiao 02", "Regiao 03"]
    regionals = [r for r in regional_order if r in df_plot["Regional"].dropna().unique()]
    if not regionals:
        regionals = list(df_plot["Regional"].dropna().unique())

    # --- cores por regime (somente 2) usando ciclo padrão ---
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["C0", "C1"])
    regime_colors = {"Alto Fuste": color_cycle[0], "Talhadia": color_cycle[1]}

    def style_ax(ax):
        # remove “moldura” superior e direita
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linestyle="--", alpha=0.20)

    def add_panel_label(ax, letter, reg_idx):
        # (a1) no canto superior esquerdo dentro do eixo
        ax.text(
            0.02, 0.98, f"({letter}{reg_idx})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10
        )

    def scatter_by_regime(ax, d, x_col, y_col):
        tmp = d[[x_col, y_col, "Regime"]].dropna()
        if tmp.empty:
            return

        for rgm in ["Alto Fuste", "Talhadia"]:
            sub = tmp[tmp["Regime"] == rgm]
            if sub.empty:
                continue
            ax.scatter(
                sub[x_col], sub[y_col],
                s=16, alpha=0.75,
                c=regime_colors.get(rgm, None),
                edgecolors="none",
                label=rgm
            )

    def get_limits(df_in, x_col, y_col):
        t = df_in[[x_col, y_col]].dropna()
        if t.empty:
            return None
        xmin, xmax = t[x_col].min(), t[x_col].max()
        ymin, ymax = t[y_col].min(), t[y_col].max()
        # margem de 3%
        dx = (xmax - xmin) * 0.03 if xmax > xmin else 1
        dy = (ymax - ymin) * 0.03 if ymax > ymin else 1
        return (xmin - dx, xmax + dx, ymin - dy, ymax + dy)

    # -------------------------
    # GRID 1: CAMPO / IPC (3 colunas)
    # Colunas: DAP (VTCC no X), HT (VTCC no X), IDADE (VTCC no Y; Idade no X)
    # -------------------------
    campo_specs = [
        ("a.", y, x_dap, "VTCC (m³/ha)", x_dap,  "DAP médio (cm)"),
        ("b.", y, x_ht,  "VTCC (m³/ha)", x_ht,   "Ht média (m)"),
        ("c.", x_id, y,  x_id,           "VTCC (m³/ha)", "VTCC (m³/ha)")  # EXCEÇÃO IDADE
    ]
    # ajustar ylabels do IDADE (o 5º item é ylabel real)
    # campo_specs: (letter, xcol, ycol, xlabel, ylabel, pretty_ylabel)

    fig1, axes1 = plt.subplots(
        nrows=len(regionals), ncols=3,
        figsize=(14, 3.6 * max(1, len(regionals))),
        dpi=300,
        sharex='col', sharey=False
    )
    if len(regionals) == 1:
        axes1 = np.array([axes1])  # garante 2D

    # limites por coluna (para comparação)
    campo_limits = []
    for j, (letter, xcol, ycol, xlabel, ylabel, pretty_y) in enumerate(campo_specs):
        if xcol is None or ycol is None:
            campo_limits.append(None)
            continue
        lim = get_limits(df_plot, xcol, ycol)
        campo_limits.append(lim)

    for i, reg in enumerate(regionals, start=1):
        dreg = df_plot[df_plot["Regional"] == reg]
        for j, (letter, xcol, ycol, xlabel, ylabel, pretty_y) in enumerate(campo_specs):
            ax = axes1[i-1, j]
            style_ax(ax)
            add_panel_label(ax, letter, i)

            if xcol is None or ycol is None:
                ax.axis("off")
                continue

            scatter_by_regime(ax, dreg, xcol, ycol)

            # y-label só na primeira coluna (para reduzir poluição visual)
            if j == 0:
                ax.set_ylabel(pretty_y)
            else:
                ax.set_ylabel("")

            # x-label só na última linha
            if i == len(regionals):
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel("")

            # aplicar limites por coluna
            lim = campo_limits[j]
            if lim:
                ax.set_xlim(lim[0], lim[1])
                ax.set_ylim(lim[2], lim[3])

    # legenda dentro da figura (canto inferior direito)
    handles, labels = axes1[0, 0].get_legend_handles_labels()
    if handles:
        axes1[0, 0].legend(
            handles, labels,
            loc="lower right",
            frameon=True,
            fontsize=9
        )

    # --- GARANTE: 1 ylabel por coluna (apenas na primeira linha) ---
    nrows, ncols = axes1.shape

    # limpa todos os ylabels
    for r in range(nrows):
        for c in range(ncols):
            axes1[r, c].set_ylabel("")

    for i in range(ncols):
        # define apenas no topo de cada coluna
        axes1[i, 0].set_ylabel("DAP médio (cm)")
        axes1[i, 1].set_ylabel("Ht média (m)")
        axes1[i, 2].set_ylabel("VTCC (m³/ha)")  # coluna da idade (VTCC no Y)

    for ax in axes1.flatten():
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # --- remover xticks das duas primeiras linhas ---
    nrows, ncols = axes1.shape

    for r in range(nrows - 1):  # todas menos a última linha
        for c in range(ncols):
            axes1[r, c].tick_params(
                axis='x',
                which='both',
                bottom=False,
                labelbottom=False
            )

    fig1.subplots_adjust(wspace=0.20, hspace=0.05, bottom=0.10)
    out_campo = out / "Fig_EDA_GRID_CAMPO.png"
    fig1.savefig(out_campo, dpi=300)
    plt.close(fig1)
    figures["EDA_GRID_CAMPO"] = str(out_campo)

    # -------------------------
    # GRID 2: LiDAR (3 colunas) — VTCC no X sempre
    # -------------------------
    lidar_specs = [
        ("d.", y, x_p90, "VTCC (m³/ha)", x_p90, "P90 (m)"),
        ("e.", y, x_cub, "VTCC (m³/ha)", x_cub, "CUB"),
        ("f.", y, x_sd,  "VTCC (m³/ha)", x_sd,  "StdDev (m)")
    ]

    fig2, axes2 = plt.subplots(
        nrows=len(regionals), ncols=3,
        figsize=(14, 3.6 * max(1, len(regionals))),
        dpi=300,
        sharex='col', sharey='col'
    )
    if len(regionals) == 1:
        axes2 = np.array([axes2])

    lidar_limits = []
    for j, (letter, xcol, ycol, xlabel, ylabel, pretty_y) in enumerate(lidar_specs):
        if xcol is None or ycol is None:
            lidar_limits.append(None)
            continue
        lim = get_limits(df_plot, xcol, ycol)
        lidar_limits.append(lim)

    for i, reg in enumerate(regionals, start=1):
        dreg = df_plot[df_plot["Regional"] == reg]
        for j, (letter, xcol, ycol, xlabel, ylabel, pretty_y) in enumerate(lidar_specs):
            ax = axes2[i-1, j]
            style_ax(ax)
            add_panel_label(ax, letter, i)

            if xcol is None or ycol is None:
                ax.axis("off")
                continue

            scatter_by_regime(ax, dreg, xcol, ycol)

            if j == 0:
                ax.set_ylabel(pretty_y)
            else:
                ax.set_ylabel("")

            if i == len(regionals):
                ax.set_xlabel(xlabel)
            else:
                ax.set_xlabel("")

            lim = lidar_limits[j]
            if lim:
                ax.set_xlim(lim[0], lim[1])
                ax.set_ylim(lim[2], lim[3])

        # legenda dentro da figura (canto inferior direito)
        handles, labels = axes1[0, 0].get_legend_handles_labels()
        if handles:
            axes2[0, 0].legend(
                handles, labels,
                loc="lower right",
                frameon=True,
                fontsize=9
            )

    # --- GARANTE: 1 ylabel por coluna (apenas na primeira linha) ---
    nrows, ncols = axes2.shape

    for r in range(nrows):
        for c in range(ncols):
            axes2[r, c].set_ylabel("")

    for i in range(ncols):
        axes2[i, 0].set_ylabel("Z P90 (m)")
        axes2[i, 1].set_ylabel("Z Kurt (média cúbica)")
        axes2[i, 2].set_ylabel("Z σ (m)")

    for ax in axes2.flatten():
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # --- remover xticks das duas primeiras linhas ---
    nrows, ncols = axes1.shape

    for r in range(nrows - 1):  # todas menos a última linha
        for c in range(ncols):
            axes2[r, c].tick_params(
                axis='x',
                which='both',
                bottom=False,
                labelbottom=False
            )

    fig2.subplots_adjust(wspace=0.20, hspace=0.05, bottom=0.10)
    out_lidar = out / "Fig_EDA_GRID_LIDAR.png"
    fig2.savefig(out_lidar, dpi=300)
    plt.close(fig2)
    figures["EDA_GRID_LIDAR"] = str(out_lidar)

    print(f"EDA: grids salvos em:\n  - {out_campo}\n  - {out_lidar}")


    return figures

# =============================================================================
# TABELA 1: CARACTERIZACAO DA AREA DE ESTUDO
# =============================================================================

def table_study_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera tabela de caracterizacao da area de estudo.

    Tabela 1. Caracterizacao da area de estudo por regional.
    """
    df = prepare_data(df)

    summary = df.groupby('Regional').agg({
        'LOTE_CODIGO': 'nunique',  # Numero de lotes
        'DESCTIPOPROPRIEDADE': 'count',  # Numero de parcelas
        'Área.corrigida (m²)': ['sum', 'mean'],
        'Idade (meses)': ['min', 'max', 'mean'],
    }).reset_index()

    # Flatten column names
    summary.columns = [
        'Regional', 'N_Lotes', 'N_Parcelas',
        'Area_Total', 'Area_Media',
        'Idade_Min', 'Idade_Max', 'Idade_Media'
    ]

    # Formatar valores
    summary['Area_Total_ha'] = (summary['Area_Total'] / 10000).round(2)
    summary['Area_Media_m2'] = summary['Area_Media'].round(0).astype(int)
    summary['Amplitude_Idade'] = summary.apply(
        lambda r: f"{r['Idade_Min']:.1f} - {r['Idade_Max']:.1f}", axis=1
    )
    summary['Idade_Media'] = summary['Idade_Media'].round(2)

    # Selecionar colunas finais
    result = summary[[
        'Regional', 'N_Lotes', 'N_Parcelas',
        'Area_Total_ha', 'Area_Media_m2',
        'Amplitude_Idade', 'Idade_Media'
    ]].copy()

    result.columns = [
        'Regional', 'Lotes (n)', 'Parcelas (n)',
        'Area Total (ha)', 'Area Media (m²)',
        'Amplitude Idade (meses)', 'Idade Media (meses)'
    ]

    # Adicionar linha de total
    total = pd.DataFrame([{
        'Regional': 'Total',
        'Lotes (n)': result['Lotes (n)'].sum(),
        'Parcelas (n)': result['Parcelas (n)'].sum(),
        'Area Total (ha)': result['Area Total (ha)'].sum(),
        'Area Media (m²)': result['Area Media (m²)'].mean().round(0),
        'Amplitude Idade (meses)': f"{df['Idade (meses)'].min():.1f} - {df['Idade (meses)'].max():.1f}",
        'Idade Media (meses)': df['Idade (meses)'].mean().round(2)
    }])

    result = pd.concat([result, total], ignore_index=True)

    return result


# =============================================================================
# TABELA 2: ESTATISTICAS DESCRITIVAS DAS VARIAVEIS DE CAMPO
# =============================================================================

def table_field_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera tabela de estatisticas descritivas das variaveis de campo.

    Tabela 2. Estatisticas descritivas das variaveis dendrometricas.
    """
    df = prepare_data(df)

    # Variaveis de interesse
    variables = {
        'Idade (meses)': 'Idade (meses)',
        'Dap.médio (cm)': 'DAP (cm)',
        'HT.média (m)': 'Ht (m)',
        'VTCC(m³/ha)': 'V (m³/ha)',
        'Fustes (n/ha)': 'N (arv/ha)',
        'Área.corrigida (m²)': 'Area (m²)'
    }

    stats_list = []

    for col, label in variables.items():
        if col in df.columns:
            stats = {
                'Variavel': label,
                'n': df[col].count(),
                'Media': df[col].mean(),
                'DP': df[col].std(),
                'CV (%)': (df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else 0,
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Mediana': df[col].median()
            }
            stats_list.append(stats)

    result = pd.DataFrame(stats_list)

    # Formatar valores
    for col in ['Media', 'DP', 'Min', 'Max', 'Mediana']:
        result[col] = result[col].round(2)
    result['CV (%)'] = result['CV (%)'].round(1)
    result['n'] = result['n'].astype(int)

    return result


# =============================================================================
# TABELA 3: ESTATISTICAS POR REGIME E REGIONAL
# =============================================================================

def table_by_regime_regional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera tabela de estatisticas por regime e regional.

    Tabela 3. Caracteristicas dendrometricas por regime de manejo e regional.
    """
    df = prepare_data(df)

    # Agregar por Regime e Regional
    agg_funcs = {
        'TALHAO_CODIGO': 'count',
        'Dap.médio (cm)': ['mean', 'std'],
        'HT.média (m)': ['mean', 'std'],
        'VTCC(m³/ha)': ['mean', 'std'],
        'Fustes (n/ha)': ['mean', 'std'],
        'Idade (meses)': ['mean', 'std']
    }

    grouped = df.groupby(['Regime', 'Regional']).agg(agg_funcs).reset_index()

    # Flatten columns
    grouped.columns = [
        'Regime', 'Regional', 'n',
        'DAP_mean', 'DAP_std',
        'HT_mean', 'HT_std',
        'V_mean', 'V_std',
        'N_mean', 'N_std',
        'Idade_mean', 'Idade_std'
    ]

    # Combinar mean ± std
    result = pd.DataFrame({
        'Regime': grouped['Regime'],
        'Regional': grouped['Regional'],
        'n': grouped['n'],
        'Idade (meses)': grouped.apply(
            lambda r: format_mean_std(r['Idade_mean'], r['Idade_std'], 1), axis=1
        ),
        'DAP (cm)': grouped.apply(
            lambda r: format_mean_std(r['DAP_mean'], r['DAP_std'], 2), axis=1
        ),
        'Ht (m)': grouped.apply(
            lambda r: format_mean_std(r['HT_mean'], r['HT_std'], 2), axis=1
        ),
        'V (m³/ha)': grouped.apply(
            lambda r: format_mean_std(r['V_mean'], r['V_std'], 2), axis=1
        ),
        'N (arv/ha)': grouped.apply(
            lambda r: format_mean_std(r['N_mean'], r['N_std'], 0), axis=1
        ),
    })

    return result.sort_values(['Regime', 'Regional'])


# =============================================================================
# TABELA 4: METRICAS LIDAR
# =============================================================================

def table_lidar_metrics() -> pd.DataFrame:
    """
    Gera tabela descritiva das metricas LiDAR utilizadas.

    Tabela 4. Metricas derivadas dos dados LiDAR.
    """
    metrics = [
        ('Zmax', 'Altura maxima', 'Valor maximo de Z normalizado', 'm'),
        ('P90', 'Percentil 90', '90º percentil das alturas normalizadas', 'm'),
        ('P60', 'Percentil 60', '60º percentil das alturas normalizadas', 'm'),
        ('Zmean', 'Altura media', 'Media das alturas normalizadas', 'm'),
        ('Zsd', 'Desvio padrao', 'Desvio padrao das alturas', 'm'),
        ('Zcv', 'Coef. variacao', 'Coeficiente de variacao das alturas', '%'),
        ('Zkurt', 'Curtose', 'Curtose da distribuicao de alturas', '-'),
        ('Zskew', 'Assimetria', 'Assimetria da distribuicao de alturas', '-'),
    ]

    result = pd.DataFrame(metrics, columns=[
        'Abreviacao', 'Metrica', 'Descricao', 'Unidade'
    ])

    return result


# =============================================================================
# TABELA 5: ESTATISTICAS DAS METRICAS LIDAR
# =============================================================================

def table_lidar_statistics(df_lidar: pd.DataFrame) -> pd.DataFrame:
    """
    Gera tabela de estatisticas descritivas das metricas LiDAR.

    Parameters
    ----------
    df_lidar : pd.DataFrame
        DataFrame com as metricas LiDAR extraidas.

    Returns
    -------
    pd.DataFrame
        Tabela com estatisticas das metricas.
    """
    # Colunas de metricas LiDAR
    metric_cols = [col for col in df_lidar.columns
                   if any(m in col.lower() for m in ['max', 'p90', 'p60', 'mean', 'std', 'kur'])]

    stats_list = []

    for col in metric_cols:
        data = df_lidar[col].dropna()
        if len(data) > 0:
            stats = {
                'Metrica': col,
                'n': len(data),
                'Media': data.mean(),
                'DP': data.std(),
                'CV (%)': (data.std() / data.mean() * 100) if data.mean() != 0 else 0,
                'Min': data.min(),
                'Max': data.max()
            }
            stats_list.append(stats)

    result = pd.DataFrame(stats_list)

    if len(result) > 0:
        for col in ['Media', 'DP', 'Min', 'Max']:
            result[col] = result[col].round(2)
        result['CV (%)'] = result['CV (%)'].round(1)

    return result


# =============================================================================
# TABELA 6: PARAMETROS DO PROCESSAMENTO LIDAR
# =============================================================================

def table_lidar_processing() -> pd.DataFrame:
    """
    Gera tabela com parametros do processamento LiDAR.

    Tabela 5. Parametros utilizados no processamento dos dados LiDAR.
    """
    params = [
        ('Tiling', 'Tamanho do tile', '1000', 'm'),
        ('Tiling', 'Buffer', '50', 'm'),
        ('Denoising', 'Janela de busca', '100', 'm'),
        ('Denoising', 'Pontos isolados', '5', 'pontos'),
        ('Ground', 'Spike tolerance', '0.5', 'm'),
        ('Ground', 'Spike down', '2.5', 'm'),
        ('Thinning', 'Densidade alvo', '5', 'pts/m²'),
        ('Thinning', 'Metodo', 'random', '-'),
        ('DTM', 'Resolucao', '1.0', 'm'),
        ('CHM', 'Resolucao', '1.0', 'm'),
        ('Metricas', 'Resolucao da grade', '17', 'm'),
        ('Metricas', 'Percentis calculados', '60, 90', '-'),
    ]

    result = pd.DataFrame(params, columns=[
        'Etapa', 'Parametro', 'Valor', 'Unidade'
    ])

    return result


# =============================================================================
# TABELA 7: RESULTADOS DO MODELO
# =============================================================================

def table_model_results(
    model_name: str,
    r2: float,
    rmse: float,
    mae: float,
    bias: float,
    n_train: int,
    n_test: int,
    features: List[str]
) -> pd.DataFrame:
    """
    Gera tabela com resultados do modelo preditivo.

    Parameters
    ----------
    model_name : str
        Nome do modelo (ex: 'Random Forest', 'MLP').
    r2 : float
        Coeficiente de determinacao.
    rmse : float
        Raiz do erro quadratico medio.
    mae : float
        Erro absoluto medio.
    bias : float
        Vies medio.
    n_train : int
        Numero de amostras de treino.
    n_test : int
        Numero de amostras de teste.
    features : list
        Lista de features utilizadas.

    Returns
    -------
    pd.DataFrame
        Tabela com metricas do modelo.
    """
    metrics = [
        ('Modelo', model_name, '-'),
        ('Amostras (treino)', str(n_train), 'n'),
        ('Amostras (teste)', str(n_test), 'n'),
        ('R²', f'{r2:.4f}', '-'),
        ('RMSE', f'{rmse:.2f}', 'm³/ha'),
        ('RMSE (%)', f'{(rmse/np.mean([n_train, n_test])*100):.2f}', '%'),
        ('MAE', f'{mae:.2f}', 'm³/ha'),
        ('Bias', f'{bias:.2f}', 'm³/ha'),
        ('Variaveis', str(len(features)), 'n'),
    ]

    result = pd.DataFrame(metrics, columns=['Metrica', 'Valor', 'Unidade'])

    return result


# =============================================================================
# EXPORTADORES
# =============================================================================

def export_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    output_path: str,
    notes: Optional[str] = None
) -> str:
    """
    Exporta DataFrame para formato LaTeX.

    Parameters
    ----------
    df : pd.DataFrame
        Tabela a ser exportada.
    caption : str
        Legenda da tabela.
    label : str
        Label para referencia cruzada.
    output_path : str
        Caminho do arquivo de saida.
    notes : str, optional
        Notas de rodape da tabela.

    Returns
    -------
    str
        Codigo LaTeX da tabela.
    """
    # Gerar LaTeX basico
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'c' * (len(df.columns) - 1)
    )

    # Adicionar ambiente table
    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
{latex}"""

    if notes:
        full_latex += f"""\\begin{{tablenotes}}
\\small
\\item {notes}
\\end{{tablenotes}}
"""

    full_latex += "\\end{table}\n"

    # Salvar arquivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_latex)

    return full_latex


def export_to_excel(
    tables: Dict[str, pd.DataFrame],
    output_path: str
) -> None:
    """
    Exporta multiplas tabelas para Excel em abas separadas.

    Parameters
    ----------
    tables : dict
        Dicionario {nome_aba: DataFrame}.
    output_path : str
        Caminho do arquivo de saida.
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in tables.items():
            # Limitar nome da aba a 31 caracteres
            sheet_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Tabelas exportadas para: {output_path}")


def create_great_table(
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    source_note: Optional[str] = None,
    stub_column: Optional[str] = None
) -> 'GT':
    """
    Cria tabela formatada usando great_tables.

    Parameters
    ----------
    df : pd.DataFrame
        Dados da tabela.
    title : str
        Titulo da tabela.
    subtitle : str, optional
        Subtitulo.
    source_note : str, optional
        Nota de fonte.
    stub_column : str, optional
        Coluna para usar como stub (indice).

    Returns
    -------
    GT
        Objeto great_tables formatado.
    """
    if not HAS_GT:
        raise ImportError("great_tables nao instalado")

    gt = GT(df)

    # Titulo e subtitulo
    gt = gt.tab_header(title=title, subtitle=subtitle)

    # Stub (coluna de indice)
    if stub_column and stub_column in df.columns:
        gt = gt.tab_stub(rowname_col=stub_column)

    # Nota de fonte
    if source_note:
        gt = gt.tab_source_note(source_note)

    return gt


# =============================================================================
# FUNCAO PRINCIPAL
# =============================================================================

def generate_all_tables(
    data_path: str,
    output_dir: str,
    lidar_data_path: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Gera todas as tabelas para o artigo.

    Parameters
    ----------
    data_path : str
        Caminho para os dados do inventario.
    output_dir : str
        Diretorio de saida.
    lidar_data_path : str, optional
        Caminho para dados LiDAR.

    Returns
    -------
    dict
        Dicionario com todas as tabelas geradas.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GERACAO DE TABELAS PARA ARTIGO")
    print("=" * 60)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Saida: {output_dir}")

    # Carregar dados
    print("\n[1/6] Carregando dados...")
    df = load_data(data_path)
    print(f"  Registros: {len(df)}")
    # Figuras EDA / pré-seleção (para Materiais e Métodos / Resultados iniciais)
    print("\n[EDA] Gerando figuras exploratórias e pré-seleção de variáveis...")
    _ = generate_exploratory_figures(df, output_path)

    tables = {}

    # Tabela 1: Area de estudo
    print("\n[2/6] Gerando Tabela 1: Caracterizacao da area de estudo...")
    tables['Tab1_Area_Estudo'] = table_study_area(df)

    # Tabela 2: Estatisticas descritivas
    print("[3/6] Gerando Tabela 2: Estatisticas descritivas...")
    tables['Tab2_Estat_Descritivas'] = table_field_statistics(df)

    # Tabela 3: Por regime e regional
    print("[4/6] Gerando Tabela 3: Estatisticas por regime e regional...")
    tables['Tab3_Regime_Regional'] = table_by_regime_regional(df)

    # Tabela 4: Metricas LiDAR
    print("[5/6] Gerando Tabela 4: Descricao das metricas LiDAR...")
    tables['Tab4_Metricas_LiDAR'] = table_lidar_metrics()

    # Tabela 5: Parametros de processamento
    print("[6/6] Gerando Tabela 5: Parametros de processamento LiDAR...")
    tables['Tab5_Param_Processamento'] = table_lidar_processing()

    # Exportar para Excel
    excel_path = output_path / "Tabelas_Artigo.xlsx"
    export_to_excel(tables, str(excel_path))

    # Exportar para LaTeX
    latex_dir = output_path / "latex"
    latex_dir.mkdir(exist_ok=True)

    latex_configs = [
        ('Tab1_Area_Estudo', 'Caracterizacao da area de estudo por regional.', 'tab:area_estudo'),
        ('Tab2_Estat_Descritivas', 'Estatisticas descritivas das variaveis dendrometricas.', 'tab:estat_descr'),
        ('Tab3_Regime_Regional', 'Caracteristicas dendrometricas por regime de manejo e regional.', 'tab:regime_regional'),
        ('Tab4_Metricas_LiDAR', 'Metricas derivadas dos dados LiDAR.', 'tab:metricas_lidar'),
        ('Tab5_Param_Processamento', 'Parametros utilizados no processamento dos dados LiDAR.', 'tab:param_lidar'),
    ]

    for table_name, caption, label in latex_configs:
        latex_path = latex_dir / f"{table_name}.tex"
        export_to_latex(tables[table_name], caption, label, str(latex_path))

    print(f"\n{'=' * 60}")
    print("TABELAS GERADAS COM SUCESSO")
    print(f"{'=' * 60}")
    print(f"  Excel: {excel_path}")
    print(f"  LaTeX: {latex_dir}")

    return tables


# =============================================================================
# EXECUCAO
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gera tabelas formatadas para artigos cientificos"
    )
    parser.add_argument("input", help="Arquivo de dados do inventario (.xlsx ou .csv)")
    parser.add_argument("-o", "--output", default="./tables",
                        help="Diretorio de saida (default: ./tables)")
    parser.add_argument("--lidar", help="Arquivo com dados LiDAR (opcional)")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar tabelas no terminal")

    args = parser.parse_args()

    # Gerar tabelas
    tables = generate_all_tables(
        data_path=args.input,
        output_dir=args.output,
        lidar_data_path=args.lidar
    )

    # Mostrar tabelas se solicitado
    if args.show:
        for name, df in tables.items():
            print(f"\n{name}:")
            print(df.to_string(index=False))
