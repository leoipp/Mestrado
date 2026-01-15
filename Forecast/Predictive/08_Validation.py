"""
08_Validation.py - Validação do Modelo por Talhão (Áreas independentes + Volume)

Objetivo (conforme solicitado):
1) Consistência geométrica do clip (ÁREA)
   - Filtrar pelo campo Flag (não vazio)
   - Gráfico: Área CAD x Área Raster
   - Gráfico: Resíduo de área (%) x Área Raster

2) Validação volumétrica em talhões independentes (VOLUME)
   - Filtrar Flag == "Ok" e Diffmed ∈ {-1, 0, +1}
   - Gráfico: Estimado x Observado (1:1 + regressão + métricas)
   - Gráfico: Estimado x Resíduo (%)
   - Gráfico: Frequência/Distribuição de resíduos (%)
   - Mantendo o filtro acima:
       - Violino resíduos (%) por Regime
       - Violino resíduos (%) por Regional
       - Violino resíduos (%) por Regional × Regime

3) Impacto do desfasamento temporal (Diffmed)
   - Filtrar Flag == "Ok"
   - Gráfico: Diffmed x Resíduo (%) com regressão linear (r, p, R²)

Autor: Leonardo Ippolito Rodrigues
Data: 2026
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Optional, Dict, List, Tuple

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminhos
INPUT_FILE = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results\ALL_RASTER_STATS_P90_CUB_STD_CLIPPED.xlsx"
SHEET_NAME = "CONSISTIDO"
OUTPUT_DIR = Path(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results\Figures")

# Nomes das colunas (adapte conforme seu arquivo)
COL_TALHAO = 'TALHAO'
COL_OBSERVADO = 'PROD_IFPC'      # VTCC IFPc (m3/ha)
COL_PREDITO = 'Mean'             # VTCC W2W (m3/ha)
COL_AREA_CAD = 'Area CAD'        # Area cadastro (ha)
COL_AREA_RASTER = 'Area_ha'      # Area raster (ha)
COL_IDADE = 'Idade'              # Idade em meses (opcional)
COL_REGIONAL = 'REGIONAL'
COL_REGIME = 'Regime'
COL_DELTA_T = 'Diffmed'          # Diferença temporal IFPc-LiDAR (meses)

COL_FLAG = "Flag"
FLAG_OK = "OK"  # será comparado após upper().strip()

# =============================================================================
# ESTILO ARTIGO
# =============================================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

# Paleta
_cmap = plt.cm.YlGnBu

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

def get_regional_colors(regions: List[str]) -> Dict[str, tuple]:
    """Gera cores para cada regional usando YlGnBu."""
    regions = [str(r) for r in regions]
    n = len(regions)
    if n <= 1:
        return {regions[0]: _cmap(0.6)} if n == 1 else {}
    return {r: _cmap(0.25 + 0.6 * i / (n - 1)) for i, r in enumerate(sorted(set(regions)))}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.3,
                  num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementação simples de LOWESS (Locally Weighted Scatterplot Smoothing).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 5:
        return x, y

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    x_eval = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
    y_eval = np.zeros(num_points)

    n = len(x_sorted)
    k = int(np.ceil(frac * n))
    k = max(3, min(k, n))

    for i, xi in enumerate(x_eval):
        distances = np.abs(x_sorted - xi)
        idx = np.argsort(distances)[:k]
        x_local = x_sorted[idx]
        y_local = y_sorted[idx]
        d_local = distances[idx]

        d_max = d_local.max() * 1.001 if d_local.max() > 0 else 1.0
        weights = (1 - (d_local / d_max) ** 3) ** 3

        W = np.diag(weights)
        X = np.column_stack([np.ones(k), x_local])

        try:
            beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y_local, rcond=None)[0]
            y_eval[i] = beta[0] + beta[1] * xi
        except Exception:
            y_eval[i] = np.average(y_local, weights=weights)

    return x_eval, y_eval


def calculate_metrics(y_obs: np.ndarray, y_pred: np.ndarray, min_obs: float = 1e-6) -> Dict:
    """
    Métricas de validação.
    IMPORTANTE: R² aqui é o R² da regressão linear (y_pred ~ y_obs),
    igual ao exibido no gráfico (linha ajustada).
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_obs) & np.isfinite(y_pred) & (y_obs > min_obs)
    y_obs = y_obs[mask]
    y_pred = y_pred[mask]

    n = len(y_obs)
    y_mean = np.mean(y_obs) if n else np.nan

    # R² da regressão (igual ao plot)
    if n >= 3:
        slope, intercept, r, p, _ = stats.linregress(y_obs, y_pred)
        y_hat = slope * y_obs + intercept
        r2 = r2_score(y_pred, y_hat)  # R² da regressão (equivale ao r² em regressão simples)
    else:
        slope = intercept = r = p = np.nan
        r2 = np.nan

    rmse = np.sqrt(mean_squared_error(y_obs, y_pred)) if n else np.nan
    rmse_pct = (rmse / y_mean) * 100 if n and y_mean != 0 else np.nan
    mae = mean_absolute_error(y_obs, y_pred) if n else np.nan
    mae_pct = (mae / y_mean) * 100 if n and y_mean != 0 else np.nan
    bias = np.mean(y_pred - y_obs) if n else np.nan
    bias_pct = (bias / y_mean) * 100 if n and y_mean != 0 else np.nan

    return {
        'N': n,
        'R2': r2,
        'Slope': slope,
        'Intercept': intercept,
        'r': r,
        'p': p,
        'RMSE': rmse,
        'RMSE_pct': rmse_pct,
        'MAE': mae,
        'MAE_pct': mae_pct,
        'Bias': bias,
        'Bias_pct': bias_pct,
    }

def calculate_temporal_regression_metrics(df: pd.DataFrame) -> Dict:
    """
    Métricas da regressão Diffmed (x) vs resíduo % (y).
    Igual ao figC1_diffmed_vs_resid_linear.
    """
    resid_pct = compute_residuals_pct(df, COL_OBSERVADO, COL_PREDITO).values
    x = pd.to_numeric(df[COL_DELTA_T], errors="coerce").values
    y = pd.to_numeric(resid_pct, errors="coerce")

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {"N": len(x), "R2": np.nan, "slope": np.nan, "intercept": np.nan, "r": np.nan, "p": np.nan}

    slope, intercept, r, p, _ = stats.linregress(x, y)
    return {"N": len(x), "R2": r**2, "slope": slope, "intercept": intercept, "r": r, "p": p}


def save_figure(fig, output_dir: Path, filename: str):
    """Salva figura em PNG e PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{filename}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

    pdf_path = output_dir / f"{filename}.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')

    print(f"  Salvo: {filename}.png / {filename}.pdf")
    plt.close(fig)


def compute_residuals_pct(df: pd.DataFrame, col_obs: str, col_pred: str) -> pd.Series:
    """Resíduo relativo (%) = (pred - obs) / obs * 100"""
    obs = pd.to_numeric(df[col_obs], errors="coerce")
    pred = pd.to_numeric(df[col_pred], errors="coerce")
    resid = (pred - obs) / obs * 100
    resid = resid.replace([np.inf, -np.inf], np.nan)
    return resid


def linear_fit_with_stats(x: np.ndarray, y: np.ndarray) -> Dict:
    """Regressão linear simples e estatísticas."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {"ok": False}

    slope, intercept, r, p, _ = stats.linregress(x, y)
    y_hat = slope * x + intercept
    r2 = r2_score(y, y_hat) if len(x) > 2 else np.nan

    return {
        "ok": True,
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "p": p,
        "r2": r2
    }


# =============================================================================
# FIGURAS - BLOCO A (ÁREA / CLIP)
# =============================================================================

def figA1_area_cad_vs_area_raster(df: pd.DataFrame, output_dir: Path):
    """Área CAD × Área Raster (1:1)"""
    fig, ax = plt.subplots(figsize=(7, 6))

    x = pd.to_numeric(df[COL_AREA_CAD], errors="coerce").values
    y = pd.to_numeric(df[COL_AREA_RASTER], errors="coerce").values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Tamanho por área CAD
    if len(x) and np.nanmax(x) > np.nanmin(x):
        sizes = 25 + (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)) * 80
    else:
        sizes = 40

    ax.scatter(x, y, c='#555555', s=sizes, alpha=0.65, edgecolors='white', linewidth=0.5)

    lim_max = max(np.nanmax(x), np.nanmax(y)) * 1.05 if len(x) else 1
    ax.plot([0, lim_max], [0, lim_max], linestyle='--', c='black', linewidth=1.5)


    ax.set_xlabel('Área Cadastro (ha)')
    ax.set_ylabel('Área Raster (ha)')
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect('equal')

    save_figure(fig, output_dir, 'FIG_A1_area_cadastro_vs_area_raster')


def figA2_resid_area_pct_vs_area_raster(df: pd.DataFrame, output_dir: Path):
    """Resíduo de área (%) × Área Raster (ha)"""
    fig, ax = plt.subplots(figsize=(7, 5))

    area_cad = pd.to_numeric(df[COL_AREA_CAD], errors="coerce").values
    area_r = pd.to_numeric(df[COL_AREA_RASTER], errors="coerce").values
    mask = np.isfinite(area_cad) & np.isfinite(area_r) & (area_cad != 0)
    area_cad = area_cad[mask]
    area_r = area_r[mask]

    resid_area_pct = (area_r - area_cad) / area_cad * 100

    ax.scatter(area_r, resid_area_pct, c='#555555', s=40, alpha=0.55,
               edgecolors='white', linewidth=0.5)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)

    # === FORMATADOR PT-BR ===
    formatter = FuncFormatter(lambda x, pos: f"{x:.1f}".replace(".", ","))
    ax.xaxis.set_major_formatter(FuncFormatter(pt_br_smart))
    ax.yaxis.set_major_formatter(FuncFormatter(pt_br_smart))

    ax.set_xlabel('Área Raster (ha)')
    ax.set_ylabel('Resíduos (%)')
    ax.set_ylim(-10, 10)

    save_figure(fig, output_dir, 'FIG_A2_resid_area_pct_vs_area_raster')


# =============================================================================
# FIGURAS - BLOCO B (VOLUME - OK + Diffmed ∈ {-1,0,+1})
# =============================================================================

def figB1_estimado_vs_observado(df: pd.DataFrame, output_dir: Path):
    """Estimado (W2W) × Observado (IFPc): 1:1 + regressão + métricas"""
    fig, ax = plt.subplots(figsize=(7, 6))

    x = pd.to_numeric(df[COL_OBSERVADO], errors="coerce").values
    y = pd.to_numeric(df[COL_PREDITO], errors="coerce").values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    ax.scatter(x, y, c=_cmap(0.6), s=40, alpha=0.6,
               edgecolors='white', linewidth=0.5)

    lim_max = max(np.nanmax(x), np.nanmax(y)) * 1.05 if len(x) else 1
    ax.plot([0, lim_max], [0, lim_max], 'r--', linewidth=1.5, label='1:1')

    fit = linear_fit_with_stats(x, y)
    if fit.get("ok"):
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(x_line, y_line, '-', color=_cmap(0.85), linewidth=1.8,
                label=f"Regressão: y={fit['slope']:.2f}x{fit['intercept']:+.2f} (R²={fit['r2']:.3f})")

    metrics = calculate_metrics(x, y)
    textstr = (f"N = {metrics['N']}\n"
               f"R² = {metrics['R2']:.4f}\n"
               f"RMSE = {metrics['RMSE']:.2f} m³/ha ({metrics['RMSE_pct']:.1f}%)\n"
               f"MAE = {metrics['MAE']:.2f} m³/ha ({metrics['MAE_pct']:.1f}%)\n"
               f"Bias = {metrics['Bias']:+.2f} m³/ha ({metrics['Bias_pct']:+.1f}%)")
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('VTCC Observado – IFPc (m³/ha)')
    ax.set_ylabel('VTCC Estimado – W2W (m³/ha)')
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)

    save_figure(fig, output_dir, 'FIG_B1_estimado_vs_observado_1to1')


def figB2_estimado_vs_residuo(df: pd.DataFrame, output_dir: Path):
    """Estimado (W2W) × Resíduo (%)"""
    fig, ax = plt.subplots(figsize=(7, 5))

    est = pd.to_numeric(df[COL_PREDITO], errors="coerce").values
    resid_pct = compute_residuals_pct(df, COL_OBSERVADO, COL_PREDITO).values

    mask = np.isfinite(est) & np.isfinite(resid_pct)
    est = est[mask]
    resid_pct = resid_pct[mask]

    ax.scatter(est, resid_pct, c=_cmap(0.6), s=40, alpha=0.55,
               edgecolors='white', linewidth=0.5)

    xs, ys = lowess_smooth(est, resid_pct, frac=0.4)
    if len(xs) > 1:
        ax.plot(xs, ys, '-', color=_cmap(0.85), linewidth=2, label='LOWESS')

    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(20, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(-20, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax.set_xlabel('VTCC Estimado – W2W (m³/ha)')
    ax.set_ylabel('Resíduo Relativo (%)')
    ax.legend(loc='upper right', fontsize=8)

    save_figure(fig, output_dir, 'FIG_B2_estimado_vs_resid_pct')


def figB3_frequencia_residuos(df: pd.DataFrame, output_dir: Path):
    """Frequência/Distribuição de resíduos (%)"""
    fig, ax = plt.subplots(figsize=(7, 5))

    resid_pct = compute_residuals_pct(df, COL_OBSERVADO, COL_PREDITO).values
    resid_pct = resid_pct[np.isfinite(resid_pct)]

    bins = np.arange(-60, 65, 5)
    ax.hist(resid_pct, bins=bins, alpha=0.75, color=_cmap(0.6),
            edgecolor='white', linewidth=0.8)

    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)

    text = (f"N = {len(resid_pct)}\n"
            f"Média = {np.mean(resid_pct):+.1f}%\n"
            f"Mediana = {np.median(resid_pct):+.1f}%\n"
            f"Desvio = {np.std(resid_pct):.1f}%\n"
            f"|Res| < 20%: {np.mean(np.abs(resid_pct) < 20)*100:.1f}%")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Resíduo Relativo (%)')
    ax.set_ylabel('Frequência')

    save_figure(fig, output_dir, 'FIG_B3_freq_residuos')


def violin_resid_by_group(df: pd.DataFrame, output_dir: Path, group_col: str,
                          filename: str, title: str):
    """Violino de resíduos (%) por uma coluna categórica."""
    if group_col not in df.columns:
        print(f"  [SKIP] {filename}: coluna {group_col} não encontrada")
        return

    dfp = df.copy()
    dfp["resid_pct"] = compute_residuals_pct(dfp, COL_OBSERVADO, COL_PREDITO)

    dfp = dfp.dropna(subset=["resid_pct", group_col])
    dfp[group_col] = dfp[group_col].astype(str)

    # ordem por mediana
    order = dfp.groupby(group_col)["resid_pct"].median().sort_values().index.tolist()
    data = [dfp[dfp[group_col] == g]["resid_pct"].values for g in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_cmap(0.35 + 0.5 * i / max(1, len(order) - 1)))
        pc.set_alpha(0.75)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.6)

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.4)

    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(20, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(-20, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{g}\n(n={len(d)})" for g, d in zip(order, data)], fontsize=8)
    ax.set_xlabel(group_col)
    ax.set_ylabel("Resíduo Relativo (%)")
    ax.set_title(title)

    save_figure(fig, output_dir, filename)


def violin_resid_regional_x_regime(df: pd.DataFrame, output_dir: Path):
    """Violino resíduos (%) por Regional × Regime (grupos combinados)."""
    if COL_REGIONAL not in df.columns or COL_REGIME not in df.columns:
        print("  [SKIP] FIG_B6: regional/regime não encontrado")
        return

    dfp = df.copy()
    dfp["resid_pct"] = compute_residuals_pct(dfp, COL_OBSERVADO, COL_PREDITO)
    dfp = dfp.dropna(subset=["resid_pct", COL_REGIONAL, COL_REGIME])

    dfp["grp"] = dfp[COL_REGIONAL].astype(str) + " | " + dfp[COL_REGIME].astype(str)

    order = dfp.groupby("grp")["resid_pct"].median().sort_values().index.tolist()
    data = [dfp[dfp["grp"] == g]["resid_pct"].values for g in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_cmap(0.25 + 0.6 * i / max(1, len(order) - 1)))
        pc.set_alpha(0.75)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.6)

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.4)

    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(20, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(-20, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{g}\n(n={len(d)})" for g, d in zip(order, data)],
                       fontsize=7, rotation=30, ha="right")
    ax.set_xlabel("Regional | Regime")
    ax.set_ylabel("Resíduo Relativo (%)")
    ax.set_title("Resíduo (%) por Regional × Regime (violino)")

    save_figure(fig, output_dir, "FIG_B6_violin_resid_regional_x_regime")


# =============================================================================
# FIGURAS - BLOCO C (Diffmed × resíduo - OK + regressão linear)
# =============================================================================

def figC1_diffmed_vs_resid_linear(df: pd.DataFrame, output_dir: Path):
    """Diffmed × Resíduo (%) com regressão linear (Flag OK)."""
    if COL_DELTA_T not in df.columns:
        print(f"  [SKIP] FIG_C1: coluna {COL_DELTA_T} não encontrada")
        return

    dfp = df.copy()
    dfp["resid_pct"] = compute_residuals_pct(dfp, COL_OBSERVADO, COL_PREDITO)

    x = pd.to_numeric(dfp[COL_DELTA_T], errors="coerce").values
    y = pd.to_numeric(dfp["resid_pct"], errors="coerce").values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 10:
        print("  [SKIP] FIG_C1: poucos dados para regressão")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c='#555555', s=45, alpha=0.55,
               edgecolors="white", linewidth=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_ylim(-100, 100)

    fit = linear_fit_with_stats(x, y)
    if fit.get("ok"):
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax.plot(x_line, y_line, "-", color='#ff7f0e', linewidth=1.5,
                 label=f"y = {fit['slope']:.2f}x{fit['intercept']:+.2f} | R² = {fit['r2']:.4f})". replace('.', ','))

        txt = (f"y = {fit['slope']:.2f}x {fit['intercept']:+.2f}\n"
               f"r = {fit['r']:.3f} (p={fit['p']:.3f})\n"
               f"R² = {fit['r2']:.3f}")
        """ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))"""
        print(txt)

    ax.set_xlabel("Δt (meses)")
    ax.set_ylabel("Resíduos (%)")
    ax.legend(loc='lower right', fontsize=8)

    save_figure(fig, output_dir, "FIG_C1_diffmed_vs_resid_linear")

def figs_volume_triptych(df: pd.DataFrame, output_dir: Path,
                         filename: str = "FIG_B_triptych_volume",
                         title: str = "Validação por talhão (Flag=OK; Diffmed∈{-1,0,1})"):
    """
    Um único painel com 3 subplots (1x3), lado a lado:
      (1) Estimado × Observado (1:1 + regressão + métricas)
      (2) Resíduo (%) × Observado (LOWESS + faixas)
      (3) Frequência/Histograma do resíduo (%)

    Espera df já filtrado (Flag OK e Diffmed ∈ {-1,0,1}).
    """
    # ---------- dados ----------
    obs = pd.to_numeric(df[COL_OBSERVADO], errors="coerce").values
    est = pd.to_numeric(df[COL_PREDITO], errors="coerce").values
    mask = np.isfinite(obs) & np.isfinite(est) & (obs > 1e-6)
    obs = obs[mask]
    est = est[mask]
    resid_pct = (est - obs) / obs * 100
    resid_pct = resid_pct[np.isfinite(resid_pct)]

    metrics = calculate_metrics(obs, est)
    fit = linear_fit_with_stats(obs, est)

    # ---------- figura ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes

    # ==========================
    # (1) Estimado x Observado
    # ==========================
    ax1.scatter(obs, est, c='#555555', s=35, alpha=0.6,
                edgecolors="white", linewidth=0.5)

    lim_max = max(np.nanmax(obs), np.nanmax(est)) * 1.05 if len(obs) else 1
    ax1.plot([0, lim_max], [0, lim_max], c="black", linestyle="--", linewidth=1.5, label="Linha 1:1")

    if fit.get("ok"):
        x_line = np.linspace(np.nanmin(obs), np.nanmax(obs), 100)
        y_line = fit["slope"] * x_line + fit["intercept"]
        ax1.plot(x_line, y_line, "-", color='#ff7f0e', linewidth=1.5,
                 label=f"y = {fit['slope']:.2f}x{fit['intercept']:+.2f} | R² = {fit['r2']:.4f})".replace('.', ','))

    """textstr = (f"N = {metrics['N']}\n"
               f"R² = {metrics['R2']:.4f}\n"
               f"RMSE = {metrics['RMSE']:.2f} ({metrics['RMSE_pct']:.1f}%)\n"
               f"MAE = {metrics['MAE']:.2f} ({metrics['MAE_pct']:.1f}%)\n"
               f"Bias = {metrics['Bias']:+.2f} ({metrics['Bias_pct']:+.1f}%)")
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))"""

    ax1.set_xlabel("VTCC IFPC (m³/ha)")
    ax1.set_ylabel("VTCC Wall-to-Wall Médio (m³/ha)")
    ax1.set_xlim(0, lim_max)
    ax1.set_ylim(0, lim_max)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_title('(a)', fontsize=12, loc='left')

    # ==========================
    # (2) Resíduo (%) x Estimado
    # ==========================
    ax2.scatter(est, resid_pct, c='#555555', s=35, alpha=0.55,
                edgecolors="white", linewidth=0.5)
    ax2.set_ylim(-100, 100)

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.5)
    """ax2.axhline(20, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.axhline(-20, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.fill_between([np.nanmin(obs)*0.95, np.nanmax(obs)*1.05], -20, 20,
                     alpha=0.08, color="green")"""

    ax2.set_xlabel("VTCC Wall-to-Wall Médio (m³/ha)")
    ax2.set_ylabel("Resíduo Relativo (%)")
    ax2.set_title('(b)', fontsize=12, loc='left')

    # ==========================
    # (3) Frequência dos resíduos (em %)
    # ==========================
    ax3 = axes[2]

    # bins centralizados (resíduos em %)
    bin_edges = np.arange(-105, 115, 10)  # bordas
    bin_centers = np.arange(-100, 110, 10)  # centros

    counts, _ = np.histogram(resid_pct, bins=bin_edges)
    percentages = counts / len(resid_pct) * 100  # frequência em %

    bars = ax3.bar(
        bin_centers, percentages, width=8, alpha=0.7,
        color="#555555", edgecolor="white", linewidth=0.8
    )

    # linha vertical em x=0 com "altura" limitada (fração do eixo Y)
    # (ajuste o ymax conforme necessário; aqui deixo 0.23 = 23% do eixo)
    ax3.axvline(x=0, ymax=0.96, color="black", linestyle="--", linewidth=1.5)

    # rótulos acima das barras
    for bar, pct in zip(bars, percentages):
        if pct > 0:
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}".replace('.', ','),
                ha="center", va="bottom", fontsize=7
            )

    ax3.set_xlabel("Resíduos (%)")
    ax3.set_ylabel("Frequência (%)")
    ax3.set_xlim(-110, 110)
    ax3.set_xticks(np.arange(-100, 110, 20))
    ax3.grid(linestyle="--", alpha=0.3)
    ax3.set_title("(c)", fontsize=12, loc="left")

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)


    plt.tight_layout()

    # salvar
    save_figure(fig, Path(output_dir), filename)


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_validation_v2(
    input_file: str = INPUT_FILE,
    sheet_name: str = SHEET_NAME,
    output_dir: Path = OUTPUT_DIR,
):
    print("=" * 70)
    print("VALIDAÇÃO DO MODELO - PIPELINE v2")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Arquivo: {input_file}")
    print(f"Saída:   {output_dir}")
    print()

    df0 = pd.read_excel(input_file, sheet_name=sheet_name)
    print(f"Linhas (bruto): {len(df0)}")
    print(f"Colunas: {list(df0.columns)}")
    print()

    # Checagens mínimas
    required = [COL_AREA_CAD, COL_AREA_RASTER, COL_OBSERVADO, COL_PREDITO]
    missing = [c for c in required if c not in df0.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias não encontradas: {missing}")
    if COL_FLAG not in df0.columns:
        raise ValueError(f"Coluna '{COL_FLAG}' não encontrada. Ajuste COL_FLAG.")

    # Normaliza flag
    df0["Flag_std"] = df0[COL_FLAG].astype(str).str.upper().str.strip()

    # =========================================================
    # BLOCO A: ÁREA (filtra pelo campo flag - não vazio)
    # =========================================================
    df_area = df0.copy()
    df_area = df_area[df_area["Flag_std"] == FLAG_OK]
    print(f"Bloco A (ÁREA): {len(df_area)} linhas")

    figA1_area_cad_vs_area_raster(df_area, Path(output_dir))
    figA2_resid_area_pct_vs_area_raster(df_area, Path(output_dir))

    # =========================================================
    # BLOCO B: VOLUME (Flag OK + Diffmed ∈ {-1,0,+1})
    # =========================================================
    df_vol = df0.copy()
    df_vol = df_vol[df_vol["Flag_std"] == FLAG_OK]

    if COL_DELTA_T in df_vol.columns:
        df_vol[COL_DELTA_T] = pd.to_numeric(df_vol[COL_DELTA_T], errors="coerce")
        df_vol = df_vol[df_vol[COL_DELTA_T].isin([-1, 0, 1])]
    else:
        raise ValueError(f"Coluna {COL_DELTA_T} não encontrada para filtro Diffmed.")

    # Remove inválidos e evita explosão por obs=0
    df_vol[COL_OBSERVADO] = pd.to_numeric(df_vol[COL_OBSERVADO], errors="coerce")
    df_vol[COL_PREDITO] = pd.to_numeric(df_vol[COL_PREDITO], errors="coerce")
    df_vol = df_vol.dropna(subset=[COL_OBSERVADO, COL_PREDITO])
    df_vol = df_vol[df_vol[COL_OBSERVADO] > 1e-6]

    print(f"Bloco B (VOLUME, OK & Diffmed∈{{-1,0,1}}): {len(df_vol)} linhas")

    # Métricas globais do bloco B
    metrics_B = calculate_metrics(df_vol[COL_OBSERVADO].values, df_vol[COL_PREDITO].values)
    print("  Métricas (Bloco B):")
    print(f"    N:     {metrics_B['N']}")
    print(f"    R²:    {metrics_B['R2']:.4f}")
    print(f"    RMSE:  {metrics_B['RMSE']:.2f} m³/ha ({metrics_B['RMSE_pct']:.1f}%)")
    print(f"    MAE:   {metrics_B['MAE']:.2f} m³/ha ({metrics_B['MAE_pct']:.1f}%)")
    print(f"    Bias:  {metrics_B['Bias']:+.2f} m³/ha ({metrics_B['Bias_pct']:+.1f}%)")
    print()

    figs_volume_triptych(df_vol, output_dir)
    figB1_estimado_vs_observado(df_vol, Path(output_dir))
    figB2_estimado_vs_residuo(df_vol, Path(output_dir))
    figB3_frequencia_residuos(df_vol, Path(output_dir))

    # Violinos mantendo filtro do bloco B
    violin_resid_by_group(
        df_vol, Path(output_dir),
        group_col=COL_REGIME,
        filename="FIG_B4_violin_resid_por_regime",
        title="Resíduo (%) por Regime (violino)"
    )
    violin_resid_by_group(
        df_vol, Path(output_dir),
        group_col=COL_REGIONAL,
        filename="FIG_B5_violin_resid_por_regional",
        title="Resíduo (%) por Regional (violino)"
    )
    violin_resid_regional_x_regime(df_vol, Path(output_dir))

    # =========================================================
    # BLOCO C: Diffmed × resíduo (Flag OK, regressão linear)
    # =========================================================
    df_dt = df0.copy()
    df_dt = df_dt[df_dt["Flag_std"] == FLAG_OK]
    print(f"Bloco C (DIFFMED, OK): {len(df_dt)} linhas")
    # 2) métricas TEMPORAIS (diffmed vs resíduo %) — isso bate com o FIG_C1
    metrics_C_dt = calculate_temporal_regression_metrics(df_dt)
    print("  Métricas TEMPORAIS (Diffmed × Resíduo%):")
    print(f"    N:     {metrics_C_dt['N']}")
    print(f"    y = {metrics_C_dt['slope']:.2f}x {metrics_C_dt['intercept']:+.2f}")
    print(f"    r = {metrics_C_dt['r']:.3f} (p={metrics_C_dt['p']:.3f})")
    print(f"    R² = {metrics_C_dt['R2']:.3f}")
    print()

    figC1_diffmed_vs_resid_linear(df_dt, Path(output_dir))


    print()
    print("=" * 70)
    print("VALIDAÇÃO v2 CONCLUÍDA")
    print(f"Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Figuras salvas em: {output_dir}")
    print("=" * 70)

    return {
        "df_area": df_area,
        "df_vol": df_vol,
        "df_dt": df_dt,
        "metrics_volume": metrics_B
    }


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    run_validation_v2()
