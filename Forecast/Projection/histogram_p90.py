#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Histograma de Frequencia de P90 (Percentil 90 de Altura)

Calcula o P90 da altura normalizada (z - DTM) em celulas de grade
e gera histogramas comparativos para cada epoca.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
import laspy

# =============== CONFIG ===================================
DATA_DIR = Path(r"G:/PycharmProjects/Mestrado")
EPOCAS = [
    {"idade": 2.0, "ano": 2019, "las": DATA_DIR / "RDBOBA00456P17-267_759_denoised_thin_norm_2019.laz"},
    {"idade": 5.0, "ano": 2022, "las": DATA_DIR / "RDBOBA00456P17-267_759_denoised_thin_norm_2022.laz"},
    {"idade": 8.0, "ano": 2025, "las": DATA_DIR / "RDBOBA00456P17-267_759_denoised_thin_norm_2025.laz"},
    {"idade": 20.0, "ano": 2037, "las": DATA_DIR / "out_ode/projecao_ode_idade_20.0a.las"},
]
OUT_DIR = DATA_DIR / "out_ode"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tamanho da celula para calcular P90 (metros)
CELL_SIZE = 5.0


# =============== UTILS ====================================

class DTMRef:
    """DTM por kNN IDW sobre ground (class=2)."""

    def __init__(self, xg, yg, zg, k=3):
        self.pts = np.c_[xg, yg]
        self.z = zg
        self.kdt = cKDTree(self.pts)
        self.k = k

    def z_at(self, x, y):
        q = np.c_[x, y]
        d, idx = self.kdt.query(q, k=min(self.k, len(self.z)))
        if np.ndim(idx) == 0:
            return float(self.z[idx])
        zz = self.z[idx]
        w = 1.0 / (d + 1e-6)
        w /= w.sum(axis=-1, keepdims=True)
        return (zz * w).sum(axis=-1)


def load_las_points(las_path):
    """Carrega pontos de arquivo LAS/LAZ."""
    p = str(las_path)

    def to_np(a, dtype=None):
        arr = np.asarray(a)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.ascontiguousarray(arr)

    try:
        if p.lower().endswith(".laz"):
            try:
                with laspy.open(p, laz_backend=laspy.LazBackend.LazrsParallel) as f:
                    las = f.read()
            except Exception:
                with laspy.open(p, laz_backend=laspy.LazBackend.Lazrs) as f:
                    las = f.read()
        else:
            las = laspy.read(p)
    except Exception as e:
        raise SystemExit(f"[ERRO] Falha ao ler LAS: {p}\n{e}")

    return {
        "x": to_np(las.x, np.float64),
        "y": to_np(las.y, np.float64),
        "z": to_np(las.z, np.float64),
        "classification": to_np(getattr(las, "classification", np.ones_like(las.x)), np.uint8),
    }


def compute_p90_grid(x, y, zprime, cell_size=CELL_SIZE, min_pts=5):
    """
    Calcula P90 da altura normalizada em uma grade regular.

    Retorna array de valores P90 por celula.
    """
    # Definir limites da grade
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Numero de celulas
    nx = int(np.ceil((xmax - xmin) / cell_size))
    ny = int(np.ceil((ymax - ymin) / cell_size))

    # Indices de celula para cada ponto
    ix = np.clip(((x - xmin) / cell_size).astype(int), 0, nx - 1)
    iy = np.clip(((y - ymin) / cell_size).astype(int), 0, ny - 1)
    cell_idx = iy * nx + ix

    # Calcular P90 por celula
    p90_values = []
    unique_cells = np.unique(cell_idx)

    for c in unique_cells:
        mask = cell_idx == c
        if mask.sum() >= min_pts:
            p90 = np.percentile(zprime[mask], 90)
            p90_values.append(p90)

    return np.array(p90_values)


def process_epoch(epoca, dtm=None):
    """Processa uma epoca e retorna valores P90."""
    print(f"  Processando: idade {epoca['idade']} anos ({epoca['ano']})")

    P = load_las_points(epoca["las"])
    is_ground = (P["classification"] == 2)

    # Construir DTM se nao fornecido
    if dtm is None:
        if not is_ground.any():
            print(f"    [WARN] Sem pontos ground, usando z minimo como referencia")
            zref = P["z"].min()
            zprime = P["z"] - zref
        else:
            Gx, Gy, Gz = P["x"][is_ground], P["y"][is_ground], P["z"][is_ground]
            dtm = DTMRef(Gx, Gy, Gz, k=3)
            zprime = P["z"] - dtm.z_at(P["x"], P["y"])
    else:
        zprime = P["z"] - dtm.z_at(P["x"], P["y"])

    # Filtrar apenas vegetacao (acima do solo)
    veg_mask = (~is_ground) & (zprime > 0.3)
    x_veg = P["x"][veg_mask]
    y_veg = P["y"][veg_mask]
    zp_veg = zprime[veg_mask]

    print(f"    Pontos vegetacao: {veg_mask.sum()}")

    # Calcular P90 em grade
    p90_vals = compute_p90_grid(x_veg, y_veg, zp_veg, cell_size=CELL_SIZE)
    print(f"    Celulas com P90: {len(p90_vals)}")
    print(f"    P90 min/med/max: {p90_vals.min():.2f} / {np.median(p90_vals):.2f} / {p90_vals.max():.2f} m")

    return p90_vals, dtm


def plot_histograms_combined(data_dict, output_path):
    """Plota histogramas sobrepostos para todas as epocas."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    # Determinar bins comuns
    all_data = np.concatenate(list(data_dict.values()))
    bins = np.linspace(0, np.percentile(all_data, 99), 30)

    for i, (label, data) in enumerate(data_dict.items()):
        ax.hist(data, bins=bins, alpha=0.5, label=label,
                color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('P90 de Altura (m)', fontsize=12)
    ax.set_ylabel('Frequencia (celulas)', fontsize=12)
    ax.set_title(f'Distribuicao de P90 por Celula ({CELL_SIZE}m x {CELL_SIZE}m)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Adicionar estatisticas
    stats_text = []
    for label, data in data_dict.items():
        stats_text.append(f"{label}: med={np.median(data):.1f}m, n={len(data)}")
    ax.text(0.02, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvo: {output_path}")


def plot_histograms_individual(data_dict, output_path):
    """Plota histogramas individuais lado a lado."""
    n = len(data_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharey=True)

    if n == 1:
        axes = [axes]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    # Determinar bins comuns
    all_data = np.concatenate(list(data_dict.values()))
    bins = np.linspace(0, np.percentile(all_data, 99), 25)

    for i, (label, data) in enumerate(data_dict.items()):
        ax = axes[i]
        ax.hist(data, bins=bins, alpha=0.7, color=colors[i % len(colors)],
                edgecolor='white', linewidth=0.5)
        ax.set_xlabel('P90 (m)', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Estatisticas
        med = np.median(data)
        mean = np.mean(data)
        ax.axvline(med, color='red', linestyle='--', linewidth=2, label=f'Mediana: {med:.1f}m')
        ax.axvline(mean, color='orange', linestyle=':', linewidth=2, label=f'Media: {mean:.1f}m')
        ax.legend(loc='upper right', fontsize=9)

    axes[0].set_ylabel('Frequencia (celulas)', fontsize=11)

    plt.suptitle(f'P90 de Altura por Epoca (celulas {CELL_SIZE}m)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvo: {output_path}")


def plot_boxplot(data_dict, output_path):
    """Plota boxplot comparativo."""
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = list(data_dict.keys())
    data = list(data_dict.values())

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    ax.set_ylabel('P90 de Altura (m)', fontsize=12)
    ax.set_title('Comparacao de P90 entre Epocas', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Adicionar valores de mediana
    for i, (label, d) in enumerate(data_dict.items()):
        med = np.median(d)
        ax.annotate(f'{med:.1f}m', xy=(i+1, med), xytext=(i+1.2, med),
                   fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvo: {output_path}")


def main():
    print("=" * 60)
    print("HISTOGRAMA DE P90 - ANALISE MULTI-TEMPORAL")
    print("=" * 60)

    # Validar epocas
    epocas_validas = [e for e in EPOCAS if Path(e["las"]).exists()]
    if not epocas_validas:
        raise SystemExit("[ERRO] Nenhum arquivo .laz encontrado!")

    print(f"\n[INFO] Epocas encontradas: {len(epocas_validas)}")
    print(f"[INFO] Tamanho da celula: {CELL_SIZE} m")

    # Usar DTM da ultima epoca como referencia
    print("\n[1/2] Construindo DTM de referencia...")
    ref_epoca = epocas_validas[-1]
    P_ref = load_las_points(ref_epoca["las"])
    is_ground = (P_ref["classification"] == 2)
    if not is_ground.any():
        raise SystemExit("[ERRO] Epoca de referencia sem pontos ground!")

    dtm = DTMRef(P_ref["x"][is_ground], P_ref["y"][is_ground], P_ref["z"][is_ground], k=3)
    print(f"  DTM construido com {is_ground.sum()} pontos ground")

    # Processar todas as epocas
    print("\n[2/2] Calculando P90 por epoca...")
    data_dict = {}

    for epoca in epocas_validas:
        label = f"Idade {epoca['idade']:.0f}a ({epoca['ano']})"
        p90_vals, _ = process_epoch(epoca, dtm=dtm)
        data_dict[label] = p90_vals

    # Gerar graficos
    print("\n[INFO] Gerando graficos...")

    plot_histograms_combined(
        data_dict,
        OUT_DIR / "histogram_p90_combined.png"
    )

    plot_histograms_individual(
        data_dict,
        OUT_DIR / "histogram_p90_individual.png"
    )

    plot_boxplot(
        data_dict,
        OUT_DIR / "boxplot_p90.png"
    )

    # Tabela de estatisticas
    print("\n" + "=" * 60)
    print("ESTATISTICAS DE P90")
    print("=" * 60)
    print(f"{'Epoca':<25} {'N':<8} {'Min':<8} {'Med':<8} {'Mean':<8} {'Max':<8} {'Std':<8}")
    print("-" * 73)

    for label, data in data_dict.items():
        print(f"{label:<25} {len(data):<8} {data.min():<8.2f} {np.median(data):<8.2f} "
              f"{data.mean():<8.2f} {data.max():<8.2f} {data.std():<8.2f}")

    print("\n[OK] Concluido!")


if __name__ == "__main__":
    main()