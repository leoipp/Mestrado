import rasterio
import numpy as np
import matplotlib.pyplot as plt

# === Caminho do raster (substitua pelo seu arquivo .tif) ===
tif_path = r"G:\PycharmProjects\Mestrado\Data\Projection\2019-2022\delta\delta_RDBOBA00412P14_17836.tif"

band_idx = 1  # primeira banda = 1

# ===== Leitura do raster =====
with rasterio.open(tif_path) as src:
    data = src.read(band_idx)
    nodata = src.nodata

# Tratar NoData -> NaN
if nodata is not None:
    data = np.where(data == nodata, np.nan, data)

# Valores válidos (1D)
vals_all = data[~np.isnan(data)].astype(float).ravel()

# ===== Quantis e limites (Tukey 1.5*IQR) =====
q1, q2, q3 = np.percentile(vals_all, [25, 50, 75])
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

# Filtrados (entre limites)
vals_filt = vals_all[(vals_all >= lower_fence) & (vals_all <= upper_fence)]

# ===== Resumo no console =====
removed = len(vals_all) - len(vals_filt)
pct_removed = 100 * removed / len(vals_all) if len(vals_all) else 0.0
print(f"Total de pixels válidos: {len(vals_all):,}")
print(f"Q1={q1:.6g} | Mediana={q2:.6g} | Q3={q3:.6g} | IQR={iqr:.6g}")
print(f"Limite inferior: {lower_fence:.6g} | Limite superior: {upper_fence:.6g}")
print(f"Removidos por Tukey (1.5*IQR): {removed:,} ({pct_removed:.2f}%)")
if len(vals_filt) == 0:
    print("Aviso: todos os valores foram removidos pelos limites — verifique o raster.")

# ===== Boxplots lado a lado =====
fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

# Boxplot com todos os valores
axes[0].boxplot(vals_all, vert=True, showfliers=True)
axes[0].set_title("Boxplot — Todos os valores")
axes[0].set_ylabel("Valor dos pixels")

# Boxplot após remoção por limites
axes[1].boxplot(vals_filt, vert=True, showfliers=True)
axes[1].set_title("Boxplot — Após remover outliers (Tukey)")

plt.show()
