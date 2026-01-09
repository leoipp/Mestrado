from Auxiliary.ClipRasterByShape import clip_raster_by_shape
from Auxiliary.RasterStats import process_rasters
from Auxiliary.CalculoCubMean import generate_cubic_mean_raster
from Auxiliary.CalculoStdDev import generate_stddev_raster
from Auxiliary.MergeTiff import merge_tiffs, merge_batch
import os
import glob

"""indicator_dir = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\SHAPEFILE\Book_2022_Base_2021.shp"

for folder in glob.glob(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\*"):
    print(f"Processing folder: {folder}")
    if os.path.basename(folder) in ["FEATURES_CAD", "SHAPEFILE"]:
        continue

    predict_folder = os.path.join(folder, "10predict")

    # Find raster files in the predict folder
    raster_files = glob.glob(os.path.join(predict_folder, "*.LAS_volume_estimado.tif"))

    if not raster_files:
        print(f"  No raster files found in {predict_folder}")
        raster_files = glob.glob(os.path.join(predict_folder, "*NP_volume_estimado.tif"))

    for raster_path in raster_files:
        print(f"  Processing raster: {os.path.basename(raster_path)}")

        output_folder = os.path.join(predict_folder, "MAX_CUB_STD_CLIPPED")

        clip_raster_by_shape(
            raster_path=raster_path,
            shapefile_path=indicator_dir,
            output_folder=output_folder,
            name_column="REF_ID",
            nodata=None,
            crop=True,
            all_touched=False,
            compress="lzw",
            prefix="",
            suffix="",
        )"""


"""for folder in glob.glob(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\*"):
    print(f"Processing folder: {folder}")
    if os.path.basename(folder) in ["FEATURES_CAD", "SHAPEFILE"]:
        continue

    cliped = os.path.join(folder, "10predict", "P90_CUB_CLIPPED")
    process_rasters(
        input_path=cliped,
        output_excel=cliped+r"\raster_stats.xlsx",
    )"""


"""
for folder in glob.glob(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\*"):
    print(f"Processing folder: {folder}")
    if os.path.basename(folder) in ["FEATURES_CAD", "SHAPEFILE"]:
        continue

    metrics_folder = os.path.join(folder, "09metrics")

    # Procura arquivo de pontos normalizados (elev_first.txt ou similar)
    point_files = glob.glob(os.path.join(metrics_folder, "*.txt"))
    if not point_files:
        point_files = glob.glob(os.path.join(metrics_folder, "*.xyz"))
    if not point_files:
        point_files = glob.glob(os.path.join(metrics_folder, "*.las"))
    if not point_files:
        point_files = glob.glob(os.path.join(metrics_folder, "*.laz"))

    if not point_files:
        print(f"  No point files found in {metrics_folder}")
        continue

    for point_file in point_files:
        print(f"  Processing: {os.path.basename(point_file)}")

        generate_stddev_raster(
            input_file=point_file,
            output_path=metrics_folder,
            cell_size=17.0,
            epsg=31983,
            nodata_value=-9999.0,
            output_format='tif',
            min_points=1
        )


# =============================================================================
# MERGE TIFFS - Combinar rasters de múltiplas pastas em mosaicos
# =============================================================================

# Opção 1: Merge de todos os tiffs de uma métrica específica (ex: todos os P90)
output_mosaic_folder = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\MOSAICS"

# Coleta todos os rasters de uma métrica específica de todas as pastas
metric_patterns = ["*_P90.tif", "*_P60.tif", "*_maximum.tif", "*_cubmean.tif", "*_stddev.tif"]

for pattern in metric_patterns:
    print(f"\n{'='*60}")
    print(f"Coletando rasters: {pattern}")
    print('='*60)

    all_rasters = []
    for folder in glob.glob(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\*"):
        if os.path.basename(folder) in ["FEATURES_CAD", "SHAPEFILE", "MOSAICS"]:
            continue

        metrics_folder = os.path.join(folder, "09metrics")
        rasters = glob.glob(os.path.join(metrics_folder, pattern))
        all_rasters.extend(rasters)

    if not all_rasters:
        print(f"  Nenhum raster encontrado para {pattern}")
        continue

    print(f"  Encontrados: {len(all_rasters)} rasters")

    # Nome do mosaico baseado no pattern
    mosaic_name = pattern.replace("*_", "").replace(".tif", "_mosaic.tif")
    output_file = os.path.join(output_mosaic_folder, mosaic_name)

    merge_tiffs(
        input_files=all_rasters,
        output_file=output_file,
        method="first",
        nodata=-9999.0,
        compress="lzw",
    )
"""

"""
# Opção 2: Merge batch automático por sufixo (detecta patterns automaticamente)
input_tiles_folder = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\TILES"
output_mosaic_folder = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\MOSAICS"

merge_batch(
    input_folder=input_tiles_folder,
    output_folder=output_mosaic_folder,
    method="first",
    nodata=-9999.0,
    compress="lzw",
)
"""

"""
# Opção 3: Merge simples de uma pasta específica
merge_tiffs(
    input_files=r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\FOLDER\09metrics",
    output_file=r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\MOSAICS\merged.tif",
    pattern="*.tif",
    method="first",
    nodata=-9999.0,
    compress="lzw",
)
"""
import pandas as pd

df = []
for folder in glob.glob(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\*"):
    print(f"Processing folder: {folder}")
    if os.path.basename(folder) in ["FEATURES_CAD", "SHAPEFILE"]:
        continue
    main = os.path.join(folder, "10predict", "MAX_CUB_STD_CLIPPED")
    _ = pd.read_excel(main+r"\raster_stats.xlsx")
    df.append(_)
final_df = pd.concat(df, ignore_index=True)
final_df.to_excel(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results\ALL_RASTER_STATS_MAX_CUB_STD_CLIPPED.xlsx",
                  index=False)