from Auxiliary.ClipRasterByShape import clip_raster_by_shape
from Auxiliary.RasterStats import process_rasters
from Auxiliary.CalculoCubMean import generate_cubic_mean_raster
from Auxiliary.CalculoStdDev import generate_stddev_raster
from Auxiliary.MergeTiff import merge_tiffs, merge_batch
from Auxiliary.AsciiToTiff import batch_convert
import os
import glob
"""
indicator_dir = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\SHAPEFILE\Book_2022_Base_2021.shp"

for folder in glob.glob(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\*"):
    print(f"Processing folder: {folder}")
    if os.path.basename(folder) in ["FEATURES_CAD", "SHAPEFILE"]:
        continue

    predict_folder = os.path.join(folder, "10predict")

    # Find raster files in the predict folder
    raster_files = glob.glob(os.path.join(predict_folder, "*.p90_cub_std_volume_incerteza.tif"))

    if not raster_files:
        print(f"  No raster files found in {predict_folder}")
        raster_files = glob.glob(os.path.join(predict_folder, "*p90_cub_std_volume_incerteza.tif"))

    for raster_path in raster_files:
        print(f"  Processing raster: {os.path.basename(raster_path)}")

        output_folder = os.path.join(predict_folder, "P90_CUB_STD_CLIPPED_INCERTEZA")

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

    cliped = os.path.join(folder, "10predict", "P90_CUB_STD_CLIPPED")
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
"""import pandas as pd

df = []
for folder in glob.glob(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\*"):
    print(f"Processing folder: {folder}")
    if os.path.basename(folder) in ["FEATURES_CAD", "SHAPEFILE"]:
        continue
    main = os.path.join(folder, "10predict", "P90_CUB_STD_CLIPPED_INCERTEZA")
    try:
        _ = pd.read_excel(main+"\incerteza.xlsx")
        _ = _[_["VAL"] != -9999]
        df.append(_)
    except FileNotFoundError:
        print(f"  Arquivo nao encontrado: {main}")
        continue

final_df = pd.concat(df, ignore_index=True)

MAX_ROWS = 1_048_576
output = r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Results\ALL_RASTER_incerteza.xlsx"

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    for i in range(0, len(final_df), MAX_ROWS):
        chunk = final_df.iloc[i:i+MAX_ROWS]
        sheet_name = f"parte_{i//MAX_ROWS + 1}"
        chunk.to_excel(writer, sheet_name=sheet_name, index=False)

print("Excel salvo em múltiplas abas.")
"""

for folder in glob.glob(r"G:\PycharmProjects\Mestrado\Forecast\Predictive\Metrics\*"):
    if os.path.basename(folder) in ["FEATURES_CAD", "SHAPEFILE"]:
        continue
    print(f"Processing folder: {folder}")
    chm = os.path.join(folder, "07chm")
    dtm = os.path.join(folder, "04dtm")
    try:
        batch_convert(
            input_folder=chm,
            output_folder=chm
        )
        batch_convert(
            input_folder=dtm,
            output_folder=dtm
        )
        chm_files = glob.glob(os.path.join(chm, "*.tif"))
        dtm_files = glob.glob(os.path.join(dtm, "*.tif"))
        merge_tiffs(
            input_files=chm_files,
            output_file=os.path.join(chm, os.path.basename(folder)+"_CHM.tif"),
        )
        merge_tiffs(
            input_files=dtm_files,
            output_file=os.path.join(dtm, os.path.basename(folder)+"_DTM.tif"),
        )
    except Exception as e:
        print(e)