"""
ClipRasterByShape.py - Clipa raster por features de um shapefile

Clipa um raster usando cada feature de um shapefile, nomeando os outputs
com base em uma coluna específica da tabela de atributos.

Exemplo:
    Se o shapefile tem 3 features com coluna "NOME":
        - Fazenda_A
        - Fazenda_B
        - Fazenda_C

    Serão criados:
        - Fazenda_A.tif
        - Fazenda_B.tif
        - Fazenda_C.tif

Dependências:
    - rasterio
    - geopandas
    - numpy

Autor: Leonardo Ippolito Rodrigues
Ano: 2026
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime
import warnings

try:
    import rasterio
    from rasterio.mask import mask
    from rasterio.features import geometry_mask
    import geopandas as gpd
    from shapely.geometry import mapping
except ImportError as e:
    raise ImportError(f"Dependência não encontrada: {e}. Execute: pip install rasterio geopandas")


def sanitize_filename(name: str) -> str:
    """Remove caracteres inválidos para nomes de arquivo."""
    invalid_chars = '<>:"/\\|?*'
    result = str(name)
    for char in invalid_chars:
        result = result.replace(char, '_')
    # Remove espaços extras e pontos no final
    result = result.strip().rstrip('.')
    # Substitui espaços por underscore
    result = result.replace(' ', '_')
    return result


def clip_raster_by_shape(
    raster_path: str,
    shapefile_path: str,
    output_folder: str,
    name_column: str,
    nodata: Optional[float] = None,
    crop: bool = True,
    all_touched: bool = False,
    compress: str = "lzw",
    prefix: str = "",
    suffix: str = "",
) -> List[str]:
    """
    Clipa um raster usando cada feature de um shapefile.

    Args:
        raster_path: Caminho para o raster de entrada
        shapefile_path: Caminho para o shapefile
        output_folder: Pasta de saída para os rasters clipados
        name_column: Nome da coluna do shapefile que será usada para nomear os outputs
        nodata: Valor nodata para o output (se None, usa do raster original)
        crop: Se True, recorta o raster para o extent da feature
        all_touched: Se True, inclui pixels que tocam a borda do polígono
        compress: Compressão do output (lzw, deflate, none)
        prefix: Prefixo para o nome do arquivo de saída
        suffix: Sufixo para o nome do arquivo de saída (antes da extensão)

    Returns:
        Lista de caminhos dos arquivos criados
    """
    print("=" * 60)
    print("CLIP RASTER BY SHAPEFILE")
    print("=" * 60)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------------------------------------------------------------
    # 1. VALIDAÇÃO DAS ENTRADAS
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Validando entradas...")

    raster_path = Path(raster_path)
    shapefile_path = Path(shapefile_path)
    output_folder = Path(output_folder)

    if not raster_path.exists():
        raise FileNotFoundError(f"Raster não encontrado: {raster_path}")
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile não encontrado: {shapefile_path}")

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"  Raster: {raster_path.name}")
    print(f"  Shapefile: {shapefile_path.name}")
    print(f"  Output: {output_folder}")

    # -------------------------------------------------------------------------
    # 2. LEITURA DO SHAPEFILE
    # -------------------------------------------------------------------------
    print(f"\n[2/4] Lendo shapefile...")

    gdf = gpd.read_file(shapefile_path)
    n_features = len(gdf)

    print(f"  Features encontradas: {n_features}")
    print(f"  Colunas disponíveis: {list(gdf.columns)}")

    if name_column not in gdf.columns:
        raise ValueError(f"Coluna '{name_column}' não encontrada. Disponíveis: {list(gdf.columns)}")

    print(f"  Coluna para nomes: {name_column}")
    print(f"  Valores: {gdf[name_column].tolist()}")

    # -------------------------------------------------------------------------
    # 3. LEITURA DO RASTER
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Lendo raster...")

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        raster_nodata = src.nodata if nodata is None else nodata

        print(f"  CRS: {raster_crs}")
        print(f"  Bounds: {raster_bounds}")
        print(f"  Shape: {src.height} x {src.width}")
        print(f"  Bandas: {src.count}")
        print(f"  NoData: {raster_nodata}")

        # Reprojetar shapefile se necessário
        if raster_crs is None:
            print(f"\n  AVISO: Raster sem CRS definido. Assumindo mesmo CRS do shapefile.")
        elif gdf.crs != raster_crs:
            print(f"\n  Reprojetando shapefile de {gdf.crs} para {raster_crs}...")
            gdf = gdf.to_crs(raster_crs)

        # -----------------------------------------------------------------
        # 4. CLIPAGEM POR FEATURE
        # -----------------------------------------------------------------
        print(f"\n[4/4] Clipando raster por feature...")

        results = []
        errors = []

        for idx, row in gdf.iterrows():
            feature_name = str(row[name_column])
            safe_name = sanitize_filename(feature_name)
            output_name = f"{prefix}{safe_name}{suffix}.tif"
            output_path = output_folder / output_name

            print(f"\n  [{idx + 1}/{n_features}] {feature_name}")

            try:
                # Geometria da feature
                geom = row.geometry
                if geom is None or geom.is_empty:
                    print(f"    AVISO: Geometria vazia, pulando...")
                    continue

                # Clipa o raster
                out_image, out_transform = mask(
                    src,
                    [mapping(geom)],
                    crop=crop,
                    nodata=raster_nodata,
                    all_touched=all_touched,
                )

                # Metadados do output
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": raster_nodata,
                    "compress": compress if compress != "none" else None,
                })

                # Remove compress None
                if out_meta.get("compress") is None:
                    out_meta.pop("compress", None)

                # Salva
                with rasterio.open(output_path, "w", **out_meta) as dst:
                    dst.write(out_image)

                # Estatísticas
                if raster_nodata is not None:
                    valid = out_image[out_image != raster_nodata]
                else:
                    valid = out_image[np.isfinite(out_image)]

                if valid.size > 0:
                    print(f"    Shape: {out_image.shape[1]} x {out_image.shape[2]}")
                    print(f"    Pixels válidos: {valid.size:,}")
                    print(f"    Min/Max: {valid.min():.4f} / {valid.max():.4f}")
                else:
                    print(f"    AVISO: Nenhum pixel válido!")

                print(f"    Salvo: {output_path.name}")
                results.append(str(output_path))

            except Exception as e:
                print(f"    ERRO: {e}")
                errors.append((feature_name, str(e)))

    # -------------------------------------------------------------------------
    # RESUMO
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("CLIPAGEM CONCLUÍDA")
    print(f"{'=' * 60}")
    print(f"  Sucesso: {len(results)}/{n_features}")
    if errors:
        print(f"  Erros: {len(errors)}")
        for name, err in errors:
            print(f"    - {name}: {err}")
    print(f"Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


def clip_multiple_rasters(
    raster_paths: List[str],
    shapefile_path: str,
    output_folder: str,
    name_column: str,
    **kwargs
) -> dict:
    """
    Clipa múltiplos rasters pelo mesmo shapefile.

    Args:
        raster_paths: Lista de caminhos para os rasters
        shapefile_path: Caminho para o shapefile
        output_folder: Pasta base de saída
        name_column: Coluna do shapefile para nomear outputs
        **kwargs: Argumentos adicionais para clip_raster_by_shape

    Returns:
        Dicionário {raster_name: [lista de outputs]}
    """
    results = {}

    for raster_path in raster_paths:
        raster_name = Path(raster_path).stem
        raster_output = Path(output_folder) / raster_name

        print(f"\n{'#' * 60}")
        print(f"# RASTER: {raster_name}")
        print(f"{'#' * 60}")

        try:
            outputs = clip_raster_by_shape(
                raster_path=raster_path,
                shapefile_path=shapefile_path,
                output_folder=str(raster_output),
                name_column=name_column,
                **kwargs
            )
            results[raster_name] = outputs
        except Exception as e:
            print(f"ERRO no raster {raster_name}: {e}")
            results[raster_name] = []

    return results


# =============================================================================
# EXECUÇÃO VIA LINHA DE COMANDO
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clipa raster por features de um shapefile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Clip simples
  python ClipRasterByShape.py raster.tif areas.shp output/ -c NOME

  # Com prefixo e sufixo
  python ClipRasterByShape.py raster.tif areas.shp output/ -c ID --prefix "area_" --suffix "_clip"

  # Incluir pixels da borda
  python ClipRasterByShape.py raster.tif areas.shp output/ -c NOME --all-touched
        """
    )

    parser.add_argument("raster", help="Raster de entrada")
    parser.add_argument("shapefile", help="Shapefile com as features de corte")
    parser.add_argument("output", help="Pasta de saída")
    parser.add_argument("-c", "--column", required=True,
                        help="Coluna do shapefile para nomear os outputs")
    parser.add_argument("--nodata", type=float,
                        help="Valor nodata (default: usa do raster)")
    parser.add_argument("--no-crop", action="store_true",
                        help="Não recortar para o extent da feature")
    parser.add_argument("--all-touched", action="store_true",
                        help="Incluir pixels que tocam a borda do polígono")
    parser.add_argument("--compress", choices=["lzw", "deflate", "none"],
                        default="lzw", help="Compressão (default: lzw)")
    parser.add_argument("--prefix", default="",
                        help="Prefixo para nome do arquivo")
    parser.add_argument("--suffix", default="",
                        help="Sufixo para nome do arquivo")

    args = parser.parse_args()

    clip_raster_by_shape(
        raster_path=args.raster,
        shapefile_path=args.shapefile,
        output_folder=args.output,
        name_column=args.column,
        nodata=args.nodata,
        crop=(not args.no_crop),
        all_touched=args.all_touched,
        compress=args.compress,
        prefix=args.prefix,
        suffix=args.suffix,
    )
