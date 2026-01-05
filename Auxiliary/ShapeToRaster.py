"""
ShapeToRaster.py - Converte shapefile para raster GeoTIFF

Este script rasteriza um campo específico de um shapefile para um arquivo GeoTIFF,
permitindo configurar o tamanho do pixel (cellsize) e o sistema de referência (EPSG).

Dependências:
    - geopandas
    - rasterio
    - numpy

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
import numpy as np
from pathlib import Path


def shapefile_to_raster(
    shapefile_path: str,
    field: str,
    output_path: str,
    cellsize: float,
    epsg: int,
    nodata: float = -9999.0,
    dtype: str = 'float32'
) -> str:
    """
    Converte um shapefile para raster GeoTIFF baseado em um campo da tabela de atributos.

    Parameters
    ----------
    shapefile_path : str
        Caminho para o shapefile de entrada (.shp)
    field : str
        Nome do campo da tabela de atributos a ser rasterizado
    output_path : str
        Caminho para o arquivo GeoTIFF de saída (.tif)
    cellsize : float
        Tamanho do pixel em unidades do sistema de coordenadas (ex: metros)
    epsg : int
        Código EPSG do sistema de referência (ex: 31983 para SIRGAS 2000 UTM 23S)
    nodata : float, optional
        Valor para pixels sem dados (default: -9999.0)
    dtype : str, optional
        Tipo de dado do raster (default: 'float32')
        Opções: 'float32', 'float64', 'int16', 'int32', 'uint8', 'uint16'

    Returns
    -------
    str
        Caminho do arquivo raster gerado

    Raises
    ------
    FileNotFoundError
        Se o shapefile não existir
    ValueError
        Se o campo não existir na tabela de atributos

    Examples
    --------
    shapefile_to_raster(
    ...     shapefile_path='parcelas.shp',
    ...     field='VTCC',
    ...     output_path='volume.tif',
    ...     cellsize=10.0,
    ...     epsg=31983
    ... )
    """
    # Validação do arquivo de entrada
    shp_path = Path(shapefile_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile não encontrado: {shapefile_path}")

    # Lê o shapefile
    print(f"Lendo shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)

    # Valida o campo
    if field not in gdf.columns:
        available_fields = [c for c in gdf.columns if c != 'geometry']
        raise ValueError(
            f"Campo '{field}' não encontrado.\n"
            f"Campos disponíveis: {available_fields}"
        )

    # Reprojecta para o EPSG desejado se necessário
    if gdf.crs is None:
        print(f"Shapefile sem CRS definido. Assumindo EPSG:{epsg}")
        gdf = gdf.set_crs(epsg=epsg)
    elif gdf.crs.to_epsg() != epsg:
        print(f"Reprojetando de EPSG:{gdf.crs.to_epsg()} para EPSG:{epsg}")
        gdf = gdf.to_crs(epsg=epsg)

    # Calcula dimensões do raster
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds

    # Ajusta bounds para múltiplos do cellsize
    minx = np.floor(minx / cellsize) * cellsize
    miny = np.floor(miny / cellsize) * cellsize
    maxx = np.ceil(maxx / cellsize) * cellsize
    maxy = np.ceil(maxy / cellsize) * cellsize

    # Calcula número de linhas e colunas
    width = int((maxx - minx) / cellsize)
    height = int((maxy - miny) / cellsize)

    print(f"Dimensões do raster: {width} x {height} pixels")
    print(f"Cellsize: {cellsize}")
    print(f"Bounds: ({minx:.2f}, {miny:.2f}) - ({maxx:.2f}, {maxy:.2f})")

    # Cria transform (georreferenciamento)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Prepara geometrias e valores para rasterização
    shapes = []
    for idx, row in gdf.iterrows():
        value = row[field]
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            shapes.append((row.geometry, float(value)))

    if not shapes:
        raise ValueError(f"Nenhum valor válido encontrado no campo '{field}'")

    print(f"Rasterizando {len(shapes)} feições...")

    # Rasteriza
    raster = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=dtype
    )

    # Cria diretório de saída se não existir
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salva o raster
    print(f"Salvando raster: {output_path}")
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=f'EPSG:{epsg}',
        transform=transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(raster, 1)
        dst.descriptions = (field,)

    print(f"Raster criado com sucesso!")
    print(f"  - Arquivo: {output_path}")
    print(f"  - Campo: {field}")
    print(f"  - EPSG: {epsg}")
    print(f"  - Cellsize: {cellsize}")
    print(f"  - Dimensões: {width} x {height}")

    return output_path


def list_shapefile_fields(shapefile_path: str) -> list:
    """
    Lista todos os campos disponíveis em um shapefile.

    Parameters
    ----------
    shapefile_path : str
        Caminho para o shapefile

    Returns
    -------
    list
        Lista com nomes dos campos (exceto geometry)
    """
    gdf = gpd.read_file(shapefile_path)
    fields = [c for c in gdf.columns if c != 'geometry']
    return fields


# =============================================================================
# EXEMPLO DE USO
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Converte shapefile para raster GeoTIFF'
    )
    parser.add_argument('shapefile', help='Caminho do shapefile de entrada')
    parser.add_argument('field', nargs='?', help='Campo a ser rasterizado')
    parser.add_argument('output', nargs='?', help='Caminho do raster de saída (.tif)')
    parser.add_argument('--cellsize', type=float, default=17.0,
                        help='Tamanho do pixel (default: 17.0)')
    parser.add_argument('--epsg', type=int, default=31983,
                        help='Código EPSG (default: 31983 - SIRGAS 2000 UTM 23S)')
    parser.add_argument('--nodata', type=float, default=-9999.0,
                        help='Valor nodata (default: -9999.0)')
    parser.add_argument('--list-fields', action='store_true',
                        help='Lista campos disponíveis e sai')

    args = parser.parse_args()

    if args.list_fields:
        fields = list_shapefile_fields(args.shapefile)
        print("Campos disponíveis:")
        for f in fields:
            print(f"  - {f}")
    else:
        shapefile_to_raster(
            shapefile_path=args.shapefile,
            field=args.field,
            output_path=args.output,
            cellsize=args.cellsize,
            epsg=args.epsg,
            nodata=args.nodata
        )
