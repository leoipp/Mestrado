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


def shapefile_to_dummy_rasters(
    shapefile_path: str,
    field: str,
    output_dir: str,
    cellsize: float,
    epsg: int,
    nodata: float = -9999.0,
    prefix: str = None
) -> list:
    """
    Converte um campo categórico/dummy de um shapefile para múltiplos rasters binários.

    Gera um raster para cada valor único da coluna, onde:
    - 1 = onde o valor é igual ao valor da categoria
    - 0 = onde o valor é diferente

    Parameters
    ----------
    shapefile_path : str
        Caminho para o shapefile de entrada (.shp)
    field : str
        Nome do campo categórico a ser rasterizado
    output_dir : str
        Diretório para os arquivos GeoTIFF de saída
    cellsize : float
        Tamanho do pixel em unidades do sistema de coordenadas
    epsg : int
        Código EPSG do sistema de referência
    nodata : float, optional
        Valor para pixels sem dados (default: -9999.0)
    prefix : str, optional
        Prefixo para os nomes dos arquivos (default: nome do campo)

    Returns
    -------
    list
        Lista de caminhos dos arquivos raster gerados

    Examples
    --------
    shapefile_to_dummy_rasters(
    ...     shapefile_path='parcelas.shp',
    ...     field='CLASSE_USO',
    ...     output_dir='./rasters_dummy/',
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

    # Obtém valores únicos (excluindo NaN/None)
    unique_values = gdf[field].dropna().unique()
    unique_values = sorted([v for v in unique_values if v is not None])

    if len(unique_values) == 0:
        raise ValueError(f"Nenhum valor válido encontrado no campo '{field}'")

    print(f"Valores únicos encontrados em '{field}': {unique_values}")
    print(f"Serão gerados {len(unique_values)} rasters binários")

    # Calcula dimensões do raster
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds

    # Ajusta bounds para múltiplos do cellsize
    minx = np.floor(minx / cellsize) * cellsize
    miny = np.floor(miny / cellsize) * cellsize
    maxx = np.ceil(maxx / cellsize) * cellsize
    maxy = np.ceil(maxy / cellsize) * cellsize

    # Calcula número de linhas e colunas
    width = int((maxx - minx) / cellsize)
    height = int((maxy - miny) / cellsize)

    print(f"Dimensões dos rasters: {width} x {height} pixels")
    print(f"Cellsize: {cellsize}")
    print(f"Bounds: ({minx:.2f}, {miny:.2f}) - ({maxx:.2f}, {maxy:.2f})")

    # Cria transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Cria diretório de saída
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define prefixo
    if prefix is None:
        prefix = field

    output_files = []

    # Gera um raster para cada valor único
    for value in unique_values:
        # Cria nome do arquivo (sanitiza o valor para nome de arquivo)
        value_str = str(value).replace(' ', '_').replace('/', '_').replace('\\', '_')
        output_file = output_path / f"{prefix}_{value_str}.tif"

        print(f"\nProcessando: {field} = {value}")

        # Prepara geometrias com valores binários (1 para match, 0 para não-match)
        shapes_one = []  # Geometrias onde value == valor atual
        shapes_zero = []  # Geometrias onde value != valor atual

        for idx, row in gdf.iterrows():
            row_value = row[field]
            if row_value is not None and not (isinstance(row_value, float) and np.isnan(row_value)):
                if row_value == value:
                    shapes_one.append((row.geometry, 1.0))
                else:
                    shapes_zero.append((row.geometry, 0.0))

        # Rasteriza primeiro os zeros, depois os uns (para garantir que uns prevaleçam)
        all_shapes = shapes_zero + shapes_one

        if not all_shapes:
            print(f"  Aviso: Nenhuma feição encontrada para {field}={value}")
            continue

        raster = features.rasterize(
            shapes=all_shapes,
            out_shape=(height, width),
            transform=transform,
            fill=nodata,
            dtype='float32'
        )

        # Salva o raster
        print(f"  Salvando: {output_file}")
        with rasterio.open(
            str(output_file),
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=f'EPSG:{epsg}',
            transform=transform,
            nodata=nodata,
            compress='lzw'
        ) as dst:
            dst.write(raster, 1)
            dst.descriptions = (f"{field}={value}",)

        # Conta pixels
        n_ones = np.sum(raster == 1)
        n_zeros = np.sum(raster == 0)
        print(f"  Pixels com valor 1: {n_ones}")
        print(f"  Pixels com valor 0: {n_zeros}")

        output_files.append(str(output_file))

    print(f"\n{'='*60}")
    print(f"Rasters dummy criados com sucesso!")
    print(f"  - Total de arquivos: {len(output_files)}")
    print(f"  - Diretório: {output_dir}")
    print(f"  - Campo: {field}")
    print(f"  - Valores: {unique_values}")

    return output_files


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
    parser.add_argument('--dummy', action='store_true',
                        help='Trata o campo como categórico/dummy e gera um raster binário para cada valor único')
    parser.add_argument('--prefix', type=str, default=None,
                        help='Prefixo para nomes dos arquivos dummy (default: nome do campo)')

    args = parser.parse_args()

    if args.list_fields:
        fields = list_shapefile_fields(args.shapefile)
        print("Campos disponíveis:")
        for f in fields:
            print(f"  - {f}")
    elif args.dummy:
        # Modo dummy: gera múltiplos rasters binários
        if args.field is None:
            parser.error("O argumento 'field' é obrigatório no modo --dummy")
        if args.output is None:
            # Usa diretório padrão baseado no nome do shapefile
            output_dir = Path(args.shapefile).stem + "_dummy"
        else:
            output_dir = args.output

        shapefile_to_dummy_rasters(
            shapefile_path=args.shapefile,
            field=args.field,
            output_dir=output_dir,
            cellsize=args.cellsize,
            epsg=args.epsg,
            nodata=args.nodata,
            prefix=args.prefix
        )
    else:
        # Modo normal: gera um único raster
        if args.field is None or args.output is None:
            parser.error("Os argumentos 'field' e 'output' são obrigatórios no modo normal")
        shapefile_to_raster(
            shapefile_path=args.shapefile,
            field=args.field,
            output_path=args.output,
            cellsize=args.cellsize,
            epsg=args.epsg,
            nodata=args.nodata
        )
