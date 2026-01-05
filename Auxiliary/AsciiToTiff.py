"""
AsciiToTiff.py - Conversão de Raster ASCII (.asc) para GeoTIFF (.tif)

Este script converte arquivos raster no formato ASCII (ESRI ASCII Grid)
para o formato GeoTIFF, permitindo definir o sistema de referência (EPSG)
e aplicar compressão.

Formato ASCII suportado:
    ncols         100
    nrows         100
    xllcorner     500000.0
    yllcorner     7500000.0
    cellsize      10.0
    NODATA_value  -9999
    <dados em matriz>

Dependências:
    - numpy
    - rasterio

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from openpyxl.styles.builtins import output

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    raise ImportError("rasterio é necessário. Execute: pip install rasterio")


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def read_ascii_header(file_path: str) -> Dict[str, Any]:
    """
    Lê o cabeçalho de um arquivo ASCII raster.

    Parameters
    ----------
    file_path : str
        Caminho do arquivo .asc.

    Returns
    -------
    dict
        Dicionário com os parâmetros do cabeçalho:
        - ncols: número de colunas
        - nrows: número de linhas
        - xllcorner ou xllcenter: coordenada X do canto/centro inferior esquerdo
        - yllcorner ou yllcenter: coordenada Y do canto/centro inferior esquerdo
        - cellsize: tamanho da célula
        - nodata_value: valor para dados ausentes
        - header_lines: número de linhas do cabeçalho
    """
    header = {}
    header_lines = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Verifica se é linha de cabeçalho (começa com texto)
            if parts[0].lower() in ['ncols', 'nrows', 'xllcorner', 'xllcenter',
                                     'yllcorner', 'yllcenter', 'cellsize',
                                     'nodata_value', 'dx', 'dy']:
                key = parts[0].lower()
                value = parts[1]

                # Converte para tipo apropriado
                if key in ['ncols', 'nrows']:
                    header[key] = int(value)
                else:
                    header[key] = float(value)

                header_lines += 1
            else:
                # Chegou nos dados
                break

    header['header_lines'] = header_lines
    return header


def read_ascii_raster(file_path: str) -> tuple:
    """
    Lê um arquivo ASCII raster completo (cabeçalho + dados).

    Parameters
    ----------
    file_path : str
        Caminho do arquivo .asc.

    Returns
    -------
    tuple
        (data, header) onde:
        - data: numpy array 2D com os valores do raster
        - header: dicionário com parâmetros do cabeçalho
    """
    # Lê cabeçalho
    header = read_ascii_header(file_path)

    # Lê dados
    data = np.loadtxt(
        file_path,
        skiprows=header['header_lines'],
        dtype=np.float32
    )

    # Valida dimensões
    expected_shape = (header['nrows'], header['ncols'])
    if data.shape != expected_shape:
        raise ValueError(
            f"Dimensões do raster ({data.shape}) não correspondem ao cabeçalho {expected_shape}"
        )

    return data, header


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def ascii_to_tiff(
    input_file: str,
    output_file: Optional[str] = None,
    epsg: Optional[int] = None,
    compress: str = 'lzw',
    dtype: str = 'float32',
    overwrite: bool = True
) -> str:
    """
    Converte raster ASCII (.asc) para GeoTIFF (.tif).

    Parameters
    ----------
    input_file : str
        Caminho do arquivo ASCII de entrada (.asc).
    output_file : str, optional
        Caminho do arquivo TIFF de saída. Se None, usa mesmo nome com extensão .tif.
    epsg : int, optional
        Código EPSG do sistema de referência. Se None, raster sem CRS definido.
    compress : str, optional
        Método de compressão: 'lzw', 'deflate', 'packbits', 'none' (default: 'lzw').
    dtype : str, optional
        Tipo de dado: 'float32', 'float64', 'int16', 'int32' (default: 'float32').
    overwrite : bool, optional
        Se True, sobrescreve arquivo existente (default: True).

    Returns
    -------
    str
        Caminho do arquivo TIFF gerado.

    Raises
    ------
    FileNotFoundError
        Se o arquivo de entrada não existir.
    FileExistsError
        Se o arquivo de saída existir e overwrite=False.

    Examples
    --------
    ascii_to_tiff('dem.asc', 'dem.tif', epsg=31983)

    # Batch conversion
    for asc in Path('.').glob('*.asc'):
    ...     ascii_to_tiff(str(asc), epsg=31983)
    """
    print("=" * 60)
    print("CONVERSÃO ASCII PARA GEOTIFF")
    print("=" * 60)

    # Validação de entrada
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")

    # Define arquivo de saída
    if output_file is None:
        output_file = str(input_path.with_suffix('.tif'))

    output_path = Path(output_file)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Arquivo já existe: {output_file}")

    # -------------------------------------------------------------------------
    # 1. LEITURA DO ASCII
    # -------------------------------------------------------------------------
    print(f"\n[1/3] Lendo arquivo ASCII: {input_file}")

    data, header = read_ascii_raster(input_file)

    print(f"  Dimensões: {header['ncols']} x {header['nrows']}")
    print(f"  Cellsize: {header['cellsize']}")
    print(f"  NoData: {header.get('nodata_value', 'não definido')}")

    # -------------------------------------------------------------------------
    # 2. CONFIGURAÇÃO DO GEOTIFF
    # -------------------------------------------------------------------------
    print(f"\n[2/3] Configurando GeoTIFF...")

    # Calcula transform
    # ASCII usa xllcorner/yllcorner (canto inferior esquerdo)
    # Rasterio usa origem no canto superior esquerdo
    cellsize = header['cellsize']

    if 'xllcorner' in header:
        xll = header['xllcorner']
        yll = header['yllcorner']
    else:
        # xllcenter/yllcenter - ajusta para corner
        xll = header['xllcenter'] - cellsize / 2
        yll = header['yllcenter'] - cellsize / 2

    # Origem no canto superior esquerdo
    xul = xll
    yul = yll + (header['nrows'] * cellsize)

    transform = from_origin(xul, yul, cellsize, cellsize)

    # NoData
    nodata = header.get('nodata_value', -9999)

    # CRS
    crs = CRS.from_epsg(epsg) if epsg else None

    # Compressão
    compress_opts = {} if compress == 'none' else {'compress': compress}

    print(f"  EPSG: {epsg if epsg else 'não definido'}")
    print(f"  Compressão: {compress}")
    print(f"  Dtype: {dtype}")

    # -------------------------------------------------------------------------
    # 3. ESCRITA DO GEOTIFF
    # -------------------------------------------------------------------------
    print(f"\n[3/3] Salvando GeoTIFF: {output_file}")

    # Converte dtype
    data = data.astype(dtype)

    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=header['nrows'],
        width=header['ncols'],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        **compress_opts
    ) as dst:
        dst.write(data, 1)

    # Estatísticas
    valid_data = data[data != nodata]
    if len(valid_data) > 0:
        print(f"\n  Estatísticas:")
        print(f"    Min:  {valid_data.min():.4f}")
        print(f"    Max:  {valid_data.max():.4f}")
        print(f"    Mean: {valid_data.mean():.4f}")

    # Tamanho do arquivo
    output_size = output_path.stat().st_size / 1024
    input_size = input_path.stat().st_size / 1024
    print(f"\n  Tamanho: {input_size:.1f} KB -> {output_size:.1f} KB ({100*output_size/input_size:.1f}%)")

    print("\n" + "=" * 60)
    print("CONVERSÃO CONCLUÍDA")
    print("=" * 60)

    return output_file


def batch_convert(
    input_folder: str,
    output_folder: Optional[str] = None,
    pattern: str = '*.asc',
    **kwargs
) -> list:
    """
    Converte múltiplos arquivos ASCII para GeoTIFF.

    Parameters
    ----------
    input_folder : str
        Pasta com arquivos ASCII.
    output_folder : str, optional
        Pasta de saída. Se None, salva na mesma pasta.
    pattern : str
        Padrão glob para filtrar arquivos (default: '*.asc').
    **kwargs
        Argumentos adicionais para ascii_to_tiff.

    Returns
    -------
    list
        Lista de arquivos convertidos.
    """
    input_path = Path(input_folder)

    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    files = list(input_path.glob(pattern))
    print(f"Encontrados {len(files)} arquivos para converter\n")

    converted = []
    for i, file in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"Arquivo {i}/{len(files)}: {file.name}")
        print('='*60)

        output_file = output_path / file.with_suffix('.tif').name

        try:
            result = ascii_to_tiff(str(file), str(output_file), **kwargs)
            converted.append(result)
        except Exception as e:
            print(f"Erro ao converter {file.name}: {e}")

    print(f"\n\nConvertidos: {len(converted)}/{len(files)} arquivos")
    return converted


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Converte raster ASCII (.asc) para GeoTIFF (.tif)'
    )
    parser.add_argument('input', help='Arquivo ASCII de entrada ou pasta para batch')
    parser.add_argument('-o', '--output', help='Arquivo ou pasta de saída')
    parser.add_argument('--epsg', type=int, help='Código EPSG do CRS')
    parser.add_argument('--compress', choices=['lzw', 'deflate', 'packbits', 'none'],
                        default='lzw', help='Método de compressão (default: lzw)')
    parser.add_argument('--dtype', choices=['float32', 'float64', 'int16', 'int32'],
                        default='float32', help='Tipo de dado (default: float32)')
    parser.add_argument('--batch', action='store_true',
                        help='Modo batch: converte todos os .asc da pasta')
    parser.add_argument('--pattern', default='*.asc',
                        help='Padrão para modo batch (default: *.asc)')

    args = parser.parse_args()

    if args.batch or Path(args.input).is_dir():
        batch_convert(
            input_folder=args.input,
            output_folder=args.output,
            pattern=args.pattern,
            epsg=args.epsg,
            compress=args.compress,
            dtype=args.dtype
        )
    else:
        ascii_to_tiff(
            input_file=args.input,
            output_file=args.output,
            epsg=args.epsg,
            compress=args.compress,
            dtype=args.dtype
        )
