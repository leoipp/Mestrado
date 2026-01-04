"""
MergeTiff.py - Mosaico de Rasters GeoTIFF

Este script combina múltiplos arquivos GeoTIFF em um único mosaico,
resolvendo sobreposições e mantendo o georreferenciamento.

Funcionalidades:
    - Merge de múltiplos TIFFs em um único arquivo
    - Múltiplos métodos de resolução de sobreposição (first, last, max, min, mean)
    - Suporte a diferentes resoluções (reamostragem automática)
    - Compressão configurável (LZW, DEFLATE, etc.)
    - Processamento em lote por padrão de nome

Dependências:
    - rasterio
    - numpy

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Callable
from datetime import datetime

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    raise ImportError("rasterio é necessário. Execute: pip install rasterio")


# =============================================================================
# MÉTODOS DE MERGE
# =============================================================================

def method_first(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Mantém o primeiro valor válido (padrão do rasterio)."""
    mask = np.empty_like(old_data, dtype='bool')
    np.equal(old_data, old_nodata, out=mask)
    np.copyto(old_data, new_data, where=mask)


def method_last(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Sobrescreve com o último valor válido."""
    mask = np.empty_like(new_data, dtype='bool')
    np.not_equal(new_data, new_nodata, out=mask)
    np.copyto(old_data, new_data, where=mask)


def method_max(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Mantém o valor máximo nas sobreposições."""
    mask = np.empty_like(new_data, dtype='bool')
    np.not_equal(new_data, new_nodata, out=mask)
    np.maximum(old_data, new_data, out=old_data, where=mask)


def method_min(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Mantém o valor mínimo nas sobreposições."""
    # Primeiro, substitui nodata por valor alto para não afetar o mínimo
    old_copy = np.where(old_data == old_nodata, np.inf, old_data)
    new_copy = np.where(new_data == new_nodata, np.inf, new_data)
    result = np.minimum(old_copy, new_copy)
    np.copyto(old_data, result, where=(result != np.inf))


def method_mean(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Calcula a média nas sobreposições."""
    old_valid = old_data != old_nodata
    new_valid = new_data != new_nodata
    both_valid = old_valid & new_valid

    # Onde ambos são válidos, faz a média
    np.copyto(old_data, (old_data + new_data) / 2, where=both_valid)
    # Onde só o novo é válido, copia o novo
    np.copyto(old_data, new_data, where=(~old_valid & new_valid))


# Dicionário de métodos disponíveis
MERGE_METHODS = {
    'first': method_first,
    'last': method_last,
    'max': method_max,
    'min': method_min,
    'mean': method_mean
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_raster_info(file_path: str) -> dict:
    """
    Obtém informações de um arquivo raster.

    Parameters
    ----------
    file_path : str
        Caminho do arquivo raster.

    Returns
    -------
    dict
        Dicionário com informações do raster.
    """
    with rasterio.open(file_path) as src:
        return {
            'path': file_path,
            'crs': src.crs,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
            'resolution': src.res,
            'transform': src.transform
        }


def validate_rasters(file_list: List[str]) -> dict:
    """
    Valida compatibilidade entre rasters para merge.

    Parameters
    ----------
    file_list : list
        Lista de caminhos dos arquivos.

    Returns
    -------
    dict
        Informações agregadas dos rasters.

    Raises
    ------
    ValueError
        Se os rasters não forem compatíveis.
    """
    if not file_list:
        raise ValueError("Lista de arquivos vazia")

    infos = [get_raster_info(f) for f in file_list]

    # Verifica CRS
    crs_set = set(str(info['crs']) for info in infos if info['crs'])
    if len(crs_set) > 1:
        print(f"  Aviso: CRS diferentes detectados: {crs_set}")
        print("  O merge usará o CRS do primeiro arquivo")

    # Verifica número de bandas
    band_counts = set(info['count'] for info in infos)
    if len(band_counts) > 1:
        raise ValueError(f"Número de bandas inconsistente: {band_counts}")

    # Verifica dtype
    dtypes = set(info['dtype'] for info in infos)
    if len(dtypes) > 1:
        print(f"  Aviso: dtypes diferentes: {dtypes}")

    # Calcula bounds totais
    all_bounds = [info['bounds'] for info in infos]
    total_bounds = (
        min(b.left for b in all_bounds),
        min(b.bottom for b in all_bounds),
        max(b.right for b in all_bounds),
        max(b.top for b in all_bounds)
    )

    return {
        'count': len(file_list),
        'crs': infos[0]['crs'],
        'band_count': infos[0]['count'],
        'dtype': infos[0]['dtype'],
        'nodata': infos[0]['nodata'],
        'total_bounds': total_bounds,
        'resolutions': [info['resolution'] for info in infos]
    }


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def merge_tiffs(
    input_files: Union[str, List[str]],
    output_file: str,
    pattern: str = '*.tif',
    method: str = 'first',
    nodata: Optional[float] = None,
    resolution: Optional[float] = None,
    resampling: str = 'nearest',
    compress: str = 'lzw',
    dtype: Optional[str] = None,
    overwrite: bool = True
) -> str:
    """
    Combina múltiplos arquivos GeoTIFF em um único mosaico.

    Parameters
    ----------
    input_files : str or list
        Pasta contendo os TIFFs ou lista de caminhos dos arquivos.
    output_file : str
        Caminho do arquivo de saída.
    pattern : str, optional
        Padrão glob para filtrar arquivos se input_files for pasta (default: '*.tif').
    method : str, optional
        Método de resolução de sobreposição:
        - 'first': mantém primeiro valor válido (default)
        - 'last': sobrescreve com último valor
        - 'max': valor máximo
        - 'min': valor mínimo
        - 'mean': média dos valores
    nodata : float, optional
        Valor nodata para o mosaico. Se None, usa o valor do primeiro arquivo.
    resolution : float, optional
        Resolução do mosaico. Se None, usa a do primeiro arquivo.
    resampling : str, optional
        Método de reamostragem: 'nearest', 'bilinear', 'cubic' (default: 'nearest').
    compress : str, optional
        Compressão: 'lzw', 'deflate', 'packbits', 'none' (default: 'lzw').
    dtype : str, optional
        Tipo de dado de saída. Se None, usa o tipo do primeiro arquivo.
    overwrite : bool, optional
        Se True, sobrescreve arquivo existente (default: True).

    Returns
    -------
    str
        Caminho do arquivo de saída.

    Examples
    --------
    # Merge de todos os TIFFs em uma pasta
    merge_tiffs('./tiles/', 'mosaico.tif')

    # Merge com filtro de nome
    merge_tiffs('./tiles/', 'dem.tif', pattern='*_dem.tif')

    # Merge usando valor máximo nas sobreposições
    merge_tiffs('./tiles/', 'chm_max.tif', method='max')

    # Lista específica de arquivos
    merge_tiffs(['tile1.tif', 'tile2.tif'], 'merged.tif')
    """
    print("=" * 60)
    print("MERGE DE RASTERS GEOTIFF")
    print("=" * 60)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------------------------------------------------------------
    # 1. COLETA DOS ARQUIVOS
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Coletando arquivos...")

    if isinstance(input_files, str):
        input_path = Path(input_files)
        if input_path.is_dir():
            file_list = sorted(input_path.glob(pattern))
            file_list = [str(f) for f in file_list]
            print(f"  Pasta: {input_files}")
            print(f"  Padrão: {pattern}")
        else:
            file_list = [input_files]
    else:
        file_list = list(input_files)

    if not file_list:
        raise ValueError(f"Nenhum arquivo encontrado com padrão '{pattern}'")

    print(f"  Arquivos encontrados: {len(file_list)}")

    # -------------------------------------------------------------------------
    # 2. VALIDAÇÃO
    # -------------------------------------------------------------------------
    print(f"\n[2/4] Validando rasters...")

    info = validate_rasters(file_list)

    print(f"  CRS: {info['crs']}")
    print(f"  Bandas: {info['band_count']}")
    print(f"  Dtype: {info['dtype']}")
    print(f"  NoData: {info['nodata']}")
    print(f"  Bounds: {info['total_bounds']}")

    # Verifica arquivo de saída
    output_path = Path(output_file)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Arquivo já existe: {output_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 3. MERGE
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Executando merge...")
    print(f"  Método: {method}")

    # Abre todos os arquivos
    src_files = [rasterio.open(f) for f in file_list]

    try:
        # Configuração do merge
        merge_kwargs = {}

        # Método de merge
        if method in MERGE_METHODS:
            merge_kwargs['method'] = MERGE_METHODS[method]
        else:
            raise ValueError(f"Método inválido: {method}. Use: {list(MERGE_METHODS.keys())}")

        # NoData
        if nodata is not None:
            merge_kwargs['nodata'] = nodata
        elif info['nodata'] is not None:
            merge_kwargs['nodata'] = info['nodata']

        # Resolução
        if resolution is not None:
            merge_kwargs['res'] = resolution

        # Reamostragem
        resampling_methods = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'lanczos': Resampling.lanczos
        }
        if resampling in resampling_methods:
            merge_kwargs['resampling'] = resampling_methods[resampling]

        # Executa merge
        mosaic, out_transform = merge(src_files, **merge_kwargs)

        print(f"  Dimensões do mosaico: {mosaic.shape[2]} x {mosaic.shape[1]} pixels")
        print(f"  Bandas: {mosaic.shape[0]}")

    finally:
        # Fecha todos os arquivos
        for src in src_files:
            src.close()

    # -------------------------------------------------------------------------
    # 4. SALVAMENTO
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Salvando mosaico: {output_file}")

    # Configurações de saída
    out_dtype = dtype if dtype else info['dtype']
    out_nodata = nodata if nodata is not None else info['nodata']

    compress_opts = {} if compress == 'none' else {'compress': compress}

    # Metadados
    out_meta = {
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'count': mosaic.shape[0],
        'dtype': out_dtype,
        'crs': info['crs'],
        'transform': out_transform,
        'nodata': out_nodata,
        **compress_opts
    }

    # Salva
    with rasterio.open(output_file, 'w', **out_meta) as dst:
        dst.write(mosaic.astype(out_dtype))

    # Estatísticas
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Tamanho: {output_size:.2f} MB")

    # Estatísticas dos dados
    valid_data = mosaic[mosaic != out_nodata]
    if len(valid_data) > 0:
        print(f"\n  Estatísticas:")
        print(f"    Min:  {valid_data.min():.4f}")
        print(f"    Max:  {valid_data.max():.4f}")
        print(f"    Mean: {valid_data.mean():.4f}")

    print("\n" + "=" * 60)
    print("MERGE CONCLUÍDO")
    print(f"Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return output_file


def merge_by_groups(
    input_folder: str,
    output_folder: str,
    group_pattern: str,
    **kwargs
) -> List[str]:
    """
    Agrupa arquivos por padrão de nome e faz merge separado de cada grupo.

    Parameters
    ----------
    input_folder : str
        Pasta com os arquivos.
    output_folder : str
        Pasta de saída.
    group_pattern : str
        Parte comum do nome para agrupar (ex: '_dem' agrupa todos *_dem.tif).
    **kwargs
        Argumentos adicionais para merge_tiffs.

    Returns
    -------
    list
        Lista de arquivos gerados.

    Examples
    --------
    # Agrupa por sufixo (tile_001_dem.tif, tile_002_dem.tif -> dem_mosaic.tif)
    merge_by_groups('./tiles/', './mosaics/', '_dem')
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Encontra todos os TIFFs
    all_files = list(input_path.glob('*.tif'))

    # Agrupa por padrão
    groups = {}
    for f in all_files:
        if group_pattern in f.stem:
            # Extrai o identificador do grupo
            group_id = f.stem.split(group_pattern)[-1] if group_pattern in f.stem else group_pattern
            group_key = group_pattern.strip('_')

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(str(f))

    print(f"Grupos encontrados: {list(groups.keys())}")

    results = []
    for group_name, files in groups.items():
        print(f"\n{'='*60}")
        print(f"Processando grupo: {group_name} ({len(files)} arquivos)")
        print('='*60)

        output_file = output_path / f"{group_name}_mosaic.tif"
        try:
            result = merge_tiffs(files, str(output_file), **kwargs)
            results.append(result)
        except Exception as e:
            print(f"Erro no grupo {group_name}: {e}")

    return results


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Combina múltiplos GeoTIFFs em um mosaico'
    )
    parser.add_argument('input', help='Pasta com TIFFs ou arquivo único')
    parser.add_argument('output', help='Arquivo de saída')
    parser.add_argument('-p', '--pattern', default='*.tif',
                        help='Padrão glob para filtrar arquivos (default: *.tif)')
    parser.add_argument('-m', '--method', choices=['first', 'last', 'max', 'min', 'mean'],
                        default='first', help='Método de merge (default: first)')
    parser.add_argument('--nodata', type=float, help='Valor nodata')
    parser.add_argument('--resolution', type=float, help='Resolução do mosaico')
    parser.add_argument('--resampling', choices=['nearest', 'bilinear', 'cubic'],
                        default='nearest', help='Método de reamostragem')
    parser.add_argument('--compress', choices=['lzw', 'deflate', 'packbits', 'none'],
                        default='lzw', help='Compressão (default: lzw)')
    parser.add_argument('--dtype', choices=['float32', 'float64', 'int16', 'int32', 'uint8', 'uint16'],
                        help='Tipo de dado de saída')

    args = parser.parse_args()

    merge_tiffs(
        input_files=args.input,
        output_file=args.output,
        pattern=args.pattern,
        method=args.method,
        nodata=args.nodata,
        resolution=args.resolution,
        resampling=args.resampling,
        compress=args.compress,
        dtype=args.dtype
    )
