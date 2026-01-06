"""
GenerateInteractionRasters.py - Geração de Rasters de Interação para Predição

Este script gera rasters de interação a partir de rasters base (variáveis contínuas)
e rasters indicadores (variáveis dummy/categóricas), replicando a lógica de
feature engineering do 04_RandomForestTrain.py.

Workflow:
    1. Carrega rasters base (ex: Elev_P90.tif, Elev_P60.tif)
    2. Carrega rasters indicadores/dummy (ex: REGIONAL_GN.tif, ROTACAO_1.tif)
    3. Multiplica cada indicador por cada base: indicador * base
    4. Salva os rasters de interação com nomenclatura padrão

Nomenclatura de saída:
    {indicador}__x__{base}.tif
    Ex: REGIONAL_GN__x__Elev_P90.tif

Autor: Leonardo Ippolito Rodrigues
Data: 2026
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import rasterio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


def load_raster(raster_path: str) -> Tuple[np.ndarray, dict]:
    """
    Carrega um raster e retorna os dados e metadados.

    Parameters
    ----------
    raster_path : str
        Caminho para o arquivo raster.

    Returns
    -------
    tuple
        (dados como numpy array, metadados do rasterio)
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        meta = src.meta.copy()
    return data, meta


def resample_to_reference(source_path: str, reference_path: str,
                          resampling_method: str = 'nearest',
                          nodata: float = -9999.0) -> Tuple[np.ndarray, dict]:
    """
    Reamostra um raster para coincidir com a grade de referência.

    Áreas fora do extent original do source ficam como nodata.

    Parameters
    ----------
    source_path : str
        Caminho do raster a ser reamostrado.
    reference_path : str
        Caminho do raster de referência (define a grade de saída).
    resampling_method : str
        Método de reamostragem: 'nearest', 'bilinear', 'cubic'.
        Use 'nearest' para dados categóricos/dummy.
    nodata : float
        Valor nodata para áreas sem dados.

    Returns
    -------
    tuple
        (dados reamostrados, metadados da referência)
    """
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS

    # Mapeia método de reamostragem
    resampling_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic
    }
    resampling = resampling_map.get(resampling_method, Resampling.nearest)

    # CRS padrão caso não exista
    default_crs = CRS.from_epsg(31983)  # SIRGAS 2000 UTM 23S

    with rasterio.open(reference_path) as ref:
        ref_meta = ref.meta.copy()
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs if ref.crs is not None else default_crs

    with rasterio.open(source_path) as src:
        src_data = src.read(1)
        src_transform = src.transform
        src_crs = src.crs if src.crs is not None else ref_crs
        src_nodata = src.nodata if src.nodata is not None else nodata

    # Cria array de destino PREENCHIDO com nodata
    dst_data = np.full(ref_shape, nodata, dtype=np.float32)

    # Reamostra - apenas preenche onde há dados no source
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=nodata
    )

    return dst_data, ref_meta


def save_raster(data: np.ndarray, meta: dict, output_path: str, description: str = None):
    """
    Salva um array numpy como raster GeoTIFF.

    Parameters
    ----------
    data : np.ndarray
        Dados do raster.
    meta : dict
        Metadados do rasterio.
    output_path : str
        Caminho de saída.
    description : str, optional
        Descrição da banda.
    """
    # Atualiza metadados
    meta.update({
        'dtype': 'float32',
        'compress': 'lzw'
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(data.astype('float32'), 1)
        if description:
            dst.descriptions = (description,)


def find_rasters_by_pattern(
    directory: str,
    patterns: List[str],
    exact_names: List[str] = None
) -> Dict[str, str]:
    """
    Encontra rasters em um diretório baseado em padrões ou nomes exatos.

    Parameters
    ----------
    directory : str
        Diretório para buscar.
    patterns : list
        Lista de prefixos/padrões para buscar (ex: ['REGIONAL_', 'ROTACAO_']).
    exact_names : list, optional
        Lista de nomes exatos de arquivos (sem extensão).

    Returns
    -------
    dict
        Dicionário {nome_variavel: caminho_completo}
    """
    directory = Path(directory)
    found = {}

    for tif_file in directory.glob('*.tif'):
        stem = tif_file.stem  # Nome sem extensão

        # Verifica nomes exatos
        if exact_names:
            if stem in exact_names:
                found[stem] = str(tif_file)
                continue

        # Verifica padrões (prefixos)
        for pattern in patterns:
            if stem.startswith(pattern) or pattern in stem:
                found[stem] = str(tif_file)
                break

    return found


def sanitize_name(name: str) -> str:
    """
    Sanitiza nome para uso em arquivos.
    """
    return name.replace(' ', '_').replace('/', '_').replace('\\', '_')


def generate_interaction_rasters(
    base_rasters: Dict[str, str],
    indicator_rasters: Dict[str, str],
    output_dir: str,
    nodata: float = -9999.0,
    verbose: bool = True
) -> List[str]:
    """
    Gera rasters de interação multiplicando indicadores por bases.

    Parameters
    ----------
    base_rasters : dict
        Dicionário {nome: caminho} dos rasters base (variáveis contínuas).
        Ex: {'Elev_P90': '/path/to/Elev_P90.tif'}
    indicator_rasters : dict
        Dicionário {nome: caminho} dos rasters indicadores (dummy 0/1).
        Ex: {'REGIONAL_GN': '/path/to/REGIONAL_GN.tif'}
    output_dir : str
        Diretório para salvar os rasters de interação.
    nodata : float
        Valor nodata.
    verbose : bool
        Se True, imprime progresso.

    Returns
    -------
    list
        Lista de caminhos dos rasters gerados.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_files = []
    total_interactions = len(base_rasters) * len(indicator_rasters)

    if verbose:
        print(f"\n{'='*70}")
        print("GERAÇÃO DE RASTERS DE INTERAÇÃO")
        print(f"{'='*70}")
        print(f"Rasters base: {len(base_rasters)}")
        for name in base_rasters:
            print(f"  - {name}")
        print(f"\nRasters indicadores: {len(indicator_rasters)}")
        for name in indicator_rasters:
            print(f"  - {name}")
        print(f"\nTotal de interações a gerar: {total_interactions}")
        print(f"Diretório de saída: {output_dir}")
        print(f"{'='*70}\n")

    # Carrega primeiro raster base para obter metadados de referência
    first_base_name = list(base_rasters.keys())[0]
    first_base_path = base_rasters[first_base_name]
    ref_data, ref_meta = load_raster(first_base_path)
    ref_shape = ref_data.shape

    if verbose:
        print(f"Grade de referência: {ref_shape[1]} x {ref_shape[0]} pixels")

    # Cache para rasters já carregados
    base_cache = {}
    indicator_cache = {}

    interaction_count = 0

    for ind_name, ind_path in indicator_rasters.items():
        # Carrega indicador (com cache e reamostragem se necessário)
        if ind_name not in indicator_cache:
            ind_data, _ = load_raster(ind_path)

            # Verifica se precisa reamostrar
            if ind_data.shape != ref_shape:
                if verbose:
                    print(f"  Reamostrando {ind_name}: {ind_data.shape} -> {ref_shape}")
                ind_data, _ = resample_to_reference(
                    source_path=ind_path,
                    reference_path=first_base_path,
                    resampling_method='nearest',  # nearest para dados categóricos
                    nodata=nodata  # áreas fora do extent = nodata
                )

            indicator_cache[ind_name] = ind_data

        indicator_data = indicator_cache[ind_name]

        for base_name, base_path in base_rasters.items():
            # Carrega base (com cache)
            if base_name not in base_cache:
                base_cache[base_name], _ = load_raster(base_path)

            base_data = base_cache[base_name]

            # Cria nome da interação
            ind_clean = sanitize_name(ind_name)
            base_clean = sanitize_name(base_name)
            interaction_name = f"{ind_clean}__x__{base_clean}"

            # Calcula interação
            interaction_data = np.where(
                (indicator_data == nodata) | (base_data == nodata),
                nodata,
                indicator_data * base_data
            )

            # Salva raster
            output_file = output_path / f"{interaction_name}.tif"
            save_raster(
                interaction_data,
                ref_meta,
                str(output_file),
                description=f"{ind_name} * {base_name}"
            )

            output_files.append(str(output_file))
            interaction_count += 1

            if verbose:
                print(f"[{interaction_count}/{total_interactions}] {interaction_name}")

    if verbose:
        print(f"\n{'='*70}")
        print(f"CONCLUÍDO: {len(output_files)} rasters de interação gerados")
        print(f"{'='*70}\n")

    return output_files


def generate_from_config(
    output_dir: str,
    base_names: List[str],
    indicator_prefixes: List[str],
    base_dir: str = None,
    indicator_dir: str = None,
    raster_dir: str = None,
    nodata: float = -9999.0
) -> List[str]:
    """
    Gera rasters de interação a partir de configuração simplificada.

    Esta função busca automaticamente os rasters nos diretórios especificados
    baseado nos nomes e prefixos fornecidos.

    Parameters
    ----------
    output_dir : str
        Diretório para salvar os rasters de interação.
    base_names : list
        Nomes dos rasters base (variáveis contínuas).
        Ex: ['Elev_P90', 'Elev_P60', 'Elev_maximum']
    indicator_prefixes : list
        Prefixos dos rasters indicadores.
        Ex: ['REGIONAL_', 'ROTACAO_']
    base_dir : str, optional
        Diretório contendo os rasters base.
        Se None, usa raster_dir.
    indicator_dir : str, optional
        Diretório contendo os rasters indicadores.
        Se None, usa raster_dir.
    raster_dir : str, optional
        Diretório padrão para ambos os tipos de raster (retrocompatibilidade).
    nodata : float
        Valor nodata.

    Returns
    -------
    list
        Lista de caminhos dos rasters gerados.

    Examples
    --------
    # Diretórios separados
    generate_from_config(
    ...     base_dir='./rasters/lidar/',
    ...     indicator_dir='./rasters/dummy/',
    ...     output_dir='./rasters/interactions/',
    ...     base_names=['Elev_P90', 'Elev_P60', 'Elev_maximum'],
    ...     indicator_prefixes=['REGIONAL_', 'ROTACAO_']
    ... )

    # Mesmo diretório (retrocompatível)
    generate_from_config(
    ...     raster_dir='./rasters/',
    ...     output_dir='./rasters/interactions/',
    ...     base_names=['Elev_P90', 'Elev_P60', 'Elev_maximum'],
    ...     indicator_prefixes=['REGIONAL_', 'ROTACAO_']
    ... )
    """
    # Resolve diretórios
    if base_dir is None:
        base_dir = raster_dir
    if indicator_dir is None:
        indicator_dir = raster_dir

    if base_dir is None:
        raise ValueError("Deve especificar base_dir ou raster_dir")
    if indicator_dir is None:
        raise ValueError("Deve especificar indicator_dir ou raster_dir")

    base_path = Path(base_dir)
    indicator_path = Path(indicator_dir)

    # Encontra rasters base
    base_rasters = {}
    for name in base_names:
        # Tenta com e sem underscores
        possible_names = [
            name,
            name.replace(' ', '_'),
            name.replace('_', ' ')
        ]
        for pname in possible_names:
            tif_path = base_path / f"{pname}.tif"
            if tif_path.exists():
                base_rasters[pname] = str(tif_path)
                break

    if not base_rasters:
        raise FileNotFoundError(
            f"Nenhum raster base encontrado em {base_dir}.\n"
            f"Procurados: {base_names}"
        )

    # Encontra rasters indicadores
    indicator_rasters = find_rasters_by_pattern(
        str(indicator_path),
        patterns=indicator_prefixes
    )

    if not indicator_rasters:
        raise FileNotFoundError(
            f"Nenhum raster indicador encontrado em {indicator_dir}.\n"
            f"Prefixos procurados: {indicator_prefixes}"
        )

    return generate_interaction_rasters(
        base_rasters=base_rasters,
        indicator_rasters=indicator_rasters,
        output_dir=output_dir,
        nodata=nodata
    )


def generate_all_model_rasters(
    output_dir: str = None,
    base_dir: str = None,
    indicator_dir: str = None,
    raster_dir: str = None,
    include_interactions: bool = True,
    nodata: float = -9999.0
) -> Dict[str, List[str]]:
    """
    Gera todos os rasters necessários para o modelo RF, incluindo interações.

    Esta é a função principal para preparar os rasters para predição espacial.
    Replica exatamente a configuração do 04_RandomForestTrain.py.

    Parameters
    ----------
    output_dir : str, optional
        Diretório para os rasters de interação.
        Default: {base_dir}/interactions
    base_dir : str, optional
        Diretório contendo os rasters base (LiDAR).
        Se None, usa raster_dir.
    indicator_dir : str, optional
        Diretório contendo os rasters indicadores (dummy).
        Se None, usa raster_dir.
    raster_dir : str, optional
        Diretório padrão para ambos (retrocompatibilidade).
    include_interactions : bool
        Se True, gera rasters de interação.
    nodata : float
        Valor nodata.

    Returns
    -------
    dict
        Dicionário com:
        - 'base': lista de rasters base
        - 'indicators': lista de rasters indicadores
        - 'interactions': lista de rasters de interação
        - 'all': lista completa de todos os rasters

    Examples
    --------
    # Diretórios separados
    result = generate_all_model_rasters(
    ...     base_dir='./Data/Rasters/LiDAR/',
    ...     indicator_dir='./Data/Rasters/Dummy/',
    ...     output_dir='./Data/Rasters/Interactions/'
    ... )

    # Mesmo diretório
    result = generate_all_model_rasters(
    ...     raster_dir='./Data/Rasters/'
    ... )
    """
    # Configuração do modelo (espelha 04_RandomForestTrain.py)
    BASE_FEATURES = ['Elev_P90', 'Elev_P60', 'Elev_maximum']
    INDICATOR_PREFIXES = ['REGIONAL_', 'ROTACAO_', 'Idade_']

    # Resolve diretórios
    if base_dir is None:
        base_dir = raster_dir
    if indicator_dir is None:
        indicator_dir = raster_dir

    if base_dir is None:
        raise ValueError("Deve especificar base_dir ou raster_dir")

    base_path = Path(base_dir)
    indicator_path = Path(indicator_dir) if indicator_dir else base_path

    if output_dir is None:
        output_dir = base_path / 'interactions'

    result = {
        'base': [],
        'indicators': [],
        'interactions': [],
        'all': []
    }

    # Encontra rasters base
    for name in BASE_FEATURES:
        for variant in [name, name.replace('_', ' '), name.replace(' ', '_')]:
            tif_path = base_path / f"{variant}.tif"
            if tif_path.exists():
                result['base'].append(str(tif_path))
                break

    # Encontra rasters indicadores
    indicator_rasters = find_rasters_by_pattern(str(indicator_path), INDICATOR_PREFIXES)
    result['indicators'] = list(indicator_rasters.values())

    print(f"Diretório base: {base_dir}")
    print(f"Diretório indicadores: {indicator_dir}")
    print(f"Rasters base encontrados: {len(result['base'])}")
    for p in result['base']:
        print(f"  - {Path(p).name}")
    print(f"Rasters indicadores encontrados: {len(result['indicators'])}")
    for p in result['indicators']:
        print(f"  - {Path(p).name}")

    # Gera interações se solicitado
    if include_interactions and result['base'] and result['indicators']:
        base_dict = {Path(p).stem: p for p in result['base']}

        result['interactions'] = generate_interaction_rasters(
            base_rasters=base_dict,
            indicator_rasters=indicator_rasters,
            output_dir=str(output_dir),
            nodata=nodata
        )

    # Consolida todos
    result['all'] = result['base'] + result['indicators'] + result['interactions']

    return result


# =============================================================================
# EXEMPLO DE USO / CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Gera rasters de interação para modelo Random Forest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Modo automático com mesmo diretório
  python GenerateInteractionRasters.py --auto --raster-dir ./rasters/

  # Modo automático com diretórios separados
  python GenerateInteractionRasters.py --auto \\
      --base-dir ./rasters/lidar/ \\
      --indicator-dir ./rasters/dummy/ \\
      -o ./rasters/interactions/

  # Modo manual (especifica bases e indicadores)
  python GenerateInteractionRasters.py \\
      --base-dir ./rasters/lidar/ \\
      --indicator-dir ./rasters/dummy/ \\
      -o ./interactions/ \\
      --base Elev_P90 Elev_P60 Elev_maximum \\
      --indicators REGIONAL_ ROTACAO_

  # Listar rasters disponíveis
  python GenerateInteractionRasters.py --list --raster-dir ./rasters/
  python GenerateInteractionRasters.py --list --base-dir ./lidar/ --indicator-dir ./dummy/
        """
    )

    # Diretórios
    parser.add_argument('--raster-dir',
                        help='Diretório único com todos os rasters (base e indicadores)')
    parser.add_argument('--base-dir',
                        help='Diretório com rasters base/LiDAR (variáveis contínuas)')
    parser.add_argument('--indicator-dir',
                        help='Diretório com rasters indicadores/dummy')
    parser.add_argument('-o', '--output',
                        help='Diretório de saída (default: {base_dir}/interactions)')

    # Modos
    parser.add_argument('--auto', action='store_true',
                        help='Usa configuração automática do modelo RF')
    parser.add_argument('--list', action='store_true',
                        help='Lista rasters disponíveis nos diretórios')

    # Configuração manual
    parser.add_argument('--base', nargs='+',
                        help='Nomes dos rasters base (variáveis contínuas)')
    parser.add_argument('--indicators', nargs='+',
                        help='Prefixos dos rasters indicadores')

    # Opções
    parser.add_argument('--nodata', type=float, default=-9999.0,
                        help='Valor nodata (default: -9999.0)')

    args = parser.parse_args()

    # Valida que pelo menos um diretório foi especificado
    if not args.raster_dir and not args.base_dir and not args.indicator_dir:
        parser.print_help()
        print("\nERRO: Especifique --raster-dir ou --base-dir/--indicator-dir")
        exit(1)

    # Resolve diretórios
    base_dir = args.base_dir or args.raster_dir
    indicator_dir = args.indicator_dir or args.raster_dir

    if args.list:
        # Lista rasters disponíveis
        print("\n" + "="*60)
        if base_dir:
            base_path = Path(base_dir)
            tifs = sorted(base_path.glob('*.tif'))
            print(f"Rasters BASE em {base_dir}:")
            for tif in tifs:
                print(f"  - {tif.stem}")
            print(f"  Total: {len(tifs)} arquivos")

        if indicator_dir and indicator_dir != base_dir:
            indicator_path = Path(indicator_dir)
            tifs = sorted(indicator_path.glob('*.tif'))
            print(f"\nRasters INDICADORES em {indicator_dir}:")
            for tif in tifs:
                print(f"  - {tif.stem}")
            print(f"  Total: {len(tifs)} arquivos")
        print("="*60)

    elif args.auto:
        # Modo automático
        output_dir = args.output or str(Path(base_dir) / 'interactions')
        result = generate_all_model_rasters(
            base_dir=base_dir,
            indicator_dir=indicator_dir,
            output_dir=output_dir,
            nodata=args.nodata
        )
        print(f"\nResumo:")
        print(f"  Base: {len(result['base'])}")
        print(f"  Indicadores: {len(result['indicators'])}")
        print(f"  Interações: {len(result['interactions'])}")
        print(f"  Total: {len(result['all'])}")

    elif args.base and args.indicators:
        # Modo manual
        output_dir = args.output or str(Path(base_dir) / 'interactions')
        generate_from_config(
            base_dir=base_dir,
            indicator_dir=indicator_dir,
            output_dir=output_dir,
            base_names=args.base,
            indicator_prefixes=args.indicators,
            nodata=args.nodata
        )

    else:
        parser.print_help()
        print("\nERRO: Use --auto para modo automático ou especifique --base e --indicators")
