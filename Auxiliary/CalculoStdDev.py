"""
CalculoStdDev.py - Cálculo de Desvio Padrão de Elevação para Raster

Este script calcula o desvio padrão (standard deviation) da elevação de pontos LiDAR
e gera um raster GeoTIFF. O desvio padrão é uma métrica importante em inventário
florestal para caracterizar a heterogeneidade vertical da estrutura do dossel.

Fórmula do Desvio Padrão:
    stddev = sqrt(mean((z - mean(z))²))

O desvio padrão indica a variabilidade das alturas dentro de cada célula,
sendo útil para identificar áreas com estrutura vertical homogênea ou heterogênea.

Entradas:
    - Arquivo de pontos (.txt, .csv, .xyz) com colunas X, Y, Z
    - Ou arquivo LAS/LAZ diretamente

Saídas:
    - Raster GeoTIFF (.tif) com o desvio padrão por célula
    - Opcional: ASCII (.asc)

Dependências:
    - pandas
    - numpy
    - rasterio (opcional, para saída GeoTIFF)
    - laspy (opcional, para leitura de arquivos LAS/LAZ)

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Aviso: rasterio não instalado. Apenas saída ASCII disponível.")

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def load_points(
    file_path: str,
    x_col: str = 'x',
    y_col: str = 'y',
    z_col: str = 'z'
) -> pd.DataFrame:
    """
    Carrega pontos de diferentes formatos de arquivo.

    Parameters
    ----------
    file_path : str
        Caminho do arquivo de pontos (.txt, .csv, .xyz, .las, .laz).
    x_col : str
        Nome da coluna X (para arquivos tabulares).
    y_col : str
        Nome da coluna Y (para arquivos tabulares).
    z_col : str
        Nome da coluna Z (para arquivos tabulares).

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas 'x', 'y', 'z'.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    suffix = path.suffix.lower()

    # Arquivos LAS/LAZ
    if suffix in ['.las', '.laz']:
        if not HAS_LASPY:
            raise ImportError("laspy não instalado. Execute: pip install laspy")

        las = laspy.read(file_path)
        df = pd.DataFrame({
            'x': las.x,
            'y': las.y,
            'z': las.z
        })

    # Arquivos de texto
    elif suffix in ['.txt', '.csv', '.xyz', '.pts']:
        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['x', 'y', 'z'])
        except Exception:
            df = pd.read_csv(file_path, sep=',', header=None, names=['x', 'y', 'z'])

        if df.iloc[0]['x'] in ['x', 'X', x_col]:
            df = pd.read_csv(file_path, sep=r'\s+|,', engine='python')
            df.columns = df.columns.str.lower()

    else:
        raise ValueError(f"Formato não suportado: {suffix}")

    for col in ['x', 'y', 'z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    n_before = len(df)
    df = df.dropna(subset=['x', 'y', 'z'])
    n_removed = n_before - len(df)

    if n_removed > 0:
        print(f"  Removidos {n_removed} pontos com valores inválidos")

    return df


def calculate_grid_dimensions(
    df: pd.DataFrame,
    cell_size: float
) -> Tuple[float, float, float, float, int, int]:
    """
    Calcula as dimensões do grid raster.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas 'x', 'y'.
    cell_size : float
        Tamanho da célula em unidades do sistema de coordenadas.

    Returns
    -------
    tuple
        (x_min, y_min, x_max, y_max, n_cols, n_rows)
    """
    x_min = np.floor(df['x'].min() / cell_size) * cell_size
    y_min = np.floor(df['y'].min() / cell_size) * cell_size
    x_max = np.ceil(df['x'].max() / cell_size) * cell_size
    y_max = np.ceil(df['y'].max() / cell_size) * cell_size

    n_cols = int((x_max - x_min) / cell_size)
    n_rows = int((y_max - y_min) / cell_size)

    return x_min, y_min, x_max, y_max, n_cols, n_rows


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def generate_stddev_raster(
    input_file: str,
    output_path: str,
    cell_size: float = 17.0,
    epsg: Optional[int] = None,
    nodata_value: float = -9999.0,
    output_format: str = 'tif',
    min_points: int = 2
) -> str:
    """
    Gera raster de desvio padrão da elevação a partir de pontos.

    Parameters
    ----------
    input_file : str
        Caminho do arquivo de pontos (.txt, .csv, .xyz, .las, .laz).
    output_path : str
        Diretório de saída ou caminho do arquivo.
    cell_size : float, optional
        Tamanho da célula em metros (default: 17.0).
    epsg : int, optional
        Código EPSG do sistema de referência. Se None, raster sem CRS.
    nodata_value : float, optional
        Valor para células sem dados (default: -9999.0).
    output_format : str, optional
        Formato de saída: 'tif', 'asc' ou 'both' (default: 'tif').
    min_points : int, optional
        Número mínimo de pontos por célula (default: 2).
        Nota: stddev requer pelo menos 2 pontos.

    Returns
    -------
    str
        Caminho do arquivo raster gerado.

    Examples
    --------
    generate_stddev_raster(
        input_file='pontos.txt',
        output_path='output_folder/',
        cell_size=17,
        epsg=31983
    )
    """
    print("=" * 60)
    print("CÁLCULO DE DESVIO PADRÃO - RASTER")
    print("=" * 60)

    # Garante mínimo de 2 pontos para stddev
    min_points = max(min_points, 2)

    # -------------------------------------------------------------------------
    # 1. CARREGAMENTO DOS PONTOS
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Carregando pontos: {input_file}")
    df = load_points(input_file)
    print(f"  Total de pontos: {len(df):,}")

    # -------------------------------------------------------------------------
    # 2. CÁLCULO DAS DIMENSÕES DO GRID
    # -------------------------------------------------------------------------
    print(f"\n[2/4] Calculando dimensões do grid...")
    x_min, y_min, x_max, y_max, n_cols, n_rows = calculate_grid_dimensions(df, cell_size)

    print(f"  Extensão: ({x_min:.2f}, {y_min:.2f}) - ({x_max:.2f}, {y_max:.2f})")
    print(f"  Dimensões: {n_cols} x {n_rows} células")
    print(f"  Cell size: {cell_size} m")

    # -------------------------------------------------------------------------
    # 3. CÁLCULO DO DESVIO PADRÃO POR CÉLULA
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Calculando desvio padrão por célula...")

    # Atribui cada ponto a uma célula do raster
    df['row'] = ((y_max - df['y']) / cell_size).astype(int)
    df['col'] = ((df['x'] - x_min) / cell_size).astype(int)

    # Limita aos bounds
    df['row'] = df['row'].clip(0, n_rows - 1)
    df['col'] = df['col'].clip(0, n_cols - 1)

    # Agrupa e calcula desvio padrão
    grouped = df.groupby(['row', 'col']).agg(
        stddev_z=('z', 'std'),
        n_points=('z', 'count')
    ).reset_index()

    # Filtra por número mínimo de pontos
    n_before = len(grouped)
    grouped = grouped[grouped['n_points'] >= min_points]
    print(f"  Células filtradas (< {min_points} pontos): {n_before - len(grouped)}")

    print(f"  Células com dados: {len(grouped):,} de {n_rows * n_cols:,} ({100*len(grouped)/(n_rows*n_cols):.1f}%)")

    # Cria matriz do raster
    grid = np.full((n_rows, n_cols), nodata_value, dtype=np.float32)

    for _, row in grouped.iterrows():
        r, c = int(row['row']), int(row['col'])
        if 0 <= r < n_rows and 0 <= c < n_cols:
            grid[r, c] = row['stddev_z']

    # Estatísticas
    valid_values = grid[grid != nodata_value]
    if len(valid_values) > 0:
        print(f"  Estatísticas do desvio padrão:")
        print(f"    Min:  {valid_values.min():.2f}")
        print(f"    Max:  {valid_values.max():.2f}")
        print(f"    Mean: {valid_values.mean():.2f}")

    # -------------------------------------------------------------------------
    # 4. EXPORTAÇÃO DO RASTER
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Salvando raster...")

    # Prepara caminho de saída
    output_dir = Path(output_path)
    input_stem = Path(input_file).stem

    if output_dir.is_dir() or not output_dir.suffix:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_base = output_dir / input_stem
    else:
        if output_dir.suffix.lower() in ['.tif', '.tiff', '.asc']:
            output_base = output_dir.with_suffix('')
        else:
            output_base = output_dir

    files_created = []

    # Salva GeoTIFF
    if output_format in ['tif', 'both']:
        if not HAS_RASTERIO:
            print("  Aviso: rasterio não disponível, salvando apenas ASCII")
            output_format = 'asc'
        else:
            tif_path = str(output_base) + '_stddev.tif'

            transform = from_bounds(x_min, y_min, x_max, y_max, n_cols, n_rows)

            with rasterio.open(
                tif_path,
                'w',
                driver='GTiff',
                height=n_rows,
                width=n_cols,
                count=1,
                dtype='float32',
                crs=f'EPSG:{epsg}' if epsg else None,
                transform=transform,
                nodata=nodata_value,
                compress='lzw'
            ) as dst:
                dst.write(grid, 1)
                dst.descriptions = ('Elev_StdDev',)

            print(f"  GeoTIFF salvo: {tif_path}")
            files_created.append(tif_path)

    # Salva ASCII
    if output_format in ['asc', 'both']:
        asc_path = str(output_base) + '_stddev.asc'

        with open(asc_path, 'w') as f:
            f.write(f"ncols         {n_cols}\n")
            f.write(f"nrows         {n_rows}\n")
            f.write(f"xllcorner     {x_min}\n")
            f.write(f"yllcorner     {y_min}\n")
            f.write(f"cellsize      {cell_size}\n")
            f.write(f"NODATA_value  {nodata_value}\n")
            np.savetxt(f, grid, fmt="%.3f")

        print(f"  ASCII salvo: {asc_path}")
        files_created.append(asc_path)

        if epsg:
            prj_path = str(output_base) + '_stddev.prj'
            try:
                from rasterio.crs import CRS
                crs = CRS.from_epsg(epsg)
                with open(prj_path, 'w') as f:
                    f.write(crs.to_wkt())
                print(f"  PRJ salvo: {prj_path}")
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)

    return files_created[0] if files_created else None


# =============================================================================
# FUNÇÕES ADICIONAIS
# =============================================================================

def batch_process(
    input_folder: str,
    output_folder: str,
    pattern: str = '*.txt',
    **kwargs
) -> list:
    """
    Processa múltiplos arquivos de pontos em lote.

    Parameters
    ----------
    input_folder : str
        Pasta com arquivos de entrada.
    output_folder : str
        Pasta para salvar os rasters.
    pattern : str
        Padrão glob para filtrar arquivos (default: '*.txt').
    **kwargs
        Argumentos adicionais para generate_stddev_raster.

    Returns
    -------
    list
        Lista de arquivos processados.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob(pattern))
    print(f"Encontrados {len(files)} arquivos para processar")

    processed = []
    for i, file in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"Arquivo {i}/{len(files)}: {file.name}")
        print('='*60)

        try:
            result = generate_stddev_raster(
                str(file),
                str(output_path),
                **kwargs
            )
            processed.append(result)
        except Exception as e:
            print(f"Erro ao processar {file.name}: {e}")

    print(f"\n\nProcessados: {len(processed)}/{len(files)} arquivos")
    return processed


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Calcula desvio padrão de elevação e gera raster'
    )
    parser.add_argument('input', help='Arquivo de pontos (.txt, .csv, .las, .laz)')
    parser.add_argument('output', help='Diretório ou caminho do raster de saída')
    parser.add_argument('--cellsize', type=float, default=17.0,
                        help='Tamanho da célula (default: 17.0)')
    parser.add_argument('--epsg', type=int, default=None,
                        help='Código EPSG do sistema de referência')
    parser.add_argument('--nodata', type=float, default=-9999.0,
                        help='Valor nodata (default: -9999.0)')
    parser.add_argument('--format', choices=['tif', 'asc', 'both'], default='tif',
                        help='Formato de saída (default: tif)')
    parser.add_argument('--min-points', type=int, default=2,
                        help='Mínimo de pontos por célula (default: 2)')

    args = parser.parse_args()

    generate_stddev_raster(
        input_file=args.input,
        output_path=args.output,
        cell_size=args.cellsize,
        epsg=args.epsg,
        nodata_value=args.nodata,
        output_format=args.format,
        min_points=args.min_points
    )
