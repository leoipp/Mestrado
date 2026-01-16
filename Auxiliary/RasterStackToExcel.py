"""
RasterStackToExcel.py - Empilha Rasters e Exporta Pixels para Excel

Recebe n rasters, empilha e gera um Excel onde cada linha representa
um pixel e cada coluna representa o valor de cada raster.

Uso:
    # Varios rasters
    python RasterStackToExcel.py raster1.tif raster2.tif raster3.tif -o pixels.xlsx

    # Pasta com rasters
    python RasterStackToExcel.py pasta_rasters/ -o pixels.xlsx

    # Com coordenadas
    python RasterStackToExcel.py pasta_rasters/ -o pixels.xlsx --coords

    # Filtrar por extensao
    python RasterStackToExcel.py pasta_rasters/ -o pixels.xlsx -p "*.tif"

Autor: Leonardo Ippolito Rodrigues
Ano: 2026
Projeto: Mestrado - Predicao de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime

try:
    import rasterio
    from rasterio.transform import xy
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    raise ImportError("rasterio e necessario. Execute: pip install rasterio")


def align_raster_to_reference(
    source_path: str,
    ref_transform,
    ref_crs,
    ref_width: int,
    ref_height: int,
    ref_nodata: float = -9999.0,
) -> np.ndarray:
    """
    Alinha um raster ao grid de referencia usando reproject.

    Parameters
    ----------
    source_path : str
        Caminho do raster a ser alinhado.
    ref_transform : Affine
        Transform do raster de referencia.
    ref_crs : CRS
        CRS do raster de referencia.
    ref_width, ref_height : int
        Dimensoes do raster de referencia.
    ref_nodata : float
        Valor nodata de referencia.

    Returns
    -------
    np.ndarray
        Array alinhado ao grid de referencia.
    """
    with rasterio.open(source_path) as src:
        src_data = src.read(1).astype('float32')
        src_nodata = src.nodata if src.nodata is not None else ref_nodata
        src_crs = src.crs if src.crs is not None else ref_crs
        src_transform = src.transform

        # Cria array de destino
        aligned = np.full((ref_height, ref_width), np.nan, dtype='float32')

        # Reprojeta/alinha
        reproject(
            source=src_data,
            destination=aligned,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest
        )

        return aligned


def stack_rasters_with_sampling(
    reference_rasters: List[str],
    sample_rasters: List[str],
    include_coords: bool = False,
    nodata: Optional[float] = None,
    remove_nodata_rows: bool = True,
) -> pd.DataFrame:
    """
    Processa cada raster de referencia individualmente, alinha os sample rasters
    ao mesmo grid usando reproject, e concatena tudo em um unico DataFrame.

    Cada raster de referencia pode ter shape diferente. Para cada um, os sample
    rasters sao reprojetados/alinhados ao mesmo grid.

    Parameters
    ----------
    reference_rasters : list of str
        Lista de rasters de referencia (podem ter shapes diferentes entre si).
        Cada um sera processado individualmente.
    sample_rasters : list of str
        Lista de rasters a serem alinhados (podem ter shapes/CRS diferentes).
    include_coords : bool
        Se True, inclui colunas X e Y com coordenadas geograficas.
    nodata : float, optional
        Valor nodata a usar. Se None, usa o valor de cada arquivo.
    remove_nodata_rows : bool
        Se True, remove linhas onde TODOS os valores sao nodata.

    Returns
    -------
    pd.DataFrame
        DataFrame concatenado com pixel_id, source, VAL e valores alinhados.
    """
    if not reference_rasters:
        raise ValueError("Nenhum raster de referencia fornecido")

    print(f"\n[INFO] Processando {len(reference_rasters)} rasters de referencia...")
    print(f"[INFO] Processando {len(sample_rasters)} rasters para alinhar...")

    # Processa cada raster de referencia individualmente
    all_dfs = []

    for ref_idx, ref_raster_path in enumerate(reference_rasters):
        ref_path = Path(ref_raster_path)
        ref_name = ref_path.stem

        print(f"\n[{ref_idx+1}/{len(reference_rasters)}] {ref_path.name}")

        with rasterio.open(ref_raster_path) as ref:
            ref_shape = ref.shape
            ref_height, ref_width = ref_shape
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_nodata_val = nodata if nodata is not None else ref.nodata
            if ref_nodata_val is None:
                ref_nodata_val = -9999.0

            # Le dados do raster de referencia
            ref_data = ref.read(1).astype('float32')

        # Se referencia nao tem CRS, pega do primeiro sample raster
        if ref_crs is None:
            for sp in sample_rasters:
                with rasterio.open(sp) as src:
                    if src.crs is not None:
                        ref_crs = src.crs
                        print(f"  [INFO] CRS herdado de {Path(sp).name}: {ref_crs}")
                        break

        print(f"  Shape: {ref_shape}, CRS: {ref_crs}")

        # Cria dicionario de dados - VAL contem valores do raster de referencia
        data_dict = {'VAL': ref_data.flatten()}

        # Alinha cada sample raster ao grid de referencia
        for i, sample_path in enumerate(sample_rasters):
            sample_name = Path(sample_path).stem
            print(f"  Alinhando [{i+1}/{len(sample_rasters)}] {sample_name}...", end=" ")

            try:
                aligned = align_raster_to_reference(
                    source_path=sample_path,
                    ref_transform=ref_transform,
                    ref_crs=ref_crs,
                    ref_width=ref_width,
                    ref_height=ref_height,
                    ref_nodata=ref_nodata_val,
                )
                data_dict[sample_name] = aligned.flatten()
                print("OK")
            except Exception as e:
                print(f"ERRO: {e}")
                data_dict[sample_name] = np.full(ref_height * ref_width, np.nan)

        # Cria DataFrame
        df = pd.DataFrame(data_dict)
        df.insert(0, 'pixel_id', range(len(df)))
        df.insert(1, 'source', ref_name)

        # Adiciona coordenadas se solicitado
        if include_coords:
            rows, cols = ref_shape
            row_indices, col_indices = np.meshgrid(
                np.arange(rows), np.arange(cols), indexing='ij'
            )
            xs, ys = xy(ref_transform, row_indices.flatten(), col_indices.flatten())
            df.insert(2, 'X', xs)
            df.insert(3, 'Y', ys)

        # Remove linhas com nodata em TODOS os rasters
        if remove_nodata_rows:
            raster_cols = [c for c in df.columns if c not in ['pixel_id', 'source', 'X', 'Y']]
            n_before = len(df)

            # Mascara: True se TODOS os valores sao nodata/nan
            all_nodata_mask = pd.Series([True] * len(df))
            for col in raster_cols:
                col_data = df[col]
                col_mask = col_data.isna()
                if ref_nodata_val is not None:
                    col_mask = col_mask | np.isclose(col_data.fillna(0), ref_nodata_val, rtol=1e-5)
                all_nodata_mask = all_nodata_mask & col_mask

            df = df[~all_nodata_mask].reset_index(drop=True)
            n_removed = n_before - len(df)
            if n_removed > 0:
                print(f"  Removidos {n_removed} pixels nodata, restam {len(df)}")

        all_dfs.append(df)
        print(f"  {len(df)} pixels validos")

    # Concatena todos os DataFrames
    print("\n[INFO] Concatenando resultados...")
    df_final = pd.concat(all_dfs, ignore_index=True)

    # Reseta pixel_id para ser unico
    df_final['pixel_id'] = range(len(df_final))

    print(f"[INFO] DataFrame final: {df_final.shape[0]} linhas x {df_final.shape[1]} colunas")

    return df_final


def stack_rasters_to_dataframe(
    raster_paths: List[str],
    include_coords: bool = False,
    nodata: Optional[float] = None,
    remove_nodata_rows: bool = True,
) -> pd.DataFrame:
    """
    Empilha multiplos rasters e retorna um DataFrame com pixels nas linhas.

    Parameters
    ----------
    raster_paths : list of str
        Lista de caminhos para os arquivos raster.
    include_coords : bool
        Se True, inclui colunas X e Y com coordenadas geograficas.
    nodata : float, optional
        Valor nodata a usar. Se None, usa o valor do primeiro raster.
    remove_nodata_rows : bool
        Se True, remove linhas onde TODOS os valores sao nodata.

    Returns
    -------
    pd.DataFrame
        DataFrame com pixel_id nas linhas e valores de cada raster nas colunas.

    Raises
    ------
    ValueError
        Se os rasters nao tiverem o mesmo shape ou CRS.
    """
    if not raster_paths:
        raise ValueError("Nenhum raster fornecido")

    print(f"\n[INFO] Processando {len(raster_paths)} rasters...")

    # Abre primeiro raster para referencia
    with rasterio.open(raster_paths[0]) as ref:
        ref_shape = ref.shape
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_nodata = nodata if nodata is not None else ref.nodata

        print(f"[INFO] Shape de referencia: {ref_shape}")
        print(f"[INFO] CRS: {ref_crs}")
        print(f"[INFO] NoData: {ref_nodata}")

    # Le todos os rasters e verifica compatibilidade
    data_dict = {}
    nodata_values = []

    for i, raster_path in enumerate(raster_paths):
        path = Path(raster_path)
        col_name = path.stem

        print(f"  [{i+1}/{len(raster_paths)}] Lendo {path.name}...", end=" ")

        with rasterio.open(raster_path) as src:
            # Verifica shape
            if src.shape != ref_shape:
                raise ValueError(
                    f"Shape incompativel: {path.name} tem {src.shape}, "
                    f"esperado {ref_shape}"
                )

            # Verifica CRS
            if src.crs != ref_crs:
                print(f"\n[AVISO] CRS diferente em {path.name}: {src.crs} vs {ref_crs}")

            # Le dados
            band_data = src.read(1).flatten()
            data_dict[col_name] = band_data

            # Guarda nodata
            file_nodata = src.nodata if src.nodata is not None else ref_nodata
            nodata_values.append(file_nodata)

            print(f"OK ({band_data.size} pixels)")

    # Cria DataFrame
    df = pd.DataFrame(data_dict)

    # Adiciona indice de pixel
    df.insert(0, 'pixel_id', range(len(df)))

    # Adiciona coluna com basename do primeiro raster (identificador da origem)
    first_raster_name = Path(raster_paths[0]).stem
    df.insert(1, 'source', first_raster_name)

    # Adiciona coordenadas se solicitado
    if include_coords:
        print("[INFO] Calculando coordenadas...")
        rows, cols = ref_shape
        row_indices, col_indices = np.meshgrid(
            np.arange(rows), np.arange(cols), indexing='ij'
        )
        row_flat = row_indices.flatten()
        col_flat = col_indices.flatten()

        # Converte indices para coordenadas geograficas (centro do pixel)
        xs, ys = xy(ref_transform, row_flat, col_flat)
        df.insert(1, 'X', xs)
        df.insert(2, 'Y', ys)
        df.insert(3, 'row', row_flat)
        df.insert(4, 'col', col_flat)

    # Remove linhas com nodata em TODOS os rasters
    if remove_nodata_rows and ref_nodata is not None:
        raster_cols = [c for c in df.columns if c not in ['pixel_id', 'source', 'X', 'Y', 'row', 'col']]
        n_before = len(df)

        # Mascara: True se TODOS os valores sao nodata ou nan
        all_nodata_mask = pd.Series([True] * len(df))
        for col in raster_cols:
            col_mask = (
                df[col].isna() |
                np.isclose(df[col], ref_nodata, rtol=1e-5, equal_nan=True)
            )
            all_nodata_mask = all_nodata_mask & col_mask

        df = df[~all_nodata_mask].reset_index(drop=True)
        n_after = len(df)

        if n_before != n_after:
            print(f"[INFO] Removidos {n_before - n_after} pixels com nodata em todos rasters")

    print(f"[INFO] DataFrame final: {df.shape[0]} linhas x {df.shape[1]} colunas")

    return df


def process_rasters_to_excel(
    input_paths: Union[str, List[str]],
    output_excel: str,
    pattern: str = "*.tif",
    include_coords: bool = False,
    nodata: Optional[float] = None,
    remove_nodata_rows: bool = True,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Processa rasters e exporta tabela de pixels para Excel.

    Parameters
    ----------
    input_paths : str or list of str
        Caminho para pasta com rasters OU lista de caminhos de rasters.
    output_excel : str
        Caminho para o arquivo Excel de saida.
    pattern : str
        Padrao glob para filtrar arquivos (se input for pasta).
    include_coords : bool
        Se True, inclui coordenadas X, Y.
    nodata : float, optional
        Valor nodata.
    remove_nodata_rows : bool
        Se True, remove linhas onde todos valores sao nodata.
    chunk_size : int, optional
        Se fornecido, processa em chunks para economizar memoria.

    Returns
    -------
    pd.DataFrame
        DataFrame com os resultados.
    """
    print("=" * 60)
    print("RASTER STACK TO EXCEL")
    print("=" * 60)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Coleta arquivos
    if isinstance(input_paths, str):
        input_path = Path(input_paths)
        if input_path.is_dir():
            files = sorted(input_path.glob(pattern))
            print(f"Pasta: {input_path}")
            print(f"Padrao: {pattern}")
        else:
            files = [input_path]
    else:
        files = [Path(p) for p in input_paths]

    files = [str(f) for f in files if f.exists()]

    print(f"Arquivos encontrados: {len(files)}")
    for f in files:
        print(f"  - {Path(f).name}")

    if not files:
        raise ValueError("Nenhum arquivo raster encontrado")

    # Processa
    df = stack_rasters_to_dataframe(
        raster_paths=files,
        include_coords=include_coords,
        nodata=nodata,
        remove_nodata_rows=remove_nodata_rows,
    )

    # Exporta para Excel
    output_path = Path(output_excel)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verifica tamanho (Excel tem limite de ~1M linhas)
    MAX_EXCEL_ROWS = 1048576
    if len(df) > MAX_EXCEL_ROWS:
        print(f"\n[AVISO] DataFrame tem {len(df)} linhas, maior que limite Excel ({MAX_EXCEL_ROWS})")
        print("[AVISO] Exportando como CSV em vez de Excel...")
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSalvo em: {csv_path}")
    else:
        df.to_excel(output_excel, index=False, sheet_name='Pixels')
        print(f"\nSalvo em: {output_excel}")

    print(f"\n{'=' * 60}")
    print("RESUMO")
    print("=" * 60)
    print(f"Total de pixels: {len(df)}")
    print(f"Colunas: {list(df.columns)}")
    print(f"\nPrimeiras 5 linhas:")
    print(df.head().to_string(index=False))

    return df


def process_rasters_to_excel_with_sampling(
    reference_rasters: List[str],
    sample_rasters: List[str],
    output_excel: str,
    include_coords: bool = False,
    nodata: Optional[float] = None,
    remove_nodata_rows: bool = True,
) -> pd.DataFrame:
    """
    Processa rasters com sampling e exporta para Excel.

    Usa os rasters de referencia como base de coordenadas e faz sampling
    dos rasters maiores nas mesmas posicoes.

    Parameters
    ----------
    reference_rasters : list of str
        Rasters com mesmo shape (o primeiro define a referencia).
    sample_rasters : list of str
        Rasters a serem amostrados (podem ter shapes diferentes).
    output_excel : str
        Caminho para o arquivo Excel de saida.
    include_coords : bool
        Se True, inclui coordenadas X, Y.
    nodata : float, optional
        Valor nodata.
    remove_nodata_rows : bool
        Se True, remove linhas onde todos valores sao nodata.

    Returns
    -------
    pd.DataFrame
        DataFrame com os resultados.
    """
    print("=" * 60)
    print("RASTER STACK TO EXCEL (COM SAMPLING)")
    print("=" * 60)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nRasters de referencia: {len(reference_rasters)}")
    for f in reference_rasters:
        print(f"  - {Path(f).name}")
    print(f"\nRasters para sampling: {len(sample_rasters)}")
    for f in sample_rasters:
        print(f"  - {Path(f).name}")

    # Processa
    df = stack_rasters_with_sampling(
        reference_rasters=reference_rasters,
        sample_rasters=sample_rasters,
        include_coords=include_coords,
        nodata=nodata,
        remove_nodata_rows=remove_nodata_rows,
    )

    # Exporta para Excel
    output_path = Path(output_excel)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    MAX_EXCEL_ROWS = 1048576
    if len(df) > MAX_EXCEL_ROWS:
        print(f"\n[AVISO] DataFrame tem {len(df)} linhas, maior que limite Excel ({MAX_EXCEL_ROWS})")
        print("[AVISO] Exportando como CSV em vez de Excel...")
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSalvo em: {csv_path}")
    else:
        df.to_excel(output_excel, index=False, sheet_name='Pixels')
        print(f"\nSalvo em: {output_excel}")

    print(f"\n{'=' * 60}")
    print("RESUMO")
    print("=" * 60)
    print(f"Total de pixels: {len(df)}")
    print(f"Colunas: {list(df.columns)}")
    print(f"\nPrimeiras 5 linhas:")
    print(df.head().to_string(index=False))

    return df


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def get_pixel_value_at_coords(
    raster_paths: List[str],
    x: float,
    y: float,
) -> dict:
    """
    Extrai valor de multiplos rasters em uma coordenada especifica.

    Parameters
    ----------
    raster_paths : list of str
        Lista de caminhos para os rasters.
    x, y : float
        Coordenadas geograficas.

    Returns
    -------
    dict
        Dicionario {nome_raster: valor}.
    """
    result = {'X': x, 'Y': y}

    for raster_path in raster_paths:
        path = Path(raster_path)
        with rasterio.open(raster_path) as src:
            # Converte coordenada para indices
            row, col = src.index(x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                value = src.read(1)[row, col]
                result[path.stem] = float(value)
            else:
                result[path.stem] = None

    return result


# =============================================================================
# EXECUCAO
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Empilha rasters e exporta pixels para Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Varios rasters (mesmo shape)
  python RasterStackToExcel.py r1.tif r2.tif r3.tif -o pixels.xlsx

  # Pasta com rasters
  python RasterStackToExcel.py pasta/ -o pixels.xlsx

  # Com coordenadas X, Y
  python RasterStackToExcel.py pasta/ -o pixels.xlsx --coords

  # Manter pixels com nodata
  python RasterStackToExcel.py pasta/ -o pixels.xlsx --keep-nodata

  # Padrao diferente
  python RasterStackToExcel.py pasta/ -o pixels.xlsx -p "*_clip.tif"

  # COM SAMPLING (rasters de tamanhos diferentes):
  # Usa rasters de referencia (pequenos) e faz sampling dos rasters maiores
  python RasterStackToExcel.py ref1.tif ref2.tif -o pixels.xlsx --sample pasta_grandes/
  python RasterStackToExcel.py ref1.tif ref2.tif -o pixels.xlsx --sample grande1.tif grande2.tif
        """
    )

    parser.add_argument("input", nargs="+",
                        help="Rasters de referencia ou pasta com rasters")
    parser.add_argument("-o", "--output", default="raster_pixels.xlsx",
                        help="Arquivo Excel de saida (default: raster_pixels.xlsx)")
    parser.add_argument("-p", "--pattern", default="*.tif",
                        help="Padrao glob para filtrar arquivos (default: *.tif)")
    parser.add_argument("--coords", action="store_true",
                        help="Incluir coordenadas X, Y")
    parser.add_argument("--nodata", type=float,
                        help="Valor nodata (default: usa do arquivo)")
    parser.add_argument("--keep-nodata", action="store_true",
                        help="Manter linhas com nodata em todos rasters")
    parser.add_argument("--sample", nargs="+",
                        help="Rasters ou pasta para fazer sampling (podem ter shapes diferentes)")
    parser.add_argument("--sample-pattern", default="*.tif",
                        help="Padrao glob para filtrar rasters de sampling (default: *.tif)")

    args = parser.parse_args()

    # Coleta rasters de referencia
    if len(args.input) == 1 and Path(args.input[0]).is_dir():
        ref_files = sorted(Path(args.input[0]).glob(args.pattern))
        ref_files = [str(f) for f in ref_files]
    else:
        ref_files = [f for f in args.input if Path(f).exists()]

    # Modo com sampling
    if args.sample:
        # Coleta rasters para sampling
        sample_files = []
        for s in args.sample:
            s_path = Path(s)
            if s_path.is_dir():
                sample_files.extend([str(f) for f in sorted(s_path.glob(args.sample_pattern))])
            elif s_path.exists():
                sample_files.append(str(s_path))

        process_rasters_to_excel_with_sampling(
            reference_rasters=ref_files,
            sample_rasters=sample_files,
            output_excel=args.output,
            include_coords=args.coords,
            nodata=args.nodata,
            remove_nodata_rows=not args.keep_nodata,
        )
    else:
        # Modo normal (todos mesmo shape)
        process_rasters_to_excel(
            input_paths=ref_files,
            output_excel=args.output,
            pattern=args.pattern,
            include_coords=args.coords,
            nodata=args.nodata,
            remove_nodata_rows=not args.keep_nodata,
        )
