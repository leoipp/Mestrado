"""
MergeTiff.py - Mosaico de Rasters GeoTIFF (ROBUSTO)

Este script combina múltiplos arquivos GeoTIFF em um único mosaico,
resolvendo sobreposições e mantendo o georreferenciamento.

Correções/robustez adicionadas:
- Trata NoData = None (muito comum em float32) definindo um nodata efetivo seguro
- Remove valores absurdos (ex.: ~1e+38), NaN e Inf antes de salvar
- Evita "upside down" no mosaico final (pixel height sempre negativo)
- Salva com BIGTIFF=IF_SAFER e tiled=True (essencial para mosaicos grandes)
- Predictor=3 para floats com compressão (melhor tamanho/performance)
- Métodos max/min/mean corrigidos para ignorar nodata dos dois lados
- Estatísticas finais robustas (não quebram com nodata None)
- merge_by_groups corrigido (agrupamento real por sufixo/padrão)

Dependências:
- rasterio
- numpy

Autor: Leonardo Ippolito Rodrigues
Revisão crítica/patch: 2026
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Union, Dict
from datetime import datetime

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    raise ImportError("rasterio é necessário. Execute: pip install rasterio")


# =============================================================================
# MÉTODOS DE MERGE (CORRIGIDOS PARA API RASTERIO >= 1.3)
# =============================================================================
# IMPORTANTE: A partir do rasterio 1.3, os parâmetros old_nodata e new_nodata
# são MÁSCARAS BOOLEANAS (True = nodata), não valores float!
# =============================================================================


def method_first(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Mantém o primeiro valor válido (preenche apenas onde old é nodata)."""
    # old_nodata/new_nodata são máscaras: True onde é nodata
    # Onde old é nodata (~old_valid) e new é válido (~new_nodata), copia
    old_is_nodata = old_nodata
    new_is_valid = ~new_nodata
    np.copyto(old_data, new_data, where=(old_is_nodata & new_is_valid))


def method_last(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Sobrescreve com o último valor válido (onde new é válido)."""
    new_is_valid = ~new_nodata
    np.copyto(old_data, new_data, where=new_is_valid)


def method_max(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Mantém o valor máximo nas sobreposições."""
    old_is_valid = ~old_nodata
    new_is_valid = ~new_nodata

    # Onde só o novo é válido, copia
    np.copyto(old_data, new_data, where=(old_nodata & new_is_valid))

    # Onde ambos válidos, faz max
    both_valid = old_is_valid & new_is_valid
    np.maximum(old_data, new_data, out=old_data, where=both_valid)


def method_min(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Mantém o valor mínimo nas sobreposições."""
    old_is_valid = ~old_nodata
    new_is_valid = ~new_nodata

    # Onde só o novo é válido, copia
    np.copyto(old_data, new_data, where=(old_nodata & new_is_valid))

    # Onde ambos válidos, faz min
    both_valid = old_is_valid & new_is_valid
    np.minimum(old_data, new_data, out=old_data, where=both_valid)


def method_mean(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Calcula média nas sobreposições."""
    old_is_valid = ~old_nodata
    new_is_valid = ~new_nodata
    both_valid = old_is_valid & new_is_valid

    # Média onde ambos válidos
    np.copyto(old_data, (old_data + new_data) / 2.0, where=both_valid)

    # Onde só o novo é válido -> copia
    np.copyto(old_data, new_data, where=(old_nodata & new_is_valid))


MERGE_METHODS = {
    "first": method_first,
    "last": method_last,
    "max": method_max,
    "min": method_min,
    "mean": method_mean,
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_raster_info(file_path: str) -> dict:
    """Obtém informações de um arquivo raster."""
    with rasterio.open(file_path) as src:
        return {
            "path": file_path,
            "crs": src.crs,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "resolution": src.res,
            "transform": src.transform,
        }


def validate_rasters(file_list: List[str]) -> dict:
    """Valida compatibilidade entre rasters para merge."""
    if not file_list:
        raise ValueError("Lista de arquivos vazia")

    infos = [get_raster_info(f) for f in file_list]

    # CRS
    crs_set = set(str(info["crs"]) for info in infos if info["crs"] is not None)
    if len(crs_set) > 1:
        print(f"  Aviso: CRS diferentes detectados: {crs_set}")
        print("  O merge usará o CRS do primeiro arquivo (não nulo, se houver).")

    # Bandas
    band_counts = set(info["count"] for info in infos)
    if len(band_counts) > 1:
        raise ValueError(f"Número de bandas inconsistente: {band_counts}")

    # dtype
    dtypes = set(info["dtype"] for info in infos)
    if len(dtypes) > 1:
        print(f"  Aviso: dtypes diferentes: {dtypes} (o output usará o do primeiro)")

    # bounds totais
    all_bounds = [info["bounds"] for info in infos]
    total_bounds = (
        min(b.left for b in all_bounds),
        min(b.bottom for b in all_bounds),
        max(b.right for b in all_bounds),
        max(b.top for b in all_bounds),
    )

    # CRS de referência: primeiro não-nulo, senão None
    ref_crs = None
    for inf in infos:
        if inf["crs"] is not None:
            ref_crs = inf["crs"]
            break

    return {
        "count": len(file_list),
        "crs": ref_crs,
        "band_count": infos[0]["count"],
        "dtype": infos[0]["dtype"],
        "nodata": infos[0]["nodata"],
        "total_bounds": total_bounds,
        "resolutions": [info["resolution"] for info in infos],
    }


def _pick_effective_nodata(
    nodata_arg: Optional[float],
    info_nodata,
    out_dtype: str,
    fallback_float: float = -9999.0,
) -> Optional[float]:
    """
    Define um nodata efetivo robusto.
    - Se nodata_arg foi passado -> usa
    - Senão usa info_nodata
    - Se ainda for None e dtype é float -> fallback_float
    """
    nd = nodata_arg if nodata_arg is not None else info_nodata
    if nd is None and np.issubdtype(np.dtype(out_dtype), np.floating):
        nd = float(fallback_float)
    return nd


def _sanitize_mosaic(
    mosaic: np.ndarray,
    out_dtype: str,
    out_nodata: Optional[float],
    huge_threshold: float = 1e20,
) -> np.ndarray:
    """
    Limpa NaN/Inf e valores gigantes (ex.: ~1e+38) que bagunçam simbologia/estatística.
    Para float: substitui por out_nodata (se definido). Se out_nodata None, substitui por 0.
    """
    arr = mosaic.astype(out_dtype, copy=True)

    if np.issubdtype(arr.dtype, np.floating):
        rep = out_nodata if out_nodata is not None else 0.0
        # NaN/Inf
        arr[~np.isfinite(arr)] = rep
        # fill values gigantes (positivos e negativos)
        arr[np.abs(arr) > huge_threshold] = rep

    return arr


def _ensure_north_up(mosaic: np.ndarray, transform: Affine) -> (np.ndarray, Affine):
    """
    Garante que pixel height (transform.e) seja NEGATIVO (north-up).
    Se vier positivo, faz flip vertical nos dados e corrige o transform.
    """
    if transform.e < 0:
        return mosaic, transform

    # flip vertical nas linhas
    mosaic2 = mosaic[:, ::-1, :]

    # corrige transform: e negativo e ajusta f
    new_transform = Affine(
        transform.a, transform.b, transform.c,
        transform.d, -transform.e, transform.f + transform.e * mosaic.shape[1]
    )
    return mosaic2, new_transform


def _print_basic_checks(output_file: str):
    """Imprime checks básicos do arquivo gerado."""
    p = Path(output_file)
    if not p.exists():
        raise RuntimeError(f"Arquivo de saída não foi criado: {output_file}")
    if p.stat().st_size == 0:
        raise RuntimeError(f"Arquivo de saída foi criado mas está vazio (0 bytes): {output_file}")

    with rasterio.open(output_file) as ds:
        print(f"  OK: CRS={ds.crs}")
        print(f"  OK: Bounds={ds.bounds}")
        print(f"  OK: Res={ds.res}")
        print(f"  OK: Transform.e (pixel height)={ds.transform.e}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def merge_tiffs(
    input_files: Union[str, List[str]],
    output_file: str,
    pattern: str = "*.tif",
    method: str = "first",
    nodata: Optional[float] = None,
    resolution: Optional[float] = None,
    resampling: str = "nearest",
    compress: str = "lzw",
    dtype: Optional[str] = None,
    overwrite: bool = True,
    huge_threshold: float = 1e20,
) -> str:
    """
    Combina múltiplos arquivos GeoTIFF em um único mosaico.
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

    print(f"  CRS (referência): {info['crs']}")
    print(f"  Bandas: {info['band_count']}")
    print(f"  Dtype (referência): {info['dtype']}")
    print(f"  NoData (1º arquivo): {info['nodata']}")
    print(f"  Bounds: {info['total_bounds']}")

    # Saída
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
        merge_kwargs = {}

        # Método
        if method in MERGE_METHODS:
            merge_kwargs["method"] = MERGE_METHODS[method]
        else:
            raise ValueError(f"Método inválido: {method}. Use: {list(MERGE_METHODS.keys())}")

        # Dtype de saída desejado
        out_dtype = dtype if dtype else info["dtype"]

        # NoData: só passa para o merge se o arquivo de entrada tiver nodata definido
        # ou se o usuário passou explicitamente. Caso contrário, deixa o merge usar
        # o nodata do próprio arquivo (mesmo que seja None).
        if nodata is not None:
            merge_kwargs["nodata"] = float(nodata)
            print(f"  Usando nodata explícito: {nodata}")
        elif info["nodata"] is not None:
            merge_kwargs["nodata"] = float(info["nodata"])
            print(f"  Usando nodata do arquivo: {info['nodata']}")

        # Resolução
        if resolution is not None:
            merge_kwargs["res"] = float(resolution)

        # Reamostragem
        resampling_methods = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "lanczos": Resampling.lanczos,
        }
        if resampling in resampling_methods:
            merge_kwargs["resampling"] = resampling_methods[resampling]
        else:
            raise ValueError(f"Resampling inválido: {resampling}")

        # Executa merge (mosaic em memória)
        mosaic, out_transform = merge(src_files, **merge_kwargs)

        # Garante north-up
        mosaic, out_transform = _ensure_north_up(mosaic, out_transform)

        print(f"  Dimensões do mosaico: {mosaic.shape[2]} x {mosaic.shape[1]} pixels")
        print(f"  Bandas: {mosaic.shape[0]}")
        print(f"  Transform.e (pixel height): {out_transform.e}")

    finally:
        for src in src_files:
            src.close()

    # -------------------------------------------------------------------------
    # 4. SALVAMENTO (ROBUSTO)
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Salvando mosaico: {output_file}")

    out_dtype = dtype if dtype else info["dtype"]
    out_nodata = _pick_effective_nodata(nodata, info["nodata"], out_dtype)

    # sanitiza dados (remove NaN/Inf e valores gigantes)
    mosaic_to_write = _sanitize_mosaic(mosaic, out_dtype=out_dtype, out_nodata=out_nodata, huge_threshold=huge_threshold)

    compress_opts = {} if compress == "none" else {"compress": compress}

    out_meta = {
        "driver": "GTiff",
        "height": mosaic_to_write.shape[1],
        "width": mosaic_to_write.shape[2],
        "count": mosaic_to_write.shape[0],
        "dtype": out_dtype,
        "crs": info["crs"],
        "transform": out_transform,
        "nodata": out_nodata,
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
        **compress_opts,
    }

    # Melhor compactação p/ float
    if np.issubdtype(np.dtype(out_dtype), np.floating) and compress != "none":
        out_meta["predictor"] = 3

    # Escreve
    with rasterio.open(output_file, "w", **out_meta) as dst:
        dst.write(mosaic_to_write)

    # Checkpoints
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Tamanho: {size_mb:.2f} MB")

    # Estatísticas (robustas)
    if out_nodata is None:
        valid = mosaic_to_write[np.isfinite(mosaic_to_write)]
    else:
        valid = mosaic_to_write[(mosaic_to_write != out_nodata) & np.isfinite(mosaic_to_write)]

    if valid.size > 0:
        print("\n  Estatísticas (válidos):")
        print(f"    Min:  {float(valid.min()):.4f}")
        print(f"    Max:  {float(valid.max()):.4f}")
        print(f"    Mean: {float(valid.mean()):.4f}")
    else:
        print("\n  Aviso: mosaico sem pixels válidos após sanitização (tudo nodata).")

    # Confirma leitura do arquivo final
    _print_basic_checks(output_file)

    print("\n" + "=" * 60)
    print("MERGE CONCLUÍDO")
    print(f"Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return output_file


# =============================================================================
# MERGE POR GRUPOS (CORRIGIDO)
# =============================================================================

def merge_by_groups(
    input_folder: str,
    output_folder: str,
    group_pattern: str,
    **kwargs
) -> List[str]:
    """
    Faz mosaicos separados para cada "grupo" identificado por `group_pattern`.

    Exemplos:
      group_pattern = "_max"   -> agrupa todos os *{group_pattern}.tif (ex: elev_max, p95_max, etc.)
      group_pattern = "_chm"   -> agrupa todos os *{group_pattern}.tif

    Observação:
      Se seus arquivos são como:
        elev_max_tile001.tif, elev_max_tile002.tif, p95_max_tile001.tif ...
      então group_pattern="_max" e o grupo será pelo prefixo antes de "_max".
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_path.glob("*.tif"))
    if not all_files:
        raise ValueError("Nenhum .tif encontrado na pasta de entrada.")

    groups: Dict[str, List[str]] = {}

    for f in all_files:
        stem = f.stem
        if group_pattern not in stem:
            continue

        # Grupo = parte antes do group_pattern (ex: "elev" em "elev_max_tile001")
        prefix = stem.split(group_pattern)[0].rstrip("_-")
        group_name = prefix if prefix else group_pattern.strip("_-")
        groups.setdefault(group_name, []).append(str(f))

    print(f"Grupos encontrados: {list(groups.keys())}")

    results: List[str] = []
    for group_name, files in groups.items():
        print(f"\n{'='*60}")
        print(f"Processando grupo: {group_name} ({len(files)} arquivos)")
        print(f"{'='*60}")

        out_file = output_path / f"{group_name}{group_pattern}_mosaic.tif"
        try:
            results.append(merge_tiffs(files, str(out_file), **kwargs))
        except Exception as e:
            print(f"Erro no grupo {group_name}: {e}")

    return results


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combina múltiplos GeoTIFFs em um mosaico (robusto)"
    )
    parser.add_argument("input", help="Pasta com TIFFs ou arquivo único")
    parser.add_argument("output", help="Arquivo de saída")
    parser.add_argument("-p", "--pattern", default="*.tif",
                        help="Padrão glob para filtrar arquivos (default: *.tif)")
    parser.add_argument("-m", "--method", choices=list(MERGE_METHODS.keys()),
                        default="first", help="Método de merge (default: first)")
    parser.add_argument("--nodata", type=float, help="Valor nodata (se não informar, tenta usar do 1º; se None e float, usa -9999)")
    parser.add_argument("--resolution", type=float, help="Resolução do mosaico (se None, usa a do rasterio.merge)")
    parser.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "lanczos"],
                        default="nearest", help="Método de reamostragem (default: nearest)")
    parser.add_argument("--compress", choices=["lzw", "deflate", "packbits", "none"],
                        default="lzw", help="Compressão (default: lzw)")
    parser.add_argument("--dtype", choices=["float32", "float64", "int16", "int32", "uint8", "uint16"],
                        help="Tipo de dado de saída (se None, usa o do primeiro)")
    parser.add_argument("--no-overwrite", action="store_true", help="Não sobrescrever arquivo de saída se existir")
    parser.add_argument("--huge-threshold", type=float, default=1e20,
                        help="Valores maiores que isso (em float) viram nodata (default: 1e20)")

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
        dtype=args.dtype,
        overwrite=(not args.no_overwrite),
        huge_threshold=args.huge_threshold,
    )
