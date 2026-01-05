"""
RasterStats.py - Estatisticas de Rasters de Producao

Calcula estatisticas (media, soma, etc.) de rasters clipados e exporta
os resultados para Excel.

Uso:
    # Um raster
    python RasterStats.py raster.tif -o resultado.xlsx

    # Pasta com varios rasters (batch)
    python RasterStats.py pasta_rasters/ -o resultado.xlsx

    # Com estatisticas extras
    python RasterStats.py pasta_rasters/ -o resultado.xlsx --stats mean sum std min max

Autor: Leonardo Ippolito Rodrigues
Ano: 2026
Projeto: Mestrado - Predicao de Volume Florestal com LiDAR
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime
from scipy import stats as scipy_stats

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    raise ImportError("rasterio e necessario. Execute: pip install rasterio")


def get_raster_stats(
    raster_path: str,
    nodata: Optional[float] = None,
    stats: List[str] = ['mean']
) -> Dict:
    """
    Calcula estatisticas de um raster.

    Parameters
    ----------
    raster_path : str
        Caminho para o arquivo raster.
    nodata : float, optional
        Valor nodata. Se None, usa o do arquivo.
    stats : list
        Lista de estatisticas: 'mean', 'sum', 'std', 'min', 'max', 'median', 'count'

    Returns
    -------
    dict
        Dicionario com nome do arquivo e estatisticas.
    """
    path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        file_nodata = src.nodata if nodata is None else nodata

        # Mascara de pixels validos
        if file_nodata is not None:
            valid_mask = (data != file_nodata) & np.isfinite(data)
        else:
            valid_mask = np.isfinite(data)

        valid_data = data[valid_mask]

        result = {
            'Arquivo': path.stem,
            'Pixels_Total': data.size,
            'Pixels_Validos': valid_data.size,
        }

        if valid_data.size > 0:
            n = valid_data.size
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data, ddof=1)  # ddof=1 para amostra

            # Calcula estatisticas solicitadas
            stat_funcs = {
                'mean': lambda x: mean_val,
                'sum': np.sum,
                'std': lambda x: std_val,
                'min': np.min,
                'max': np.max,
                'median': np.median,
                'count': len,
            }

            for stat in stats:
                if stat in stat_funcs:
                    if stat == 'count':
                        result[stat.capitalize()] = n
                    else:
                        result[stat.capitalize()] = float(stat_funcs[stat](valid_data))
                else:
                    result[stat.capitalize()] = None

            # Erro padrao e erro de amostragem (t 95%)
            if n > 1:
                se = std_val / np.sqrt(n)  # Erro padrao
                t_value = scipy_stats.t.ppf(0.975, df=n-1)  # t critico bicaudal 95%
                ea = t_value * se  # Erro absoluto de amostragem
                ea_pct = (ea / mean_val) * 100 if mean_val != 0 else 0  # Erro relativo (%)

                result['SE'] = round(se, 4)  # Erro padrao
                result['t_95'] = round(t_value, 4)  # Valor t
                result['EA'] = round(ea, 4)  # Erro absoluto
                result['EA_pct'] = round(ea_pct, 2)  # Erro de amostragem %
                result['IC_inf'] = round(mean_val - ea, 4)  # IC inferior
                result['IC_sup'] = round(mean_val + ea, 4)  # IC superior
            else:
                result['SE'] = None
                result['t_95'] = None
                result['EA'] = None
                result['EA_pct'] = None
                result['IC_inf'] = None
                result['IC_sup'] = None
        else:
            for stat in stats:
                result[stat.capitalize()] = None
            result['SE'] = None
            result['t_95'] = None
            result['EA'] = None
            result['EA_pct'] = None
            result['IC_inf'] = None
            result['IC_sup'] = None

        # Area (se possivel calcular)
        try:
            pixel_area = abs(src.res[0] * src.res[1])
            result['Area_ha'] = round((valid_data.size * pixel_area) / 10000, 4)
        except:
            result['Area_ha'] = None

    return result


def process_rasters(
    input_path: str,
    output_excel: str,
    pattern: str = "*.tif",
    nodata: Optional[float] = None,
    stats: List[str] = ['mean'],
) -> pd.DataFrame:
    """
    Processa um ou mais rasters e exporta estatisticas para Excel.

    Parameters
    ----------
    input_path : str
        Caminho para um raster ou pasta com rasters.
    output_excel : str
        Caminho para o arquivo Excel de saida.
    pattern : str
        Padrao glob para filtrar arquivos (default: *.tif).
    nodata : float, optional
        Valor nodata. Se None, usa o do arquivo.
    stats : list
        Lista de estatisticas a calcular.

    Returns
    -------
    pd.DataFrame
        DataFrame com os resultados.
    """
    print("=" * 60)
    print("ESTATISTICAS DE RASTERS")
    print("=" * 60)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    input_path = Path(input_path)

    # Coleta arquivos
    if input_path.is_dir():
        files = sorted(input_path.glob(pattern))
        print(f"Pasta: {input_path}")
        print(f"Padrao: {pattern}")
    else:
        files = [input_path]

    print(f"Arquivos encontrados: {len(files)}")
    print(f"Estatisticas: {stats}")

    if not files:
        raise ValueError(f"Nenhum arquivo encontrado em {input_path}")

    # Processa cada arquivo
    results = []
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {f.name}...", end=" ")
        try:
            result = get_raster_stats(str(f), nodata=nodata, stats=stats)
            results.append(result)
            if 'Mean' in result and result['Mean'] is not None:
                print(f"Media: {result['Mean']:.2f}")
            else:
                print("OK")
        except Exception as e:
            print(f"ERRO: {e}")
            results.append({
                'Arquivo': f.stem,
                'Erro': str(e)
            })

    # Cria DataFrame
    df = pd.DataFrame(results)

    # Adiciona linha de totais/medias se houver mais de um arquivo
    if len(df) > 1 and 'Mean' in df.columns:
        totals = {'Arquivo': 'TOTAL/MEDIA'}

        if 'Pixels_Validos' in df.columns:
            totals['Pixels_Total'] = df['Pixels_Total'].sum()
            totals['Pixels_Validos'] = df['Pixels_Validos'].sum()

        if 'Area_ha' in df.columns:
            totals['Area_ha'] = df['Area_ha'].sum()

        for stat in stats:
            col = stat.capitalize()
            if col in df.columns:
                if stat == 'sum':
                    totals[col] = df[col].sum()
                elif stat == 'count':
                    totals[col] = df[col].sum()
                else:
                    totals[col] = df[col].mean()

        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    # Exporta para Excel
    output_path = Path(output_excel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False, sheet_name='Estatisticas')

    print(f"\n{'=' * 60}")
    print("RESULTADO")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSalvo em: {output_excel}")

    return df


def volume_total_by_area(
    raster_path: str,
    nodata: Optional[float] = None
) -> Dict:
    """
    Calcula volume total (soma * area do pixel) de um raster de volume (m3/ha).

    Parameters
    ----------
    raster_path : str
        Caminho para o raster de volume.
    nodata : float, optional
        Valor nodata.

    Returns
    -------
    dict
        Dicionario com volume medio, area e volume total.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        file_nodata = src.nodata if nodata is None else nodata

        if file_nodata is not None:
            valid_mask = (data != file_nodata) & np.isfinite(data)
        else:
            valid_mask = np.isfinite(data)

        valid_data = data[valid_mask]

        # Area do pixel em hectares
        pixel_area_m2 = abs(src.res[0] * src.res[1])
        pixel_area_ha = pixel_area_m2 / 10000

        # Calculos
        n_pixels = valid_data.size
        area_total_ha = n_pixels * pixel_area_ha
        volume_medio = float(np.mean(valid_data)) if n_pixels > 0 else 0
        volume_total = volume_medio * area_total_ha

        return {
            'arquivo': Path(raster_path).stem,
            'pixels_validos': n_pixels,
            'area_pixel_ha': pixel_area_ha,
            'area_total_ha': round(area_total_ha, 4),
            'volume_medio_m3_ha': round(volume_medio, 2),
            'volume_total_m3': round(volume_total, 2),
        }


# =============================================================================
# EXECUCAO
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calcula estatisticas de rasters e exporta para Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Um raster
  python RasterStats.py volume.tif -o resultado.xlsx

  # Pasta com rasters
  python RasterStats.py pasta_clips/ -o resultado.xlsx

  # Com estatisticas extras
  python RasterStats.py pasta/ -o resultado.xlsx --stats mean sum min max

  # Padrao diferente
  python RasterStats.py pasta/ -o resultado.xlsx -p "*_volume.tif"
        """
    )

    parser.add_argument("input", help="Raster ou pasta com rasters")
    parser.add_argument("-o", "--output", default="raster_stats.xlsx",
                        help="Arquivo Excel de saida (default: raster_stats.xlsx)")
    parser.add_argument("-p", "--pattern", default="*.tif",
                        help="Padrao glob para filtrar arquivos (default: *.tif)")
    parser.add_argument("--nodata", type=float,
                        help="Valor nodata (default: usa do arquivo)")
    parser.add_argument("--stats", nargs="+",
                        default=["mean", "sum", "std", "min", "max", "median"],
                        choices=["mean", "sum", "std", "min", "max", "median", "count"],
                        help="Estatisticas a calcular (default: mean)")

    args = parser.parse_args()

    process_rasters(
        input_path=args.input,
        output_excel=args.output,
        pattern=args.pattern,
        nodata=args.nodata,
        stats=args.stats,
    )
