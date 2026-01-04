"""
PredictVolume.py - Predição de Volume Florestal a partir de Rasters LiDAR

Este script aplica um modelo Random Forest treinado para predizer volume
florestal (VTCC) a partir de rasters de métricas LiDAR.

Funcionalidades:
    - Carrega modelo treinado (.pkl) do pipeline 04_RandomForestTrain.py
    - Lê múltiplos rasters (uma banda por variável)
    - Alinha e reprojeta rasters automaticamente
    - Gera raster de predição (volume estimado)
    - Gera raster de incerteza (desvio padrão entre árvores do RF)
    - Suporte a processamento em lote por talhão

Entradas:
    - Rasters GeoTIFF das variáveis (Elev_P90.tif, Elev_P60.tif, etc.)
    - Modelo treinado (.pkl)
    - Variáveis auxiliares (ROTACAO, REGIONAL, Idade) como rasters TIF

Saídas:
    - {nome}_volume_estimado.tif: Predição de VTCC (m³/ha)
    - {nome}_volume_incerteza.tif: Desvio padrão da predição

Dependências:
    - numpy
    - rasterio
    - joblib
    - scikit-learn

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
except ImportError:
    raise ImportError("rasterio é necessário. Execute: pip install rasterio")

try:
    import joblib
except ImportError:
    raise ImportError("joblib é necessário. Execute: pip install joblib")


# =============================================================================
# CONFIGURAÇÕES PADRÃO
# =============================================================================

# Variáveis esperadas pelo modelo (ordem importa!)
DEFAULT_FEATURE_ORDER = [
    'Elev P90',
    'Elev P60',
    'Elev maximum',
    'ROTACAO',
    'REGIONAL',
    'Idade (meses)'
]

# Mapeamento de nomes de arquivo para variáveis
# Usado para detectar automaticamente variáveis a partir de nomes de arquivos
DEFAULT_FILE_MAPPING = {
    # Métricas LiDAR
    'elev_p90': 'Elev P90',
    'elev p90': 'Elev P90',
    'p90': 'Elev P90',
    'elev_p60': 'Elev P60',
    'elev p60': 'Elev P60',
    'p60': 'Elev P60',
    'elev_maximum': 'Elev maximum',
    'elev_max': 'Elev maximum',
    'maximum': 'Elev maximum',
    'max': 'Elev maximum',
    # Variáveis auxiliares (rasters)
    'rotacao': 'ROTACAO',
    'rotation': 'ROTACAO',
    'regional': 'REGIONAL',
    'region': 'REGIONAL',
    'idade': 'Idade (meses)',
    'age': 'Idade (meses)',
    'idade_meses': 'Idade (meses)',
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def load_model(model_path: str):
    """
    Carrega modelo Random Forest treinado.

    Parameters
    ----------
    model_path : str
        Caminho do arquivo .pkl do modelo.

    Returns
    -------
    sklearn.ensemble.RandomForestRegressor
        Modelo carregado.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = joblib.load(model_path)
    print(f"Modelo carregado: {model_path}")

    # Verifica se tem feature_names
    if hasattr(model, 'feature_names_in_'):
        print(f"  Features: {list(model.feature_names_in_)}")
    elif hasattr(model, 'feature_names'):
        print(f"  Features: {model.feature_names}")

    if hasattr(model, 'n_estimators'):
        print(f"  Árvores: {model.n_estimators}")

    return model


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
        Dicionário com metadados do raster.
    """
    with rasterio.open(file_path) as src:
        return {
            'path': file_path,
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'nodata': src.nodata,
            'dtype': src.dtypes[0],
            'bounds': src.bounds
        }


def align_raster(
    source_path: str,
    reference_info: dict,
    resampling_method: Resampling = Resampling.bilinear
) -> Tuple[np.ndarray, float]:
    """
    Lê e alinha um raster ao raster de referência.

    Parameters
    ----------
    source_path : str
        Caminho do raster a ser alinhado.
    reference_info : dict
        Informações do raster de referência.
    resampling_method : Resampling
        Método de reamostragem.

    Returns
    -------
    tuple
        (array alinhado, valor nodata)
    """
    with rasterio.open(source_path) as src:
        band = src.read(1).astype('float32')
        nodata = src.nodata if src.nodata is not None else np.nan

        # Verifica se precisa reprojetar/alinhar
        needs_align = (
            src.crs != reference_info['crs'] or
            src.transform != reference_info['transform'] or
            src.width != reference_info['width'] or
            src.height != reference_info['height']
        )

        if needs_align:
            aligned = np.empty(
                (reference_info['height'], reference_info['width']),
                dtype='float32'
            )

            reproject(
                source=band,
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_info['transform'],
                dst_crs=reference_info['crs'],
                resampling=resampling_method
            )
            band = aligned

        return band, nodata


def predict_with_uncertainty(
    model,
    X: np.ndarray,
    n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Faz predição com estimativa de incerteza usando todas as árvores do RF.

    Parameters
    ----------
    model : RandomForestRegressor
        Modelo treinado.
    X : np.ndarray
        Dados de entrada (n_samples, n_features).
    n_jobs : int
        Número de jobs paralelos.

    Returns
    -------
    tuple
        (média das predições, desvio padrão das predições)
    """
    # Predição por cada árvore
    all_predictions = np.array([
        tree.predict(X) for tree in model.estimators_
    ])

    # Média e desvio padrão
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)

    return mean_pred, std_pred


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def predict_volume(
    raster_paths: Dict[str, str],
    model: Union[str, object],
    output_dir: str,
    output_name: Optional[str] = None,
    feature_order: Optional[List[str]] = None,
    nodata_output: float = -9999.0,
    compress: str = 'lzw',
    calculate_uncertainty: bool = True
) -> Dict[str, str]:
    """
    Prediz volume florestal a partir de rasters de métricas LiDAR.

    Parameters
    ----------
    raster_paths : dict
        Dicionário mapeando nome da variável para caminho do raster (.tif).
        Todas as 6 variáveis devem ser fornecidas como rasters:
            {
                'Elev P90': 'metricas/p90.tif',
                'Elev P60': 'metricas/p60.tif',
                'Elev maximum': 'metricas/max.tif',
                'ROTACAO': 'metricas/rotacao.tif',
                'REGIONAL': 'metricas/regional.tif',
                'Idade (meses)': 'metricas/idade.tif'
            }
    model : str or RandomForestRegressor
        Caminho do modelo (.pkl) ou modelo já carregado.
    output_dir : str
        Diretório para salvar os rasters de saída.
    output_name : str, optional
        Nome base para os arquivos de saída. Se None, usa nome do diretório.
    feature_order : list, optional
        Ordem das features esperada pelo modelo.
    nodata_output : float
        Valor nodata para os rasters de saída.
    compress : str
        Método de compressão ('lzw', 'deflate', 'none').
    calculate_uncertainty : bool
        Se True, calcula e salva raster de incerteza.

    Returns
    -------
    dict
        Dicionário com caminhos dos arquivos gerados.

    Examples
    --------
    predict_volume(
        raster_paths={
            'Elev P90': 'metricas/p90.tif',
            'Elev P60': 'metricas/p60.tif',
            'Elev maximum': 'metricas/max.tif',
            'ROTACAO': 'metricas/rotacao.tif',
            'REGIONAL': 'metricas/regional.tif',
            'Idade (meses)': 'metricas/idade.tif'
        },
        model='Models/RandomForestRegressor.pkl',
        output_dir='resultados/'
    )
    """
    print("=" * 70)
    print("PREDIÇÃO DE VOLUME FLORESTAL")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------------------------------------------------------------
    # 1. CARREGAMENTO DO MODELO
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Carregando modelo...")

    if isinstance(model, str):
        model = load_model(model)

    # Obtém ordem das features do modelo
    if feature_order is None:
        if hasattr(model, 'feature_names_in_'):
            feature_order = list(model.feature_names_in_)
        elif hasattr(model, 'feature_names'):
            feature_order = list(model.feature_names)
        else:
            feature_order = DEFAULT_FEATURE_ORDER
            print(f"  Aviso: usando ordem de features padrão")

    print(f"  Features esperadas: {feature_order}")

    # -------------------------------------------------------------------------
    # 2. VALIDAÇÃO DAS ENTRADAS
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Validando entradas...")

    # Verifica se todas as features estão disponíveis
    missing = [f for f in feature_order if f not in raster_paths]
    if missing:
        raise ValueError(f"Features faltando: {missing}")

    # Verifica se todos os arquivos existem
    for var_name, raster_path in raster_paths.items():
        if not Path(raster_path).exists():
            raise FileNotFoundError(f"Raster não encontrado: {raster_path} ({var_name})")

    print(f"  Variáveis: {list(raster_paths.keys())}")

    # -------------------------------------------------------------------------
    # 3. CARREGAMENTO E ALINHAMENTO DOS RASTERS
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Carregando e alinhando rasters...")

    # Primeiro raster como referência (usa a primeira feature na ordem)
    first_raster_path = raster_paths[feature_order[0]]
    ref_info = get_raster_info(first_raster_path)

    print(f"  Referência: {Path(first_raster_path).name}")
    print(f"  Dimensões: {ref_info['width']} x {ref_info['height']}")
    print(f"  CRS: {ref_info['crs']}")

    # Carrega todos os rasters na ordem correta
    feature_arrays = []
    nodata_values = []

    for feature_name in feature_order:
        raster_path = raster_paths[feature_name]
        band, nodata = align_raster(raster_path, ref_info)
        feature_arrays.append(band)
        nodata_values.append(nodata)
        print(f"    {feature_name}: {Path(raster_path).name} "
              f"(min={np.nanmin(band):.2f}, max={np.nanmax(band):.2f})")

    # -------------------------------------------------------------------------
    # 4. PREDIÇÃO
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Executando predição...")

    # Empilha features em (H, W, n_features)
    stack = np.stack(feature_arrays, axis=-1)
    height, width, n_features = stack.shape
    n_pixels = height * width

    # Achata para (n_pixels, n_features)
    flat_stack = stack.reshape(-1, n_features)

    # Cria máscara de pixels válidos
    valid_mask = np.ones(n_pixels, dtype=bool)
    for i, nodata in enumerate(nodata_values):
        band = flat_stack[:, i]
        if np.isnan(nodata):
            valid_mask &= ~np.isnan(band)
        else:
            valid_mask &= (band != nodata) & ~np.isnan(band)

    n_valid = np.sum(valid_mask)
    print(f"  Pixels válidos: {n_valid:,} de {n_pixels:,} ({100*n_valid/n_pixels:.1f}%)")

    # Extrai dados válidos
    valid_data = flat_stack[valid_mask]

    # Predição com incerteza
    print(f"  Predizendo com {model.n_estimators} árvores...")
    mean_pred, std_pred = predict_with_uncertainty(model, valid_data)

    # Reconstrói rasters
    predicted_full = np.full(n_pixels, nodata_output, dtype='float32')
    predicted_full[valid_mask] = mean_pred.astype('float32')
    predicted_raster = predicted_full.reshape((height, width))

    if calculate_uncertainty:
        uncertainty_full = np.full(n_pixels, nodata_output, dtype='float32')
        uncertainty_full[valid_mask] = std_pred.astype('float32')
        uncertainty_raster = uncertainty_full.reshape((height, width))

    # Estatísticas
    valid_pred = predicted_raster[predicted_raster != nodata_output]
    print(f"\n  Estatísticas da predição:")
    print(f"    Min:  {valid_pred.min():.2f} m³/ha")
    print(f"    Max:  {valid_pred.max():.2f} m³/ha")
    print(f"    Mean: {valid_pred.mean():.2f} m³/ha")
    print(f"    Std:  {valid_pred.std():.2f} m³/ha")

    # -------------------------------------------------------------------------
    # 5. SALVAMENTO DOS RESULTADOS
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Salvando resultados...")

    # Cria diretório de saída
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define nome base
    if output_name is None:
        output_name = Path(first_raster_path).parent.name

    # Perfil de saída
    with rasterio.open(first_raster_path) as src:
        out_profile = src.profile.copy()

    compress_opts = {} if compress == 'none' else {'compress': compress}
    out_profile.update(
        dtype='float32',
        count=1,
        nodata=nodata_output,
        **compress_opts
    )

    # Salva predição
    pred_path = output_path / f"{output_name}_volume_estimado.tif"
    with rasterio.open(pred_path, 'w', **out_profile) as dst:
        dst.write(predicted_raster, 1)
        dst.descriptions = ('VTCC_estimado_m3ha',)
    print(f"  Volume estimado: {pred_path}")

    # Salva incerteza
    output_files = {'prediction': str(pred_path)}

    if calculate_uncertainty:
        unc_path = output_path / f"{output_name}_volume_incerteza.tif"
        with rasterio.open(unc_path, 'w', **out_profile) as dst:
            dst.write(uncertainty_raster, 1)
            dst.descriptions = ('VTCC_incerteza_std',)
        print(f"  Incerteza: {unc_path}")
        output_files['uncertainty'] = str(unc_path)

    print("\n" + "=" * 70)
    print("PREDIÇÃO CONCLUÍDA")
    print(f"Término: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return output_files


def predict_volume_batch(
    input_folders: List[str],
    model: Union[str, object],
    output_dir: str,
    raster_pattern: Dict[str, str],
    **kwargs
) -> List[Dict[str, str]]:
    """
    Processa múltiplos talhões em lote.

    Parameters
    ----------
    input_folders : list
        Lista de pastas, cada uma contendo rasters de um talhão.
    model : str or object
        Modelo ou caminho do modelo.
    output_dir : str
        Diretório de saída.
    raster_pattern : dict
        Mapeamento de variável para padrão glob de nome de arquivo.
        Exemplo:
            {
                'Elev P90': '*p90*.tif',
                'Elev P60': '*p60*.tif',
                'Elev maximum': '*max*.tif',
                'ROTACAO': '*rotacao*.tif',
                'REGIONAL': '*regional*.tif',
                'Idade (meses)': '*idade*.tif'
            }
    **kwargs
        Argumentos adicionais para predict_volume.

    Returns
    -------
    list
        Lista de dicionários com arquivos gerados por talhão.

    Examples
    --------
    predict_volume_batch(
        input_folders=['talhao_01/', 'talhao_02/', 'talhao_03/'],
        model='model.pkl',
        output_dir='resultados/',
        raster_pattern={
            'Elev P90': '*p90*.tif',
            'Elev P60': '*p60*.tif',
            'Elev maximum': '*max*.tif',
            'ROTACAO': '*rotacao*.tif',
            'REGIONAL': '*regional*.tif',
            'Idade (meses)': '*idade*.tif'
        }
    )
    """
    # Carrega modelo uma vez
    if isinstance(model, str):
        model = load_model(model)

    results = []

    for i, folder in enumerate(input_folders, 1):
        folder_path = Path(folder)
        print(f"\n{'='*70}")
        print(f"TALHÃO {i}/{len(input_folders)}: {folder_path.name}")
        print('='*70)

        # Monta dicionário de rasters para este talhão
        raster_paths = {}
        missing_vars = []

        for var_name, pattern in raster_pattern.items():
            matches = list(folder_path.glob(pattern))
            if matches:
                raster_paths[var_name] = str(matches[0])
            else:
                missing_vars.append(var_name)

        if missing_vars:
            print(f"  Erro: rasters não encontrados para: {missing_vars}")
            continue

        try:
            result = predict_volume(
                raster_paths=raster_paths,
                model=model,
                output_dir=output_dir,
                output_name=folder_path.name,
                **kwargs
            )
            results.append(result)
        except Exception as e:
            print(f"  Erro ao processar {folder_path.name}: {e}")

    print(f"\n\nProcessados: {len(results)}/{len(input_folders)} talhões")
    return results


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Predição de volume florestal a partir de rasters LiDAR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplo:
  python PredictVolume.py --model model.pkl --output resultados/ \\
      --p90 metricas/p90.tif --p60 metricas/p60.tif --max metricas/max.tif \\
      --rotacao metricas/rotacao.tif --regional metricas/regional.tif \\
      --idade metricas/idade.tif
        """
    )
    parser.add_argument('--model', required=True, help='Caminho do modelo .pkl')
    parser.add_argument('--output', required=True, help='Diretório de saída')
    parser.add_argument('--p90', required=True, help='Raster de Elev P90')
    parser.add_argument('--p60', required=True, help='Raster de Elev P60')
    parser.add_argument('--max', required=True, help='Raster de Elev maximum')
    parser.add_argument('--rotacao', required=True,
                        help='Raster de ROTACAO (.tif)')
    parser.add_argument('--regional', required=True,
                        help='Raster de REGIONAL (.tif)')
    parser.add_argument('--idade', required=True,
                        help='Raster de Idade em meses (.tif)')
    parser.add_argument('--name', help='Nome base para os arquivos de saída')
    parser.add_argument('--no-uncertainty', action='store_true',
                        help='Não calcular raster de incerteza')

    args = parser.parse_args()

    # Monta dicionário de rasters
    raster_paths = {
        'Elev P90': args.p90,
        'Elev P60': args.p60,
        'Elev maximum': args.max,
        'ROTACAO': args.rotacao,
        'REGIONAL': args.regional,
        'Idade (meses)': args.idade
    }

    predict_volume(
        raster_paths=raster_paths,
        model=args.model,
        output_dir=args.output,
        output_name=args.name,
        calculate_uncertainty=not args.no_uncertainty
    )
