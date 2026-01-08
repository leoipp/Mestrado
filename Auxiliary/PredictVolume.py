"""
PredictVolume.py - Predição de Volume Florestal a partir de Rasters LiDAR

Este script aplica modelos treinados (Random Forest ou MLP) para predizer volume
florestal (VTCC) a partir de rasters de métricas LiDAR.

Funcionalidades:
    - Carrega modelo treinado (.pkl ou .pt)
    - Suporte a Random Forest (.pkl) e MLP/PyTorch (.pt)
    - Lê múltiplos rasters (uma banda por variável)
    - Alinha e reprojeta rasters automaticamente
    - Gera raster de predição (volume estimado)
    - Gera raster de incerteza (desvio padrão entre árvores do RF)
    - Suporte a processamento em lote por talhão

Entradas:
    - Rasters GeoTIFF das variáveis (Elev_P90.tif, Elev_P60.tif, etc.)
    - Modelo treinado (.pkl para sklearn ou .pt para PyTorch)
    - Variáveis auxiliares (ROTACAO, REGIONAL, Idade) como rasters TIF

Saídas:
    - {nome}_volume_estimado.tif: Predição de VTCC (m³/ha)
    - {nome}_volume_incerteza.tif: Desvio padrão da predição (apenas RF)

Dependências:
    - numpy
    - rasterio
    - joblib
    - scikit-learn
    - torch (opcional, para modelos MLP)

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import numpy as np
import re
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

# PyTorch (opcional, para modelos MLP)
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# CLASSE MLP (compatível com 05_NeuralNetworkTrain.py)
# =============================================================================

if HAS_TORCH:
    class MLP(nn.Module):
        """Multi-Layer Perceptron para regressão."""

        def __init__(
            self,
            input_size: int,
            hidden_sizes: List[int] = [64, 32],
            dropout_rate: float = 0.2,
            activation: str = 'relu'
        ):
            super(MLP, self).__init__()

            activations = {
                'relu': nn.ReLU(),
                'leaky_relu': nn.LeakyReLU(0.1),
                'elu': nn.ELU(),
                'tanh': nn.Tanh(),
                'selu': nn.SELU()
            }
            self.activation = activations.get(activation, nn.ReLU())

            layers = []
            prev_size = input_size

            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    self.activation,
                    nn.Dropout(dropout_rate)
                ])
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)


class MLPWrapper:
    """
    Wrapper para modelo MLP compatível com sklearn API.

    Encapsula o modelo PyTorch + scaler para uso transparente
    com o pipeline de predição.
    """

    def __init__(self, model, scaler, feature_names, device=None):
        self.model = model
        self.scaler = scaler
        self.feature_names_in_ = np.array(feature_names)
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu') if HAS_TORCH else None)
        self._is_mlp = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz valores normalizando automaticamente."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions


# =============================================================================
# FUNÇÕES AUXILIARES - MAPEAMENTO DE FEATURES
# =============================================================================

def get_model_features(model) -> List[str]:
    """
    Extrai a lista de features esperadas pelo modelo.

    Parameters
    ----------
    model : object
        Modelo treinado (sklearn ou similar).

    Returns
    -------
    list
        Lista de nomes das features na ordem esperada pelo modelo.

    Raises
    ------
    ValueError
        Se não for possível extrair as features do modelo.
    """
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    elif hasattr(model, 'feature_names'):
        return list(model.feature_names)
    elif hasattr(model, 'n_features_in_'):
        raise ValueError(
            f"Modelo tem {model.n_features_in_} features mas não armazena os nomes. "
            "Forneça feature_order manualmente."
        )
    else:
        raise ValueError(
            "Não foi possível extrair features do modelo. "
            "Forneça feature_order manualmente."
        )


def normalize_feature_name(name: str) -> str:
    """
    Normaliza nome de feature para comparação.
    Remove espaços extras, converte para minúsculas, substitui espaços por underscores.
    """
    return re.sub(r'\s+', '_', name.strip().lower())


def build_reverse_mapping(features: List[str]) -> Dict[str, str]:
    """
    Constrói mapeamento reverso de padrões de arquivo para nomes de features.
    """
    mapping = {}
    for feature in features:
        normalized = normalize_feature_name(feature)
        mapping[normalized] = feature
        if normalized.startswith('elev_'):
            mapping[normalized[5:]] = feature
        match = re.match(r'elev[_\s]?p(\d+)', normalized)
        if match:
            p_num = match.group(1)
            mapping[f'p{p_num}'] = feature
            mapping[f'elev_p{p_num}'] = feature
        if 'maximum' in normalized or 'max' in normalized:
            mapping['max'] = feature
            mapping['maximum'] = feature
            mapping['elev_max'] = feature
        if 'minimum' in normalized or 'min' in normalized:
            mapping['min'] = feature
            mapping['minimum'] = feature
        if 'rotacao' in normalized or 'rotation' in normalized:
            mapping['rotacao'] = feature
            mapping['rotation'] = feature
        if 'regional' in normalized:
            mapping['regional'] = feature
        if 'idade' in normalized or 'age' in normalized:
            mapping['idade'] = feature
            mapping['age'] = feature
    return mapping


def detect_feature_from_filename(filename: str, features: List[str]) -> Optional[str]:
    """Detecta qual feature corresponde a um nome de arquivo."""
    name = normalize_feature_name(Path(filename).stem)
    mapping = build_reverse_mapping(features)
    if name in mapping:
        return mapping[name]
    for pattern, feature in mapping.items():
        if pattern in name or name in pattern:
            return feature
    for feature in features:
        feat_normalized = normalize_feature_name(feature)
        match = re.search(r'p(\d+)', feat_normalized)
        if match:
            p_num = match.group(1)
            if re.search(rf'p{p_num}(?!\d)', name):
                return feature
    return None


def auto_detect_rasters(
    raster_dir: str,
    features: List[str],
    extensions: List[str] = ['.tif', '.tiff']
) -> Dict[str, str]:
    """Auto-detecta rasters em um diretório baseado nas features do modelo."""
    raster_dir = Path(raster_dir)
    if not raster_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {raster_dir}")
    raster_files = []
    for ext in extensions:
        raster_files.extend(raster_dir.glob(f'*{ext}'))
        raster_files.extend(raster_dir.glob(f'*{ext.upper()}'))
    raster_paths = {}
    unmatched_files = []
    for raster_file in raster_files:
        feature = detect_feature_from_filename(raster_file.name, features)
        if feature and feature not in raster_paths:
            raster_paths[feature] = str(raster_file)
        elif not feature:
            unmatched_files.append(raster_file.name)
    missing = [f for f in features if f not in raster_paths]
    if missing:
        print(f"\nFeatures não encontradas: {missing}")
        print(f"Arquivos não mapeados: {unmatched_files}")
        raise FileNotFoundError(
            f"Não foi possível encontrar rasters para: {missing}\n"
            f"Forneça manualmente usando --raster 'Feature Name=caminho.tif'"
        )
    return raster_paths


# =============================================================================
# FUNÇÕES AUXILIARES - MODELO E RASTER
# =============================================================================

def load_model(model_path: str):
    """
    Carrega modelo treinado (Random Forest .pkl ou MLP .pt).

    Parameters
    ----------
    model_path : str
        Caminho do arquivo .pkl (sklearn) ou .pt (PyTorch).

    Returns
    -------
    object
        Modelo carregado (RandomForestRegressor ou MLPWrapper).

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    ImportError
        Se tentar carregar .pt sem PyTorch instalado.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    suffix = path.suffix.lower()

    # -------------------------------------------------------------------------
    # Modelo PyTorch (.pt)
    # -------------------------------------------------------------------------
    if suffix == '.pt':
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch é necessário para carregar modelos .pt. "
                "Execute: pip install torch"
            )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)

        # Extrai configuração do modelo
        config = checkpoint['model_config']
        feature_names = checkpoint['feature_names']
        scaler = checkpoint['scaler']

        # Reconstrói o modelo MLP
        model = MLP(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            dropout_rate=config['dropout_rate'],
            activation=config['activation']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Cria wrapper compatível com sklearn API
        wrapper = MLPWrapper(model, scaler, feature_names, device)

        print(f"Modelo MLP carregado: {model_path}")
        print(f"  Device: {device}")
        print(f"  Features ({len(feature_names)}): {feature_names}")
        print(f"  Arquitetura: {config['hidden_sizes']}")

        return wrapper

    # -------------------------------------------------------------------------
    # Modelo sklearn (.pkl)
    # -------------------------------------------------------------------------
    else:
        data = joblib.load(model_path)

        # Handle dict format with model + feature_names
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            if 'feature_names' in data:
                model.feature_names = data['feature_names']
            print(f"Modelo carregado (formato dict): {model_path}")
        else:
            model = data
            print(f"Modelo carregado: {model_path}")

        # Verifica se tem feature_names
        try:
            features = get_model_features(model)
            print(f"  Features ({len(features)}): {features}")
        except ValueError as e:
            print(f"  Aviso: {e}")

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
    target_crs: Optional[CRS] = None,
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
    target_crs : CRS, optional
        CRS alvo. Se fornecido, todos os rasters serao reprojetados para este CRS.
        Se None, usa o CRS do raster de referência.
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

        # Determina CRS de destino
        dst_crs = target_crs if target_crs is not None else reference_info['crs']

        # Determina CRS de origem (se nao tiver, assume o CRS alvo)
        src_crs = src.crs if src.crs is not None else dst_crs

        # Verifica se precisa reprojetar/alinhar
        needs_align = (
            src.transform != reference_info['transform'] or
            src.width != reference_info['width'] or
            src.height != reference_info['height'] or
            src_crs != dst_crs
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
                src_crs=src_crs,
                dst_transform=reference_info['transform'],
                dst_crs=dst_crs,
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
    Faz predição com estimativa de incerteza.

    Para Random Forest, calcula o desvio padrão entre as árvores.
    Para MLP e outros modelos, retorna incerteza zero.

    Parameters
    ----------
    model : object
        Modelo treinado (RandomForestRegressor, MLPWrapper, ou similar).
    X : np.ndarray
        Dados de entrada (n_samples, n_features).
    n_jobs : int
        Número de jobs paralelos (apenas para RF).

    Returns
    -------
    tuple
        (média das predições, desvio padrão das predições)

    Notes
    -----
    - Random Forest: incerteza = std entre árvores
    - MLP/outros: incerteza = 0 (não disponível)
    """
    # Verifica se é MLP (wrapper)
    if hasattr(model, '_is_mlp') and model._is_mlp:
        mean_pred = model.predict(X)
        std_pred = np.zeros_like(mean_pred)

    # Random Forest: usa estimators_ para calcular incerteza
    elif hasattr(model, 'estimators_'):
        all_predictions = np.array([
            tree.predict(X) for tree in model.estimators_
        ])
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)

    # Outros modelos sklearn
    else:
        mean_pred = model.predict(X)
        std_pred = np.zeros_like(mean_pred)

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
    target_crs: Optional[Union[str, int, CRS]] = None,
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
    target_crs : str, int or CRS, optional
        CRS alvo para reprojetar todos os rasters. Pode ser EPSG code (int),
        string (ex: 'EPSG:31983') ou objeto CRS. Se None, usa o CRS do primeiro raster.
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
        try:
            feature_order = get_model_features(model)
        except ValueError as e:
            raise ValueError(
                f"{e}\nForneça feature_order manualmente ou use um modelo "
                "treinado com feature_names_in_"
            )

    print(f"  Features esperadas ({len(feature_order)}): {feature_order}")

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

    # Converte target_crs para objeto CRS se necessario
    crs_obj = None
    if target_crs is not None:
        if isinstance(target_crs, CRS):
            crs_obj = target_crs
        elif isinstance(target_crs, int):
            crs_obj = CRS.from_epsg(target_crs)
        elif isinstance(target_crs, str):
            crs_obj = CRS.from_string(target_crs)
        print(f"  CRS alvo: {crs_obj}")

    # Primeiro raster como referência (usa a primeira feature na ordem)
    first_raster_path = raster_paths[feature_order[0]]
    ref_info = get_raster_info(first_raster_path)

    # Se target_crs foi fornecido, atualiza o ref_info
    if crs_obj is not None:
        ref_info['crs'] = crs_obj

    print(f"  Referência: {Path(first_raster_path).name}")
    print(f"  Dimensões: {ref_info['width']} x {ref_info['height']}")
    print(f"  CRS: {ref_info['crs']}")

    # Carrega todos os rasters na ordem correta
    feature_arrays = []
    nodata_values = []

    for feature_name in feature_order:
        raster_path = raster_paths[feature_name]
        band, nodata = align_raster(raster_path, ref_info, target_crs=crs_obj)
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
    if hasattr(model, 'n_estimators'):
        print(f"  Predizendo com {model.n_estimators} árvores...")
    else:
        print(f"  Predizendo...")
    mean_pred, std_pred = predict_with_uncertainty(model, valid_data)

    # Reconstrói rasters
    predicted_full = np.full(n_pixels, nodata_output, dtype='float32')
    predicted_full[valid_mask] = mean_pred.astype('float32')
    predicted_raster = predicted_full.reshape((height, width))

    if calculate_uncertainty and np.any(std_pred > 0):
        uncertainty_full = np.full(n_pixels, nodata_output, dtype='float32')
        uncertainty_full[valid_mask] = std_pred.astype('float32')
        uncertainty_raster = uncertainty_full.reshape((height, width))
    else:
        calculate_uncertainty = False

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


def predict_volume_from_dir(
    raster_dir: str,
    model: Union[str, object],
    output_dir: str,
    output_name: Optional[str] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Prediz volume auto-detectando rasters em um diretório.

    Parameters
    ----------
    raster_dir : str
        Diretório contendo os rasters.
    model : str or object
        Caminho do modelo (.pkl) ou modelo já carregado.
    output_dir : str
        Diretório para salvar os rasters de saída.
    output_name : str, optional
        Nome base para os arquivos de saída.
    **kwargs
        Argumentos adicionais para predict_volume.

    Returns
    -------
    dict
        Dicionário com caminhos dos arquivos gerados.
    """
    if isinstance(model, str):
        model_obj = load_model(model)
    else:
        model_obj = model

    features = get_model_features(model_obj)
    print(f"\nAuto-detectando rasters em: {raster_dir}")
    print(f"Features necessárias: {features}")

    raster_paths = auto_detect_rasters(raster_dir, features)
    print(f"\nRasters detectados:")
    for feature, path in raster_paths.items():
        print(f"  {feature}: {Path(path).name}")

    return predict_volume(
        raster_paths=raster_paths,
        model=model_obj,
        output_dir=output_dir,
        output_name=output_name or Path(raster_dir).name,
        **kwargs
    )


def predict_volume_batch(
    input_folders: List[str],
    model: Union[str, object],
    output_dir: str,
    raster_pattern: Optional[Dict[str, str]] = None,
    auto_detect: bool = True,
    **kwargs
) -> List[Dict[str, str]]:
    """
    Processa múltiplos talhões em lote com auto-detecção de features.

    Parameters
    ----------
    input_folders : list
        Lista de pastas, cada uma contendo rasters de um talhão.
    model : str or object
        Modelo ou caminho do modelo.
    output_dir : str
        Diretório de saída.
    raster_pattern : dict, optional
        Mapeamento de variável para padrão glob. Se None, usa auto-detecção.
    auto_detect : bool
        Se True, usa auto-detecção de rasters baseada nas features do modelo.
    **kwargs
        Argumentos adicionais para predict_volume.

    Returns
    -------
    list
        Lista de dicionários com arquivos gerados por talhão.
    """
    if isinstance(model, str):
        model = load_model(model)

    if auto_detect and raster_pattern is None:
        features = get_model_features(model)
        print(f"\nModo auto-detecção ativado")
        print(f"Features do modelo: {features}")

    results = []

    for i, folder in enumerate(input_folders, 1):
        folder_path = Path(folder)
        print(f"\n{'='*70}")
        print(f"TALHÃO {i}/{len(input_folders)}: {folder_path.name}")
        print('='*70)

        try:
            if auto_detect and raster_pattern is None:
                raster_paths = auto_detect_rasters(str(folder_path), features)
            else:
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
# EXECUÇÃO CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Predição de volume florestal a partir de rasters LiDAR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:

  # Auto-detecção de rasters em um diretório
  python PredictVolume.py --model model.pkl --raster-dir metricas/ --output resultados/

  # Especificando rasters manualmente
  python PredictVolume.py --model model.pkl --output resultados/ \\
      --raster "Elev P90=metricas/p90.tif" \\
      --raster "Elev P60=metricas/p60.tif" \\
      --raster "Elev maximum=metricas/max.tif"

  # Listar features do modelo
  python PredictVolume.py --model model.pkl --list-features

  # Processamento em lote
  python PredictVolume.py --model model.pkl --output resultados/ \\
      --batch talhao_01/ talhao_02/ talhao_03/
        """
    )

    parser.add_argument('--model', required=True, help='Caminho do modelo .pkl')
    parser.add_argument('--output', help='Diretório de saída')
    parser.add_argument('--raster-dir', help='Diretório com rasters (auto-detecção)')
    parser.add_argument('--raster', action='append', metavar='FEATURE=PATH',
                        help='Raster no formato "Nome Feature=caminho.tif" (repetível)')
    parser.add_argument('--batch', nargs='+', metavar='DIR',
                        help='Processar múltiplos diretórios em lote')
    parser.add_argument('--name', help='Nome base para os arquivos de saída')
    parser.add_argument('--no-uncertainty', action='store_true',
                        help='Não calcular raster de incerteza')
    parser.add_argument('--list-features', action='store_true',
                        help='Listar features do modelo e sair')

    args = parser.parse_args()

    # Modo: listar features
    if args.list_features:
        model = load_model(args.model)
        try:
            features = get_model_features(model)
            print(f"\nFeatures do modelo ({len(features)}):")
            for i, f in enumerate(features, 1):
                print(f"  {i}. {f}")
        except ValueError as e:
            print(f"\nErro: {e}")
        sys.exit(0)

    # Validação de argumentos
    if not args.output:
        parser.error("--output é obrigatório para predição")

    if not args.raster_dir and not args.raster and not args.batch:
        parser.error("Forneça --raster-dir, --raster ou --batch")

    # Modo: processamento em lote
    if args.batch:
        predict_volume_batch(
            input_folders=args.batch,
            model=args.model,
            output_dir=args.output,
            auto_detect=True,
            calculate_uncertainty=not args.no_uncertainty
        )
        sys.exit(0)

    # Modo: auto-detecção em diretório
    if args.raster_dir:
        predict_volume_from_dir(
            raster_dir=args.raster_dir,
            model=args.model,
            output_dir=args.output,
            output_name=args.name,
            calculate_uncertainty=not args.no_uncertainty
        )
        sys.exit(0)

    # Modo: rasters especificados manualmente
    if args.raster:
        raster_paths = {}
        for r in args.raster:
            if '=' not in r:
                parser.error(f"Formato inválido: '{r}'. Use 'Feature=caminho.tif'")
            feature, path = r.split('=', 1)
            raster_paths[feature.strip()] = path.strip()

        predict_volume(
            raster_paths=raster_paths,
            model=args.model,
            output_dir=args.output,
            output_name=args.name,
            calculate_uncertainty=not args.no_uncertainty
        )
