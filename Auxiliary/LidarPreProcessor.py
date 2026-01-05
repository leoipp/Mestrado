"""
LidarPreProcessor.py - Pipeline de Processamento de Dados LiDAR com LAStools

Este script implementa um pipeline completo de pre-processamento de nuvens de
pontos LiDAR usando ferramentas do LAStools, preparando os dados para analise
de metricas florestais.

Pipeline de Processamento:
    1. Info/Catalog   - Gera relatorio com estatisticas da nuvem de pontos
    2. Tiling         - Divide a nuvem em tiles para processamento paralelo
    3. Denoising      - Remove ruidos e pontos isolados
    4. Ground/DTM     - Classifica pontos de solo e gera modelo digital de terreno
    5. Thinning       - Reduz densidade de pontos mantendo representatividade
    6. Normalization  - Normaliza alturas (z_lidar - z_terreno)
    7. CHM            - Gera modelo de altura de copa (Canopy Height Model)
    8. DSM            - Gera modelo digital de superficie
    9. Metrics        - Calcula metricas de grade (percentis, max, kurtosis, etc.)

Requisitos:
    - LAStools instalado (https://rapidlasso.com/lastools/)
    - Python 3.8+

Autor: Leonardo Ippolito Rodrigues
Data: 2026
Projeto: Mestrado - Predicao de Volume Florestal com LiDAR
"""

import subprocess
import multiprocessing
import math
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from datetime import datetime


# =============================================================================
# CONFIGURACAO
# =============================================================================

@dataclass
class LidarConfig:
    """
    Configuracao do pipeline de processamento LiDAR.

    Attributes
    ----------
    lastools_path : str
        Caminho para a pasta bin do LAStools.
    input_files : list
        Lista de arquivos .las ou .laz para processar.
    output_base_dir : str
        Diretorio base para salvar os resultados.
    n_cores : int
        Numero de cores para processamento paralelo. Default: todos disponiveis.

    Tiling Parameters
    -----------------
    tile_size : int
        Tamanho do tile em metros (default: 1000).
    buffer : int
        Buffer entre tiles em metros (default: 50).

    Denoising Parameters
    --------------------
    noise_step : int
        Tamanho da janela para deteccao de ruido (default: 100).
    noise_isolated : int
        Numero minimo de vizinhos para nao ser ruido (default: 5).

    Ground/DTM Parameters
    ---------------------
    dtm_resolution : float
        Resolucao do DTM em metros (default: 1.0).
    ground_spike : float
        Tolerancia para spikes no terreno (default: 0.5).
    ground_spike_down : float
        Tolerancia para spikes abaixo do terreno (default: 2.5).

    Thinning Parameters
    -------------------
    thin_density : float
        Densidade alvo de pontos por m2 (default: 5).
    thin_method : str
        Metodo de thinning: 'random', 'highest', 'lowest' (default: 'random').

    CHM/DSM Parameters
    ------------------
    chm_resolution : float
        Resolucao do CHM em metros (default: 1.0).
    dsm_resolution : float
        Resolucao do DSM em metros (default: 1.0).

    Metrics Parameters
    ------------------
    metrics_resolution : float
        Resolucao da grade de metricas em metros (default: 17).
    percentiles : list
        Lista de percentis a calcular (default: [60, 90]).
    """
    # Caminhos
    lastools_path: str = "G:\\LAStools\\bin"
    input_files: List[str] = field(default_factory=list)
    output_base_dir: str = ""
    n_cores: int = field(default_factory=multiprocessing.cpu_count)

    # Tiling
    tile_size: int = 1000
    buffer: int = 50

    # Denoising
    noise_step: int = 100
    noise_isolated: int = 5

    # Ground/DTM
    dtm_resolution: float = 1.0
    ground_spike: float = 0.5
    ground_spike_down: float = 2.5

    # Thinning
    thin_density: float = 5.0
    thin_method: str = "random"

    # CHM/DSM
    chm_resolution: float = 1.0
    dsm_resolution: float = 1.0

    # Metrics
    metrics_resolution: float = 17.0
    percentiles: List[int] = field(default_factory=lambda: [60, 90])

    # Controle de execucao
    start_step: int = 1
    end_step: int = 9

    def __post_init__(self):
        """Valida configuracao apos inicializacao."""
        if isinstance(self.n_cores, Callable):
            self.n_cores = self.n_cores()


# Nomes dos subdiretorios para cada etapa
STEP_DIRS = {
    1: "00info",
    2: "01tiles",
    3: "02clean",
    4: "03gnd",
    5: "04dtm",
    6: "05thin",
    7: "06norm",
    8: "07chm",
    9: "08dsm",
    10: "09metrics"
}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura sistema de logging.

    Parameters
    ----------
    log_file : str, optional
        Caminho para arquivo de log. Se None, usa apenas console.

    Returns
    -------
    logging.Logger
        Logger configurado.
    """
    logger = logging.getLogger("LidarPreProcessor")
    logger.setLevel(logging.DEBUG)

    # Formato
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Evita adicionar handlers duplicados
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (adiciona apenas se especificado e não existir)
    if log_file:
        # Verifica se já existe um FileHandler para este arquivo
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file).resolve())
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# Logger global
logger = logging.getLogger("LidarPreProcessor")


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def run_command(
    command: str,
    description: str = "",
    check: bool = True
) -> subprocess.CompletedProcess:
    """
    Executa comando no shell e captura saida.

    Parameters
    ----------
    command : str
        Comando a executar.
    description : str
        Descricao do comando para logging.
    check : bool
        Se True, levanta excecao em caso de erro.

    Returns
    -------
    subprocess.CompletedProcess
        Resultado da execucao.

    Raises
    ------
    subprocess.CalledProcessError
        Se o comando falhar e check=True.
    """
    if description:
        logger.info(f"Executando: {description}")
    logger.debug(f"Comando: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )

        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.debug(f"  {line}")

        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")

        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao executar comando: {e}")
        if e.stdout:
            logger.error(f"Saida: {e.stdout}")
        if e.stderr:
            logger.error(f"Erro: {e.stderr}")
        raise


def get_tool_path(config: LidarConfig, tool_name: str) -> str:
    """
    Retorna caminho completo para ferramenta LAStools.

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    tool_name : str
        Nome da ferramenta (ex: 'lasinfo', 'lastile').

    Returns
    -------
    str
        Caminho completo do executavel.
    """
    # Adiciona sufixo 64 se nao presente
    if not tool_name.endswith('64'):
        tool_name = f"{tool_name}64"

    # Adiciona extensao .exe
    if not tool_name.endswith('.exe'):
        tool_name = f"{tool_name}.exe"

    return str(Path(config.lastools_path) / tool_name)


def ensure_dir(path: str) -> Path:
    """
    Cria diretorio se nao existir.

    Parameters
    ----------
    path : str
        Caminho do diretorio.

    Returns
    -------
    Path
        Objeto Path do diretorio.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def calculate_thin_window(density: float) -> float:
    """
    Calcula tamanho da janela para thinning baseado na densidade alvo.

    Parameters
    ----------
    density : float
        Densidade alvo de pontos por m2.

    Returns
    -------
    float
        Tamanho da janela em metros.
    """
    return round(math.sqrt(1 / density), 2)


# =============================================================================
# ETAPAS DO PIPELINE
# =============================================================================

def step_01_info(
    config: LidarConfig,
    input_file: str,
    work_dir: Path
) -> Path:
    """
    Etapa 1: Gera catalogo/info da nuvem de pontos.

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    input_file : str
        Arquivo LAS/LAZ de entrada.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Caminho do arquivo de catalogo gerado.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 1: CATALOG/INFO")
    logger.info("=" * 60)

    info_dir = ensure_dir(work_dir / STEP_DIRS[1])
    catalog_file = info_dir / "catalog.txt"

    cmd = (
        f'"{get_tool_path(config, "lasinfo")}" '
        f'-cpu64 -i "{input_file}" -merged '
        f'-o "{catalog_file}" -cd -histo gps_time 20'
    )

    run_command(cmd, "Gerando catalogo da nuvem de pontos")
    logger.info(f"Catalogo salvo: {catalog_file}")

    return catalog_file


def step_02_tiling(
    config: LidarConfig,
    input_file: str,
    work_dir: Path
) -> Path:
    """
    Etapa 2: Divide nuvem de pontos em tiles.

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    input_file : str
        Arquivo LAS/LAZ de entrada.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Diretorio com os tiles gerados.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 2: TILING")
    logger.info("=" * 60)

    tiles_dir = ensure_dir(work_dir / STEP_DIRS[2])
    output_pattern = tiles_dir / "tile.laz"

    cmd = (
        f'"{get_tool_path(config, "lastile")}" '
        f'-i "{input_file}" '
        f'-tile_size {config.tile_size} '
        f'-buffer {config.buffer} '
        f'-o "{output_pattern}"'
    )

    run_command(cmd, f"Criando tiles de {config.tile_size}m com buffer de {config.buffer}m")

    n_tiles = len(list(tiles_dir.glob("*.laz")))
    logger.info(f"Tiles criados: {n_tiles} arquivos em {tiles_dir}")

    return tiles_dir


def step_03_denoise(
    config: LidarConfig,
    tiles_dir: Path,
    work_dir: Path
) -> Path:
    """
    Etapa 3: Remove ruidos da nuvem de pontos.

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    tiles_dir : Path
        Diretorio com tiles de entrada.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Diretorio com arquivos limpos.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 3: DENOISING")
    logger.info("=" * 60)

    clean_dir = ensure_dir(work_dir / STEP_DIRS[3])

    cmd = (
        f'"{get_tool_path(config, "lasnoise")}" '
        f'-i "{tiles_dir / "*.laz"}" '
        f'-odir "{clean_dir}" -odix _denoised -olaz '
        f'-step {config.noise_step} '
        f'-isolated {config.noise_isolated} '
        f'-remove_noise '
        f'-cores {config.n_cores}'
    )

    run_command(
        cmd,
        f"Removendo ruido (step={config.noise_step}, isolated={config.noise_isolated})"
    )

    n_files = len(list(clean_dir.glob("*.laz")))
    logger.info(f"Arquivos limpos: {n_files} em {clean_dir}")

    return clean_dir


def step_04_ground(
    config: LidarConfig,
    clean_dir: Path,
    work_dir: Path
) -> Path:
    """
    Etapa 4a: Classifica pontos de solo (ground).

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    clean_dir : Path
        Diretorio com arquivos limpos.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Diretorio com arquivos classificados.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 4a: GROUND CLASSIFICATION")
    logger.info("=" * 60)

    gnd_dir = ensure_dir(work_dir / STEP_DIRS[4])

    cmd = (
        f'"{get_tool_path(config, "lasground_new")}" '
        f'-i "{clean_dir / "*.laz"}" '
        f'-odir "{gnd_dir}" -odix _gnd -olaz '
        f'-nature '
        f'-spike {config.ground_spike} '
        f'-spike_down {config.ground_spike_down} '
        f'-cores {config.n_cores}'
    )

    run_command(
        cmd,
        f"Classificando solo (spike={config.ground_spike}, spike_down={config.ground_spike_down})"
    )

    n_files = len(list(gnd_dir.glob("*.laz")))
    logger.info(f"Arquivos classificados: {n_files} em {gnd_dir}")

    return gnd_dir


def step_05_dtm(
    config: LidarConfig,
    gnd_dir: Path,
    work_dir: Path
) -> Path:
    """
    Etapa 4b: Gera modelo digital de terreno (DTM).

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    gnd_dir : Path
        Diretorio com arquivos de ground.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Diretorio com arquivos DTM.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 4b: DTM GENERATION")
    logger.info("=" * 60)

    dtm_dir = ensure_dir(work_dir / STEP_DIRS[5])

    cmd = (
        f'"{get_tool_path(config, "las2dem")}" '
        f'-i "{gnd_dir / "*.laz"}" '
        f'-odir "{dtm_dir}" -oasc '
        f'-keep_class 2 '
        f'-use_tile_bb '
        f'-step {config.dtm_resolution} '
        f'-cores {config.n_cores}'
    )

    run_command(cmd, f"Gerando DTM (resolucao={config.dtm_resolution}m)")

    n_files = len(list(dtm_dir.glob("*.asc")))
    logger.info(f"Arquivos DTM: {n_files} em {dtm_dir}")

    return dtm_dir


def step_06_thin(
    config: LidarConfig,
    clean_dir: Path,
    work_dir: Path
) -> Path:
    """
    Etapa 5: Reduz densidade de pontos (thinning).

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    clean_dir : Path
        Diretorio com arquivos limpos.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Diretorio com arquivos rarefacionados.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 5: THINNING")
    logger.info("=" * 60)

    thin_dir = ensure_dir(work_dir / STEP_DIRS[6])
    window_size = calculate_thin_window(config.thin_density)

    cmd = (
        f'"{get_tool_path(config, "lasthin")}" '
        f'-i "{clean_dir / "*.laz"}" '
        f'-odir "{thin_dir}" -odix _thin -olaz '
        f'-step {window_size} '
        f'-{config.thin_method} '
        f'-cores {config.n_cores}'
    )

    run_command(
        cmd,
        f"Thinning (densidade={config.thin_density} pts/m2, janela={window_size}m, metodo={config.thin_method})"
    )

    n_files = len(list(thin_dir.glob("*.laz")))
    logger.info(f"Arquivos rarefacionados: {n_files} em {thin_dir}")

    return thin_dir


def step_07_normalize(
    config: LidarConfig,
    thin_dir: Path,
    work_dir: Path
) -> Path:
    """
    Etapa 6: Normaliza alturas (z_lidar - z_terreno).

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    thin_dir : Path
        Diretorio com arquivos thin.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Diretorio com arquivos normalizados.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 6: NORMALIZATION")
    logger.info("=" * 60)

    norm_dir = ensure_dir(work_dir / STEP_DIRS[7])

    cmd = (
        f'"{get_tool_path(config, "lasground_new")}" '
        f'-i "{thin_dir / "*.laz"}" '
        f'-odir "{norm_dir}" -odix _norm -olaz '
        f'-nature '
        f'-spike {config.ground_spike} '
        f'-compute_height -replace_z '
        f'-cores {config.n_cores}'
    )

    run_command(cmd, "Normalizando alturas (z = altura acima do solo)")

    n_files = len(list(norm_dir.glob("*.laz")))
    logger.info(f"Arquivos normalizados: {n_files} em {norm_dir}")

    return norm_dir


def step_08_chm(
    config: LidarConfig,
    norm_dir: Path,
    work_dir: Path
) -> Path:
    """
    Etapa 7: Gera modelo de altura de copa (CHM).

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    norm_dir : Path
        Diretorio com arquivos normalizados.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Diretorio com arquivos CHM.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 7: CHM (Canopy Height Model)")
    logger.info("=" * 60)

    chm_dir = ensure_dir(work_dir / STEP_DIRS[8])

    cmd = (
        f'"{get_tool_path(config, "lasgrid")}" '
        f'-i "{norm_dir / "*.laz"}" '
        f'-odir "{chm_dir}" -oasc '
        f'-step {config.chm_resolution} '
        f'-highest '
        f'-use_tile_bb '
        f'-cores {config.n_cores}'
    )

    run_command(cmd, f"Gerando CHM (resolucao={config.chm_resolution}m)")

    n_files = len(list(chm_dir.glob("*.asc")))
    logger.info(f"Arquivos CHM: {n_files} em {chm_dir}")

    return chm_dir


def step_09_dsm(
    config: LidarConfig,
    thin_dir: Path,
    work_dir: Path
) -> Path:
    """
    Etapa 8: Gera modelo digital de superficie (DSM).

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    thin_dir : Path
        Diretorio com arquivos thin.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Diretorio com arquivos DSM.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 8: DSM (Digital Surface Model)")
    logger.info("=" * 60)

    dsm_dir = ensure_dir(work_dir / STEP_DIRS[9])

    cmd = (
        f'"{get_tool_path(config, "lasgrid")}" '
        f'-i "{thin_dir / "*.laz"}" '
        f'-odir "{dsm_dir}" -oasc -odix _dsm '
        f'-step {config.dsm_resolution} '
        f'-highest '
        f'-use_tile_bb '
        f'-cores {config.n_cores}'
    )

    run_command(cmd, f"Gerando DSM (resolucao={config.dsm_resolution}m)")

    n_files = len(list(dsm_dir.glob("*.asc")))
    logger.info(f"Arquivos DSM: {n_files} em {dsm_dir}")

    return dsm_dir


def step_10_metrics(
    config: LidarConfig,
    norm_dir: Path,
    work_dir: Path
) -> Path:
    """
    Etapa 9: Calcula metricas de grade.

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    norm_dir : Path
        Diretorio com arquivos normalizados.
    work_dir : Path
        Diretorio de trabalho.

    Returns
    -------
    Path
        Diretorio com arquivos de metricas.
    """
    logger.info("=" * 60)
    logger.info("ETAPA 9: GRID METRICS")
    logger.info("=" * 60)

    metrics_dir = ensure_dir(work_dir / STEP_DIRS[10])

    # Formata percentis para o comando
    percentile_args = " ".join([str(p) for p in config.percentiles])

    # 1. Calcular metricas com lascanopy
    cmd_canopy = (
        f'"{get_tool_path(config, "lascanopy")}" '
        f'-i "{norm_dir / "*.laz"}" '
        f'-step {config.metrics_resolution} '
        f'-max -p {percentile_args} -kur '
        f'-odir "{metrics_dir}" -oasc '
        f'-use_tile_bb '
        f'-cores {config.n_cores}'
    )

    run_command(
        cmd_canopy,
        f"Calculando metricas (resolucao={config.metrics_resolution}m, percentis={config.percentiles})"
    )

    # 2. Exportar pontos para calculo da media cubica
    cmd_export = (
        f'"{get_tool_path(config, "las2txt")}" '
        f'-i "{norm_dir / "*.laz"}" '
        f'-odir "{metrics_dir}" '
        f'-parse xyz '
        f'-cores {config.n_cores}'
    )

    run_command(cmd_export, "Exportando pontos XYZ para media cubica")

    n_files = len(list(metrics_dir.glob("*.asc")))
    n_txt = len(list(metrics_dir.glob("*.txt")))
    logger.info(f"Metricas: {n_files} arquivos ASC, {n_txt} arquivos TXT em {metrics_dir}")

    return metrics_dir


# =============================================================================
# FUNCAO PRINCIPAL
# =============================================================================

def process_lidar(
    config: LidarConfig,
    input_file: str
) -> Dict[str, Path]:
    """
    Executa pipeline completo de processamento LiDAR.

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline.
    input_file : str
        Arquivo LAS/LAZ de entrada.

    Returns
    -------
    dict
        Dicionario com caminhos dos diretorios de cada etapa.
    """
    input_path = Path(input_file)
    project_name = input_path.stem

    # Define diretorio de trabalho
    if config.output_base_dir:
        work_dir = Path(config.output_base_dir) / project_name
    else:
        work_dir = Path.cwd() / "LiDAR_Output" / project_name

    work_dir = ensure_dir(work_dir)

    # Configura logging para arquivo
    log_file = work_dir / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(str(log_file))

    logger.info("=" * 70)
    logger.info("PIPELINE DE PROCESSAMENTO LIDAR")
    logger.info("=" * 70)
    logger.info(f"Projeto: {project_name}")
    logger.info(f"Entrada: {input_file}")
    logger.info(f"Saida: {work_dir}")
    logger.info(f"Cores: {config.n_cores}")
    logger.info(f"Etapas: {config.start_step} a {config.end_step}")
    logger.info(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Resultados
    results = {
        'work_dir': work_dir,
        'input_file': input_path
    }

    # Variaveis para armazenar diretorios intermediarios
    tiles_dir = None
    clean_dir = None
    gnd_dir = None
    thin_dir = None
    norm_dir = None

    try:
        # Etapa 1: Info/Catalog
        if config.start_step <= 1 <= config.end_step:
            results['catalog'] = step_01_info(config, input_file, work_dir)

        # Etapa 2: Tiling
        if config.start_step <= 2 <= config.end_step:
            tiles_dir = step_02_tiling(config, input_file, work_dir)
            results['tiles'] = tiles_dir
        else:
            tiles_dir = work_dir / STEP_DIRS[2]

        # Etapa 3: Denoising
        if config.start_step <= 3 <= config.end_step:
            clean_dir = step_03_denoise(config, tiles_dir, work_dir)
            results['clean'] = clean_dir
        else:
            clean_dir = work_dir / STEP_DIRS[3]

        # Etapa 4: Ground Classification
        if config.start_step <= 4 <= config.end_step:
            gnd_dir = step_04_ground(config, clean_dir, work_dir)
            results['ground'] = gnd_dir
        else:
            gnd_dir = work_dir / STEP_DIRS[4]

        # Etapa 5: DTM
        if config.start_step <= 5 <= config.end_step:
            results['dtm'] = step_05_dtm(config, gnd_dir, work_dir)

        # Etapa 6: Thinning
        if config.start_step <= 6 <= config.end_step:
            thin_dir = step_06_thin(config, clean_dir, work_dir)
            results['thin'] = thin_dir
        else:
            thin_dir = work_dir / STEP_DIRS[6]

        # Etapa 7: Normalization
        if config.start_step <= 7 <= config.end_step:
            norm_dir = step_07_normalize(config, thin_dir, work_dir)
            results['norm'] = norm_dir
        else:
            norm_dir = work_dir / STEP_DIRS[7]

        # Etapa 8: CHM
        if config.start_step <= 8 <= config.end_step:
            results['chm'] = step_08_chm(config, norm_dir, work_dir)

        # Etapa 9: DSM
        if config.start_step <= 9 <= config.end_step:
            results['dsm'] = step_09_dsm(config, thin_dir, work_dir)

        # Etapa 10: Metrics (considera como etapa 9 para compatibilidade)
        if config.start_step <= 9 <= config.end_step:
            results['metrics'] = step_10_metrics(config, norm_dir, work_dir)

        logger.info("=" * 70)
        logger.info("PROCESSAMENTO CONCLUIDO COM SUCESSO")
        logger.info(f"Termino: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Erro no processamento: {e}")
        raise

    return results


def process_batch(config: LidarConfig) -> List[Dict[str, Path]]:
    """
    Processa multiplos arquivos LiDAR em lote.

    Parameters
    ----------
    config : LidarConfig
        Configuracao do pipeline com lista de arquivos em input_files.

    Returns
    -------
    list
        Lista de dicionarios com resultados de cada arquivo.
    """
    results = []
    n_files = len(config.input_files)

    logger.info("=" * 70)
    logger.info(f"PROCESSAMENTO EM LOTE: {n_files} arquivos")
    logger.info("=" * 70)

    for i, input_file in enumerate(config.input_files, 1):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# ARQUIVO {i}/{n_files}: {Path(input_file).name}")
        logger.info(f"{'#' * 70}\n")

        try:
            result = process_lidar(config, input_file)
            results.append(result)
        except Exception as e:
            logger.error(f"Erro ao processar {input_file}: {e}")
            results.append({'error': str(e), 'input_file': input_file})

    # Resumo
    successful = sum(1 for r in results if 'error' not in r)
    logger.info(f"\n{'=' * 70}")
    logger.info(f"LOTE CONCLUIDO: {successful}/{n_files} arquivos processados")
    logger.info("=" * 70)

    return results


# =============================================================================
# INTERFACE DE LINHA DE COMANDO
# =============================================================================

if __name__ == '__main__':
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description='Pipeline de processamento de dados LiDAR com LAStools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Processar um arquivo
  python LidarPreProcessor.py -i dados/floresta.las -o resultados/

  # Processar multiplos arquivos
  python LidarPreProcessor.py -i "dados/*.las" -o resultados/

  # Processar apenas etapas especificas (ex: 3 a 7)
  python LidarPreProcessor.py -i dados/floresta.las -o resultados/ --start 3 --end 7

  # Configuracao personalizada
  python LidarPreProcessor.py -i dados/floresta.las -o resultados/ \\
      --tile-size 500 --metrics-res 20 --percentiles 50 75 90 95
        """
    )

    # Argumentos obrigatorios
    parser.add_argument('-i', '--input', required=True,
                        help='Arquivo(s) LAS/LAZ de entrada (aceita wildcards)')
    parser.add_argument('-o', '--output', required=True,
                        help='Diretorio base de saida')

    # Configuracao do LAStools
    parser.add_argument('--lastools', default='G:\\LAStools\\bin',
                        help='Caminho para pasta bin do LAStools')
    parser.add_argument('--cores', type=int, default=multiprocessing.cpu_count(),
                        help='Numero de cores (default: todos)')

    # Controle de etapas
    parser.add_argument('--start', type=int, default=1,
                        help='Etapa inicial (1-9, default: 1)')
    parser.add_argument('--end', type=int, default=9,
                        help='Etapa final (1-9, default: 9)')

    # Parametros de tiling
    parser.add_argument('--tile-size', type=int, default=1000,
                        help='Tamanho do tile em metros (default: 1000)')
    parser.add_argument('--buffer', type=int, default=50,
                        help='Buffer entre tiles em metros (default: 50)')

    # Parametros de denoising
    parser.add_argument('--noise-step', type=int, default=100,
                        help='Janela para deteccao de ruido (default: 100)')
    parser.add_argument('--noise-isolated', type=int, default=5,
                        help='Vizinhos minimos para nao ser ruido (default: 5)')

    # Parametros de ground
    parser.add_argument('--spike', type=float, default=0.5,
                        help='Tolerancia para spikes (default: 0.5)')
    parser.add_argument('--spike-down', type=float, default=2.5,
                        help='Tolerancia para spikes abaixo (default: 2.5)')

    # Parametros de thinning
    parser.add_argument('--thin-density', type=float, default=5.0,
                        help='Densidade alvo pts/m2 (default: 5)')
    parser.add_argument('--thin-method', choices=['random', 'highest', 'lowest'],
                        default='random', help='Metodo de thinning (default: random)')

    # Parametros de resolucao
    parser.add_argument('--dtm-res', type=float, default=1.0,
                        help='Resolucao do DTM em metros (default: 1)')
    parser.add_argument('--chm-res', type=float, default=1.0,
                        help='Resolucao do CHM em metros (default: 1)')
    parser.add_argument('--dsm-res', type=float, default=1.0,
                        help='Resolucao do DSM em metros (default: 1)')
    parser.add_argument('--metrics-res', type=float, default=17.0,
                        help='Resolucao da grade de metricas (default: 17)')

    # Parametros de metricas
    parser.add_argument('--percentiles', type=int, nargs='+', default=[60, 90],
                        help='Percentis a calcular (default: 60 90)')

    args = parser.parse_args()

    # Expande wildcards nos arquivos de entrada
    input_files = glob.glob(args.input)
    if not input_files:
        print(f"Erro: Nenhum arquivo encontrado para '{args.input}'")
        exit(1)

    # Cria configuracao
    config = LidarConfig(
        lastools_path=args.lastools,
        input_files=input_files,
        output_base_dir=args.output,
        n_cores=args.cores,
        tile_size=args.tile_size,
        buffer=args.buffer,
        noise_step=args.noise_step,
        noise_isolated=args.noise_isolated,
        ground_spike=args.spike,
        ground_spike_down=args.spike_down,
        thin_density=args.thin_density,
        thin_method=args.thin_method,
        dtm_resolution=args.dtm_res,
        chm_resolution=args.chm_res,
        dsm_resolution=args.dsm_res,
        metrics_resolution=args.metrics_res,
        percentiles=args.percentiles,
        start_step=args.start,
        end_step=args.end
    )

    # Configura logging inicial
    setup_logging()

    # Executa processamento
    if len(input_files) == 1:
        process_lidar(config, input_files[0])
    else:
        process_batch(config)
