#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NEURAL ODE PARA Metrics DE CRESCIMENTO FLORESTAL
================================================================================

CONCEITO GERAL
--------------
Este script usa Neural ODEs (Ordinary Differential Equations) para modelar o
crescimento de uma floresta ao longo do tempo. A ideia central e tratar o
crescimento como um processo continuo no tempo, descrito por uma equacao
diferencial:

    dX/dt = f_theta(X, t, features)

Onde:
    - X = posicao 3D de cada ponto da nuvem (x, y, z)
    - t = tempo (idade da floresta em anos, normalizado)
    - f_theta = rede neural que aprende a "velocidade" de crescimento
    - features = caracteristicas locais de cada ponto

DADOS DE ENTRADA
----------------
IMPORTANTE: Este script assume que os dados de entrada JA ESTAO NORMALIZADOS:
    - Coordenada Z = altura acima do solo (z' = z_original - DTM)
    - Ou seja, z=0 representa o nivel do solo

Isso simplifica o pipeline pois nao e necessario:
    - Construir/carregar DTM
    - Separar pontos de solo (ground)
    - Calcular altura normalizada durante processamento

VANTAGENS DA NEURAL ODE
-----------------------
1. CONTINUIDADE TEMPORAL: Diferente de modelos discretos, a ODE permite
   projetar para QUALQUER idade, nao apenas para instantes especificos.

2. FISICA EMBUTIDA: Podemos adicionar restricoes fisicas (ex: crescimento
   predominantemente vertical, sem movimento horizontal excessivo).

3. CONSERVACAO DE ESTRUTURA: O modelo move pontos existentes ao inves de
   criar/deletar, preservando a estrutura espacial da nuvem.

4. EXTRAPOLACAO: Treinado com dados de 2-8 anos, pode projetar para 10, 15,
   20 anos de forma suave e fisicamente plausivel.

FLUXO DE DADOS
--------------
1. ENTRADA: Nuvens de pontos LiDAR normalizadas de diferentes epocas

2. PRE-PROCESSAMENTO:
   - Filtrar pontos com z > 0.3 (remover ruido proximo ao solo)
   - Extrair features locais (densidade, normais, etc.)
   - Centralizar coordenadas XY

3. TREINAMENTO:
   - Integrar ODE de t0 para t1 (ex: idade 2 -> 5 anos)
   - Comparar nuvem projetada com nuvem real usando Chamfer Distance
   - Ajustar pesos da rede para minimizar a distancia

4. INFERENCIA:
   - Carregar nuvem de referencia (mais recente)
   - Integrar ODE ate a idade desejada
   - Salvar nuvem projetada em formato LAS

ARQUITETURA DA REDE
-------------------
A rede GrowthDynamics recebe:
    [x_rel, y_rel, z, t, feat1, feat2, feat3, feat4, feat5]

E produz:
    [vx, vy, vz] = velocidade de cada ponto em m/ano

Estrutura:
    Input (9) -> Linear(128) -> SiLU -> Linear(128) -> SiLU ->
    Linear(128) -> SiLU -> Linear(64) -> SiLU -> Linear(3)

RESTRICOES FISICAS
------------------
1. ESCALA DE VELOCIDADE: vx, vy limitados a 0.1x, vz a 1.0x
   (arvores crescem verticalmente, nao se movem horizontalmente)

2. HEIGHT FACTOR: Pontos mais baixos crescem mais rapido
   (simulando competicao por luz - pontos no topo crescem menos)

3. REGULARIZACAO: Penalizacao para velocidades negativas e drift horizontal

USO
---
    # Treinar e projetar (padrao)
    python neural_ode_forest.py

    # Apenas comparar epocas
    python neural_ode_forest.py compare

    # Apenas inferir para idade especifica
    python neural_ode_forest.py infer 15

SAIDAS
------
    out_ode/
        ode_model_best.pth    - Melhor modelo treinado
        ode_model_final.pth   - Modelo final
        projecao_ode_idade_X.las - Nuvem projetada para idade X

REFERENCIAS
-----------
- Chen et al. (2018) "Neural Ordinary Differential Equations"
- torchdiffeq: https://github.com/rtqichen/torchdiffeq

Autor: Leonardo
Data: 2025
================================================================================
"""

import numpy as np
import numpy.random as npr
from pathlib import Path
from tqdm import tqdm
import laspy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================================
# DEPENDENCIAS EXTERNAS
# ================================================================================
# torchdiffeq: Biblioteca para resolver ODEs com PyTorch, permitindo
# backpropagation atraves do solver (metodo adjunto)
try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError:
    raise ImportError("Instale: pip install torchdiffeq")

from sklearn.neighbors import NearestNeighbors


# ================================================================================
# CONFIGURACAO
# ================================================================================
# Diretorio raiz do projeto
DATA_DIR = Path(r"G:/PycharmProjects/Mestrado")

# Diretorio com arquivos LAZ organizados por ano de sobrevoo
PROJECAO_DIR = DATA_DIR / "Forecast/Projection/Metrics"


def scan_laz_files(base_dir, ref_id_filter=None):
    """
    Escaneia arquivos LAZ e extrai idade do nome do arquivo.

    Formato esperado: REF_ID_tile_IDADE_denoised_thin_norm.laz
    Exemplo: GNVIVI00580P18-002_210_70_denoised_thin_norm.laz
        - REF_ID: GNVIVI00580P18-002
        - tile: 210
        - idade: 70 meses

    Parametros
    ----------
    base_dir : Path
        Diretorio base (Metrics)
    ref_id_filter : str, opcional
        Filtrar por REF_ID especifico (ex: 'GNVIVI00580P18-002')

    Retorna
    -------
    dict
        Dicionario {ref_id: [{"idade": float, "las": Path, "ano": str}, ...]}
    """
    from collections import defaultdict
    import re

    results = defaultdict(list)
    # Formato: *_TILE_IDADE_denoised_thin_norm.laz
    pattern = re.compile(r'(.+)_(\d+)_(\d+)_denoised_thin_norm\.laz$')

    for laz_file in base_dir.rglob('*_denoised_thin_norm.laz'):
        match = pattern.match(laz_file.name)
        if not match:
            continue

        ref_id = match.group(1)       # Ex: GNVIVI00580P18-002
        tile = match.group(2)         # Ex: 210
        idade = int(match.group(3))   # Ex: 70 (meses)

        # Extrair ano do sobrevoo da pasta
        try:
            rel_path = laz_file.relative_to(base_dir)
            ano_sobrevoo = rel_path.parts[0]
        except:
            ano_sobrevoo = ''

        if ref_id_filter is None or ref_id == ref_id_filter:
            results[ref_id].append({
                "idade": idade,  # em meses
                "las": laz_file,
                "ano_sobrevoo": ano_sobrevoo,
                "tile": tile
            })

    # Ordenar cada lista por idade
    for ref_id in results:
        results[ref_id] = sorted(results[ref_id], key=lambda x: x["idade"])

    return dict(results)


def select_epochs_for_training(ref_id):
    """
    Seleciona epocas de um REF_ID especifico para treinamento.

    Parametros
    ----------
    ref_id : str
        Identificador do talhao (ex: 'GNVIVI00580P18-002')

    Retorna
    -------
    list
        Lista de epocas no formato [{"idade": float, "las": Path}, ...]
    """
    all_data = scan_laz_files(PROJECAO_DIR, ref_id_filter=ref_id)

    if ref_id not in all_data:
        print(f"[WARN] REF_ID '{ref_id}' nao encontrado.")
        all_refs = list(scan_laz_files(PROJECAO_DIR).keys())
        print(f"[INFO] REF_IDs disponiveis: {all_refs[:10]}...")
        return []

    epocas = all_data[ref_id]
    print(f"[INFO] Encontradas {len(epocas)} epocas para {ref_id}:")
    for e in epocas:
        print(f"  - Idade: {e['idade']} meses | "
              f"Ano: {e['ano_sobrevoo']} | Tile: {e['tile']} | {e['las'].name}")

    return [{"idade": e["idade"], "las": e["las"]} for e in epocas]


def list_available_ref_ids():
    """Lista todos os REF_IDs disponiveis com multiplas epocas."""
    all_data = scan_laz_files(PROJECAO_DIR)

    # Filtrar apenas REF_IDs com pelo menos 2 epocas
    multi_epoch = {k: v for k, v in all_data.items() if len(v) >= 2}

    print(f"[INFO] REF_IDs com 2+ epocas: {len(multi_epoch)}")
    for ref_id, epocas in sorted(multi_epoch.items(), key=lambda x: -len(x[1]))[:20]:
        idades = [str(e['idade']) for e in epocas]
        print(f"  {ref_id}: {len(epocas)} epocas ({', '.join(idades)} meses)")

    return multi_epoch


def load_all_epochs(min_epocas=2):
    """
    Carrega todos os arquivos LAZ de todos os REF_IDs para treinamento conjunto.

    Parametros
    ----------
    min_epocas : int
        Numero minimo de epocas por REF_ID para incluir

    Retorna
    -------
    list
        Lista de todas as epocas no formato [{"idade": int, "las": Path, "ref_id": str}, ...]
    """
    all_data = scan_laz_files(PROJECAO_DIR)

    # Filtrar REF_IDs com minimo de epocas
    multi_epoch = {k: v for k, v in all_data.items() if len(v) >= min_epocas}

    # Coletar todas as epocas
    all_epochs = []
    for ref_id, epocas in multi_epoch.items():
        for e in epocas:
            all_epochs.append({
                "idade": e["idade"],
                "las": e["las"],
                "ref_id": ref_id,
                "tile": e["tile"],
                "ano_sobrevoo": e["ano_sobrevoo"]
            })

    # Ordenar por idade
    all_epochs = sorted(all_epochs, key=lambda x: x["idade"])

    # Estatisticas
    idades_unicas = sorted(set(e["idade"] for e in all_epochs))
    print(f"[INFO] Carregados {len(all_epochs)} arquivos de {len(multi_epoch)} REF_IDs")
    print(f"[INFO] Idades unicas: {len(idades_unicas)} ({min(idades_unicas)} - {max(idades_unicas)} meses)")

    return all_epochs


# ================================================================================
# SELECAO DE DADOS PARA TREINAMENTO
# ================================================================================
# Modo de treinamento: 'all' para todos os dados, ou especificar REF_ID
MODO_TREINO = 'all'  # 'all' ou REF_ID especifico (ex: 'RDBOBA00412P14-178')

if MODO_TREINO == 'all':
    # Carregar todos os dados
    EPOCAS = load_all_epochs(min_epocas=2)
else:
    # Carregar apenas um REF_ID
    EPOCAS = select_epochs_for_training(MODO_TREINO)

# Qual epoca usar como referencia para inferencia
# 'last' = usar a mais recente
REF_INDEX = 'last'

# Idade alvo para projecao (meses)
ID_ALVO = 120  # 10 anos

# -------------------- HIPERPARAMETROS --------------------
KNN = 16           # Numero de vizinhos para calcular features locais
MAX_SAMP = 150000  # Maximo de pontos por epoca (subsampling se necessario)
LR = 5e-4          # Taxa de aprendizado (Learning Rate)
EPOCHS = 50        # Numero de epocas de treinamento
BATCH_PTS = 16384  # Pontos por batch (limitado pela memoria GPU)
MIN_HEIGHT = 0.3   # Altura minima para considerar vegetacao (metros)

# Diretorio de saida
OUT_DIR = DATA_DIR / "out_ode"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dispositivo de computacao (GPU se disponivel, senao CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Usando dispositivo: {DEVICE}")


# ================================================================================
# FUNCOES DE CARREGAMENTO DE DADOS
# ================================================================================

def load_las_points(las_path):
    """
    Carrega pontos de um arquivo LAS/LAZ.

    IMPORTANTE: Assume que z ja esta normalizado (altura acima do solo).

    Parametros
    ----------
    las_path : str ou Path
        Caminho do arquivo LAS ou LAZ

    Retorna
    -------
    dict
        Dicionario com:
        - x, y: coordenadas horizontais (float64)
        - z: altura normalizada acima do solo (float64)
        - intensity: intensidade do retorno (float32)
        - hdr: cabecalho do arquivo LAS
    """
    p = str(las_path)

    def to_np(a, dtype=None):
        """Converte para numpy array contiguo em memoria."""
        arr = np.asarray(a)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.ascontiguousarray(arr)

    try:
        # Arquivos LAZ sao comprimidos - tentar backend paralelo primeiro
        if p.lower().endswith(".laz"):
            try:
                with laspy.open(p, laz_backend=laspy.LazBackend.LazrsParallel) as f:
                    las = f.read()
            except Exception:
                with laspy.open(p, laz_backend=laspy.LazBackend.Lazrs) as f:
                    las = f.read()
        else:
            las = laspy.read(p)
    except Exception as e:
        raise SystemExit(f"[ERRO] Falha ao ler LAS: {p}\n{e}")

    return {
        "x": to_np(las.x, np.float64),
        "y": to_np(las.y, np.float64),
        "z": to_np(las.z, np.float64),  # Ja normalizado!
        "intensity": to_np(getattr(las, "intensity", np.zeros_like(las.x)), np.float32),
        "hdr": las.header,
    }


def _validate_and_sort_epocas(epocas):
    """
    Valida e ordena as epocas por idade.

    Filtra epocas cujos arquivos nao existem e ordena por idade crescente.
    Requer pelo menos 2 epocas validas para treinamento.
    """
    valid = []
    for e in epocas:
        p = Path(e["las"])
        if p.exists():
            entry = {"idade": int(e["idade"]), "las": p}
            # Preservar ref_id se disponivel
            if "ref_id" in e:
                entry["ref_id"] = e["ref_id"]
            if "tile" in e:
                entry["tile"] = e["tile"]
            valid.append(entry)

    if len(valid) < 2:
        raise SystemExit("[ERRO] Sao necessarias pelo menos 2 epocas validas.")

    return sorted(valid, key=lambda d: d["idade"])


def _resolve_ref_index(epocas_validas, ref_index):
    """
    Resolve o indice da epoca de referencia.

    Parametros
    ----------
    epocas_validas : list
        Lista de epocas validadas
    ref_index : str ou int
        'last' para ultima, ou indice numerico
    """
    n = len(epocas_validas)
    if isinstance(ref_index, str) and ref_index.lower() == 'last':
        return n - 1
    if isinstance(ref_index, int) and -n <= ref_index < n:
        return ref_index % n
    return n - 1


# ================================================================================
# EXTRACAO DE FEATURES LOCAIS
# ================================================================================

def compute_local_feats(X, z, intensity, k=KNN):
    """
    Calcula features locais para cada ponto da nuvem.

    Essas features ajudam a rede neural a entender o contexto espacial
    de cada ponto, permitindo diferentes taxas de crescimento baseadas
    na posicao relativa na vegetacao.

    Parametros
    ----------
    X : ndarray [N, 3]
        Coordenadas XYZ dos pontos (z ja normalizado)
    z : ndarray [N]
        Altura normalizada (igual a X[:, 2])
    intensity : ndarray [N]
        Intensidade do retorno LiDAR
    k : int
        Numero de vizinhos para calculo

    Retorna
    -------
    features : ndarray [N, 5]
        Matriz de features:
        - [0] densidade: inverso da distancia media aos vizinhos
        - [1] normal_z: componente Z da normal local (planaridade)
        - [2] var_z: variancia das alturas na vizinhanca
        - [3] rank: posicao relativa na vizinhanca (0=base, 1=topo)
        - [4] intensity: intensidade normalizada [0, 1]

    INTERPRETACAO DAS FEATURES
    --------------------------
    densidade: Alta = regiao densa (interior da copa)
               Baixa = regiao esparsa (bordas, galhos isolados)

    normal_z:  ~1 = superficie horizontal (folhas no topo)
               ~0 = superficie vertical (tronco, galhos laterais)

    var_z:     Alta = estrutura vertical variada
               Baixa = camada uniforme

    rank:      ~1 = ponto no topo da vizinhanca
               ~0 = ponto na base da vizinhanca

    intensity: Reflete a refletividade da superficie
               (folhas verdes vs galhos secos)
    """
    # Construir estrutura KNN
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X)), algorithm='kd_tree').fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)

    # ---------- FEATURE 1: Densidade ----------
    # Inverso da distancia media aos vizinhos (exclui o proprio ponto)
    dens = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-6)

    # ---------- FEATURE 2 e 3: Normal local e variancia ----------
    normals_z = np.zeros(len(X), dtype=np.float32)
    var_z = np.zeros(len(X), dtype=np.float32)

    for i in range(len(X)):
        P = X[idxs[i]]  # Pontos na vizinhanca
        if len(P) > 2:
            # Calcular normal via PCA (autovetor do menor autovalor)
            C = np.cov(P.T)  # Matriz de covariancia 3x3
            w, v = np.linalg.eigh(C)  # Autovalores e autovetores
            n = v[:, np.argmin(w)]  # Normal = autovetor do menor autovalor
            normals_z[i] = abs(n[2])  # Componente Z (0 a 1)

        # Variancia das alturas na vizinhanca
        var_z[i] = z[idxs[i]].var() if len(idxs[i]) > 1 else 0.0

    # ---------- FEATURE 4: Rank vertical ----------
    # Posicao relativa do ponto dentro da sua vizinhanca
    rank = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        z_loc = z[idxs[i]]
        m = z_loc.max() - z_loc.min()
        rank[i] = 0.0 if m < 1e-6 else (z[i] - z_loc.min()) / m

    # ---------- FEATURE 5: Intensidade normalizada ----------
    inten = intensity / (intensity.max() + 1e-6) if intensity.max() > 0 else intensity

    # Empilhar todas as features em matriz [N, 5]
    return np.stack([dens, normals_z, var_z, rank, inten], axis=1).astype(np.float32)


# ================================================================================
# CLASSE: GrowthDynamics (Dinamica de Crescimento)
# ================================================================================

class GrowthDynamics(nn.Module):
    """
    Rede neural que define a dinamica de crescimento florestal.

    MODELO GERAL: dz/dt = f(z, idade, features)

    A rede aprende a taxa de crescimento vertical como funcao de:
    - z: altura atual do ponto (metros)
    - idade: idade da floresta (normalizada)
    - features: caracteristicas locais do ponto

    NAO usa coordenadas XY pois arvores nao se movem horizontalmente.

    RESTRICOES FISICAS
    ------------------
    1. Crescimento sempre positivo (Softplus na saida)
    2. Pontos mais altos crescem mais devagar (competicao por luz)
    3. Taxa diminui com a idade (saturacao)

    Arquitetura
    -----------
    Input: [z, idade, f1, f2, f3, f4, f5] = 7 dimensoes

    Hidden layers:
        Linear(7 -> 128) + SiLU
        Linear(128 -> 128) + SiLU
        Linear(128 -> 64) + SiLU
        Linear(64 -> 1) + Softplus

    Output: dz/dt (taxa de crescimento em m/mes)
    """

    def __init__(self, feat_dim=5, hidden_dim=128):
        super().__init__()

        # Dimensao de entrada: z (1) + idade (1) + features (feat_dim)
        input_dim = 1 + 1 + feat_dim

        # Rede MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # Inicializar com taxa de crescimento positiva
        self.net[-1].bias.data = torch.tensor([0.1])

    def forward(self, t, state):
        """
        Calcula dz/dt para cada ponto no tempo t.

        Parametros
        ----------
        t : tensor escalar
            Idade atual (normalizada entre 0 e 1)
        state : tensor [B, N, 1+feat_dim]
            Estado atual:
            - state[..., :1] = z (altura)
            - state[..., 1:] = features (constantes)

        Retorna
        -------
        dstate/dt : tensor [B, N, 1+feat_dim]
            - [..., :1] = dz/dt (taxa de crescimento)
            - [..., 1:] = zeros (features nao mudam)
        """
        z = state[..., :1]        # [B, N, 1] - altura
        feats = state[..., 1:]    # [B, N, feat_dim] - features

        # Expandir idade t para cada ponto
        B, N, _ = z.shape
        t_val = t.view(1, 1, 1).expand(B, N, 1)

        # Input: [z, idade, features]
        inp = torch.cat([z, t_val, feats], dim=-1)

        # Taxa de crescimento (Softplus garante valor positivo)
        dz = F.softplus(self.net(inp))  # [B, N, 1]

        # Fator de saturacao: pontos mais altos crescem mais devagar
        # Crescimento diminui logaritmicamente com a altura
        height_factor = 1.0 / (1.0 + z / 20.0)  # ~1 em z=0, ~0.5 em z=20m
        dz = dz * height_factor

        # Features nao mudam
        dfeats = torch.zeros_like(feats)

        return torch.cat([dz, dfeats], dim=-1)


# ================================================================================
# CLASSE: NeuralODEForest (Modelo Principal)
# ================================================================================

class NeuralODEForest(nn.Module):
    """
    Modelo Neural ODE para projecao de crescimento florestal.

    MODELO SIMPLIFICADO: apenas crescimento vertical (z)

    Integra a ODE: dz/dt = f(z, idade, features)
    para projetar alturas de uma idade para outra.

    Parametros
    ----------
    feat_dim : int
        Numero de features por ponto
    hidden_dim : int
        Dimensao das camadas ocultas da rede
    use_adjoint : bool
        Se True, usa metodo adjunto (menos memoria)
    """

    def __init__(self, feat_dim=5, hidden_dim=128, use_adjoint=True):
        super().__init__()
        self.dynamics = GrowthDynamics(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.use_adjoint = use_adjoint
        self.ode_solver = odeint_adjoint if use_adjoint else odeint

    def forward(self, z, features, t_span):
        """
        Integra a ODE de crescimento ao longo do tempo.

        Parametros
        ----------
        z : tensor [B, N, 1]
            Alturas iniciais dos pontos
        features : tensor [B, N, feat_dim]
            Features locais de cada ponto
        t_span : tensor [T]
            Idades para integracao (normalizadas)

        Retorna
        -------
        z_trajectory : tensor [T, B, N, 1]
            Alturas dos pontos em cada tempo de t_span
        """
        # Concatenar estado inicial: [z, features]
        state0 = torch.cat([z, features], dim=-1)  # [B, N, 1+feat_dim]

        # Resolver ODE usando metodo Dormand-Prince (dopri5)
        trajectory = self.ode_solver(
            self.dynamics,
            state0,
            t_span,
            method='dopri5',
            rtol=1e-3,
            atol=1e-4,
            options={'max_num_steps': 1000}
        )

        # Retornar apenas z (sem features)
        return trajectory[..., :1]  # [T, B, N, 1]


# ================================================================================
# PREPARACAO DE DADOS (SIMPLIFICADA - DADOS JA NORMALIZADOS)
# ================================================================================

def sample_epoch(P, max_samp=MAX_SAMP):
    """
    Amostra e processa pontos de uma epoca.

    IMPORTANTE: Assume que z ja esta normalizado (altura acima do solo).

    Parametros
    ----------
    P : dict
        Dados da nuvem carregados por load_las_points()
    max_samp : int
        Maximo de pontos a amostrar

    Retorna
    -------
    dict ou None
        - X: coordenadas [N, 3] com z normalizado
        - z: alturas (igual a X[:, 2]) [N]
        - intensity: intensidades [N]
        - F: features locais [N, 5]
        None se nao houver pontos validos
    """
    x, y, z = P["x"], P["y"], P["z"]
    intensity = P["intensity"]

    # Filtrar pontos acima da altura minima (remover ruido proximo ao solo)
    mask = z > MIN_HEIGHT
    x, y, z = x[mask], y[mask], z[mask]
    intensity = intensity[mask]

    N = len(x)
    if N == 0:
        return None

    # Subsampling se necessario
    if N > max_samp:
        sel = npr.choice(N, max_samp, replace=False)
        x, y, z, intensity = x[sel], y[sel], z[sel], intensity[sel]

    # Construir matriz de coordenadas
    X = np.c_[x, y, z].astype(np.float32)

    # Calcular features locais
    F = compute_local_feats(X, z.astype(np.float32), intensity, k=min(KNN, len(X)))

    return {
        "X": X,
        "z": z.astype(np.float32),
        "intensity": intensity.astype(np.float32),
        "F": F
    }


def prepare_data():
    """
    Prepara dados de todas as epocas para treinamento.

    Retorna
    -------
    epochs_data : list
        Lista de dicts com dados de cada epoca
    epocas_validas : list
        Epocas validadas e ordenadas
    ref_idx : int
        Indice da epoca de referencia
    """
    epocas_validas = _validate_and_sort_epocas(EPOCAS)
    ref_idx = _resolve_ref_index(epocas_validas, REF_INDEX)

    # Resumo das epocas
    idades = [e['idade'] for e in epocas_validas]
    ref_ids = set(e.get('ref_id', 'N/A') for e in epocas_validas)
    print(f"[INFO] Epocas validas: {len(epocas_validas)} arquivos")
    print(f"[INFO] REF_IDs: {len(ref_ids)}")
    print(f"[INFO] Idades: {min(idades)} - {max(idades)} meses")
    print(f"[INFO] Referencia: idx={ref_idx}, idade={epocas_validas[ref_idx]['idade']} meses")

    # Processar todas as epocas
    print(f"[INFO] Carregando e processando {len(epocas_validas)} arquivos...")
    epochs_data = []
    for i, e in enumerate(tqdm(epocas_validas, desc="Carregando LAZ")):
        P = load_las_points(e["las"])
        samp = sample_epoch(P)
        if samp is not None:
            epochs_data.append({
                "idade": e["idade"],
                "ref_id": e.get("ref_id", "unknown"),
                "tile": e.get("tile", "0"),
                "las": e["las"],
                "data": samp
            })

    print(f"[INFO] {len(epochs_data)} epocas processadas com sucesso")
    return epochs_data, epocas_validas, ref_idx


# ================================================================================
# FUNCOES DE PERDA (LOSS FUNCTIONS)
# ================================================================================

def chamfer_loss(pred, target):
    """
    Calcula a Chamfer Distance entre duas nuvens de pontos.

    A Chamfer Distance e uma metrica comum para comparar nuvens de pontos.
    Ela mede a "distancia" bidirecional:

        CD = mean(min_j ||p_i - t_j||^2) + mean(min_i ||t_j - p_i||^2)

    Primeiro termo: para cada ponto predito, distancia ao ponto mais proximo no target
    Segundo termo: para cada ponto target, distancia ao ponto mais proximo na predicao

    Parametros
    ----------
    pred : tensor [B, N, 3]
        Nuvem predita
    target : tensor [B, M, 3]
        Nuvem alvo

    Retorna
    -------
    loss : tensor escalar
        Chamfer distance media
    """
    # Usar float64 para precisao numerica
    pred64 = pred.double()
    target64 = target.double()

    # Matriz de distancias pairwise [B, N, M]
    D = torch.cdist(pred64, target64, p=2) ** 2

    # Distancia minima para cada ponto predito -> target
    d_pred = D.min(dim=2).values.mean()

    # Distancia minima para cada ponto target -> predito
    d_target = D.min(dim=1).values.mean()

    return (d_pred + d_target).float()


def velocity_regularization(model, xyz, features, t):
    """
    Regularizacao para garantir velocidades fisicamente plaussiveis.

    Esta funcao penaliza:
    1. Drift horizontal excessivo (arvores nao andam!)
    2. Crescimento negativo (decaimento)
    3. Velocidades muito altas

    Parametros
    ----------
    model : NeuralODEForest
        Modelo para calcular velocidades
    xyz : tensor [B, N, 3]
        Posicoes dos pontos
    features : tensor [B, N, feat_dim]
        Features dos pontos
    t : float
        Tempo normalizado

    Retorna
    -------
    loss : tensor escalar
        Termo de regularizacao
    """
    t_tensor = torch.tensor([t], dtype=torch.float32, device=xyz.device)
    state = torch.cat([xyz, features], dim=-1)

    with torch.enable_grad():
        v_full = model.dynamics(t_tensor, state)
        v = v_full[..., :3]  # Velocidades xyz

    # Penalizar drift horizontal (vx, vy)
    loss_xy = (v[..., :2] ** 2).mean()

    # Penalizar crescimento negativo (vz < 0)
    loss_neg_z = F.relu(-v[..., 2]).mean()

    # Penalizar velocidades muito altas
    loss_mag = (v ** 2).mean()

    # Pesos dos termos
    return 0.1 * loss_xy + 0.5 * loss_neg_z + 0.01 * loss_mag


# ================================================================================
# TREINAMENTO
# ================================================================================

def _build_training_pairs(epochs_data):
    """
    Agrupa epocas por ref_id+tile e cria pares de treinamento.

    Para treinar a ODE, precisamos de pares (t0, t1) do MESMO local
    em tempos diferentes.

    Retorna
    -------
    list
        Lista de tuplas (src_epoch, tgt_epoch) para treinamento
    """
    from collections import defaultdict

    # Agrupar por ref_id + tile
    groups = defaultdict(list)
    for e in epochs_data:
        key = f"{e['ref_id']}_{e['tile']}"
        groups[key].append(e)

    # Ordenar cada grupo por idade e criar pares
    pairs = []
    for key, epochs in groups.items():
        epochs_sorted = sorted(epochs, key=lambda x: x["idade"])
        # Criar pares consecutivos
        for i in range(len(epochs_sorted) - 1):
            pairs.append((epochs_sorted[i], epochs_sorted[i + 1]))

    return pairs


def train():
    """
    Loop principal de treinamento do modelo Neural ODE.

    ESTRATEGIA DE TREINAMENTO
    -------------------------
    1. Agrupar epocas por ref_id+tile (mesmo local)
    2. Para cada par de epocas do mesmo local (t0, t1):
       - Amostrar pontos da epoca t0 (source)
       - Amostrar pontos da epoca t1 (target)
       - Integrar ODE de t0 ate t1
       - Calcular Chamfer loss entre predicao e target
       - Adicionar termos de regularizacao
       - Backpropagation e atualizacao de pesos

    3. Salvar melhor modelo (menor loss)

    NORMALIZACAO
    ------------
    - Idade: normalizada para [0, 1]
    - Z: altura normalizada (acima do solo)

    Retorna
    -------
    model : NeuralODEForest
        Modelo treinado
    age_min, age_range : float
        Parametros de normalizacao de idade
    """
    print("=" * 50)
    print("NEURAL ODE FOREST - TREINAMENTO")
    print("=" * 50)

    epochs_data, epocas_validas, ref_idx = prepare_data()

    if len(epochs_data) < 2:
        raise SystemExit("[ERRO] Menos de 2 epocas com dados validos.")

    # ---------- CRIAR PARES DE TREINAMENTO ----------
    training_pairs = _build_training_pairs(epochs_data)
    print(f"[INFO] Pares de treinamento: {len(training_pairs)}")

    if len(training_pairs) == 0:
        raise SystemExit("[ERRO] Nenhum par de treinamento encontrado.")

    print(f"[INFO] Modelo simplificado: apenas crescimento vertical (z)")

    # ---------- MODELO E OTIMIZADOR ----------
    model = NeuralODEForest(feat_dim=5, hidden_dim=128, use_adjoint=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Normalizar idades para [0, 1]
    age_min = min(e["idade"] for e in epochs_data)
    age_max = max(e["idade"] for e in epochs_data)
    age_range = max(age_max - age_min, 1.0)

    def normalize_age(t):
        return (t - age_min) / age_range

    print(f"[INFO] Idade range: [{age_min}, {age_max}] meses -> normalizado [0, 1]")

    best_loss = float('inf')

    # ---------- LOOP DE TREINAMENTO ----------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []

        # Shuffle dos pares para cada epoca
        npr.shuffle(training_pairs)

        # Treinar em pares do mesmo local
        for src, tgt in training_pairs:
            t0 = normalize_age(src["idade"])
            t1 = normalize_age(tgt["idade"])

            # Amostrar pontos
            Ns = len(src["data"]["X"])
            Nt = len(tgt["data"]["X"])

            idx_s = npr.choice(Ns, min(BATCH_PTS, Ns), replace=False)
            idx_t = npr.choice(Nt, min(BATCH_PTS, Nt), replace=False)

            # Preparar dados - apenas z e features
            z_src = src["data"]["z"][idx_s].reshape(-1, 1)
            F_src = src["data"]["F"][idx_s]
            z_tgt = tgt["data"]["z"][idx_t]

            # Converter para tensores
            z_src_t = torch.from_numpy(z_src).float().unsqueeze(0).to(DEVICE)
            feats_src = torch.from_numpy(F_src).float().unsqueeze(0).to(DEVICE)
            z_tgt_t = torch.from_numpy(z_tgt).float().to(DEVICE)

            t_span = torch.tensor([t0, t1], dtype=torch.float32, device=DEVICE)

            # Forward pass
            optimizer.zero_grad()

            try:
                # Integrar ODE de t0 para t1
                z_trajectory = model(z_src_t, feats_src, t_span)  # [2, B, N, 1]
                z_pred = z_trajectory[-1, 0, :, 0]  # [N] - alturas preditas em t1

                # ---------- CALCULAR LOSSES ----------
                # 1. Comparar distribuicoes de z (estatisticas)
                # MSE entre percentis
                percentiles = [10, 25, 50, 75, 90, 95]
                loss_percentiles = 0.0
                for p in percentiles:
                    p_pred = torch.quantile(z_pred, p / 100.0)
                    p_tgt = torch.quantile(z_tgt_t, p / 100.0)
                    loss_percentiles += (p_pred - p_tgt) ** 2
                loss_percentiles /= len(percentiles)

                # 2. MSE entre medias
                loss_mean = (z_pred.mean() - z_tgt_t.mean()) ** 2

                # 3. MSE entre desvios padrao
                loss_std = (z_pred.std() - z_tgt_t.std()) ** 2

                # 4. Regularizacao: crescimento deve ser positivo
                dz = z_pred - z_src_t[0, :, 0]
                loss_neg = F.relu(-dz).mean() * 0.1

                # Loss total
                loss = loss_percentiles + loss_mean + loss_std + loss_neg

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                epoch_losses.append(loss.item())

            except Exception as e:
                print(f"[WARN] Erro no batch: {e}")
                continue

        # Atualizar learning rate
        scheduler.step()

        # ---------- LOGGING E SALVAMENTO ----------
        if epoch_losses:
            mean_loss = np.mean(epoch_losses)

            # Salvar melhor modelo
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save({
                    'model': model.state_dict(),
                    'age_min': age_min,
                    'age_range': age_range,
                }, OUT_DIR / "ode_model_best.pth")

            # Log periodico
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {mean_loss:.6f} | "
                      f"Best: {best_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Salvar modelo final
    torch.save({
        'model': model.state_dict(),
        'age_min': age_min,
        'age_range': age_range,
    }, OUT_DIR / "ode_model_final.pth")

    print(f"\n[INFO] Modelo salvo em {OUT_DIR}")
    print(f"[INFO] Melhor loss: {best_loss:.6f}")

    return model, age_min, age_range


# ================================================================================
# INFERENCIA
# ================================================================================

def infer_to_age(idade_alvo, model_path=None):
    """
    Projeta a nuvem de pontos para uma idade alvo.

    Carrega o modelo treinado e integra a ODE da epoca de referencia
    ate a idade desejada, gerando uma nova nuvem de pontos.

    MODELO SIMPLIFICADO: apenas crescimento vertical (z)
    Coordenadas XY sao preservadas - apenas Z e atualizado.

    Parametros
    ----------
    idade_alvo : int
        Idade da floresta desejada (meses)
    model_path : Path, opcional
        Caminho do modelo. Default: ode_model_best.pth

    Retorna
    -------
    out_path : Path
        Caminho do arquivo LAS gerado
    Xnew : ndarray [N, 3]
        Coordenadas dos pontos projetados (z normalizado)
    """
    print("=" * 50)
    print(f"Metrics PARA IDADE {idade_alvo} MESES")
    print("=" * 50)

    # ---------- CARREGAR MODELO ----------
    if model_path is None:
        model_path = OUT_DIR / "ode_model_best.pth"

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    age_min = checkpoint['age_min']
    age_range = checkpoint['age_range']

    model = NeuralODEForest(feat_dim=5, hidden_dim=128, use_adjoint=False).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # ---------- CARREGAR DADOS DE REFERENCIA ----------
    epocas_validas = _validate_and_sort_epocas(EPOCAS)
    ref_idx = _resolve_ref_index(epocas_validas, REF_INDEX)
    ref_path = epocas_validas[ref_idx]["las"]
    idade_src = epocas_validas[ref_idx]["idade"]

    Pref = load_las_points(ref_path)

    # Filtrar pontos acima da altura minima
    mask = Pref["z"] > MIN_HEIGHT
    x_orig = Pref["x"][mask].astype(np.float32)
    y_orig = Pref["y"][mask].astype(np.float32)
    z_orig = Pref["z"][mask].astype(np.float32)
    inten = Pref["intensity"][mask].astype(np.float32)

    # Construir matriz XYZ para calculo de features
    Xv = np.c_[x_orig, y_orig, z_orig]

    # Calcular features locais
    print("[INFO] Computando features locais...")
    Fv = compute_local_feats(Xv, z_orig, inten, k=min(KNN, len(Xv)))

    # ---------- PREPARAR INTEGRACAO ----------
    t0 = (idade_src - age_min) / age_range
    t1 = (idade_alvo - age_min) / age_range

    print(f"[INFO] Integrando de t={t0:.3f} ({idade_src} meses) "
          f"para t={t1:.3f} ({idade_alvo} meses)")

    # ---------- INFERENCIA EM BATCHES ----------
    B = 32768  # Pontos por batch
    z_new = np.zeros_like(z_orig)
    n_steps = 5  # Passos intermediarios

    t_span = torch.linspace(t0, t1, n_steps, device=DEVICE)

    with torch.no_grad():
        for s in tqdm(range(0, len(z_orig), B), desc="Projetando"):
            e = min(s + B, len(z_orig))

            # Preparar tensores - apenas z e features
            z_batch = torch.from_numpy(z_orig[s:e].reshape(-1, 1)).float().unsqueeze(0).to(DEVICE)
            feats = torch.from_numpy(Fv[s:e]).float().unsqueeze(0).to(DEVICE)

            # Integrar ODE (modelo retorna apenas z)
            z_trajectory = model(z_batch, feats, t_span)  # [T, B, N, 1]

            # Extrair z final
            z_pred = z_trajectory[-1, 0, :, 0].cpu().numpy()  # [N]

            # Garantir que z >= altura minima
            z_pred = np.maximum(z_pred, MIN_HEIGHT)

            z_new[s:e] = z_pred

            # Log do primeiro batch
            if s == 0:
                dz = z_pred - z_orig[s:e]
                print(f"[INFO] Crescimento mediano em z: {np.median(dz):.3f} m")
                print(f"[INFO] Crescimento p95 em z: {np.percentile(dz, 95):.3f} m")

    # ---------- ESTATISTICAS ----------
    dz_total = z_new - z_orig
    print(f"\n[STATS] Crescimento vertical (dz):")
    print(f"  min={dz_total.min():.2f} m")
    print(f"  max={dz_total.max():.2f} m")
    print(f"  mediana={np.median(dz_total):.2f} m")
    print(f"  media={dz_total.mean():.2f} m")

    # Construir Xnew com XY original e Z atualizado
    Xnew = np.c_[x_orig, y_orig, z_new]

    # ---------- ESCREVER ARQUIVO LAS ----------
    # Criar header baseado no original
    ref_hdr = Pref["hdr"]
    hdr = laspy.LasHeader(point_format=ref_hdr.point_format, version=ref_hdr.version)
    hdr.scales = ref_hdr.scales
    hdr.offsets = ref_hdr.offsets

    # Preservar CRS se disponivel
    try:
        crs = ref_hdr.parse_crs()
        if crs is not None:
            hdr.add_crs(crs)
    except Exception:
        pass

    # Criar arquivo de saida
    las_out = laspy.LasData(hdr)
    las_out.x = Xnew[:, 0]
    las_out.y = Xnew[:, 1]
    las_out.z = Xnew[:, 2]  # z normalizado

    # Classificacao baseada na altura normalizada
    z_new = Xnew[:, 2]
    cls = np.where(z_new < 2.0, 3,        # Low vegetation
          np.where(z_new < 5.0, 4, 5))    # Medium / High vegetation
    cls = cls.astype(np.uint8)

    # Intensidade normalizada
    vint = (inten / (inten.max() + 1e-6) * 100).astype(np.uint16)

    las_out.intensity = vint
    las_out.classification = cls

    n = len(las_out.x)
    las_out.return_number = np.ones(n, dtype=np.uint8)
    las_out.number_of_returns = np.ones(n, dtype=np.uint8)

    # Salvar
    out_path = OUT_DIR / f"projecao_ode_idade_{idade_alvo}m.las"
    las_out.write(str(out_path))
    print(f"\n[OK] Gerado: {out_path}")

    return out_path, Xnew


# ================================================================================
# DIAGNOSTICO
# ================================================================================

def compare_epochs():
    """
    Compara estatisticas entre epocas para diagnostico.

    Util para verificar a consistencia dos dados antes do treinamento
    e para validar se o crescimento observado e razoavel.
    """
    print("=" * 50)
    print("DIAGNOSTICO: COMPARACAO ENTRE EPOCAS")
    print("=" * 50)

    epochs_data, epocas_validas, ref_idx = prepare_data()

    print("\nEstatisticas de altura normalizada (z):\n")
    print(f"{'Idade':<8} {'N pts':<10} {'z min':<8} {'z max':<8} {'z med':<8} {'z mean':<8}")
    print("-" * 60)

    for ep in epochs_data:
        z = ep["data"]["z"]
        print(f"{ep['idade']:<8.1f} {len(z):<10} {z.min():<8.2f} "
              f"{z.max():<8.2f} {np.median(z):<8.2f} {z.mean():<8.2f}")

    # Crescimento observado
    print("\nCrescimento observado entre epocas:")
    for i in range(len(epochs_data) - 1):
        e1 = epochs_data[i]
        e2 = epochs_data[i + 1]
        dt = e2["idade"] - e1["idade"]
        dz_med = np.median(e2["data"]["z"]) - np.median(e1["data"]["z"])
        dz_max = e2["data"]["z"].max() - e1["data"]["z"].max()
        print(f"  {e1['idade']} -> {e2['idade']} meses (dt={dt}): "
              f"dz_med = {dz_med:.2f} m ({dz_med / dt:.3f} m/mes), "
              f"dz_max = {dz_max:.2f} m ({dz_max / dt:.3f} m/mes)")


# ================================================================================
# PONTO DE ENTRADA
# ================================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Apenas diagnostico
        compare_epochs()

    elif len(sys.argv) > 1 and sys.argv[1] == "infer":
        # Apenas inferencia
        idade = float(sys.argv[2]) if len(sys.argv) > 2 else ID_ALVO
        infer_to_age(idade)

    else:
        # Fluxo completo: diagnostico -> treino -> inferencia
        compare_epochs()
        print("\n")
        train()
        print("\n")
        infer_to_age(ID_ALVO)
