"""
03_AplicacaoModelos.py - Aplicacao dos Modelos de Projecao aos Rasters

Este script aplica os modelos exponenciais ajustados (Gompertz/Logistico/Exp.Modificado)
aos rasters de metricas LiDAR para projecao temporal.

Formula de projecao:
    z2 = z1 * modelo(idade_futura) / modelo(idade_atual)

Estrutura de entrada:
    - Metrics Projected/cubmean/  -> Z Kurt
    - Metrics Projected/p90/      -> Z P90
    - Metrics Projected/stddev/   -> Z sigma

Nomenclatura dos arquivos TIF:
    GNSACR00627P12-084_125_122.tif
    - [0:2]   -> Regional (GN, NE, RD)
    - [11]    -> Regime: 'R' = Talhadia, 'P' = Alto fuste
    - _XXX_   -> Primeiro numero apos _ = idade atual (meses)
    - _XXX    -> Segundo numero apos _ = idade futura (meses)

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predicao de Volume Florestal com LiDAR
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURACOES GLOBAIS
# =============================================================================

# Diretorio base dos rasters de metricas
METRICS_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Metrics Projected"

# Arquivo com modelos ajustados
MODELS_FILE = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\models\modelos_exponenciais_resumo.xlsx"

# Mapeamento de subfolders para variaveis no Excel
FOLDER_VAR_MAP = {
    "cubmean": "Z Kurt",
    "p90": "Z P90",
    "stddev": "Z σ"
}

# =============================================================================
# DEFINICAO DOS MODELOS
# =============================================================================

def gompertz(x, a, b, c):
    """Modelo de Gompertz: y = a * exp(-b * exp(-c * x))"""
    return a * np.exp(-b * np.exp(-c * x))


def logistico(x, a, b, c):
    """Modelo Logistico: y = a / (1 + b * exp(-c * x))"""
    return a / (1 + b * np.exp(-c * x))


def exponencial_modificado(x, a, b):
    """Modelo Exponencial Modificado: y = a * exp(b / x)"""
    return a * np.exp(b / x)


def get_model_prediction(modelo_nome, x, a, b, c=None):
    """Retorna a predicao do modelo baseado no nome."""
    if modelo_nome == 'Gompertz':
        return gompertz(x, a, b, c)
    elif modelo_nome == 'Logístico':
        return logistico(x, a, b, c)
    elif modelo_nome == 'Exp. Modificado':
        return exponencial_modificado(x, a, b)
    else:
        # Fallback para Gompertz
        return gompertz(x, a, b, c)


# =============================================================================
# FUNCOES DE EXTRACAO DE INFORMACOES DO NOME DO ARQUIVO
# =============================================================================

def extrair_info_arquivo(filename):
    """
    Extrai informacoes do nome do arquivo TIF.

    Exemplo: GNSACR00627P12-084_125_122.tif
    - Regional: GN (primeiras 2 letras)
    - Regime: P -> Alto fuste, R -> Talhadia (posicao 11)
    - Idade atual: 125 (primeiro numero apos _)
    - Idade futura: 122 (segundo numero apos segundo _)

    Returns:
        dict com regional, regime, idade_futura, idade_atual, grupo
        ou None se nao conseguir extrair
    """
    # Remove extensao
    base_name = os.path.splitext(filename)[0]

    # Extrai regional (primeiras 2 letras)
    if len(base_name) < 12:
        return None

    regional = base_name[:2].upper()

    # Extrai regime (posicao 11)
    char_regime = base_name[11].upper()
    if char_regime == 'R':
        regime = 'Talhadia'
    elif char_regime == 'P':
        regime = 'Alto fuste'
    else:
        print(f"    [AVISO] Caractere de regime desconhecido '{char_regime}' em {filename}")
        regime = 'Alto fuste'  # Default

    # Extrai idades usando regex
    # Padrao: _XXX_XXX onde XXX sao numeros
    match = re.search(r'_(\d+)_(\d+)(?:\.tif)?$', base_name, re.IGNORECASE)
    if not match:
        print(f"    [AVISO] Nao foi possivel extrair idades de {filename}")
        return None

    idade_atual = int(match.group(1))
    idade_futura = int(match.group(2))

    grupo = f"{regional}_{regime}"

    return {
        'regional': regional,
        'regime': regime,
        'grupo': grupo,
        'idade_futura': idade_futura,
        'idade_atual': idade_atual
    }


# =============================================================================
# FUNCAO DE PROJECAO DO RASTER
# =============================================================================

def projetar_raster(input_path, output_path, modelo_info, idade_futura, idade_atual):
    """
    Aplica a projecao temporal ao raster.

    Formula: z2 = z1 * modelo(idade_futura) / modelo(idade_atual)

    Args:
        input_path: Caminho do raster de entrada
        output_path: Caminho do raster de saida
        modelo_info: Dict com parametros do modelo (modelo, a, b, c, k_opt)
        idade_futura: Idade futura em meses
        idade_atual: Idade atual em meses

    Returns:
        True se sucesso, False se erro
    """
    try:
        # Calcula valores do modelo nas duas idades
        modelo_nome = modelo_info['modelo']
        a = modelo_info['a']
        b = modelo_info['b']
        c = modelo_info['c'] if pd.notna(modelo_info.get('c')) else None

        zhat_futuro = get_model_prediction(modelo_nome, idade_futura, a, b, c)
        zhat_atual = get_model_prediction(modelo_nome, idade_atual, a, b, c)

        # Evita divisao por zero
        if zhat_atual <= 0 or not np.isfinite(zhat_atual):
            print(f"    [ERRO] Zhat atual invalido: {zhat_atual}")
            return False

        # Fator de projecao
        fator = zhat_futuro / zhat_atual

        # Le raster de entrada
        with rasterio.open(input_path) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            nodata = src.nodata

            # Cria mascara de nodata
            if nodata is not None:
                mask = (data == nodata) | np.isnan(data)
            else:
                mask = np.isnan(data)

            # Aplica projecao: z2 = z1 * fator
            data_proj = data * fator

            # Restaura nodata
            data_proj[mask] = nodata if nodata is not None else np.nan

        # Atualiza profile para saida
        profile.update(dtype=rasterio.float32)

        # Salva raster projetado
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data_proj, 1)

        return True

    except Exception as e:
        print(f"    [ERRO] Falha ao projetar raster: {e}")
        return False


# =============================================================================
# FUNCAO PRINCIPAL DE PROCESSAMENTO
# =============================================================================

def processar_pasta(folder_name, var_name, modelos_df):
    """
    Processa todos os TIFs de uma pasta de metricas.

    Args:
        folder_name: Nome da pasta (cubmean, p90, stddev)
        var_name: Nome da variavel no Excel (Z Kurt, Z P90, Z sigma)
        modelos_df: DataFrame com modelos filtrados (melhor = 'X')
    """
    input_folder = os.path.join(METRICS_DIR, folder_name)
    output_folder = os.path.join(METRICS_DIR, f"{folder_name}_projected")

    # Cria pasta de saida se nao existir
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PROCESSANDO: {folder_name} -> {var_name}")
    print(f"{'='*60}")
    print(f"  Entrada:  {input_folder}")
    print(f"  Saida:    {output_folder}")

    # Filtra modelos para esta variavel
    modelos_var = modelos_df[modelos_df['variavel'] == var_name].copy()
    print(f"  Modelos disponiveis: {len(modelos_var)}")

    if len(modelos_var) == 0:
        print(f"  [ERRO] Nenhum modelo encontrado para {var_name}")
        return

    # Lista grupos disponiveis
    grupos_disponiveis = modelos_var['grupo'].unique()
    print(f"  Grupos: {list(grupos_disponiveis)}")

    # Lista arquivos TIF na pasta
    if not os.path.exists(input_folder):
        print(f"  [ERRO] Pasta nao encontrada: {input_folder}")
        return

    tif_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]
    print(f"  Arquivos TIF encontrados: {len(tif_files)}")

    # Contadores
    processados = 0
    erros = 0
    sem_modelo = 0

    # Processa cada arquivo
    for tif_file in tif_files:
        input_path = os.path.join(input_folder, tif_file)
        output_path = os.path.join(output_folder, tif_file)

        # Extrai informacoes do nome
        info = extrair_info_arquivo(tif_file)
        if info is None:
            print(f"    [SKIP] {tif_file} - Nao foi possivel extrair informacoes")
            erros += 1
            continue

        # Busca modelo para o grupo
        grupo = info['grupo']
        modelo_row = modelos_var[modelos_var['grupo'] == grupo]

        if len(modelo_row) == 0:
            print(f"    [SKIP] {tif_file} - Modelo nao encontrado para grupo '{grupo}'")
            sem_modelo += 1
            continue

        modelo_info = modelo_row.iloc[0].to_dict()

        # Aplica projecao
        sucesso = projetar_raster(
            input_path,
            output_path,
            modelo_info,
            info['idade_futura'],
            info['idade_atual']
        )

        if sucesso:
            processados += 1
            # Calcula fator para log
            zhat_futuro = get_model_prediction(
                modelo_info['modelo'],
                info['idade_futura'],
                modelo_info['a'],
                modelo_info['b'],
                modelo_info['c'] if pd.notna(modelo_info.get('c')) else None
            )
            zhat_atual = get_model_prediction(
                modelo_info['modelo'],
                info['idade_atual'],
                modelo_info['a'],
                modelo_info['b'],
                modelo_info['c'] if pd.notna(modelo_info.get('c')) else None
            )
            fator = zhat_futuro / zhat_atual
            print(f"    [OK] {tif_file} | {grupo} | i1={info['idade_atual']} -> i2={info['idade_futura']} | fator={fator:.4f}")
        else:
            erros += 1

    # Resumo
    print(f"\n  --- Resumo {folder_name} ---")
    print(f"  Processados com sucesso: {processados}")
    print(f"  Sem modelo disponivel:   {sem_modelo}")
    print(f"  Erros:                   {erros}")
    print(f"  Total:                   {len(tif_files)}")


# =============================================================================
# EXECUCAO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("APLICACAO DOS MODELOS DE PROJECAO AOS RASTERS")
    print("="*60)

    # Carrega modelos ajustados (apenas os melhores)
    print(f"\nCarregando modelos de: {MODELS_FILE}")

    try:
        df_modelos = pd.read_excel(MODELS_FILE, sheet_name='modelo_gompertz')
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelos: {e}")
        exit(1)

    # Filtra apenas modelos marcados como melhor
    df_melhores = df_modelos[df_modelos['melhor'] == 'X'].copy()

    print(f"  Total de modelos: {len(df_modelos)}")
    print(f"  Modelos selecionados (melhor=X): {len(df_melhores)}")

    # Mostra resumo dos modelos
    print("\n  Modelos disponiveis:")
    for _, row in df_melhores.iterrows():
        print(f"    {row['variavel']:<8} | {row['grupo']:<15} | {row['modelo']:<16}")

    # Processa cada pasta de metricas
    for folder_name, var_name in FOLDER_VAR_MAP.items():
        processar_pasta(folder_name, var_name, df_melhores)

    print("\n" + "="*60)
    print("PROCESSAMENTO CONCLUIDO")
    print("="*60)
