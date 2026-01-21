"""
02_ModelFitting.py - Ajuste de Modelo Exponencial para Taxa de Crescimento (Delta)
+ Ajuste de Múltiplos Modelos para variável = f(idade): Gompertz, Logístico, Exp. Modificado
+ Cálculo do parâmetro k via minimização de RMSE (projeção compatível)

Este script ajusta:
Modelo 1 (Delta):
    delta = a * exp(b * v1)

Modelos para variável = f(idade) - seleciona o melhor por AIC:
    - Gompertz:           y = a * exp(-b * exp(-c * x))
    - Logístico:          y = a / (1 + b * exp(-c * x))
    - Exp. Modificado:    y = a * exp(b / x)

E, para cada grupo (regional + regime), calcula o parâmetro k por minimização
do RMSE na projeção compatível:

    Z2_hat(k) = Zhat(i2) * (Z1 / Zhat(i1))^k

O k ótimo é aquele que minimiza o RMSE de Z2_hat em relação a Z2.

Estratificação:
    - Regional: primeiros 2 caracteres do REF_ID
    - Regime: caracteres 11:13 do REF_ID
        - 'R' ou 'R_' -> Talhadia
        - Outros -> Alto fuste

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para salvar sem abrir

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminhos dos arquivos de entrada (dados limpos)
INPUT_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\cleaned"
INPUT_FILE_1 = rf"{INPUT_DIR}\cubmean_comp_cleaned.xlsx"
INPUT_FILE_2 = rf"{INPUT_DIR}\p90_comp_cleaned.xlsx"
INPUT_FILE_3 = rf"{INPUT_DIR}\stddev_comp_cleaned.xlsx"

# Nomes das variáveis
VAR1 = "Z Kurt"
VAR2 = "Z P90"
VAR3 = "Z σ"

# Mapeamento variável -> nome da coluna na sheet long
VAR_COL_MAP = {
    VAR1: "Z Kurt",
    VAR2: "Z P90",
    VAR3: "Z σ"
}

# Sheets com dados
SHEET_DELTA = "pares_delta"
SHEET_LONG = "long"

# Sheet com dados de pares/delta (compatibilidade)
SHEET_NAME = "pares_delta"

# Diretório de saída
OUTPUT_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuração de estilo dos gráficos
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "legend.fontsize": 9
})

# Cores para visualização
COLORS_REGIME = {
    'Talhadia': '#d62728',
    'Alto fuste': '#2ca02c'
}

# =============================================================================
# COLUNAS ESPERADAS NA SHEET pares_delta PARA AJUSTAR k
# =============================================================================
PAIR_I1_COL = "i1"
PAIR_I2_COL = "i2"
PAIR_V1_COL = "v1"
PAIR_V2_COL = "v2"


# =============================================================================
# DEFINIÇÃO DOS MODELOS
# =============================================================================

def exponencial(x, a, b):
    """
    Modelo Exponencial Simples.
    y = a * exp(b * x)
    """
    return a * np.exp(b * x)


def gompertz(x, a, b, c):
    """
    Modelo de Gompertz.
    y = a * exp(-b * exp(-c * x))
    """
    return a * np.exp(-b * np.exp(-c * x))


def logistico(x, a, b, c):
    """
    Modelo Logístico.
    y = a / (1 + b * exp(-c * x))
    """
    return a / (1 + b * np.exp(-c * x))


def exponencial_modificado(x, a, b):
    """
    Modelo Exponencial Modificado.
    y = a * exp(b / x)
    """
    return a * np.exp(b / x)


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def extrair_regional(ref_id):
    """Extrai os primeiros 2 caracteres do REF_ID como regional."""
    if pd.isna(ref_id) or len(str(ref_id)) < 2:
        return 'XX'
    return str(ref_id)[:2].upper()


def extrair_regime(ref_id):
    """
    Extrai o regime de manejo do REF_ID (caracteres 11:13).

    Retorna:
        'Talhadia' se começar com 'R'
        'Alto fuste' caso contrário
    """
    if pd.isna(ref_id) or len(str(ref_id)) < 13:
        return 'Alto fuste'

    codigo = str(ref_id)[11:13].upper()

    # R ou R_ indica Talhadia (reforma)
    if codigo.startswith('R'):
        return 'Talhadia'
    else:
        return 'Alto fuste'


def preparar_dados(df):
    """Prepara o DataFrame adicionando colunas regional e regime."""
    df = df.copy()
    df['regional'] = df['REF_ID'].apply(extrair_regional)
    df['regime'] = df['REF_ID'].apply(extrair_regime)
    df['grupo'] = df['regional'] + '_' + df['regime']
    return df


# =============================================================================
# FUNÇÕES DE AJUSTE - MODELO DELTA
# =============================================================================

def fit_exponential(x, y, grupo_name):
    """
    Ajusta o modelo exponencial aos dados.
    Retorna dict com params, r2, rmse, n, success
    """
    try:
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 5:
            print(f"      [AVISO] {grupo_name}: Dados insuficientes ({len(x_clean)} pontos)")
            return {'success': False, 'n': len(x_clean)}

        # Chutes iniciais
        a0 = np.max(y_clean)
        b0 = -0.1

        bounds = ([0, -np.inf], [np.inf, np.inf])

        params, cov = curve_fit(
            exponencial, x_clean, y_clean,
            p0=[a0, b0],
            bounds=bounds,
            maxfev=10000
        )

        y_pred = exponencial(x_clean, *params)

        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        rmse = np.sqrt(np.mean((y_clean - y_pred) ** 2))

        n = len(y_clean)
        k = len(params)
        aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else np.inf

        return {
            'params': params,
            'a': params[0],
            'b': params[1],
            'r2': r2,
            'rmse': rmse,
            'aic': aic,
            'n': n,
            'success': True
        }
    except Exception as e:
        print(f"      [ERRO] {grupo_name}: {e}")
        return {'success': False, 'n': len(x) if hasattr(x, '__len__') else 0}


def ajustar_por_grupo(df, var_name):
    """Ajusta modelo exponencial para cada combinação de regional + regime."""
    resultados = []
    grupos = df.groupby(['regional', 'regime'])
    print(f"\n    Grupos encontrados: {len(grupos)}")

    for (regional, regime), grupo_df in grupos:
        grupo_name = f"{regional}_{regime}"
        print(f"\n      [{grupo_name}] n={len(grupo_df)}")

        x = grupo_df['v1'].values
        y = grupo_df['delta'].values

        resultado = fit_exponential(x, y, grupo_name)

        if resultado['success']:
            print(f"        R² = {resultado['r2']:.4f}")
            print(f"        RMSE = {resultado['rmse']:.4f}")
            print(f"        Equação: delta = {resultado['a']:.6f} * exp({resultado['b']:.6f} * v1)")

            resultados.append({
                'variavel': var_name,
                'regional': regional,
                'regime': regime,
                'grupo': grupo_name,
                'n': resultado['n'],
                'a': resultado['a'],
                'b': resultado['b'],
                'r2': resultado['r2'],
                'rmse': resultado['rmse'],
                'aic': resultado['aic'],
                'equacao': f"delta = {resultado['a']:.6f} * exp({resultado['b']:.6f} * v1)"
            })
        else:
            resultados.append({
                'variavel': var_name,
                'regional': regional,
                'regime': regime,
                'grupo': grupo_name,
                'n': resultado['n'],
                'a': None,
                'b': None,
                'r2': None,
                'rmse': None,
                'aic': None,
                'equacao': 'FALHOU'
            })

    return pd.DataFrame(resultados)


# =============================================================================
# FUNÇÕES DE AJUSTE - MODELO GOMPERTZ + AJUSTE DO k
# =============================================================================

def fit_gompertz(x, y, grupo_name):
    """
    Ajusta Gompertz: y = a * exp(-b * exp(-c * x))
    Retorna dict com params, r2, rmse, n, success
    """
    try:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 10:
            print(f"      [AVISO] {grupo_name}: Dados insuficientes ({len(x_clean)} pontos)")
            return {'success': False, 'n': len(x_clean)}

        a0 = np.max(y_clean) * 1.1
        b0 = 2.0
        c0 = 0.03  # meses

        bounds = ([0, 0, 0], [np.inf, 20, 1])

        params, cov = curve_fit(
            gompertz, x_clean, y_clean,
            p0=[a0, b0, c0],
            bounds=bounds,
            maxfev=20000
        )

        y_pred = gompertz(x_clean, *params)

        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        rmse = np.sqrt(np.mean((y_clean - y_pred) ** 2))

        n = len(y_clean)
        k = len(params)
        aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else np.inf

        return {
            'params': params,
            'a': params[0],
            'b': params[1],
            'c': params[2],
            'r2': r2,
            'rmse': rmse,
            'aic': aic,
            'n': n,
            'success': True
        }
    except Exception as e:
        print(f"      [ERRO] {grupo_name}: {e}")
        return {'success': False, 'n': len(x) if hasattr(x, '__len__') else 0}


def fit_logistico(x, y, grupo_name):
    """
    Ajusta Logístico: y = a / (1 + b * exp(-c * x))
    Retorna dict com params, r2, rmse, n, success
    """
    try:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 10:
            print(f"      [AVISO] {grupo_name}: Dados insuficientes ({len(x_clean)} pontos)")
            return {'success': False, 'n': len(x_clean)}

        a0 = np.max(y_clean) * 1.1
        b0 = 5.0
        c0 = 0.05

        bounds = ([0, 0, 0], [np.inf, 100, 1])

        params, cov = curve_fit(
            logistico, x_clean, y_clean,
            p0=[a0, b0, c0],
            bounds=bounds,
            maxfev=20000
        )

        y_pred = logistico(x_clean, *params)

        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        rmse = np.sqrt(np.mean((y_clean - y_pred) ** 2))

        n = len(y_clean)
        k = len(params)
        aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else np.inf

        return {
            'modelo': 'Logístico',
            'params': params,
            'a': params[0],
            'b': params[1],
            'c': params[2],
            'r2': r2,
            'rmse': rmse,
            'aic': aic,
            'n': n,
            'success': True
        }
    except Exception as e:
        print(f"      [ERRO] {grupo_name}: {e}")
        return {'success': False, 'n': len(x) if hasattr(x, '__len__') else 0}


def fit_exponencial_modificado(x, y, grupo_name):
    """
    Ajusta Exponencial Modificado: y = a * exp(b / x)
    Retorna dict com params, r2, rmse, n, success
    """
    try:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 10:
            print(f"      [AVISO] {grupo_name}: Dados insuficientes ({len(x_clean)} pontos)")
            return {'success': False, 'n': len(x_clean)}

        # Chutes iniciais - a é a assíntota, b é negativo para crescimento
        a0 = np.max(y_clean) * 1.2
        b0 = -np.mean(x_clean)

        bounds = ([0, -np.inf], [np.inf, 0])

        params, cov = curve_fit(
            exponencial_modificado, x_clean, y_clean,
            p0=[a0, b0],
            bounds=bounds,
            maxfev=20000
        )

        y_pred = exponencial_modificado(x_clean, *params)

        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        rmse = np.sqrt(np.mean((y_clean - y_pred) ** 2))

        n = len(y_clean)
        k = len(params)
        aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else np.inf

        return {
            'modelo': 'Exp. Modificado',
            'params': params,
            'a': params[0],
            'b': params[1],
            'c': None,  # Modelo com 2 parâmetros
            'r2': r2,
            'rmse': rmse,
            'aic': aic,
            'n': n,
            'success': True
        }
    except Exception as e:
        print(f"      [ERRO] {grupo_name}: {e}")
        return {'success': False, 'n': len(x) if hasattr(x, '__len__') else 0}


def fit_k_for_projection_generic(df_pairs_group, model_func, model_params, grupo_name, modelo_nome='Gompertz'):
    """
    Calcula k por minimização de RMSE para projeção compatível:
        Z2_hat(k) = Zhat(i2) * (v1 / Zhat(i1))^k

    O k ótimo é o que minimiza o RMSE entre Z2_hat e Z2 em uma grade.
    """
    required = [PAIR_I1_COL, PAIR_I2_COL, PAIR_V1_COL, PAIR_V2_COL]
    for col in required:
        if col not in df_pairs_group.columns:
            print(f"      [AVISO] {grupo_name}: coluna ausente em pares_delta: '{col}'")
            return {"success": False, "n": 0}

    I1 = df_pairs_group[PAIR_I1_COL].to_numpy(dtype=float)
    I2 = df_pairs_group[PAIR_I2_COL].to_numpy(dtype=float)
    Z1 = df_pairs_group[PAIR_V1_COL].to_numpy(dtype=float)
    Z2 = df_pairs_group[PAIR_V2_COL].to_numpy(dtype=float)

    zhat1 = model_func(I1, *model_params)
    zhat2 = model_func(I2, *model_params)

    mask = np.isfinite(I1) & np.isfinite(I2) & np.isfinite(Z1) & np.isfinite(Z2) \
           & (I2 > I1) & (I1 > 0) & (I2 > 0) \
           & (Z1 > 0) & (Z2 > 0) \
           & np.isfinite(zhat1) & np.isfinite(zhat2) \
           & (zhat1 > 0) & (zhat2 > 0)

    I1 = I1[mask]; I2 = I2[mask]; Z1 = Z1[mask]; Z2 = Z2[mask]
    zhat1 = zhat1[mask]; zhat2 = zhat2[mask]

    n = len(Z2)
    if n < 10:
        print(f"      [AVISO] {grupo_name}: pares insuficientes para ajustar k (n={n})")
        return {"success": False, "n": n}

    # Buscar k ótimo por minimização de RMSE
    r = Z1 / zhat1
    k_grid = np.linspace(0, 2, 401)
    rmse_vals = []

    for k in k_grid:
        Z2_hat = zhat2 * (r ** k)
        rmse_vals.append(np.sqrt(np.mean((Z2_hat - Z2) ** 2)))

    rmse_vals = np.array(rmse_vals, dtype=float)
    if not np.isfinite(rmse_vals).any():
        print(f"      [AVISO] {grupo_name}: RMSE inválido para grade de k")
        return {"success": False, "n": n}

    best_idx = int(np.nanargmin(rmse_vals))
    k_opt = float(k_grid[best_idx])
    k_mean = float('nan')
    k_std = float('nan')

    # Calcular métricas de projeção com k_opt
    Z2_hat = zhat2 * (r ** k_opt)

    rmse = float(rmse_vals[best_idx])
    mae = float(np.mean(np.abs(Z2_hat - Z2)))
    bias = float(np.mean(Z2_hat - Z2))

    ss_res = np.sum((Z2 - Z2_hat) ** 2)
    ss_tot = np.sum((Z2 - np.mean(Z2)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "success": True,
        "n": n,
        "n_k_valid": int(np.isfinite(rmse_vals).sum()),
        "k": k_opt,
        "k_mean": k_mean,
        "k_std": k_std,
        "rmse_proj": rmse,
        "mae_proj": mae,
        "bias_proj": bias,
        "r2_proj": r2
    }


def fit_k_for_projection(df_pairs_group, gompertz_params, grupo_name):
    """
    Calcula k por minimização de RMSE para projeção compatível.
    (Função legada mantida para compatibilidade - usa fit_k_for_projection_generic internamente)
    """
    return fit_k_for_projection_generic(
        df_pairs_group=df_pairs_group,
        model_func=gompertz,
        model_params=gompertz_params,
        grupo_name=grupo_name,
        modelo_nome='Gompertz'
    )


def ajustar_modelos_long_por_grupo(df_long, df_pairs, var_name, var_col):
    """
    Ajusta múltiplos modelos (Gompertz, Logístico, Exp. Modificado) por grupo (regional+regime).
    Salva TODOS os ajustes e marca o melhor com 'X' na coluna 'melhor'.
    Para o melhor modelo de cada grupo, ajusta k usando os pares (i1,i2,v1,v2).
    """
    resultados = []
    grupos = df_long.groupby(['regional', 'regime'])
    print(f"\n    Grupos encontrados: {len(grupos)}")

    for (regional, regime), grupo_df in grupos:
        grupo_name = f"{regional}_{regime}"
        print(f"\n      [{grupo_name}] n={len(grupo_df)}")

        x = grupo_df['IDADE'].values
        y = grupo_df[var_col].values

        # Ajustar todos os modelos
        modelos_ajustados = []

        # 1. Gompertz
        res_gompertz = fit_gompertz(x, y, grupo_name)
        if res_gompertz['success']:
            res_gompertz['modelo'] = 'Gompertz'
            res_gompertz['equacao'] = f"{var_col} = {res_gompertz['a']:.4f} * exp(-{res_gompertz['b']:.4f} * exp(-{res_gompertz['c']:.6f} * idade))"
            res_gompertz['model_func'] = gompertz
            res_gompertz['model_params'] = (res_gompertz['a'], res_gompertz['b'], res_gompertz['c'])
            modelos_ajustados.append(res_gompertz)

        # 2. Logístico
        res_logistico = fit_logistico(x, y, grupo_name)
        if res_logistico['success']:
            res_logistico['equacao'] = f"{var_col} = {res_logistico['a']:.4f} / (1 + {res_logistico['b']:.4f} * exp(-{res_logistico['c']:.6f} * idade))"
            res_logistico['model_func'] = logistico
            res_logistico['model_params'] = (res_logistico['a'], res_logistico['b'], res_logistico['c'])
            modelos_ajustados.append(res_logistico)

        # 3. Exponencial Modificado
        res_expmod = fit_exponencial_modificado(x, y, grupo_name)
        if res_expmod['success']:
            res_expmod['equacao'] = f"{var_col} = {res_expmod['a']:.4f} * exp({res_expmod['b']:.4f} / idade)"
            res_expmod['model_func'] = exponencial_modificado
            res_expmod['model_params'] = (res_expmod['a'], res_expmod['b'])
            modelos_ajustados.append(res_expmod)

        if len(modelos_ajustados) == 0:
            # Nenhum modelo convergiu
            resultados.append({
                'variavel': var_name,
                'regional': regional,
                'regime': regime,
                'grupo': grupo_name,
                'n': len(x),
                'modelo': None,
                'melhor': '',
                'a': None,
                'b': None,
                'c': None,
                'r2': None,
                'rmse': None,
                'aic': None,
                'equacao': 'FALHOU',
                'k_opt': None,
                'k_mean': None,
                'k_std': None,
                'rmse_proj': None,
                'mae_proj': None,
                'bias_proj': None,
                'r2_proj': None,
                'n_pairs_k': None,
                'n_k_valid': None,
            })
            continue

        # Comparar modelos e exibir métricas
        print(f"        Modelos ajustados:")
        for m in modelos_ajustados:
            c_str = f", c={m['c']:.6f}" if m['c'] is not None else ""
            print(f"          {m['modelo']:18s} | R²={m['r2']:.4f} | RMSE={m['rmse']:.4f} | AIC={m['aic']:.2f} | a={m['a']:.4f}, b={m['b']:.4f}{c_str}")

        # Selecionar melhor modelo por AIC
        melhor = min(modelos_ajustados, key=lambda m: m['aic'])
        melhor_aic = melhor['aic']
        print(f"        >> MELHOR: {melhor['modelo']} (AIC={melhor_aic:.2f})")
        print(f"        Equação: {melhor['equacao']}")

        # Ajuste do k com dados pareados do MESMO grupo (apenas para o melhor modelo)
        df_pairs_group = df_pairs[
            (df_pairs['regional'] == regional) &
            (df_pairs['regime'] == regime)
        ].copy()

        k_result = fit_k_for_projection_generic(
            df_pairs_group=df_pairs_group,
            model_func=melhor['model_func'],
            model_params=melhor['model_params'],
            grupo_name=grupo_name,
            modelo_nome=melhor['modelo']
        )

        if k_result["success"]:
            print(f"        k* (RMSE) = {k_result['k']:.4f} | k_std = {k_result['k_std']:.4f} | RMSE_proj = {k_result['rmse_proj']:.4f} | R²_proj = {k_result['r2_proj']:.4f}")
        else:
            print(f"        [AVISO] k não ajustado (n_pairs={k_result.get('n',0)})")

        # Salvar TODOS os modelos ajustados, marcando o melhor com 'X'
        for m in modelos_ajustados:
            is_melhor = (m['aic'] == melhor_aic)
            resultados.append({
                'variavel': var_name,
                'regional': regional,
                'regime': regime,
                'grupo': grupo_name,
                'n': m['n'],
                'modelo': m['modelo'],
                'melhor': 'X' if is_melhor else '',
                'a': m['a'],
                'b': m['b'],
                'c': m['c'],
                'r2': m['r2'],
                'rmse': m['rmse'],
                'aic': m['aic'],
                'equacao': m['equacao'],
                # k só é calculado para o melhor modelo (via equação analítica)
                'k_opt': k_result.get('k', None) if is_melhor else None,
                'k_mean': k_result.get('k_mean', None) if is_melhor else None,
                'k_std': k_result.get('k_std', None) if is_melhor else None,
                'rmse_proj': k_result.get('rmse_proj', None) if is_melhor else None,
                'mae_proj': k_result.get('mae_proj', None) if is_melhor else None,
                'bias_proj': k_result.get('bias_proj', None) if is_melhor else None,
                'r2_proj': k_result.get('r2_proj', None) if is_melhor else None,
                'n_pairs_k': k_result.get('n', None) if is_melhor else None,
                'n_k_valid': k_result.get('n_k_valid', None) if is_melhor else None,
            })

    return pd.DataFrame(resultados)


# Alias para compatibilidade
def ajustar_gompertz_por_grupo(df_long, df_pairs, var_name, var_col):
    """Alias para ajustar_modelos_long_por_grupo (compatibilidade)."""
    return ajustar_modelos_long_por_grupo(df_long, df_pairs, var_name, var_col)


# =============================================================================
# FUNÇÕES DE PLOTAGEM - DELTA
# =============================================================================

def plot_ajustes_por_grupo(df, var_name, df_resultados, output_path):
    """Plota os dados e curvas ajustadas para cada grupo em subplots."""
    grupos_validos = df_resultados[df_resultados['r2'].notna()]['grupo'].unique()
    n_grupos = len(grupos_validos)

    if n_grupos == 0:
        print("    Nenhum grupo com ajuste válido para plotar.")
        return

    n_cols = min(3, n_grupos)
    n_rows = int(np.ceil(n_grupos / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_grupos == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, grupo in enumerate(grupos_validos):
        ax = axes[idx]
        grupo_df = df[df['grupo'] == grupo]
        resultado = df_resultados[df_resultados['grupo'] == grupo].iloc[0]

        x = grupo_df['v1'].values
        y = grupo_df['delta'].values

        regime = resultado['regime']
        cor = COLORS_REGIME.get(regime, '#1f77b4')

        ax.scatter(x, y, alpha=0.4, s=15, c=cor, label='Dados')

        if resultado['r2'] is not None:
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = exponencial(x_smooth, resultado['a'], resultado['b'])
            ax.plot(x_smooth, y_smooth, 'k-', linewidth=2, label=f"R²={resultado['r2']:.3f}")

        ax.set_xlabel('v1 (valor inicial)')
        ax.set_ylabel('delta')
        ax.set_title(f"{grupo}\n(n={resultado['n']})")
        ax.legend(loc='best', fontsize=8)
        ax.grid(linestyle='--', alpha=0.5)

    for idx in range(n_grupos, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Ajuste Exponencial por Grupo - {var_name}\ndelta = a * exp(b * v1)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Gráfico salvo: {output_path}")


def plot_comparativo_regimes(df, var_name, df_resultados, output_path):
    """Plota comparativo entre regimes (Talhadia vs Alto fuste) por regional."""
    regionais = df['regional'].unique()
    n_regionais = len(regionais)

    fig, axes = plt.subplots(1, n_regionais, figsize=(5 * n_regionais, 5))
    if n_regionais == 1:
        axes = [axes]

    for idx, regional in enumerate(regionais):
        ax = axes[idx]
        regional_df = df[df['regional'] == regional]

        for regime in ['Talhadia', 'Alto fuste']:
            regime_df = regional_df[regional_df['regime'] == regime]
            if len(regime_df) == 0:
                continue

            x = regime_df['v1'].values
            y = regime_df['delta'].values
            cor = COLORS_REGIME.get(regime, '#1f77b4')

            ax.scatter(x, y, alpha=0.3, s=10, c=cor, label=f'{regime} (n={len(regime_df)})')

            resultado = df_resultados[
                (df_resultados['regional'] == regional) &
                (df_resultados['regime'] == regime)
            ]

            if len(resultado) > 0 and resultado.iloc[0]['r2'] is not None:
                res = resultado.iloc[0]
                x_smooth = np.linspace(x.min(), x.max(), 100)
                y_smooth = exponencial(x_smooth, res['a'], res['b'])
                ax.plot(x_smooth, y_smooth, c=cor, linewidth=2, linestyle='-')

        ax.set_xlabel('v1 (valor inicial)')
        ax.set_ylabel('delta')
        ax.set_title(f'Regional: {regional}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(linestyle='--', alpha=0.5)

    plt.suptitle(f'Comparativo de Regimes por Regional - {var_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Gráfico comparativo salvo: {output_path}")


def plot_geral(df, var_name, output_path):
    """Plota todos os dados coloridos por regime."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for regime in ['Talhadia', 'Alto fuste']:
        regime_df = df[df['regime'] == regime]
        if len(regime_df) == 0:
            continue

        cor = COLORS_REGIME.get(regime, '#1f77b4')
        ax.scatter(regime_df['v1'], regime_df['delta'],
                   alpha=0.4, s=15, c=cor, label=f'{regime} (n={len(regime_df)})')

    ax.set_xlabel('v1 (valor inicial)')
    ax.set_ylabel('delta (taxa de variação)')
    ax.set_title(f'Dados de Delta - {var_name}\nColorido por Regime de Manejo')
    ax.legend(loc='best')
    ax.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Gráfico geral salvo: {output_path}")


# =============================================================================
# FUNÇÕES DE PLOTAGEM - MODELOS LONG (Gompertz/Logístico/Exp. Modificado)
# =============================================================================

def _get_model_prediction(modelo_nome, x, a, b, c):
    """Retorna a predição do modelo baseado no nome."""
    if modelo_nome == 'Gompertz':
        return gompertz(x, a, b, c)
    elif modelo_nome == 'Logístico':
        return logistico(x, a, b, c)
    elif modelo_nome == 'Exp. Modificado':
        return exponencial_modificado(x, a, b)
    else:
        # Fallback para Gompertz
        return gompertz(x, a, b, c)


def plot_ajustes_gompertz_por_grupo(df, var_name, var_col, df_resultados, output_path):
    """Plota os dados e curvas ajustadas do melhor modelo para cada grupo em subplots."""
    # Filtrar apenas os melhores modelos para plotagem
    df_melhores = df_resultados[df_resultados['melhor'] == 'X'] if 'melhor' in df_resultados.columns else df_resultados
    grupos_validos = df_melhores[df_melhores['r2'].notna()]['grupo'].unique()
    n_grupos = len(grupos_validos)

    if n_grupos == 0:
        print("    Nenhum grupo com ajuste válido para plotar.")
        return

    n_cols = min(3, n_grupos)
    n_rows = int(np.ceil(n_grupos / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_grupos == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, grupo in enumerate(grupos_validos):
        ax = axes[idx]
        grupo_df = df[df['grupo'] == grupo]
        resultado = df_melhores[df_melhores['grupo'] == grupo].iloc[0]

        x = grupo_df['IDADE'].values
        y = grupo_df[var_col].values

        regime = resultado['regime']
        cor = COLORS_REGIME.get(regime, '#1f77b4')

        ax.scatter(x, y, alpha=0.4, s=15, c=cor, label='Dados')

        if resultado['r2'] is not None:
            x_smooth = np.linspace(x.min(), x.max(), 100)
            modelo_nome = resultado.get('modelo', 'Gompertz')
            y_smooth = _get_model_prediction(modelo_nome, x_smooth, resultado['a'], resultado['b'], resultado['c'])
            ax.plot(x_smooth, y_smooth, 'k-', linewidth=2, label=f"{modelo_nome}\nR²={resultado['r2']:.3f}")

        ax.set_xlabel('Idade (meses)')
        ax.set_ylabel(var_col)
        ax.set_title(f"{grupo}\n(n={resultado['n']})")
        ax.legend(loc='best', fontsize=8)
        ax.grid(linestyle='--', alpha=0.5)

    for idx in range(n_grupos, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Melhor Modelo por Grupo - {var_name} = f(Idade)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Gráfico salvo: {output_path}")


def plot_comparativo_gompertz_regimes(df, var_name, var_col, df_resultados, output_path):
    """Plota comparativo entre regimes para o melhor modelo por regional."""
    # Filtrar apenas os melhores modelos para plotagem
    df_melhores = df_resultados[df_resultados['melhor'] == 'X'] if 'melhor' in df_resultados.columns else df_resultados
    regionais = df['regional'].unique()
    n_regionais = len(regionais)

    fig, axes = plt.subplots(1, n_regionais, figsize=(5 * n_regionais, 5))
    if n_regionais == 1:
        axes = [axes]

    for idx, regional in enumerate(regionais):
        ax = axes[idx]
        regional_df = df[df['regional'] == regional]

        for regime in ['Talhadia', 'Alto fuste']:
            regime_df = regional_df[regional_df['regime'] == regime]
            if len(regime_df) == 0:
                continue

            x = regime_df['IDADE'].values
            y = regime_df[var_col].values
            cor = COLORS_REGIME.get(regime, '#1f77b4')

            ax.scatter(x, y, alpha=0.3, s=10, c=cor, label=f'{regime} (n={len(regime_df)})')

            resultado = df_melhores[
                (df_melhores['regional'] == regional) &
                (df_melhores['regime'] == regime)
            ]

            if len(resultado) > 0 and resultado.iloc[0]['r2'] is not None:
                res = resultado.iloc[0]
                x_smooth = np.linspace(x.min(), x.max(), 100)
                modelo_nome = res.get('modelo', 'Gompertz')
                y_smooth = _get_model_prediction(modelo_nome, x_smooth, res['a'], res['b'], res['c'])
                ax.plot(x_smooth, y_smooth, c=cor, linewidth=2, linestyle='-')

        ax.set_xlabel('Idade (meses)')
        ax.set_ylabel(var_col)
        ax.set_title(f'Regional: {regional}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(linestyle='--', alpha=0.5)

    plt.suptitle(f'Comparativo Modelos - Regimes por Regional - {var_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Gráfico comparativo salvo: {output_path}")


def plot_geral_gompertz(df, var_name, var_col, output_path):
    """Plota todos os dados (Idade vs Variável) coloridos por regime."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for regime in ['Talhadia', 'Alto fuste']:
        regime_df = df[df['regime'] == regime]
        if len(regime_df) == 0:
            continue

        cor = COLORS_REGIME.get(regime, '#1f77b4')
        ax.scatter(regime_df['IDADE'], regime_df[var_col],
                   alpha=0.4, s=15, c=cor, label=f'{regime} (n={len(regime_df)})')

    ax.set_xlabel('Idade (meses)')
    ax.set_ylabel(var_col)
    ax.set_title(f'Dados de {var_name} vs Idade\nColorido por Regime de Manejo')
    ax.legend(loc='best')
    ax.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Gráfico geral salvo: {output_path}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def process_variable(input_file, var_name, file_prefix):
    """
    Processa uma variável: carrega, prepara, ajusta e plota.
    Retorna: (df_resultados_delta, df_resultados_gompertz)
    """
    print(f"\n{'='*60}")
    print(f"PROCESSANDO: {var_name}")
    print(f"{'='*60}")

    # =========================================================================
    # PARTE 1: MODELO DELTA = f(v1)
    # =========================================================================
    print(f"\n  --- MODELO DELTA ---")

    df_pairs = pd.read_excel(input_file, sheet_name=SHEET_NAME)
    print(f"  Registros carregados: {len(df_pairs)}")

    df_pairs = preparar_dados(df_pairs)

    print(f"\n  Distribuição por Regional:")
    for reg, count in df_pairs['regional'].value_counts().items():
        print(f"    {reg}: {count}")

    print(f"\n  Distribuição por Regime:")
    for reg, count in df_pairs['regime'].value_counts().items():
        print(f"    {reg}: {count}")

    print(f"\n  Gerando gráfico geral...")
    plot_geral(df_pairs, var_name, rf"{OUTPUT_DIR}\{file_prefix}_delta_geral.png")

    print(f"\n  Ajustando modelos por grupo (regional + regime)...")
    df_resultados_delta = ajustar_por_grupo(df_pairs, var_name)

    print(f"\n  Gerando gráficos por grupo...")
    plot_ajustes_por_grupo(df_pairs, var_name, df_resultados_delta,
                           rf"{OUTPUT_DIR}\{file_prefix}_ajustes_grupos.png")

    print(f"\n  Gerando gráfico comparativo de regimes...")
    plot_comparativo_regimes(df_pairs, var_name, df_resultados_delta,
                             rf"{OUTPUT_DIR}\{file_prefix}_comparativo_regimes.png")

    # =========================================================================
    # PARTE 2: MODELOS LONG - Variável = f(Idade) + Ajuste k por pares
    # =========================================================================
    print(f"\n  --- MODELOS LONG (Gompertz/Logístico/Exp.Mod.) + k por pares ---")

    df_long = pd.read_excel(input_file, sheet_name=SHEET_LONG)
    print(f"  Registros carregados (long): {len(df_long)}")

    df_long = preparar_dados(df_long)

    var_col = VAR_COL_MAP.get(var_name, var_name)

    print(f"\n  Distribuição por Regional (long):")
    for reg, count in df_long['regional'].value_counts().items():
        print(f"    {reg}: {count}")

    print(f"\n  Distribuição por Regime (long):")
    for reg, count in df_long['regime'].value_counts().items():
        print(f"    {reg}: {count}")

    print(f"\n  Gerando gráfico geral...")
    plot_geral_gompertz(df_long, var_name, var_col, rf"{OUTPUT_DIR}\{file_prefix}_long_geral.png")

    print(f"\n  Ajustando múltiplos modelos por grupo (regional + regime) + k por pares...")
    df_resultados_gompertz = ajustar_gompertz_por_grupo(df_long, df_pairs, var_name, var_col)

    print(f"\n  Gerando gráficos por grupo...")
    plot_ajustes_gompertz_por_grupo(df_long, var_name, var_col, df_resultados_gompertz,
                                    rf"{OUTPUT_DIR}\{file_prefix}_long_ajustes_grupos.png")

    print(f"\n  Gerando gráfico comparativo de regimes...")
    plot_comparativo_gompertz_regimes(df_long, var_name, var_col, df_resultados_gompertz,
                                      rf"{OUTPUT_DIR}\{file_prefix}_long_comparativo_regimes.png")

    return df_resultados_delta, df_resultados_gompertz


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("AJUSTE DE MODELOS")
    print("Modelo 1: delta = a * exp(b * v1)")
    print("Modelos var=f(idade): Gompertz, Logístico, Exp. Modificado")
    print("Seleção automática do melhor modelo por AIC")
    print("Extra: Cálculo de k via equação analítica (mediana dos pares)")
    print("Estratificado por: Regional + Regime")
    print("="*60)

    all_results_delta = []
    all_results_gompertz = []

    print("\n" + "-"*60)
    print(f"Processando {VAR1}...")
    results_delta_var1, results_gompertz_var1 = process_variable(INPUT_FILE_1, VAR1, "zkurt")
    all_results_delta.append(results_delta_var1)
    all_results_gompertz.append(results_gompertz_var1)

    print("\n" + "-"*60)
    print(f"Processando {VAR2}...")
    results_delta_var2, results_gompertz_var2 = process_variable(INPUT_FILE_2, VAR2, "zp90")
    all_results_delta.append(results_delta_var2)
    all_results_gompertz.append(results_gompertz_var2)

    print("\n" + "-"*60)
    print(f"Processando {VAR3}...")
    results_delta_var3, results_gompertz_var3 = process_variable(INPUT_FILE_3, VAR3, "zstddev")
    all_results_delta.append(results_delta_var3)
    all_results_gompertz.append(results_gompertz_var3)

    df_all_delta = pd.concat(all_results_delta, ignore_index=True)
    df_all_gompertz = pd.concat(all_results_gompertz, ignore_index=True)

    # =========================================================================
    # RESUMO MODELO DELTA
    # =========================================================================
    print("\n" + "="*60)
    print("RESUMO GERAL - MODELO DELTA")
    print("="*60)

    df_delta_success = df_all_delta[df_all_delta['r2'].notna()].copy()
    print(f"\nTotal de ajustes bem sucedidos: {len(df_delta_success)} de {len(df_all_delta)}")

    print(f"\n{'Variável':<10} {'Grupo':<20} {'n':>6} {'R²':>8} {'RMSE':>10} {'a':>12} {'b':>12}")
    print("-" * 80)

    for _, row in df_delta_success.iterrows():
        print(f"{row['variavel']:<10} {row['grupo']:<20} {row['n']:>6} "
              f"{row['r2']:>8.4f} {row['rmse']:>10.4f} {row['a']:>12.6f} {row['b']:>12.6f}")

    # =========================================================================
    # RESUMO MODELOS LONG (+k)
    # =========================================================================
    print("\n" + "="*60)
    print("RESUMO GERAL - TODOS OS MODELOS (X = melhor por AIC)")
    print("="*60)

    df_gompertz_success = df_all_gompertz[df_all_gompertz['r2'].notna()].copy()
    df_melhores = df_gompertz_success[df_gompertz_success['melhor'] == 'X'] if 'melhor' in df_gompertz_success.columns else df_gompertz_success
    print(f"\nTotal de ajustes: {len(df_gompertz_success)} ({len(df_melhores)} melhores)")

    print(f"\n{'Variável':<8} {'Grupo':<18} {'Modelo':<16} {'X':>1} {'n':>5} {'R²':>7} {'RMSE':>8} {'AIC':>10} {'k*':>7}")
    print("-" * 100)

    for _, row in df_gompertz_success.iterrows():
        kopt = row.get('k_opt', None)
        modelo = row.get('modelo', 'Gompertz')
        melhor = row.get('melhor', '')
        kopt_s = f"{kopt:.4f}" if pd.notna(kopt) else ""
        aic_s = f"{row['aic']:.2f}" if pd.notna(row['aic']) else "NA"

        print(f"{row['variavel']:<8} {row['grupo']:<18} {modelo:<16} {melhor:>1} {row['n']:>5} "
              f"{row['r2']:>7.4f} {row['rmse']:>8.4f} {aic_s:>10} {kopt_s:>7}")

    # =========================================================================
    # EXPORTAR RESULTADOS
    # =========================================================================
    output_excel = rf"{OUTPUT_DIR}\modelos_exponenciais_resumo.xlsx"

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_all_delta.to_excel(writer, sheet_name='modelo_delta', index=False)
        df_all_gompertz.to_excel(writer, sheet_name='modelo_gompertz', index=False)

    print(f"\n Resultados exportados para: {output_excel}")
    print(f"    - Sheet 'modelo_delta': delta = a * exp(b * v1)")
    print(f"    - Sheet 'modelo_gompertz': TODOS os modelos por grupo (X = melhor por AIC)")

    # =========================================================================
    # ESTATÍSTICAS POR VARIÁVEL
    # =========================================================================
    print("\n" + "-"*60)
    print("ESTATÍSTICAS POR VARIÁVEL - MODELO DELTA")
    print("-"*60)

    for var in [VAR1, VAR2, VAR3]:
        var_results = df_delta_success[df_delta_success['variavel'] == var]
        if len(var_results) > 0:
            print(f"\n  {var}:")
            print(f"    Grupos ajustados: {len(var_results)}")
            print(f"    R² médio: {var_results['r2'].mean():.4f}")
            print(f"    R² range: [{var_results['r2'].min():.4f}, {var_results['r2'].max():.4f}]")
            print(f"    Melhor grupo: {var_results.loc[var_results['r2'].idxmax(), 'grupo']} "
                  f"(R²={var_results['r2'].max():.4f})")

    print("\n" + "-"*60)
    print("ESTATÍSTICAS POR VARIÁVEL - MELHORES MODELOS (+k)")
    print("-"*60)

    for var in [VAR1, VAR2, VAR3]:
        # Usar apenas os melhores modelos para estatísticas
        var_results = df_melhores[df_melhores['variavel'] == var]
        if len(var_results) > 0:
            print(f"\n  {var}:")
            print(f"    Grupos ajustados: {len(var_results)}")
            print(f"    R² médio (curva): {var_results['r2'].mean():.4f}")
            print(f"    RMSE médio (curva): {var_results['rmse'].mean():.4f}")

            # Contagem de modelos selecionados como melhores
            if 'modelo' in var_results.columns:
                model_counts = var_results['modelo'].value_counts()
                print(f"    Melhores por tipo: {dict(model_counts)}")

            if 'k_opt' in var_results.columns:
                k_valid = var_results['k_opt'].dropna()
                if len(k_valid) > 0:
                    print(f"    k* médio: {k_valid.mean():.4f}")
                    print(f"    k* range: [{k_valid.min():.4f}, {k_valid.max():.4f}]")

            print(f"    Melhor grupo (curva): {var_results.loc[var_results['r2'].idxmax(), 'grupo']} "
                  f"(R²={var_results['r2'].max():.4f})")

    print("\n" + "="*60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("="*60)
