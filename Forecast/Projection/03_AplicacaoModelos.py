"""
03_AplicacaoModelos.py - Aplicação dos Modelos Ajustados aos Dados de Pares

Este script aplica os modelos ajustados no script 02_ModelFitting.py aos dados de pares_delta:

1. Modelo Delta:
    delta_est = a * exp(b * v1)
    v2_est_delta = v1 + delta_est

2. Modelo de Idade (Gompertz/Logístico/Exp.Mod.) com k:
    v2_est_idade = Zhat(i2) * (v1 / Zhat(i1))^k

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

# Arquivo com modelos ajustados
MODELS_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\models"
MODELS_FILE = rf"{MODELS_DIR}\modelos_exponenciais_resumo.xlsx"

# Nomes das variáveis
VAR1 = "Z Kurt"
VAR2 = "Z P90"
VAR3 = "Z σ"

# Mapeamento variável -> arquivo de entrada
VAR_FILE_MAP = {
    VAR1: INPUT_FILE_1,
    VAR2: INPUT_FILE_2,
    VAR3: INPUT_FILE_3,
}

# Mapeamento variável -> nome da coluna na sheet long
VAR_COL_MAP = {
    VAR1: "Z Kurt",
    VAR2: "Z P90",
    VAR3: "Z σ"
}

# Sheets com dados
SHEET_DELTA = "pares_delta"

# Colunas esperadas na sheet pares_delta
PAIR_I1_COL = "i1"
PAIR_I2_COL = "i2"
PAIR_V1_COL = "v1"
PAIR_V2_COL = "v2"

# Diretório de saída
OUTPUT_DIR = r"G:\PycharmProjects\Mestrado\Forecast\Projection\Exports\aplicacao"
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
# DEFINIÇÃO DOS MODELOS
# =============================================================================

def exponencial(x, a, b):
    """Modelo Exponencial: y = a * exp(b * x)"""
    return a * np.exp(b * x)


def gompertz(x, a, b, c):
    """Modelo de Gompertz: y = a * exp(-b * exp(-c * x))"""
    return a * np.exp(-b * np.exp(-c * x))


def logistico(x, a, b, c):
    """Modelo Logístico: y = a / (1 + b * exp(-c * x))"""
    return a / (1 + b * np.exp(-c * x))


def exponencial_modificado(x, a, b):
    """Modelo Exponencial Modificado: y = a * exp(b / x)"""
    return a * np.exp(b / x)


def get_model_prediction(modelo_nome, x, a, b, c=None):
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
    Retorna 'Talhadia' se começar com 'R', 'Alto fuste' caso contrário.
    """
    if pd.isna(ref_id) or len(str(ref_id)) < 13:
        return 'Alto fuste'

    codigo = str(ref_id)[11:13].upper()

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


def calcular_metricas(y_obs, y_pred):
    """Calcula métricas de avaliação."""
    mask = np.isfinite(y_obs) & np.isfinite(y_pred)
    y_obs = y_obs[mask]
    y_pred = y_pred[mask]

    if len(y_obs) == 0:
        return {'n': 0, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'r2': np.nan, 'mape': np.nan}

    rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
    mae = np.mean(np.abs(y_obs - y_pred))
    bias = np.mean(y_pred - y_obs)

    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # MAPE (evitando divisão por zero)
    mask_nonzero = y_obs != 0
    if np.sum(mask_nonzero) > 0:
        mape = np.mean(np.abs((y_obs[mask_nonzero] - y_pred[mask_nonzero]) / y_obs[mask_nonzero])) * 100
    else:
        mape = np.nan

    return {
        'n': len(y_obs),
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'r2': r2,
        'mape': mape
    }


# =============================================================================
# FUNÇÕES DE APLICAÇÃO DOS MODELOS
# =============================================================================

# Filtro de consistência
MIN_IDADE_I1 = 24  # Idade mínima i1 em meses


def aplicar_modelo_delta(df, modelo_delta):
    """
    Aplica o modelo delta aos dados.
    delta_est = a * exp(b * v1)
    v2_est_delta = (delta_est * di * v1) + v1

    Onde di = i2 - i1 (diferença de idade em meses)

    Filtro: i1 >= 24 meses
    """
    df = df.copy()
    df['delta_est'] = np.nan
    df['di'] = np.nan
    df['v2_est_delta'] = np.nan

    for idx, row in df.iterrows():
        grupo = row['grupo']
        v1 = row[PAIR_V1_COL]
        i1 = row[PAIR_I1_COL]
        i2 = row[PAIR_I2_COL]

        # Filtro de consistência: i1 >= 24 meses
        if pd.isna(i1) or i1 < MIN_IDADE_I1:
            continue

        # Buscar parâmetros do modelo para este grupo
        modelo = modelo_delta[modelo_delta['grupo'] == grupo]

        if len(modelo) == 0 or pd.isna(modelo.iloc[0]['a']):
            continue

        a = modelo.iloc[0]['a']
        b = modelo.iloc[0]['b']

        # Calcular diferença de idade
        di = i2 - i1

        # Aplicar modelo
        delta_est = exponencial(v1, a, b)
        v2_est = (delta_est * di * v1) + v1

        df.at[idx, 'delta_est'] = delta_est
        df.at[idx, 'di'] = di
        df.at[idx, 'v2_est_delta'] = v2_est

    return df


def aplicar_modelo_idade(df, modelo_idade):
    """
    Aplica o modelo de idade aos dados com três métodos:

    1. v2_est_razao = (Zhat(i2) / Zhat(i1)) * v1
       Projeção simples pela razão das curvas

    2. v2_est_idade = Zhat(i2) * (v1 / Zhat(i1))^k
       Projeção com parâmetro k ajustado

    Filtro: i1 >= 24 meses
    """
    df = df.copy()
    df['zhat_i1'] = np.nan
    df['zhat_i2'] = np.nan
    df['v2_est_razao'] = np.nan
    df['v2_est_idade'] = np.nan
    df['k_usado'] = np.nan
    df['modelo_usado'] = ''

    for idx, row in df.iterrows():
        grupo = row['grupo']
        i1 = row[PAIR_I1_COL]
        i2 = row[PAIR_I2_COL]
        v1 = row[PAIR_V1_COL]

        # Filtro de consistência: i1 >= 24 meses
        if pd.isna(i1) or i1 < MIN_IDADE_I1:
            continue

        # Buscar o melhor modelo para este grupo
        modelo = modelo_idade[(modelo_idade['grupo'] == grupo) & (modelo_idade['melhor'] == 'X')]

        if len(modelo) == 0 or pd.isna(modelo.iloc[0]['a']):
            continue

        m = modelo.iloc[0]
        modelo_nome = m['modelo']
        a = m['a']
        b = m['b']
        c = m['c'] if pd.notna(m['c']) else None
        k = m['k_opt'] if pd.notna(m['k_opt']) else 1.0  # Default k=1 se não ajustado

        # Calcular Zhat(i1) e Zhat(i2)
        zhat_i1 = get_model_prediction(modelo_nome, i1, a, b, c)
        zhat_i2 = get_model_prediction(modelo_nome, i2, a, b, c)

        # Evitar divisão por zero
        if zhat_i1 <= 0 or not np.isfinite(zhat_i1):
            continue

        # Método 1: Projeção pela razão simples
        v2_est_razao = (zhat_i2 / zhat_i1) * v1

        # Método 2: Projeção com k
        ratio = v1 / zhat_i1
        v2_est_k = zhat_i2 * (ratio ** k)

        df.at[idx, 'zhat_i1'] = zhat_i1
        df.at[idx, 'zhat_i2'] = zhat_i2
        df.at[idx, 'v2_est_razao'] = v2_est_razao
        df.at[idx, 'v2_est_idade'] = v2_est_k
        df.at[idx, 'k_usado'] = k
        df.at[idx, 'modelo_usado'] = modelo_nome

    return df


# =============================================================================
# FUNÇÕES DE PLOTAGEM
# =============================================================================

def plot_observado_vs_estimado(df, var_name, col_obs, col_est, titulo, output_path):
    """Plota gráfico de observado vs estimado."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for regime in ['Talhadia', 'Alto fuste']:
        regime_df = df[df['regime'] == regime]
        if len(regime_df) == 0:
            continue

        mask = regime_df[col_est].notna()
        x = regime_df.loc[mask, col_obs]
        y = regime_df.loc[mask, col_est]

        cor = COLORS_REGIME.get(regime, '#1f77b4')
        ax.scatter(x, y, alpha=0.4, s=15, c=cor, label=f'{regime} (n={len(x)})')

    # Linha 1:1
    all_vals = pd.concat([df[col_obs], df[col_est]]).dropna()
    if len(all_vals) > 0:
        lims = [all_vals.min(), all_vals.max()]
        ax.plot(lims, lims, 'k--', linewidth=1, label='1:1')

    # Calcular métricas gerais
    mask = df[col_est].notna()
    metricas = calcular_metricas(df.loc[mask, col_obs].values, df.loc[mask, col_est].values)

    ax.set_xlabel(f'{col_obs} (Observado)')
    ax.set_ylabel(f'{col_est} (Estimado)')
    ax.set_title(f'{titulo}\n{var_name}\nR²={metricas["r2"]:.4f} | RMSE={metricas["rmse"]:.4f} | Bias={metricas["bias"]:.4f}')
    ax.legend(loc='best')
    ax.grid(linestyle='--', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Gráfico salvo: {output_path}")
    return metricas


def plot_residuos(df, var_name, col_obs, col_est, titulo, output_path):
    """Plota gráfico de resíduos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    mask = df[col_est].notna()
    df_plot = df[mask].copy()
    df_plot['residuo'] = df_plot[col_est] - df_plot[col_obs]
    df_plot['residuo_perc'] = (df_plot['residuo'] / df_plot[col_obs]) * 100

    # Resíduo vs Estimado
    ax = axes[0]
    for regime in ['Talhadia', 'Alto fuste']:
        regime_df = df_plot[df_plot['regime'] == regime]
        if len(regime_df) == 0:
            continue
        cor = COLORS_REGIME.get(regime, '#1f77b4')
        ax.scatter(regime_df[col_est], regime_df['residuo'], alpha=0.4, s=15, c=cor, label=regime)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel(f'{col_est} (Estimado)')
    ax.set_ylabel('Resíduo (Est - Obs)')
    ax.set_title(f'Resíduos vs Estimado')
    ax.legend(loc='best')
    ax.grid(linestyle='--', alpha=0.5)

    # Histograma de resíduos percentuais
    ax = axes[1]
    ax.hist(df_plot['residuo_perc'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Resíduo (%)')
    ax.set_ylabel('Frequência')
    ax.set_title(f'Distribuição dos Resíduos (%)\nMédia={df_plot["residuo_perc"].mean():.2f}%')
    ax.grid(linestyle='--', alpha=0.5)

    plt.suptitle(f'{titulo} - {var_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Gráfico de resíduos salvo: {output_path}")


# =============================================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# =============================================================================

def processar_variavel(var_name, input_file, modelo_delta, modelo_idade, file_prefix):
    """Processa uma variável: carrega dados, aplica modelos e gera resultados."""
    print(f"\n{'='*60}")
    print(f"PROCESSANDO: {var_name}")
    print(f"{'='*60}")

    # Carregar dados
    df = pd.read_excel(input_file, sheet_name=SHEET_DELTA)
    print(f"  Registros carregados: {len(df)}")

    # Preparar dados
    df = preparar_dados(df)

    # Filtrar modelos para esta variável
    modelo_delta_var = modelo_delta[modelo_delta['variavel'] == var_name].copy()
    modelo_idade_var = modelo_idade[modelo_idade['variavel'] == var_name].copy()

    print(f"  Modelos delta encontrados: {len(modelo_delta_var)} grupos")
    print(f"  Modelos idade encontrados: {len(modelo_idade_var[modelo_idade_var['melhor'] == 'X'])} grupos (melhores)")

    # Filtro de consistência
    n_filtrados = (df[PAIR_I1_COL] < MIN_IDADE_I1).sum()
    print(f"\n  Filtro de consistência: i1 >= {MIN_IDADE_I1} meses")
    print(f"  Registros filtrados (i1 < {MIN_IDADE_I1}): {n_filtrados}")
    print(f"  Registros válidos: {len(df) - n_filtrados}")

    # ==========================================================================
    # APLICAR MODELO DELTA
    # ==========================================================================
    print(f"\n  --- Aplicando Modelo Delta ---")
    df = aplicar_modelo_delta(df, modelo_delta_var)

    n_aplicados_delta = df['delta_est'].notna().sum()
    print(f"    Estimativas geradas: {n_aplicados_delta} de {len(df) - n_filtrados} válidos")

    # Métricas do modelo delta
    mask_delta = df['delta_est'].notna()
    if mask_delta.sum() > 0:
        # Métricas para delta
        metricas_delta = calcular_metricas(df.loc[mask_delta, 'delta'].values,
                                           df.loc[mask_delta, 'delta_est'].values)
        print(f"    Métricas (delta): R²={metricas_delta['r2']:.4f} | RMSE={metricas_delta['rmse']:.4f}")

        # Métricas para v2
        metricas_v2_delta = calcular_metricas(df.loc[mask_delta, PAIR_V2_COL].values,
                                              df.loc[mask_delta, 'v2_est_delta'].values)
        print(f"    Métricas (v2): R²={metricas_v2_delta['r2']:.4f} | RMSE={metricas_v2_delta['rmse']:.4f}")

    # ==========================================================================
    # APLICAR MODELO IDADE (Razão e com k)
    # ==========================================================================
    print(f"\n  --- Aplicando Modelo Idade ---")
    df = aplicar_modelo_idade(df, modelo_idade_var)

    n_aplicados_idade = df['v2_est_idade'].notna().sum()
    print(f"    Estimativas geradas: {n_aplicados_idade} de {len(df)}")

    # Métricas do modelo idade - Razão simples
    mask_idade = df['v2_est_razao'].notna()
    if mask_idade.sum() > 0:
        metricas_v2_razao = calcular_metricas(df.loc[mask_idade, PAIR_V2_COL].values,
                                              df.loc[mask_idade, 'v2_est_razao'].values)
        print(f"    Métricas Razão (v2): R²={metricas_v2_razao['r2']:.4f} | RMSE={metricas_v2_razao['rmse']:.4f}")

    # Métricas do modelo idade - com k
    mask_idade = df['v2_est_idade'].notna()
    if mask_idade.sum() > 0:
        metricas_v2_idade = calcular_metricas(df.loc[mask_idade, PAIR_V2_COL].values,
                                              df.loc[mask_idade, 'v2_est_idade'].values)
        print(f"    Métricas com k (v2): R²={metricas_v2_idade['r2']:.4f} | RMSE={metricas_v2_idade['rmse']:.4f}")

        # Contagem por modelo usado
        model_counts = df.loc[mask_idade, 'modelo_usado'].value_counts()
        print(f"    Modelos usados: {dict(model_counts)}")

    # ==========================================================================
    # GERAR GRÁFICOS
    # ==========================================================================
    print(f"\n  --- Gerando Gráficos ---")

    # Gráfico delta: observado vs estimado
    if mask_delta.sum() > 0:
        plot_observado_vs_estimado(
            df, var_name, 'delta', 'delta_est',
            'Modelo Delta: Observado vs Estimado',
            rf"{OUTPUT_DIR}\{file_prefix}_delta_obs_vs_est.png"
        )
        plot_residuos(
            df, var_name, 'delta', 'delta_est',
            'Modelo Delta',
            rf"{OUTPUT_DIR}\{file_prefix}_delta_residuos.png"
        )

    # Gráfico v2 pelo modelo delta
    if mask_delta.sum() > 0:
        plot_observado_vs_estimado(
            df, var_name, PAIR_V2_COL, 'v2_est_delta',
            'Modelo Delta: v2 Observado vs Estimado',
            rf"{OUTPUT_DIR}\{file_prefix}_v2_delta_obs_vs_est.png"
        )

    # Gráfico v2 pelo modelo de idade - Razão
    mask_razao = df['v2_est_razao'].notna()
    if mask_razao.sum() > 0:
        plot_observado_vs_estimado(
            df, var_name, PAIR_V2_COL, 'v2_est_razao',
            'Modelo Idade (Razão): v2 Observado vs Estimado',
            rf"{OUTPUT_DIR}\{file_prefix}_v2_razao_obs_vs_est.png"
        )

    # Gráfico v2 pelo modelo de idade - com k
    if mask_idade.sum() > 0:
        plot_observado_vs_estimado(
            df, var_name, PAIR_V2_COL, 'v2_est_idade',
            'Modelo Idade (com k): v2 Observado vs Estimado',
            rf"{OUTPUT_DIR}\{file_prefix}_v2_idade_obs_vs_est.png"
        )
        plot_residuos(
            df, var_name, PAIR_V2_COL, 'v2_est_idade',
            'Modelo Idade (com k)',
            rf"{OUTPUT_DIR}\{file_prefix}_v2_idade_residuos.png"
        )

    # Comparativo entre os três modelos
    if mask_delta.sum() > 0 and mask_idade.sum() > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Modelo Delta
        ax = axes[0]
        mask = df['v2_est_delta'].notna()
        for regime in ['Talhadia', 'Alto fuste']:
            regime_df = df[(df['regime'] == regime) & mask]
            if len(regime_df) == 0:
                continue
            cor = COLORS_REGIME.get(regime, '#1f77b4')
            ax.scatter(regime_df[PAIR_V2_COL], regime_df['v2_est_delta'],
                      alpha=0.4, s=15, c=cor, label=regime)

        metricas = calcular_metricas(df.loc[mask, PAIR_V2_COL].values,
                                     df.loc[mask, 'v2_est_delta'].values)
        lims = [df.loc[mask, PAIR_V2_COL].min(), df.loc[mask, PAIR_V2_COL].max()]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlabel('v2 Observado')
        ax.set_ylabel('v2 Estimado')
        ax.set_title(f'Modelo Delta\nR²={metricas["r2"]:.4f} | RMSE={metricas["rmse"]:.4f}')
        ax.legend(loc='best')
        ax.grid(linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')

        # Modelo Idade - Razão
        ax = axes[1]
        mask = df['v2_est_razao'].notna()
        for regime in ['Talhadia', 'Alto fuste']:
            regime_df = df[(df['regime'] == regime) & mask]
            if len(regime_df) == 0:
                continue
            cor = COLORS_REGIME.get(regime, '#1f77b4')
            ax.scatter(regime_df[PAIR_V2_COL], regime_df['v2_est_razao'],
                      alpha=0.4, s=15, c=cor, label=regime)

        metricas = calcular_metricas(df.loc[mask, PAIR_V2_COL].values,
                                     df.loc[mask, 'v2_est_razao'].values)
        lims = [df.loc[mask, PAIR_V2_COL].min(), df.loc[mask, PAIR_V2_COL].max()]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlabel('v2 Observado')
        ax.set_ylabel('v2 Estimado')
        ax.set_title(f'Modelo Idade (Razão)\nR²={metricas["r2"]:.4f} | RMSE={metricas["rmse"]:.4f}')
        ax.legend(loc='best')
        ax.grid(linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')

        # Modelo Idade - com k
        ax = axes[2]
        mask = df['v2_est_idade'].notna()
        for regime in ['Talhadia', 'Alto fuste']:
            regime_df = df[(df['regime'] == regime) & mask]
            if len(regime_df) == 0:
                continue
            cor = COLORS_REGIME.get(regime, '#1f77b4')
            ax.scatter(regime_df[PAIR_V2_COL], regime_df['v2_est_idade'],
                      alpha=0.4, s=15, c=cor, label=regime)

        metricas = calcular_metricas(df.loc[mask, PAIR_V2_COL].values,
                                     df.loc[mask, 'v2_est_idade'].values)
        lims = [df.loc[mask, PAIR_V2_COL].min(), df.loc[mask, PAIR_V2_COL].max()]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlabel('v2 Observado')
        ax.set_ylabel('v2 Estimado')
        ax.set_title(f'Modelo Idade (com k)\nR²={metricas["r2"]:.4f} | RMSE={metricas["rmse"]:.4f}')
        ax.legend(loc='best')
        ax.grid(linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')

        plt.suptitle(f'Comparativo de Modelos - {var_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(rf"{OUTPUT_DIR}\{file_prefix}_comparativo_modelos.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Gráfico comparativo salvo: {file_prefix}_comparativo_modelos.png")

    return df


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("APLICAÇÃO DOS MODELOS AJUSTADOS")
    print("="*60)
    print(f"\nCarregando modelos de: {MODELS_FILE}")

    # Carregar modelos ajustados
    modelo_delta = pd.read_excel(MODELS_FILE, sheet_name='modelo_delta')
    modelo_idade = pd.read_excel(MODELS_FILE, sheet_name='modelo_gompertz')

    print(f"  Modelos delta carregados: {len(modelo_delta)}")
    print(f"  Modelos idade carregados: {len(modelo_idade)}")
    print(f"  Modelos idade (melhores): {len(modelo_idade[modelo_idade['melhor'] == 'X'])}")

    # Processar cada variável
    all_results = []
    metricas_resumo = []

    # VAR1 - Z Kurt
    df_var1 = processar_variavel(VAR1, INPUT_FILE_1, modelo_delta, modelo_idade, "zkurt")
    df_var1['variavel'] = VAR1
    all_results.append(df_var1)

    # VAR2 - Z P90
    df_var2 = processar_variavel(VAR2, INPUT_FILE_2, modelo_delta, modelo_idade, "zp90")
    df_var2['variavel'] = VAR2
    all_results.append(df_var2)

    # VAR3 - Z σ
    df_var3 = processar_variavel(VAR3, INPUT_FILE_3, modelo_delta, modelo_idade, "zstddev")
    df_var3['variavel'] = VAR3
    all_results.append(df_var3)

    # ==========================================================================
    # CONSOLIDAR E EXPORTAR RESULTADOS
    # ==========================================================================
    print("\n" + "="*60)
    print("EXPORTANDO RESULTADOS")
    print("="*60)

    # Exportar resultados por variável
    output_excel = rf"{OUTPUT_DIR}\aplicacao_modelos_resultado.xlsx"

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_var1.to_excel(writer, sheet_name='Z_Kurt', index=False)
        df_var2.to_excel(writer, sheet_name='Z_P90', index=False)
        df_var3.to_excel(writer, sheet_name='Z_Sigma', index=False)

        # ======================================================================
        # MÉTRICAS GERAIS (consolidadas)
        # ======================================================================
        metricas_list = []
        for var_name, df_var in [(VAR1, df_var1), (VAR2, df_var2), (VAR3, df_var3)]:
            # Métricas modelo delta (para delta)
            mask = df_var['delta_est'].notna()
            if mask.sum() > 0:
                m = calcular_metricas(df_var.loc[mask, 'delta'].values,
                                      df_var.loc[mask, 'delta_est'].values)
                metricas_list.append({
                    'variavel': var_name,
                    'estrato': 'GERAL',
                    'modelo': 'Delta',
                    'alvo': 'delta',
                    'n': m['n'],
                    'r2': m['r2'],
                    'rmse': m['rmse'],
                    'mae': m['mae'],
                    'bias': m['bias'],
                    'mape': m['mape']
                })

            # Métricas modelo delta (para v2)
            mask = df_var['v2_est_delta'].notna()
            if mask.sum() > 0:
                m = calcular_metricas(df_var.loc[mask, PAIR_V2_COL].values,
                                      df_var.loc[mask, 'v2_est_delta'].values)
                metricas_list.append({
                    'variavel': var_name,
                    'estrato': 'GERAL',
                    'modelo': 'Delta',
                    'alvo': 'v2',
                    'n': m['n'],
                    'r2': m['r2'],
                    'rmse': m['rmse'],
                    'mae': m['mae'],
                    'bias': m['bias'],
                    'mape': m['mape']
                })

            # Métricas modelo idade - Razão (para v2)
            mask = df_var['v2_est_razao'].notna()
            if mask.sum() > 0:
                m = calcular_metricas(df_var.loc[mask, PAIR_V2_COL].values,
                                      df_var.loc[mask, 'v2_est_razao'].values)
                metricas_list.append({
                    'variavel': var_name,
                    'estrato': 'GERAL',
                    'modelo': 'Idade (Razão)',
                    'alvo': 'v2',
                    'n': m['n'],
                    'r2': m['r2'],
                    'rmse': m['rmse'],
                    'mae': m['mae'],
                    'bias': m['bias'],
                    'mape': m['mape']
                })

            # Métricas modelo idade - com k (para v2)
            mask = df_var['v2_est_idade'].notna()
            if mask.sum() > 0:
                m = calcular_metricas(df_var.loc[mask, PAIR_V2_COL].values,
                                      df_var.loc[mask, 'v2_est_idade'].values)
                metricas_list.append({
                    'variavel': var_name,
                    'estrato': 'GERAL',
                    'modelo': 'Idade (com k)',
                    'alvo': 'v2',
                    'n': m['n'],
                    'r2': m['r2'],
                    'rmse': m['rmse'],
                    'mae': m['mae'],
                    'bias': m['bias'],
                    'mape': m['mape']
                })

        # ======================================================================
        # MÉTRICAS POR ESTRATO (grupo = regional + regime)
        # ======================================================================
        for var_name, df_var in [(VAR1, df_var1), (VAR2, df_var2), (VAR3, df_var3)]:
            grupos = df_var['grupo'].unique()

            for grupo in grupos:
                df_grupo = df_var[df_var['grupo'] == grupo]

                # Métricas modelo delta (para v2) por estrato
                mask = df_grupo['v2_est_delta'].notna()
                if mask.sum() >= 5:  # Mínimo de 5 observações
                    m = calcular_metricas(df_grupo.loc[mask, PAIR_V2_COL].values,
                                          df_grupo.loc[mask, 'v2_est_delta'].values)
                    metricas_list.append({
                        'variavel': var_name,
                        'estrato': grupo,
                        'modelo': 'Delta',
                        'alvo': 'v2',
                        'n': m['n'],
                        'r2': m['r2'],
                        'rmse': m['rmse'],
                        'mae': m['mae'],
                        'bias': m['bias'],
                        'mape': m['mape']
                    })

                # Métricas modelo idade - Razão (para v2) por estrato
                mask = df_grupo['v2_est_razao'].notna()
                if mask.sum() >= 5:
                    m = calcular_metricas(df_grupo.loc[mask, PAIR_V2_COL].values,
                                          df_grupo.loc[mask, 'v2_est_razao'].values)
                    metricas_list.append({
                        'variavel': var_name,
                        'estrato': grupo,
                        'modelo': 'Idade (Razão)',
                        'alvo': 'v2',
                        'n': m['n'],
                        'r2': m['r2'],
                        'rmse': m['rmse'],
                        'mae': m['mae'],
                        'bias': m['bias'],
                        'mape': m['mape']
                    })

                # Métricas modelo idade - com k (para v2) por estrato
                mask = df_grupo['v2_est_idade'].notna()
                if mask.sum() >= 5:
                    m = calcular_metricas(df_grupo.loc[mask, PAIR_V2_COL].values,
                                          df_grupo.loc[mask, 'v2_est_idade'].values)
                    metricas_list.append({
                        'variavel': var_name,
                        'estrato': grupo,
                        'modelo': 'Idade (com k)',
                        'alvo': 'v2',
                        'n': m['n'],
                        'r2': m['r2'],
                        'rmse': m['rmse'],
                        'mae': m['mae'],
                        'bias': m['bias'],
                        'mape': m['mape']
                    })

        df_metricas = pd.DataFrame(metricas_list)
        df_metricas.to_excel(writer, sheet_name='Metricas_Resumo', index=False)

    print(f"\n  Resultados exportados para: {output_excel}")

    # ==========================================================================
    # RESUMO FINAL - MÉTRICAS GERAIS
    # ==========================================================================
    print("\n" + "="*60)
    print("RESUMO DE MÉTRICAS - GERAL")
    print("="*60)

    df_metricas_geral = df_metricas[df_metricas['estrato'] == 'GERAL']

    print(f"\n{'Variável':<10} {'Modelo':<16} {'Alvo':<6} {'n':>5} {'R²':>7} {'RMSE':>9} {'Bias':>9} {'MAPE':>7}")
    print("-" * 80)

    for _, row in df_metricas_geral.iterrows():
        print(f"{row['variavel']:<10} {row['modelo']:<16} {row['alvo']:<6} {row['n']:>5} "
              f"{row['r2']:>7.4f} {row['rmse']:>9.4f} {row['bias']:>9.4f} {row['mape']:>7.2f}%")

    # ==========================================================================
    # RESUMO FINAL - MÉTRICAS POR ESTRATO
    # ==========================================================================
    print("\n" + "="*60)
    print("RESUMO DE MÉTRICAS - POR ESTRATO (v2)")
    print("="*60)

    df_metricas_estrato = df_metricas[(df_metricas['estrato'] != 'GERAL') & (df_metricas['alvo'] == 'v2')]

    print(f"\n{'Variável':<10} {'Estrato':<20} {'Modelo':<16} {'n':>5} {'R²':>7} {'RMSE':>9} {'Bias':>9}")
    print("-" * 90)

    for var_name in [VAR1, VAR2, VAR3]:
        df_var_metricas = df_metricas_estrato[df_metricas_estrato['variavel'] == var_name]
        for _, row in df_var_metricas.iterrows():
            print(f"{row['variavel']:<10} {row['estrato']:<20} {row['modelo']:<16} {row['n']:>5} "
                  f"{row['r2']:>7.4f} {row['rmse']:>9.4f} {row['bias']:>9.4f}")
        if len(df_var_metricas) > 0:
            print()

    print("="*60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("="*60)
