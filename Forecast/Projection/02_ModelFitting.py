"""
02_ModelFitting.py - Ajuste de Modelos de Crescimento para Métricas LiDAR

Este script ajusta modelos de crescimento (Gompertz, Exponencial Modificado e Logístico)
para as métricas LiDAR (Z Kurt, Z P90, Z σ) em função da idade.

Modelos:
    1. Gompertz:              y = A * exp(-B * exp(-k * t))
    2. Exponencial Modificado: y = A * (1 - exp(-k * t)) + c
    3. Logístico:             y = L / (1 + exp(-k * (t - t0)))

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
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

# Cores para cada modelo
COLORS = {
    'data': '#1f77b4',
    'gompertz': '#d62728',
    'exponencial': '#2ca02c',
    'logistico': '#ff7f0e'
}


# =============================================================================
# DEFINIÇÃO DOS MODELOS
# =============================================================================

def gompertz(t, A, B, k):
    """
    Modelo de Gompertz.

    y = A * exp(-B * exp(-k * t))

    Parâmetros:
        A: Assíntota superior (valor máximo)
        B: Parâmetro de deslocamento
        k: Taxa de crescimento
    """
    return A * np.exp(-B * np.exp(-k * t))


def exponencial_modificado(t, A, k, c):
    """
    Modelo Exponencial Modificado (Monomolecular).

    y = A * (1 - exp(-k * t)) + c

    Parâmetros:
        A: Amplitude
        k: Taxa de crescimento
        c: Intercepto (valor inicial)
    """
    return A * (1 - np.exp(-k * t)) + c


def logistico(t, L, k, t0):
    """
    Modelo Logístico.

    y = L / (1 + exp(-k * (t - t0)))

    Parâmetros:
        L: Capacidade de suporte (valor máximo)
        k: Taxa de crescimento
        t0: Ponto de inflexão (idade no ponto médio)
    """
    return L / (1 + np.exp(-k * (t - t0)))


# =============================================================================
# FUNÇÕES DE AJUSTE
# =============================================================================

def fit_model(model_func, x, y, p0, bounds, model_name):
    """
    Ajusta um modelo aos dados.

    Retorna:
        params: Parâmetros ajustados
        r2: Coeficiente de determinação
        rmse: Raiz do erro quadrático médio
        aic: Critério de informação de Akaike
    """
    try:
        params, cov = curve_fit(model_func, x, y, p0=p0, bounds=bounds, maxfev=10000)

        # Predições
        y_pred = model_func(x, *params)

        # Métricas
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        # AIC
        n = len(y)
        k = len(params)
        aic = n * np.log(ss_res / n) + 2 * k

        return {
            'params': params,
            'r2': r2,
            'rmse': rmse,
            'aic': aic,
            'success': True
        }
    except Exception as e:
        print(f"    [ERRO] Falha ao ajustar {model_name}: {e}")
        return {
            'params': None,
            'r2': None,
            'rmse': None,
            'aic': None,
            'success': False
        }


def fit_all_models(x, y, var_name):
    """
    Ajusta todos os 3 modelos aos dados.
    """
    results = {}

    # Estimativas iniciais baseadas nos dados
    y_max = np.max(y)
    y_min = np.min(y)
    y_range = y_max - y_min
    t_mid = np.median(x)

    # --- Gompertz ---
    print(f"    Ajustando Gompertz...")
    p0_gomp = [y_max * 1.1, 2.0, 0.05]
    bounds_gomp = ([0, 0, 0.001], [y_max * 3, 10, 1])
    results['gompertz'] = fit_model(gompertz, x, y, p0_gomp, bounds_gomp, 'Gompertz')

    # --- Exponencial Modificado ---
    print(f"    Ajustando Exponencial Modificado...")
    p0_exp = [y_range, 0.05, y_min]
    bounds_exp = ([0, 0.001, -np.abs(y_min) * 2 - 10], [y_range * 3, 1, y_max])
    results['exponencial'] = fit_model(exponencial_modificado, x, y, p0_exp, bounds_exp, 'Exponencial Modificado')

    # --- Logístico ---
    print(f"    Ajustando Logístico...")
    p0_log = [y_max * 1.1, 0.1, t_mid]
    bounds_log = ([0, 0.001, 0], [y_max * 3, 1, np.max(x) * 2])
    results['logistico'] = fit_model(logistico, x, y, p0_log, bounds_log, 'Logístico')

    return results


def print_results(results, var_name):
    """
    Imprime os resultados do ajuste de forma formatada.
    """
    print(f"\n    {'Modelo':<25} {'R²':>10} {'RMSE':>12} {'AIC':>12}")
    print("    " + "-" * 60)

    for model_name, res in results.items():
        if res['success']:
            print(f"    {model_name.capitalize():<25} {res['r2']:>10.4f} {res['rmse']:>12.4f} {res['aic']:>12.2f}")
        else:
            print(f"    {model_name.capitalize():<25} {'FALHOU':>10}")

    # Melhor modelo (menor AIC)
    valid_results = {k: v for k, v in results.items() if v['success']}
    if valid_results:
        best_model = min(valid_results, key=lambda k: valid_results[k]['aic'])
        print(f"\n    >>> Melhor modelo (menor AIC): {best_model.upper()}")


# =============================================================================
# FUNÇÃO DE PLOTAGEM
# =============================================================================

def plot_models(df, var_name, results, output_path):
    """
    Plota os dados e as curvas ajustadas dos 3 modelos.
    """
    x = df['IDADE'].values
    y = df[var_name].values

    # Criar range para curvas suaves
    x_smooth = np.linspace(x.min(), x.max(), 500)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plotar dados
    ax.scatter(x, y, alpha=0.3, s=15, c=COLORS['data'], label='Dados', zorder=1)

    # Plotar cada modelo
    labels = {
        'gompertz': 'Gompertz',
        'exponencial': 'Exp. Modificado',
        'logistico': 'Logístico'
    }

    model_funcs = {
        'gompertz': gompertz,
        'exponencial': exponencial_modificado,
        'logistico': logistico
    }

    for model_name, res in results.items():
        if res['success']:
            y_smooth = model_funcs[model_name](x_smooth, *res['params'])
            ax.plot(x_smooth, y_smooth,
                    color=COLORS[model_name],
                    linewidth=2.5,
                    label=f"{labels[model_name]} (R²={res['r2']:.3f})",
                    zorder=2)

    ax.set_xlabel('Idade (meses)')
    ax.set_ylabel(var_name)
    ax.set_title(f'Ajuste de Modelos de Crescimento - {var_name}')
    ax.legend(loc='best')
    ax.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"    Gráfico salvo: {output_path}")


def plot_residuals(df, var_name, results, output_path):
    """
    Plota os resíduos de cada modelo.
    """
    x = df['IDADE'].values
    y = df[var_name].values

    model_funcs = {
        'gompertz': gompertz,
        'exponencial': exponencial_modificado,
        'logistico': logistico
    }

    labels = {
        'gompertz': 'Gompertz',
        'exponencial': 'Exp. Modificado',
        'logistico': 'Logístico'
    }

    # Contar modelos válidos
    valid_models = [k for k, v in results.items() if v['success']]
    n_models = len(valid_models)

    if n_models == 0:
        print("    Nenhum modelo válido para plotar resíduos.")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, valid_models):
        res = results[model_name]
        y_pred = model_funcs[model_name](x, *res['params'])
        residuals = y - y_pred

        ax.scatter(y_pred, residuals, alpha=0.3, s=10, c=COLORS[model_name])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Valores Preditos')
        ax.set_ylabel('Resíduos')
        ax.set_title(f'{labels[model_name]}')
        ax.grid(linestyle='--', alpha=0.5)

    plt.suptitle(f'Análise de Resíduos - {var_name}', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"    Gráfico de resíduos salvo: {output_path}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def process_variable(df, var_name, file_prefix):
    """
    Processa uma variável: ajusta modelos e gera gráficos.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSANDO: {var_name}")
    print(f"{'='*60}")
    print(f"  Registros: {len(df)}")

    x = df['IDADE'].values
    y = df[var_name].values

    print(f"  Idade: {x.min():.0f} - {x.max():.0f} meses")
    print(f"  {var_name}: {y.min():.3f} - {y.max():.3f}")

    # Ajustar modelos
    print(f"\n  Ajustando modelos...")
    results = fit_all_models(x, y, var_name)

    # Imprimir resultados
    print_results(results, var_name)

    # Plotar modelos
    print(f"\n  Gerando gráficos...")
    plot_models(df, var_name, results, rf"{OUTPUT_DIR}\{file_prefix}_modelos.png")

    # Plotar resíduos
    plot_residuals(df, var_name, results, rf"{OUTPUT_DIR}\{file_prefix}_residuos.png")

    return results


def format_equation(model_name, params):
    """
    Formata a equação do modelo com os parâmetros ajustados.
    """
    if model_name == 'gompertz':
        A, B, k = params
        return f"y = {A:.4f} * exp(-{B:.4f} * exp(-{k:.6f} * t))"

    elif model_name == 'exponencial':
        A, k, c = params
        if c >= 0:
            return f"y = {A:.4f} * (1 - exp(-{k:.6f} * t)) + {c:.4f}"
        else:
            return f"y = {A:.4f} * (1 - exp(-{k:.6f} * t)) - {abs(c):.4f}"

    elif model_name == 'logistico':
        L, k, t0 = params
        return f"y = {L:.4f} / (1 + exp(-{k:.6f} * (t - {t0:.4f})))"

    return ""


def create_summary_table(all_results):
    """
    Cria tabela resumo com todos os resultados.
    """
    rows = []

    for var_name, results in all_results.items():
        for model_name, res in results.items():
            if res['success']:
                rows.append({
                    'Variável': var_name,
                    'Modelo': model_name.capitalize(),
                    'R²': res['r2'],
                    'RMSE': res['rmse'],
                    'AIC': res['aic'],
                    'Equação': format_equation(model_name, res['params']),
                    'Parâmetros': str(np.round(res['params'], 6).tolist())
                })

    df_summary = pd.DataFrame(rows)
    return df_summary


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("AJUSTE DE MODELOS DE CRESCIMENTO")
    print("="*60)

    # Carregar dados
    print("\nCarregando dados limpos...")

    df_var1 = pd.read_excel(INPUT_FILE_1)
    print(f"  {VAR1}: {len(df_var1)} registros")

    df_var2 = pd.read_excel(INPUT_FILE_2)
    print(f"  {VAR2}: {len(df_var2)} registros")

    df_var3 = pd.read_excel(INPUT_FILE_3)
    print(f"  {VAR3}: {len(df_var3)} registros")

    # Processar cada variável
    all_results = {}

    all_results[VAR1] = process_variable(df_var1, VAR1, "zkurt")
    all_results[VAR2] = process_variable(df_var2, VAR2, "zp90")
    all_results[VAR3] = process_variable(df_var3, VAR3, "zstddev")

    # Criar tabela resumo
    print("\n" + "="*60)
    print("RESUMO GERAL")
    print("="*60)

    df_summary = create_summary_table(all_results)
    print("\n")
    print(df_summary.to_string(index=False))

    # Exportar resumo
    summary_path = rf"{OUTPUT_DIR}\modelos_resumo.xlsx"
    df_summary.to_excel(summary_path, index=False)
    print(f"\n✓ Resumo exportado para: {summary_path}")

    # Identificar melhor modelo para cada variável
    print("\n" + "-"*60)
    print("MELHOR MODELO POR VARIÁVEL (menor AIC):")
    print("-"*60)

    for var_name in [VAR1, VAR2, VAR3]:
        var_results = df_summary[df_summary['Variável'] == var_name]
        if not var_results.empty:
            best = var_results.loc[var_results['AIC'].idxmin()]
            print(f"  {var_name}: {best['Modelo']} (R²={best['R²']:.4f}, AIC={best['AIC']:.2f})")

    print("\n" + "="*60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("="*60)
