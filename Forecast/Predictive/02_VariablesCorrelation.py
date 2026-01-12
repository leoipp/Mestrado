"""
02_VariablesCorrelation.py - Seleção de Variáveis para Modelagem Florestal

Este script realiza a análise de correlação e seleção de variáveis LiDAR para
predição de volume florestal (VTCC) usando técnicas estatísticas e machine learning.

Etapas de processamento:
    1. Carregamento do DataFrame limpo (saída do 01_DataConsistency.py)
    2. Cálculo da correlação de Pearson entre variáveis numéricas e VTCC
    3. Filtragem de variáveis com correlação |r| > 0.6 (apenas métricas LiDAR)
    4. Seleção recursiva de features (RFE) com Random Forest
    5. Validação cruzada (10-fold) para otimização do número de features
    6. Análise de multicolinearidade via VIF (Variance Inflation Factor)
    7. Visualização da matriz de correlação das variáveis selecionadas

Saídas:
    - RFE_Metrics_Target.xlsx: Tabela com métricas por número de features
    - Gráficos de correlação e heatmap

Autor: Leonardo Ippolito Rodrigues
Data: 2024
Projeto: Mestrado - Predição de Volume Florestal com LiDAR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Caminhos dos arquivos
INPUT_FILE = r"G:\PycharmProjects\Mestrado\Data\DataFrames\IFC_LiDAR_Plots_RTK_Cleaned_v02.xlsx"
OUTPUT_RFE_SUMMARY = ".\Results\RFE_Metrics_Target.xlsx"

# Parâmetros de análise
CORRELATION_THRESHOLD = 0.7      # Limiar mínimo de correlação com VTCC
CV_FOLDS = 5                    # Número de folds para validação cruzada
RANDOM_STATE = 42                # Semente para reprodutibilidade

# Variável alvo
TARGET_COLUMN = 'VTCC(m³/ha)'

# Configuração de estilo dos gráficos
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

# Cores padrão
COLOR_PRIMARY = '#1f77b4'

# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================
#%% Leitura do DataFrame limpo
print("=" * 60)
print("CARREGAMENTO DOS DADOS")
print("=" * 60)
df = pd.read_excel(INPUT_FILE)
print(f"Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
# =============================================================================
# ANÁLISE DE CORRELAÇÃO
# =============================================================================
#%% Cálculo da correlação de Pearson com VTCC
print("\n" + "=" * 60)
print("ANÁLISE DE CORRELAÇÃO")
print("=" * 60)

numeric_df = df.select_dtypes(include='number')
correlations = numeric_df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)

print(f"\nCorrelações calculadas para {len(correlations)} variáveis numéricas")

#%% Histograma da distribuição das correlações
plt.figure(figsize=(10, 6))

# Define bins centralizados nos ticks (-1.0, -0.9, ..., 0.9, 1.0)
bin_edges = np.arange(-1.05, 1.15, 0.1)  # Edges entre os centros
bin_centers = np.arange(-1.0, 1.1, 0.1)  # Centros nos ticks

# Calcula frequência percentual
counts, _ = np.histogram(correlations, bins=bin_edges)
percentages = counts / len(correlations) * 100

# Plota barras centralizadas nos ticks
plt.bar(bin_centers, percentages, width=0.08, alpha=0.7,
        color=COLOR_PRIMARY, edgecolor=COLOR_PRIMARY, linewidth=0.8, zorder=2)
plt.xlabel('Coeficiente de correlação de Pearson (r)')
plt.xlim(-1.1, 1.1)
plt.xticks(np.arange(-1.0, 1.1, 0.2))
plt.ylabel('Frequência (%)')
plt.grid(linestyle='--', color='gray', alpha=0.5, zorder=1)
plt.tight_layout()
plt.show()

#%% Filtragem de variáveis com correlação significativa
# Critérios:
# 1. Correlação absoluta > CORRELATION_THRESHOLD (0.6)
# 2. Apenas métricas LiDAR (contém 'Elev' no nome)
correlations_filtered = correlations[abs(correlations) > CORRELATION_THRESHOLD]
correlations_filtered = correlations_filtered[correlations_filtered.index.str.contains('Elev')]

print(f"\nVariáveis selecionadas (|r| > {CORRELATION_THRESHOLD}, apenas LiDAR):")
print(f"  {len(correlations_filtered)} variáveis de {len(correlations)} totais")
print(correlations_filtered)

# Salva correlação VTCC x variáveis
VTCC_CORR_FILE = ".\\Results\\Pearson_VTCC_Correlation.xlsx"
correlations.to_frame(name="Pearson_r").to_excel(VTCC_CORR_FILE)
print(f"✓ Correlação VTCC salva em: {VTCC_CORR_FILE}")


# =============================================================================
# SELEÇÃO RECURSIVA DE FEATURES (RFE)
# =============================================================================
#%% Configuração do RFE com Random Forest
print("\n" + "=" * 60)
print("SELEÇÃO RECURSIVA DE FEATURES (RFE)")
print("=" * 60)

# Prepara dados para o modelo
feature_cols = correlations_filtered.index.to_list()
X = df[feature_cols].values
Y = df[[TARGET_COLUMN]].values.ravel()

print(f"\nFeatures de entrada: {len(feature_cols)}")
print(f"Amostras: {len(Y)}")

# Configuração da validação cruzada
folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
hyper_params = [{'n_features_to_select': list(range(1, X.shape[1] + 1))}]

# Modelo base: Random Forest Regressor
model = RandomForestRegressor(random_state=RANDOM_STATE)

# RFE com GridSearchCV para encontrar número ótimo de features
rfe = RFE(estimator=model)
model_cv = GridSearchCV(
    estimator=rfe,
    param_grid=hyper_params,
    scoring='neg_root_mean_squared_error',
    cv=folds,
    verbose=1,
    return_train_score=True
)

print("\nExecutando RFE com validação cruzada...")
model_cv.fit(X, Y)

# =============================================================================
# VARIÁVEIS FINAIS: APENAS RFE RANKING = 1 (support = True)
# =============================================================================
selected_features_rfe = list(np.array(feature_cols)[model_cv.best_estimator_.support_])

print("\nVariáveis finais (RFE ranking = 1):")
for v in selected_features_rfe:
    print(f"  - {v}")

# Salva lista das variáveis finais
RFE_SELECTED_FILE = ".\\Results\\RFE_Selected_Features.xlsx"
pd.DataFrame({"Feature": selected_features_rfe}).to_excel(RFE_SELECTED_FILE, index=False)
print(f"✓ Lista de variáveis RFE (ranking=1) salva em: {RFE_SELECTED_FILE}")


#%% Ranking das features
print("\n" + "-" * 40)
print("RANKING DAS FEATURES")
print("-" * 40)

ranking_sorted = sorted(
    zip(feature_cols, model_cv.best_estimator_.ranking_, model_cv.best_estimator_.support_),
    key=lambda x: x[1]
)

for i, (name, rank, selected) in enumerate(ranking_sorted):
    status = "✓" if selected else " "
    print(f"{status} {i+1:02d}. {name:30s} | Rank: {rank:2d}")

print(f"\nNúmero ótimo de features: {model_cv.best_params_['n_features_to_select']}")

#%% Cálculo de métricas para cada número de features
print("\nCalculando métricas detalhadas...")

results_df = pd.DataFrame(model_cv.cv_results_)

# Listas para armazenar métricas
features_selected_list = []
rmse_list = []
mae_list = []
bias_list = []
r2_list = []

# Itera sobre cada configuração de número de features
# Usa cross_val_predict para evitar viés otimista (overfitting)
cv_eval = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for n in results_df['param_n_features_to_select']:
    rfe_temp = RFE(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE),
        n_features_to_select=n
    )
    rfe_temp.fit(X, Y)

    # Seleciona features
    X_sel = rfe_temp.transform(X)
    selected = np.array(feature_cols)[rfe_temp.support_].tolist()
    features_selected_list.append(selected)

    # Predições via validação cruzada (out-of-sample)
    rf_model = RandomForestRegressor(random_state=RANDOM_STATE)
    y_pred_cv = cross_val_predict(rf_model, X_sel, Y, cv=cv_eval)

    # Calcula métricas de avaliação (validação cruzada)
    rmse_list.append(np.sqrt(mean_squared_error(Y, y_pred_cv)))
    mae_list.append(mean_absolute_error(Y, y_pred_cv))
    bias_list.append(np.mean(y_pred_cv - Y))
    r2_list.append(r2_score(Y, y_pred_cv))

# Adiciona métricas ao DataFrame
results_df['features_selected'] = features_selected_list
results_df['RMSE'] = rmse_list
results_df['MAE'] = mae_list
results_df['BIAS'] = bias_list
results_df['R2'] = r2_list

# Organiza tabela de resumo
summary_table = results_df[[
    'param_n_features_to_select',
    'RMSE',
    'MAE',
    'BIAS',
    'R2',
    'features_selected'
]].sort_values(by='RMSE')

summary_table.columns = ['N_Features', 'RMSE', 'MAE', 'BIAS', 'R2', 'Features_Selected']

# Cria DataFrame com ranking das variáveis
ranking_df = pd.DataFrame(ranking_sorted, columns=['Variable', 'Rank', 'Selected'])
ranking_df['Selected'] = ranking_df['Selected'].map({True: 'Sim', False: 'Nao'})
ranking_df.index = ranking_df.index + 1  # Index começa em 1
ranking_df.index.name = 'Position'

# Exporta resultados em múltiplas worksheets
with pd.ExcelWriter(OUTPUT_RFE_SUMMARY, engine='openpyxl') as writer:
    summary_table.to_excel(writer, sheet_name='Metrics', index=False)
    ranking_df.to_excel(writer, sheet_name='Ranking', index=True)

print(f"\n✓ Resumo exportado para: {OUTPUT_RFE_SUMMARY}")
print(f"  - Worksheet 'Metrics': métricas por número de features")
print(f"  - Worksheet 'Ranking': ranking das variáveis pelo RFE")
# =============================================================================
# VISUALIZAÇÃO DA MATRIZ DE CORRELAÇÃO
# =============================================================================
#%% Heatmap das variáveis selecionadas
print("\n" + "=" * 60)
print("MATRIZ DE CORRELAÇÃO DAS VARIÁVEIS SELECIONADAS")
print("=" * 60)

# Obtém colunas selecionadas pelo RFE
selected_cols = list(np.array(feature_cols)[model_cv.best_estimator_.support_])
selected_cols.append(TARGET_COLUMN)

print(f"\nVariáveis no heatmap: {len(selected_cols)}")

# Calcula matriz de correlação
corr_matrix = df[selected_cols].corr()

# =============================================================================
# HEATMAP TRIANGULAR (SEM REPETIÇÃO)
# =============================================================================

FIG_CORR_HEATMAP = ".\\Results\\Heatmap_Correlation_RFE_Rank1.png"

plt.figure(figsize=(14, 12))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

ax = sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.3,
    cbar_kws={"orientation": "horizontal", "shrink": 0.7}
)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Matriz de Correlação de Pearson (Triângulo Inferior)")
plt.tight_layout()

# Salva figura
plt.savefig(FIG_CORR_HEATMAP, dpi=300, bbox_inches="tight")
plt.show()

print(f"✓ Heatmap de correlação salvo em: {FIG_CORR_HEATMAP}")


# =============================================================================
# HEATMAP: APENAS VARIÁVEIS COM |r| > 0.6 + VTCC  + COLCHETE (SELECTED METRICS)
# =============================================================================
print("\nGerando heatmap com Pearson |r| > 0.6 + VTCC")

from matplotlib.patches import FancyArrowPatch

def add_vertical_bracket(ax, y0, y1, text, x=-0.9, text_x=-1.35,
                         lw=1.4, fontsize=10):
    """
    Desenha um colchete vertical no lado esquerdo do heatmap, cobrindo de y0 até y1.
    Coordenadas (x, y) no sistema de dados do heatmap (0..n).
    """
    y0, y1 = sorted([y0, y1])

    bracket = FancyArrowPatch(
        (x, y0), (x, y1),
        arrowstyle='-[',          # colchete
        mutation_scale=18,
        lw=lw,
        color='black',
        clip_on=False
    )
    ax.add_patch(bracket)

    ax.text(
        text_x, (y0 + y1) / 2,
        text,
        va='center', ha='right',
        fontsize=fontsize,
        color='black',
        clip_on=False
    )

# =============================================================================
# SELEÇÃO FINAL PARA HEATMAP:
# Apenas variáveis LiDAR (Elev*) + VTCC
# =============================================================================
# =============================================================================
# HEATMAP: apenas Elev* com |r(VTCC)| > threshold + VTCC
# =============================================================================
print("\nGerando heatmap: Elev* com |r(VTCC)| > threshold + VTCC")

from mpl_toolkits.axes_grid1 import make_axes_locatable

# 1) Apenas Elev* dentro do vetor de correlações com VTCC
elev_corr = correlations[correlations.index.str.startswith("Elev")]

# 2) Aplica limiar |r| > threshold
elev_strong = elev_corr[abs(elev_corr) > CORRELATION_THRESHOLD] \
    .sort_values(key=np.abs, ascending=False)

# 3) Lista final: VTCC + Elev selecionadas
vars_heatmap = [TARGET_COLUMN] + elev_strong.index.tolist()

print(f"Variáveis no heatmap: {len(vars_heatmap)}")
for v in vars_heatmap:
    print(f"  - {v}")

# 4) Matriz de correlação (Pearson) + máscara triangular
corr_matrix = df[vars_heatmap].corr(method="pearson")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 5) Figura com tamanho dinâmico
n_vars = len(vars_heatmap)
figsize = max(8, n_vars * 0.55)

FIG_CORR_HEATMAP = ".\\Results\\Heatmap_Correlation_Elev_gt_06_plus_VTCC.png"

plt.figure(figsize=(figsize, figsize), dpi=300)
ax = plt.gca()

# cria o divider vinculado ao eixo principal
divider = make_axes_locatable(ax)

# colorbar à ESQUERDA com a MESMA ALTURA do heatmap
cax = divider.append_axes(
    "left",
    size="3.5%",   # largura da barra
    pad=0.15       # distância do heatmap
)

# HEATMAP
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.3,
    annot_kws={"size": 8 if n_vars <= 18 else 7},
    cbar=True,
    cbar_ax=cax,   # controla posição/tamanho da barra
    ax=ax
)

# colorbar: ticks do lado esquerdo (como está à esquerda)
cax.yaxis.set_ticks_position("left")
cax.yaxis.set_label_position("left")

# =========================
# AJUSTES DE EIXOS (LIMPEZA)
# =========================

# X: embaixo, diagonal padrão (remove último label da ponta)
ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
xt = ax.get_xticks()
xl = [t.get_text().replace("Elev", "Z") for t in ax.get_xticklabels()]
if len(xl) > 1:
    ax.set_xticks(xt[:-1])
    ax.set_xticklabels(xl[:-1], rotation=45, ha="right")

# Y: remove yticks padrão e coloca labels na diagonal do heatmap
ax.set_yticks([])
ax.set_yticklabels([])

# Adiciona labels na diagonal (posição i,i com offset para a direita)
for i, var in enumerate(vars_heatmap[1:], start=1):  # pula o primeiro (VTCC)
    ax.text(
        i + 0.1,          # x: na diagonal, com pequeno offset para direita
        i + 0.5,          # y: centro da célula
        var.replace("Elev", "Z"),
        ha="left",
        va="center",
        fontsize=8 if n_vars <= 18 else 7,
        clip_on=False
    )

# remove spines desnecessários
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# layout final
plt.subplots_adjust(
    left=0.32,
    right=0.88,
    bottom=0.22,
    top=0.98
)

# salva figura
plt.savefig(
    FIG_CORR_HEATMAP,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.25
)
plt.show()

print(f"✓ Heatmap salvo em: {FIG_CORR_HEATMAP}")

