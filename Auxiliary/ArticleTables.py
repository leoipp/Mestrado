import pandas as pd
from great_tables import GT


#%% Dataframe reader
df = pd.read_excel(r"G:\PycharmProjects\Mestrado\Data\DataFrames\IFC_LiDAR_Plots_RTK_Cleaned.xlsx")
print(df.shape)

#%% Auxiliar column maker
df['Regional'] = df["LOTE_CODIGO"].str[:2]
region_map = {
    'GN': 'Região 01',
    'NE': 'Região 02',
    'RD': 'Região 03'
}
df['Regional'] = df['Regional'].map(region_map)
df['Regime_'] = df["LOTE_CODIGO"].str[11].map(lambda x: 'Alto Fuste' if x == 'P' else 'Talhadia')

df['Classe_idade_m'] = (df['Idade (anos)'] // 2) * 2 + (2 / 2)

#%% Pivot table
pivot_mean = df.groupby(['Regime_', 'Regional', 'Classe_idade_m']).agg({
    'DESCTIPOPROPRIEDADE': 'count',
    'Área.corrigida (m²)': 'mean',
    'Dap.médio (cm)': 'mean',
    'HT.média (m)': 'mean',
    'VTCC(m³/ha)': 'mean',
    'Fustes (n/ha)': 'mean'
}).reset_index()

# Organize values as integer
pivot_mean['Classe_idade_m'] = pivot_mean['Classe_idade_m'].astype(int)
pivot_mean['Área.corrigida (m²)'] = pivot_mean['Área.corrigida (m²)'].round().astype(int)
pivot_mean['Fustes (n/ha)'] = pivot_mean['Fustes (n/ha)'].round().astype(int)

pivot_std = df.groupby(['Regime_', 'Regional', 'Classe_idade_m']).agg({
    'DESCTIPOPROPRIEDADE': 'count',
    'Área.corrigida (m²)': 'std',
    'Dap.médio (cm)': 'std',
    'HT.média (m)': 'std',
    'VTCC(m³/ha)': 'std',
    'Fustes (n/ha)': 'std'
}).reset_index()
# Replace NaN values with 0
pivot_std = pivot_std.fillna(0)
# Organize values as integer
# Identify the columns to round (excluding group-by columns)
cols_to_round = pivot_std.columns.difference(['Área.corrigida (m²)', 'Dap.médio (cm)', 'HT.média (m)', 'VTCC(m³/ha)', 'Fustes (n/ha)'])
# Round selected columns to 2 decimals
pivot_std[cols_to_round] = pivot_std[cols_to_round].round(2)

pivot = pd.merge(pivot_mean, pivot_std, on=['Regime_', 'Regional', 'Classe_idade_m'], suffixes=('_mean', '_std'))

print(pivot)
#%% Article table
# Combine mean and std values correctly after the merge
pivot_combined = pivot.copy()

# Loop through the columns and combine the mean and std values with ±
for column in ['Área.corrigida (m²)', 'Dap.médio (cm)', 'HT.média (m)', 'VTCC(m³/ha)', 'Fustes (n/ha)']:
    # Refer to the correct column names after the merge
    mean_column = column + '_mean'
    std_column = column + '_std'
    pivot_combined[column] = pivot_combined[mean_column].round(2).astype(str) + ' ± ' + pivot_combined[std_column].round(2).astype(str)

pivot_combined['Qtd Parcelas'] = pivot_combined['DESCTIPOPROPRIEDADE_mean']
pivot_combined_cleaned = pivot_combined.drop(columns=[col for col in pivot_combined.columns if col.endswith('_std') or col.endswith('_mean')])
pivot_combined_cleaned.to_excel('OUTPUT_DADOS.xlsx')
# Create the GreatTable object from the DataFrame
table = (
    GT(pivot_combined_cleaned)
    .tab_stub(rowname_col="Regime_")
    .tab_stubhead(label='Regime')
         )

table.show()