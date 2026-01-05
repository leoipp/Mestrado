"""
ArticleTables.py - Gerador de Tabelas para Artigos Cientificos

Gera tabelas formatadas para a secao de Materiais e Metodos de artigos
cientificos sobre predicao de volume florestal com LiDAR.

Formatos de saida:
    - Excel (.xlsx)
    - LaTeX (.tex)
    - Word (.docx)
    - HTML (para visualizacao)

Tabelas geradas:
    1. Caracterizacao da area de estudo
    2. Estatisticas descritivas das variaveis de campo
    3. Metricas LiDAR utilizadas
    4. Parametros do modelo preditivo
    5. Resultados por regional/regime

Autor: Leonardo Ippolito Rodrigues
Ano: 2026
Projeto: Mestrado - Predicao de Volume Florestal com LiDAR
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime

try:
    from great_tables import GT, md, html, style, loc
    HAS_GT = True
except ImportError:
    HAS_GT = False
    print("Aviso: great_tables nao instalado. Use: pip install great-tables")


# =============================================================================
# CONFIGURACAO
# =============================================================================

class TableConfig:
    """Configuracao para geracao de tabelas."""

    # Mapeamento de regioes
    REGION_MAP = {
        'GN': 'Regiao 01',
        'NE': 'Regiao 02',
        'RD': 'Regiao 03'
    }

    # Mapeamento de regime
    REGIME_MAP = {
        'P': 'Alto Fuste',
        'T': 'Talhadia'
    }

    # Nomes das colunas para exibicao
    COLUMN_LABELS = {
        'Regional': 'Regional',
        'Regime_': 'Regime',
        'Classe_idade_m': 'Classe de Idade (anos)',
        'n': 'n',
        'Area': 'Area (m²)',
        'DAP': 'DAP (cm)',
        'HT': 'Ht (m)',
        'VTCC': 'V (m³/ha)',
        'Fustes': 'N (arv/ha)',
        'Idade': 'Idade (anos)'
    }

    # Metricas LiDAR
    LIDAR_METRICS = {
        'max': ('Altura maxima', 'Zmax', 'm'),
        'p90': ('Percentil 90', 'P90', 'm'),
        'p60': ('Percentil 60', 'P60', 'm'),
        'kur': ('Curtose', 'Kurt', '-'),
        'mean': ('Altura media', 'Zmean', 'm'),
        'std': ('Desvio padrao', 'Zsd', 'm'),
        'cv': ('Coef. variacao', 'Zcv', '%'),
        'skew': ('Assimetria', 'Zskew', '-'),
    }


# =============================================================================
# FUNCOES AUXILIARES
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega dados do inventario florestal.

    Parameters
    ----------
    filepath : str
        Caminho para o arquivo Excel ou CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados carregados.
    """
    path = Path(filepath)

    if path.suffix == '.xlsx':
        df = pd.read_excel(filepath)
    elif path.suffix == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Formato nao suportado: {path.suffix}")

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara os dados para geracao das tabelas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame bruto do inventario.

    Returns
    -------
    pd.DataFrame
        DataFrame preparado com colunas auxiliares.
    """
    df = df.copy()

    # Regional (primeiros 2 caracteres do codigo do lote)
    if 'LOTE_CODIGO' in df.columns:
        df['Regional'] = df['LOTE_CODIGO'].str[:2].map(TableConfig.REGION_MAP)

        # Regime (caracter na posicao 11)
        df['Regime'] = df['LOTE_CODIGO'].str[11].map(
            lambda x: 'Alto Fuste' if x == 'P' else 'Talhadia'
        )

    # Classe de idade (intervalos de 2 anos)
    if 'Idade (anos)' in df.columns:
        df['Classe_Idade'] = (df['Idade (anos)'] // 2) * 2 + 1

    return df


def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    """Formata valor como media ± desvio padrao."""
    if pd.isna(std) or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_range(min_val: float, max_val: float, decimals: int = 2) -> str:
    """Formata valor como intervalo (min - max)."""
    return f"{min_val:.{decimals}f} - {max_val:.{decimals}f}"


# =============================================================================
# TABELA 1: CARACTERIZACAO DA AREA DE ESTUDO
# =============================================================================

def table_study_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera tabela de caracterizacao da area de estudo.

    Tabela 1. Caracterizacao da area de estudo por regional.
    """
    df = prepare_data(df)

    summary = df.groupby('Regional').agg({
        'LOTE_CODIGO': 'nunique',  # Numero de lotes
        'DESCTIPOPROPRIEDADE': 'count',  # Numero de parcelas
        'Área.corrigida (m²)': ['sum', 'mean'],
        'Idade (anos)': ['min', 'max', 'mean'],
    }).reset_index()

    # Flatten column names
    summary.columns = [
        'Regional', 'N_Lotes', 'N_Parcelas',
        'Area_Total', 'Area_Media',
        'Idade_Min', 'Idade_Max', 'Idade_Media'
    ]

    # Formatar valores
    summary['Area_Total_ha'] = (summary['Area_Total'] / 10000).round(2)
    summary['Area_Media_m2'] = summary['Area_Media'].round(0).astype(int)
    summary['Amplitude_Idade'] = summary.apply(
        lambda r: f"{r['Idade_Min']:.1f} - {r['Idade_Max']:.1f}", axis=1
    )
    summary['Idade_Media'] = summary['Idade_Media'].round(2)

    # Selecionar colunas finais
    result = summary[[
        'Regional', 'N_Lotes', 'N_Parcelas',
        'Area_Total_ha', 'Area_Media_m2',
        'Amplitude_Idade', 'Idade_Media'
    ]].copy()

    result.columns = [
        'Regional', 'Lotes (n)', 'Parcelas (n)',
        'Area Total (ha)', 'Area Media (m²)',
        'Amplitude Idade (anos)', 'Idade Media (anos)'
    ]

    # Adicionar linha de total
    total = pd.DataFrame([{
        'Regional': 'Total',
        'Lotes (n)': result['Lotes (n)'].sum(),
        'Parcelas (n)': result['Parcelas (n)'].sum(),
        'Area Total (ha)': result['Area Total (ha)'].sum(),
        'Area Media (m²)': result['Area Media (m²)'].mean().round(0),
        'Amplitude Idade (anos)': f"{df['Idade (anos)'].min():.1f} - {df['Idade (anos)'].max():.1f}",
        'Idade Media (anos)': df['Idade (anos)'].mean().round(2)
    }])

    result = pd.concat([result, total], ignore_index=True)

    return result


# =============================================================================
# TABELA 2: ESTATISTICAS DESCRITIVAS DAS VARIAVEIS DE CAMPO
# =============================================================================

def table_field_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera tabela de estatisticas descritivas das variaveis de campo.

    Tabela 2. Estatisticas descritivas das variaveis dendrometricas.
    """
    df = prepare_data(df)

    # Variaveis de interesse
    variables = {
        'Idade (anos)': 'Idade (anos)',
        'Dap.médio (cm)': 'DAP (cm)',
        'HT.média (m)': 'Ht (m)',
        'VTCC(m³/ha)': 'V (m³/ha)',
        'Fustes (n/ha)': 'N (arv/ha)',
        'Área.corrigida (m²)': 'Area (m²)'
    }

    stats_list = []

    for col, label in variables.items():
        if col in df.columns:
            stats = {
                'Variavel': label,
                'n': df[col].count(),
                'Media': df[col].mean(),
                'DP': df[col].std(),
                'CV (%)': (df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else 0,
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Mediana': df[col].median()
            }
            stats_list.append(stats)

    result = pd.DataFrame(stats_list)

    # Formatar valores
    for col in ['Media', 'DP', 'Min', 'Max', 'Mediana']:
        result[col] = result[col].round(2)
    result['CV (%)'] = result['CV (%)'].round(1)
    result['n'] = result['n'].astype(int)

    return result


# =============================================================================
# TABELA 3: ESTATISTICAS POR REGIME E REGIONAL
# =============================================================================

def table_by_regime_regional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera tabela de estatisticas por regime e regional.

    Tabela 3. Caracteristicas dendrometricas por regime de manejo e regional.
    """
    df = prepare_data(df)

    # Agregar por Regime e Regional
    agg_funcs = {
        'DESCTIPOPROPRIEDADE': 'count',
        'Dap.médio (cm)': ['mean', 'std'],
        'HT.média (m)': ['mean', 'std'],
        'VTCC(m³/ha)': ['mean', 'std'],
        'Fustes (n/ha)': ['mean', 'std'],
        'Idade (anos)': ['mean', 'std']
    }

    grouped = df.groupby(['Regime', 'Regional']).agg(agg_funcs).reset_index()

    # Flatten columns
    grouped.columns = [
        'Regime', 'Regional', 'n',
        'DAP_mean', 'DAP_std',
        'HT_mean', 'HT_std',
        'V_mean', 'V_std',
        'N_mean', 'N_std',
        'Idade_mean', 'Idade_std'
    ]

    # Combinar mean ± std
    result = pd.DataFrame({
        'Regime': grouped['Regime'],
        'Regional': grouped['Regional'],
        'n': grouped['n'],
        'Idade (anos)': grouped.apply(
            lambda r: format_mean_std(r['Idade_mean'], r['Idade_std'], 1), axis=1
        ),
        'DAP (cm)': grouped.apply(
            lambda r: format_mean_std(r['DAP_mean'], r['DAP_std'], 2), axis=1
        ),
        'Ht (m)': grouped.apply(
            lambda r: format_mean_std(r['HT_mean'], r['HT_std'], 2), axis=1
        ),
        'V (m³/ha)': grouped.apply(
            lambda r: format_mean_std(r['V_mean'], r['V_std'], 2), axis=1
        ),
        'N (arv/ha)': grouped.apply(
            lambda r: format_mean_std(r['N_mean'], r['N_std'], 0), axis=1
        ),
    })

    return result.sort_values(['Regime', 'Regional'])


# =============================================================================
# TABELA 4: METRICAS LIDAR
# =============================================================================

def table_lidar_metrics() -> pd.DataFrame:
    """
    Gera tabela descritiva das metricas LiDAR utilizadas.

    Tabela 4. Metricas derivadas dos dados LiDAR.
    """
    metrics = [
        ('Zmax', 'Altura maxima', 'Valor maximo de Z normalizado', 'm'),
        ('P90', 'Percentil 90', '90º percentil das alturas normalizadas', 'm'),
        ('P60', 'Percentil 60', '60º percentil das alturas normalizadas', 'm'),
        ('Zmean', 'Altura media', 'Media das alturas normalizadas', 'm'),
        ('Zsd', 'Desvio padrao', 'Desvio padrao das alturas', 'm'),
        ('Zcv', 'Coef. variacao', 'Coeficiente de variacao das alturas', '%'),
        ('Zkurt', 'Curtose', 'Curtose da distribuicao de alturas', '-'),
        ('Zskew', 'Assimetria', 'Assimetria da distribuicao de alturas', '-'),
    ]

    result = pd.DataFrame(metrics, columns=[
        'Abreviacao', 'Metrica', 'Descricao', 'Unidade'
    ])

    return result


# =============================================================================
# TABELA 5: ESTATISTICAS DAS METRICAS LIDAR
# =============================================================================

def table_lidar_statistics(df_lidar: pd.DataFrame) -> pd.DataFrame:
    """
    Gera tabela de estatisticas descritivas das metricas LiDAR.

    Parameters
    ----------
    df_lidar : pd.DataFrame
        DataFrame com as metricas LiDAR extraidas.

    Returns
    -------
    pd.DataFrame
        Tabela com estatisticas das metricas.
    """
    # Colunas de metricas LiDAR
    metric_cols = [col for col in df_lidar.columns
                   if any(m in col.lower() for m in ['max', 'p90', 'p60', 'mean', 'std', 'kur'])]

    stats_list = []

    for col in metric_cols:
        data = df_lidar[col].dropna()
        if len(data) > 0:
            stats = {
                'Metrica': col,
                'n': len(data),
                'Media': data.mean(),
                'DP': data.std(),
                'CV (%)': (data.std() / data.mean() * 100) if data.mean() != 0 else 0,
                'Min': data.min(),
                'Max': data.max()
            }
            stats_list.append(stats)

    result = pd.DataFrame(stats_list)

    if len(result) > 0:
        for col in ['Media', 'DP', 'Min', 'Max']:
            result[col] = result[col].round(2)
        result['CV (%)'] = result['CV (%)'].round(1)

    return result


# =============================================================================
# TABELA 6: PARAMETROS DO PROCESSAMENTO LIDAR
# =============================================================================

def table_lidar_processing() -> pd.DataFrame:
    """
    Gera tabela com parametros do processamento LiDAR.

    Tabela 5. Parametros utilizados no processamento dos dados LiDAR.
    """
    params = [
        ('Tiling', 'Tamanho do tile', '1000', 'm'),
        ('Tiling', 'Buffer', '50', 'm'),
        ('Denoising', 'Janela de busca', '100', 'm'),
        ('Denoising', 'Pontos isolados', '5', 'pontos'),
        ('Ground', 'Spike tolerance', '0.5', 'm'),
        ('Ground', 'Spike down', '2.5', 'm'),
        ('Thinning', 'Densidade alvo', '5', 'pts/m²'),
        ('Thinning', 'Metodo', 'random', '-'),
        ('DTM', 'Resolucao', '1.0', 'm'),
        ('CHM', 'Resolucao', '1.0', 'm'),
        ('Metricas', 'Resolucao da grade', '17', 'm'),
        ('Metricas', 'Percentis calculados', '60, 90', '-'),
    ]

    result = pd.DataFrame(params, columns=[
        'Etapa', 'Parametro', 'Valor', 'Unidade'
    ])

    return result


# =============================================================================
# TABELA 7: RESULTADOS DO MODELO
# =============================================================================

def table_model_results(
    model_name: str,
    r2: float,
    rmse: float,
    mae: float,
    bias: float,
    n_train: int,
    n_test: int,
    features: List[str]
) -> pd.DataFrame:
    """
    Gera tabela com resultados do modelo preditivo.

    Parameters
    ----------
    model_name : str
        Nome do modelo (ex: 'Random Forest', 'MLP').
    r2 : float
        Coeficiente de determinacao.
    rmse : float
        Raiz do erro quadratico medio.
    mae : float
        Erro absoluto medio.
    bias : float
        Vies medio.
    n_train : int
        Numero de amostras de treino.
    n_test : int
        Numero de amostras de teste.
    features : list
        Lista de features utilizadas.

    Returns
    -------
    pd.DataFrame
        Tabela com metricas do modelo.
    """
    metrics = [
        ('Modelo', model_name, '-'),
        ('Amostras (treino)', str(n_train), 'n'),
        ('Amostras (teste)', str(n_test), 'n'),
        ('R²', f'{r2:.4f}', '-'),
        ('RMSE', f'{rmse:.2f}', 'm³/ha'),
        ('RMSE (%)', f'{(rmse/np.mean([n_train, n_test])*100):.2f}', '%'),
        ('MAE', f'{mae:.2f}', 'm³/ha'),
        ('Bias', f'{bias:.2f}', 'm³/ha'),
        ('Variaveis', str(len(features)), 'n'),
    ]

    result = pd.DataFrame(metrics, columns=['Metrica', 'Valor', 'Unidade'])

    return result


# =============================================================================
# EXPORTADORES
# =============================================================================

def export_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    output_path: str,
    notes: Optional[str] = None
) -> str:
    """
    Exporta DataFrame para formato LaTeX.

    Parameters
    ----------
    df : pd.DataFrame
        Tabela a ser exportada.
    caption : str
        Legenda da tabela.
    label : str
        Label para referencia cruzada.
    output_path : str
        Caminho do arquivo de saida.
    notes : str, optional
        Notas de rodape da tabela.

    Returns
    -------
    str
        Codigo LaTeX da tabela.
    """
    # Gerar LaTeX basico
    latex = df.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'c' * (len(df.columns) - 1)
    )

    # Adicionar ambiente table
    full_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
{latex}"""

    if notes:
        full_latex += f"""\\begin{{tablenotes}}
\\small
\\item {notes}
\\end{{tablenotes}}
"""

    full_latex += "\\end{table}\n"

    # Salvar arquivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_latex)

    return full_latex


def export_to_excel(
    tables: Dict[str, pd.DataFrame],
    output_path: str
) -> None:
    """
    Exporta multiplas tabelas para Excel em abas separadas.

    Parameters
    ----------
    tables : dict
        Dicionario {nome_aba: DataFrame}.
    output_path : str
        Caminho do arquivo de saida.
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in tables.items():
            # Limitar nome da aba a 31 caracteres
            sheet_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Tabelas exportadas para: {output_path}")


def create_great_table(
    df: pd.DataFrame,
    title: str,
    subtitle: Optional[str] = None,
    source_note: Optional[str] = None,
    stub_column: Optional[str] = None
) -> 'GT':
    """
    Cria tabela formatada usando great_tables.

    Parameters
    ----------
    df : pd.DataFrame
        Dados da tabela.
    title : str
        Titulo da tabela.
    subtitle : str, optional
        Subtitulo.
    source_note : str, optional
        Nota de fonte.
    stub_column : str, optional
        Coluna para usar como stub (indice).

    Returns
    -------
    GT
        Objeto great_tables formatado.
    """
    if not HAS_GT:
        raise ImportError("great_tables nao instalado")

    gt = GT(df)

    # Titulo e subtitulo
    gt = gt.tab_header(title=title, subtitle=subtitle)

    # Stub (coluna de indice)
    if stub_column and stub_column in df.columns:
        gt = gt.tab_stub(rowname_col=stub_column)

    # Nota de fonte
    if source_note:
        gt = gt.tab_source_note(source_note)

    return gt


# =============================================================================
# FUNCAO PRINCIPAL
# =============================================================================

def generate_all_tables(
    data_path: str,
    output_dir: str,
    lidar_data_path: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Gera todas as tabelas para o artigo.

    Parameters
    ----------
    data_path : str
        Caminho para os dados do inventario.
    output_dir : str
        Diretorio de saida.
    lidar_data_path : str, optional
        Caminho para dados LiDAR.

    Returns
    -------
    dict
        Dicionario com todas as tabelas geradas.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GERACAO DE TABELAS PARA ARTIGO")
    print("=" * 60)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Saida: {output_dir}")

    # Carregar dados
    print("\n[1/6] Carregando dados...")
    df = load_data(data_path)
    print(f"  Registros: {len(df)}")

    tables = {}

    # Tabela 1: Area de estudo
    print("\n[2/6] Gerando Tabela 1: Caracterizacao da area de estudo...")
    tables['Tab1_Area_Estudo'] = table_study_area(df)

    # Tabela 2: Estatisticas descritivas
    print("[3/6] Gerando Tabela 2: Estatisticas descritivas...")
    tables['Tab2_Estat_Descritivas'] = table_field_statistics(df)

    # Tabela 3: Por regime e regional
    print("[4/6] Gerando Tabela 3: Estatisticas por regime e regional...")
    tables['Tab3_Regime_Regional'] = table_by_regime_regional(df)

    # Tabela 4: Metricas LiDAR
    print("[5/6] Gerando Tabela 4: Descricao das metricas LiDAR...")
    tables['Tab4_Metricas_LiDAR'] = table_lidar_metrics()

    # Tabela 5: Parametros de processamento
    print("[6/6] Gerando Tabela 5: Parametros de processamento LiDAR...")
    tables['Tab5_Param_Processamento'] = table_lidar_processing()

    # Exportar para Excel
    excel_path = output_path / "Tabelas_Artigo.xlsx"
    export_to_excel(tables, str(excel_path))

    # Exportar para LaTeX
    latex_dir = output_path / "latex"
    latex_dir.mkdir(exist_ok=True)

    latex_configs = [
        ('Tab1_Area_Estudo', 'Caracterizacao da area de estudo por regional.', 'tab:area_estudo'),
        ('Tab2_Estat_Descritivas', 'Estatisticas descritivas das variaveis dendrometricas.', 'tab:estat_descr'),
        ('Tab3_Regime_Regional', 'Caracteristicas dendrometricas por regime de manejo e regional.', 'tab:regime_regional'),
        ('Tab4_Metricas_LiDAR', 'Metricas derivadas dos dados LiDAR.', 'tab:metricas_lidar'),
        ('Tab5_Param_Processamento', 'Parametros utilizados no processamento dos dados LiDAR.', 'tab:param_lidar'),
    ]

    for table_name, caption, label in latex_configs:
        latex_path = latex_dir / f"{table_name}.tex"
        export_to_latex(tables[table_name], caption, label, str(latex_path))

    print(f"\n{'=' * 60}")
    print("TABELAS GERADAS COM SUCESSO")
    print(f"{'=' * 60}")
    print(f"  Excel: {excel_path}")
    print(f"  LaTeX: {latex_dir}")

    return tables


# =============================================================================
# EXECUCAO
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gera tabelas formatadas para artigos cientificos"
    )
    parser.add_argument("input", help="Arquivo de dados do inventario (.xlsx ou .csv)")
    parser.add_argument("-o", "--output", default="./tables",
                        help="Diretorio de saida (default: ./tables)")
    parser.add_argument("--lidar", help="Arquivo com dados LiDAR (opcional)")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar tabelas no terminal")

    args = parser.parse_args()

    # Gerar tabelas
    tables = generate_all_tables(
        data_path=args.input,
        output_dir=args.output,
        lidar_data_path=args.lidar
    )

    # Mostrar tabelas se solicitado
    if args.show:
        for name, df in tables.items():
            print(f"\n{name}:")
            print(df.to_string(index=False))
