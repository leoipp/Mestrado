"""
Script para listar arquivos *_denoised_thin_norm.laz e fazer lookup de REF_ID e DATA_PLANTIO
"""
import pandas as pd
import geopandas as gpd
from pathlib import Path


def extract_key_from_filename(filename: str, suffix: str = '_denoised_thin_norm.laz') -> str:
    """Extrai a chave do nome do arquivo (tudo antes do sufixo)."""
    if filename.endswith(suffix):
        return filename[:-len(suffix)]
    return filename.split('_')[0]


def list_laz_files_with_plantio(
    laz_folder: str,
    shapefile_path: str,
    lookup_xlsx: str,
    output_xlsx: str,
    shp_key_col: str = 'Chave',
    shp_ref_col: str = 'REF_ID',
    xlsx_key_col: str = 'Projeto',
    xlsx_date_col: str = 'Data de Plantio',
    laz_pattern: str = '*_denoised_thin_norm.laz'
) -> pd.DataFrame:
    """
    Lista arquivos LAZ, busca REF_ID no shapefile e Data de Plantio no xlsx.

    Args:
        laz_folder: Pasta raiz contendo os arquivos LAZ
        shapefile_path: Caminho do shapefile com Chave e REF_ID
        lookup_xlsx: Arquivo xlsx com Projeto e Data de Plantio
        output_xlsx: Caminho do arquivo xlsx de saída
        shp_key_col: Coluna chave no shapefile
        shp_ref_col: Coluna REF_ID no shapefile
        xlsx_key_col: Coluna chave no xlsx (Projeto)
        xlsx_date_col: Coluna data de plantio no xlsx
        laz_pattern: Padrão para encontrar arquivos LAZ

    Returns:
        DataFrame com os resultados
    """
    # Ler shapefile DBF diretamente (evita erros de data invalida)
    from dbfread import DBF
    dbf_path = shapefile_path.replace('.shp', '.dbf')
    records = [record for record in DBF(dbf_path, ignore_missing_memofile=True)]
    gdf = pd.DataFrame(records)
    shp_lookup = dict(zip(gdf[shp_key_col].astype(str), gdf[shp_ref_col].astype(str)))

    # Ler xlsx de cadastro
    df_xlsx = pd.read_excel(lookup_xlsx)
    xlsx_lookup = dict(zip(df_xlsx[xlsx_key_col].astype(str), df_xlsx[xlsx_date_col]))

    # Listar arquivos LAZ
    laz_folder_path = Path(laz_folder)
    results = []

    for laz_file in laz_folder_path.rglob(laz_pattern):
        filename = laz_file.name
        chave = extract_key_from_filename(filename)

        # Extrair ano da pasta (ex: 2019, 2022, 2024, 2025)
        rel_path = laz_file.relative_to(laz_folder_path)
        ano = rel_path.parts[0] if rel_path.parts else ''

        # Lookup 1: Chave -> REF_ID (shapefile)
        ref_id = shp_lookup.get(chave, '')

        # Lookup 2: REF_ID -> Data de Plantio (xlsx)
        data_plantio = xlsx_lookup.get(ref_id, '') if ref_id else ''

        results.append({
            'CAMINHO': str(laz_file),
            'NOME_ARQUIVO': filename,
            'ANO': ano,
            'CHAVE': chave,
            'REF_ID': ref_id,
            'DATA_PLANTIO': data_plantio
        })

    # Criar DataFrame
    df_result = pd.DataFrame(results)

    # Salvar xlsx
    df_result.to_excel(output_xlsx, index=False)
    print(f"Arquivo salvo: {output_xlsx}")
    print(f"Total de arquivos encontrados: {len(df_result)}")
    print(f"Arquivos com REF_ID: {(df_result['REF_ID'] != '').sum()}")
    print(f"Arquivos com DATA_PLANTIO: {df_result['DATA_PLANTIO'].notna().sum()}")

    return df_result


if __name__ == '__main__':
    # Configuracoes
    LAZ_FOLDER = r'G:\PycharmProjects\Mestrado\Forecast\Projection\PROJECAO'
    SHAPEFILE = r'G:\PycharmProjects\Mestrado\Forecast\Projection\PROJECAO\Shapefiles\PROJECAO.shp'
    LOOKUP_XLSX = r'G:\PycharmProjects\Mestrado\Data\DataFrames\Cadastro_florestal.xlsx'
    OUTPUT_XLSX = r'G:\PycharmProjects\Mestrado\Data\DataFrames\laz_files_plantio_v2.xlsx'

    df = list_laz_files_with_plantio(
        laz_folder=LAZ_FOLDER,
        shapefile_path=SHAPEFILE,
        lookup_xlsx=LOOKUP_XLSX,
        output_xlsx=OUTPUT_XLSX,
        shp_key_col='Chave',
        shp_ref_col='REF_ID',
        xlsx_key_col='Talhão',
        xlsx_date_col='Data de Plantio'
    )

    print("\nPrimeiras linhas do resultado:")
    print(df.head(10))
