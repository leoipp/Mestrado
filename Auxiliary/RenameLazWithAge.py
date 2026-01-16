"""
Script para renomear arquivos LAZ usando coluna NEW_NAME do xlsx.
"""
import pandas as pd
from pathlib import Path


def rename_laz_files(
    xlsx_path: str,
    caminho_col: str = 'CAMINHO',
    new_name_col: str = 'NEW_NAME',
    dry_run: bool = True
) -> pd.DataFrame:
    """
    Renomeia arquivos LAZ usando o nome definido na coluna NEW_NAME.

    Args:
        xlsx_path: Caminho do arquivo xlsx
        caminho_col: Nome da coluna com o caminho atual do arquivo
        new_name_col: Nome da coluna com o novo nome do arquivo
        dry_run: Se True, apenas mostra o que seria feito sem renomear

    Returns:
        DataFrame com os caminhos antigos e novos
    """
    df = pd.read_excel(xlsx_path)

    results = []
    renamed_count = 0
    error_count = 0
    skip_count = 0

    for _, row in df.iterrows():
        caminho_antigo = Path(row[caminho_col])
        new_name = row[new_name_col]

        # Pular se NEW_NAME for vazio/NaN
        if pd.isna(new_name) or str(new_name).strip() == '':
            results.append({
                'CAMINHO_ANTIGO': str(caminho_antigo),
                'CAMINHO_NOVO': '',
                'STATUS': 'SKIP - NEW_NAME vazio'
            })
            skip_count += 1
            continue

        # Construir novo caminho (mesma pasta, novo nome)
        caminho_novo = caminho_antigo.parent / str(new_name)

        # Pular se o nome for igual
        if caminho_antigo.name == str(new_name):
            results.append({
                'CAMINHO_ANTIGO': str(caminho_antigo),
                'CAMINHO_NOVO': str(caminho_novo),
                'STATUS': 'SKIP - Nome igual'
            })
            skip_count += 1
            continue

        status = ''
        if dry_run:
            status = 'DRY_RUN'
        else:
            try:
                if caminho_antigo.exists():
                    caminho_antigo.rename(caminho_novo)
                    status = 'OK'
                    renamed_count += 1
                else:
                    status = 'ERRO - Arquivo nao encontrado'
                    error_count += 1
            except Exception as e:
                status = f'ERRO - {str(e)}'
                error_count += 1

        results.append({
            'CAMINHO_ANTIGO': str(caminho_antigo),
            'CAMINHO_NOVO': str(caminho_novo),
            'STATUS': status
        })

    df_result = pd.DataFrame(results)

    print(f"Total de arquivos: {len(df_result)}")
    print(f"Ignorados (skip): {skip_count}")
    if dry_run:
        print("MODO DRY_RUN - Nenhum arquivo foi renomeado")
        print(f"Arquivos a renomear: {len(df_result) - skip_count}")
    else:
        print(f"Arquivos renomeados: {renamed_count}")
        print(f"Erros: {error_count}")

    return df_result


if __name__ == '__main__':
    XLSX_PATH = r'G:\PycharmProjects\Mestrado\Data\DataFrames\laz_files_plantio_v2.xlsx'

    # Primeiro executa em dry_run para verificar
    print("=== DRY RUN (simulacao) ===\n")
    df = rename_laz_files(
        xlsx_path=XLSX_PATH,
        caminho_col='CAMINHO',
        new_name_col='NEW_NAME',
        dry_run=False
    )

    print("\nPrimeiras linhas:")
    print(df.head(10).to_string())

    # Para executar de verdade, mude dry_run=False
    # print("\n=== RENOMEANDO ARQUIVOS ===\n")
    # df = rename_laz_files(
    #     xlsx_path=XLSX_PATH,
    #     caminho_col='CAMINHO',
    #     new_name_col='NEW_NAME',
    #     dry_run=False
    # )
