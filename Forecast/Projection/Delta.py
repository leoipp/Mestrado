# utf-8
# usage: python Delta.py file01.tif file02.tif output_delta.tif
import rasterio
import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime


def match_raster_and_df(
        raster_path: str,
        df: pd.DataFrame,
        nome_col: str = "Talhão",
        idade_col: str = "Data de Plantio",
        referencia_data: str = "2019-12-14",  # Data de referência como argumento
        unidade: str = "anos"  # Unidade para a diferença: 'anos' ou 'meses'
) -> float:
    """
    Lê um raster, extrai o nome do arquivo, formata-o removendo os dois últimos caracteres e
    substitui "_" por "-", busca no DataFrame pela coluna 'nome' e retorna a idade
    (diferença entre a data de plantio e a data de referência em anos ou meses).

    :param raster_path: Caminho para o raster.
    :param df: DataFrame contendo as colunas 'nome' e 'idade' (data de plantio).
    :param nome_col: Nome da coluna no DataFrame que corresponde ao nome do raster.
    :param idade_col: Nome da coluna com os valores de idade (data de plantio).
    :param referencia_data: Data de referência para cálculo da idade.
    :param unidade: Unidade para a diferença ('anos' ou 'meses').
    :return: Valor da idade calculada (diferença em anos ou meses) ou None se não encontrado.
    """
    # Extrai o nome base e formata
    raster_name = os.path.splitext(os.path.basename(raster_path))[0]
    formatted_name = raster_name[:-2].replace("_", "-")

    # Busca no DataFrame
    match = df[df[nome_col] == formatted_name]

    if not match.empty:
        # Obtém a data de plantio do DataFrame
        data_plantio = match.iloc[0][idade_col]

        # Converte a data de referência para datetime
        referencia_data = datetime.strptime(referencia_data, "%Y-%m-%d")

        # Verifica se a data de plantio é válida
        if isinstance(data_plantio, pd.Timestamp):
            # Calcula a diferença entre as duas datas
            delta = referencia_data - data_plantio

            if unidade == "anos":
                idade = delta.days / 365.25  # Calculando a idade em anos (ano bissexto é considerado)
                return idade
            elif unidade == "meses":
                idade = delta.days / 30.44  # Calculando a idade em meses (média de dias por mês)
                return idade
            else:
                print("Unidade inválida. Use 'anos' ou 'meses'.")
                return None
        else:
            print("Data de plantio inválida.")
            return None
    else:
        print(f"Nome '{formatted_name}' não encontrado no DataFrame.")
        return None


def delta_raster(
        file01: str,
        file02: str,
        output_path: str,
        df: pd.DataFrame,
        referencia_data: str = "2019-12-14",
        unidade: str = "anos"
) -> None:
    """
    Calculate the delta between two rasters and save the result to a new 4-band raster:
    Band 1: data_1
    Band 2: data_2
    Band 3: delta = (data_2 - data_1) / data_1
    Band 4: idade = Difference between plant date and reference date (in years or months)

    :param file01: File path for the first raster.
    :param file02: File path for the second raster.
    :param output_path: Path to save the output raster.
    :param df: DataFrame containing the plant data.
    :param referencia_data: Reference date for the age calculation.
    :param unidade: Unit for the difference ('anos' or 'meses').
    :return: None
    """
    with rasterio.open(file01) as src1, rasterio.open(file02) as src2:
        # Check compatibility of rasters
        if (src1.crs != src2.crs or
            src1.transform != src2.transform or
            src1.width != src2.width or
            src1.height != src2.height):
            print(f"Skipping {file01}: incompatible rasters")
            return

        # Data reader
        data_1 = src1.read(1).astype(np.float32)
        data_2 = src2.read(1).astype(np.float32)

        # Delta calculation
        mask = (data_1 == src1.nodata) | (data_2 == src2.nodata) | (data_1 == 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = np.where(~mask, (data_2 - data_1) / data_1, np.nan)

        # Calculate the age (idade) using match_raster_and_df
        idade = match_raster_and_df(file01, df, referencia_data=referencia_data, unidade=unidade)

        # Profile update for the 4-band raster
        profile = src1.profile
        profile.update(
            count=4,  # We now have 4 bands
            dtype=rasterio.float32,
            nodata=np.nan
        )

        # Write the output raster with 4 bands
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data_1, 1)  # Band 1
            dst.write(data_2, 2)  # Band 2
            dst.write(delta, 3)   # Band 3 (delta)
            dst.write(np.full_like(data_1, idade, dtype=np.float32), 4)  # Band 4 (idade)


def add_age_band(raster_path, output_path, df, referencia_data='2019-12-14'):
    """
    Adiciona uma banda com a idade em meses ao raster original.
    A idade é obtida via função `match_raster_and_df`.

    :param raster_path: Caminho do raster original
    :param output_path: Caminho para salvar o novo raster com a banda extra
    :param df: DataFrame contendo dados de plantio
    :param referencia_data: Data da imagem usada como referência
    """
    idade_meses = match_raster_and_df(
        raster_path=raster_path,
        df=df,
        nome_col="Talhão",
        idade_col="Data de Plantio",
        referencia_data=referencia_data,
        unidade="meses"
    )

    if idade_meses is None:
        print(f"[WARN] Não foi possível calcular idade para {raster_path}")
        return

    # Abrir o raster original
    with rasterio.open(raster_path) as src:
        profile = src.profile
        data = src.read(1)

        # Atualizar perfil para 2 bandas
        profile.update(count=2, dtype='float32')

        # Criar banda de idade com o mesmo shape
        idade_band = np.full_like(data, idade_meses, dtype='float32')

        # Escrever novo raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data.astype('float32'), 1)
            dst.write(idade_band, 2)

    print(f"[OK] Raster com idade salvo: {output_path}")


if __name__ == "__main__":
    # --- DELTA RASTER ---
    # src_2022 = r'G:\PycharmProjects\Mestrado\Data\Projection\2022-2024\2022\Talhoes\*.tif'
    # src_2024 = r'G:\PycharmProjects\Mestrado\Data\Projection\2022-2024\2024\Talhoes\*.tif'
    # output_dir = r'G:\PycharmProjects\Mestrado\Data\Projection\2022-2024\delta'
    # df = pd.read_excel(r'G:\PycharmProjects\Mestrado\Data\Cadastro_florestal.xlsx')
    # # Get the list of files for 01 and 02
    # files_2022 = {os.path.basename(f): f for f in glob.glob(src_2022)}
    # files_2024 = {os.path.basename(f): f for f in glob.glob(src_2024)}
    #
    # # Common files between src01 and src02
    # common_files = set(files_2022.keys()) & set(files_2024.keys())
    #
    # # Loop through common files and calculate delta
    # for name in common_files:
    #     file_2022 = files_2022[name]
    #     file_2024 = files_2024[name]
    #
    #     # Calculate the delta raster
    #     delta_raster(file_2022, file_2024, os.path.join(output_dir, name), df)

    src_2024 = r'/Data/Projection\2022-2024\2024\Talhoes\*.tif'
    output_dir = r'/Data/Projection/2022-2024/2024/Talhoes_id01'
    df = pd.read_excel(r'G:\PycharmProjects\Mestrado\Data\Cadastro_florestal.xlsx')

    os.makedirs(output_dir, exist_ok=True)

    files_2024 = glob.glob(src_2024)

    for raster_path in files_2024:
        output_path = os.path.join(output_dir, os.path.basename(raster_path))
        add_age_band(raster_path, output_path, df)