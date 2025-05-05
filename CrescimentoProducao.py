import os
import glob
from .Delta import delta_raster



src_2022 = r'G:\PycharmProjects\Mestrado\Data\Crescimento_e_Producao\2022-2024\2022\Talhoes\*.tif'
src_2024 = r'G:\PycharmProjects\Mestrado\Data\Crescimento_e_Producao\2022-2024\2024\Talhoes\*.tif'
output_dir = r'G:\PycharmProjects\Mestrado\Data\Crescimento_e_Producao\2022-2024\delta'

def main() -> None:
    """
    Main function to compare rasters.
    :return:
    """
    # Get the list of files for 01 and 02
    files_2022 = {os.path.basename(f): f for f in glob.glob(src_2022)}
    files_2024 = {os.path.basename(f): f for f in glob.glob(src_2024)}

    # Common files between src01 and src02
    common_files = set(files_2022.keys()) & set(files_2024.keys())

    # Loop through common files and calculate delta
    for name in common_files:
        file_2022 = files_2022[name]
        file_2024 = files_2024[name]

        # Calculate the delta raster
        delta_raster(file_2022, file_2024, os.path.join(output_dir, name))

if __name__ == '__main__':
    main()