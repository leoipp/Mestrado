# utf-8
# usage: python Delta.py file01.tif file02.tif output_delta.tif
import rasterio
import numpy as np

def delta_raster(file01:str, file02:str, output_path:str) -> None:
    """
    Calculate the delta between two rasters and save the result to a new file.
    :param file01: File path for the first raster.
    :param file02: File path for the second raster.
    :param output_path: Path to save the output raster.
    :return: None
    """
    # Open the first and second raster files
    with rasterio.open(file01) as src1, rasterio.open(file02) as src2:
        # Check metadata compatibility
        if (src1.crs != src2.crs or
            src1.transform != src2.transform or
            src1.width != src2.width or
            src1.height != src2.height):
            print(f"Skipping {file01}: incompatible rasters")
            return

        # Read the data
        data_1 = src1.read(1).astype(np.float32)
        data_2 = src2.read(1).astype(np.float32)

        # Handle division and no-data
        mask = (data_1 == src1.nodata) | (data_2 == src2.nodata) | (data_1 == 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta = np.where(~mask, (data_2 - data_1) / data_1, np.nan)

        # Update profile for output raster
        profile = src1.profile
        profile.update(
            dtype=rasterio.float32,
            nodata=np.nan
        )

        # Save the delta raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(delta, 1)
