import laspy, json, math
from pathlib import Path

# Edite aqui:
DATA_DIR = Path(r"G:/PycharmProjects//Mestrado")
EPOCAS = [
    {"idade": 2.0,  "las": DATA_DIR/"RDBOBA00456P17-267_759_denoised_thin_norm_2019.laz"},
    {"idade": 5.0,  "las": DATA_DIR/"RDBOBA00456P17-267_759_denoised_thin_norm_2022.laz"},
    {"idade": 8.0,  "las": DATA_DIR/"RDBOBA00456P17-267_759_denoised_thin_norm_2025.laz"}
]

def inspect_las_meta(paths):
    import laspy, numpy as np
    def bbox(a): return float(np.min(a)), float(np.max(a))
    for p in paths:
        las = laspy.read(str(p))
        try:
            crs = las.header.parse_crs()
            crs_str = crs.to_wkt()[:60] + "..." if crs else "None"
        except Exception:
            crs_str = "parse_crs failed"
        print(f"\n== {p}")
        print("version:", las.header.version, "  point_format:", las.header.point_format.id)
        print("scales:", las.header.scales, " offsets:", las.header.offsets)
        print("CRS:", crs_str)
        print("bbox X:", bbox(las.x), "Y:", bbox(las.y), "Z:", bbox(las.z))

# use:
inspect_las_meta([e["las"] for e in EPOCAS])