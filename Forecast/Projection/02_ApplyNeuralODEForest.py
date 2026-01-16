#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ApplyNeuralODEForest.py
=========================

Script SOMENTE de aplicacao (inferencia) do modelo Neural ODE.

O que faz:
- Carrega um checkpoint .pth (ode_model_best.pth / ode_model_final.pth)
- Carrega um arquivo LAS/LAZ de referencia (ja normalizado: z = altura acima do solo)
- Filtra pontos com z > MIN_HEIGHT
- Recalcula features locais (KNN) para os pontos
- Integra a ODE da idade do arquivo (ou idade informada) ate a idade alvo
- Salva um LAS com a nuvem projetada (mesmo XY, novo Z)

Exemplos:
  python apply_neural_ode_forest.py \
      --model "G:/PycharmProjects/Mestrado/out_ode/ode_model_best.pth" \
      --in_las "G:/PycharmProjects/Mestrado/Forecast/Projection/PROJECAO/2023/REF_210_70_denoised_thin_norm.laz" \
      --age_src 70 \
      --age_tgt 120 \
      --out_dir "G:/PycharmProjects/Mestrado/out_ode"

  # Se o nome do arquivo tiver a idade no padrao *_TILE_IDADE_denoised_thin_norm.laz,
  # voce pode omitir --age_src:
  python apply_neural_ode_forest.py --model ... --in_las ... --age_tgt 120
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import numpy.random as npr
from tqdm import tqdm
import laspy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError as e:
    raise SystemExit("Instale: pip install torchdiffeq") from e

from sklearn.neighbors import NearestNeighbors


# -------------------- DEFAULTS (ajuste se quiser) --------------------
KNN = 16
MIN_HEIGHT = 0.3
BATCH_PTS = 32768          # pontos por batch na inferencia
N_STEPS = 5                # passos intermediarios do solver na inferencia
HIDDEN_DIM = 128           # deve bater com o treino (no seu codigo: 128)
USE_ADJOINT_IN_INFER = False  # em inferencia normalmente nao precisa adjoint


# =====================================================================
# I/O LAS
# =====================================================================

def load_las_points(las_path: Path) -> dict:
    p = str(las_path)

    def to_np(a, dtype=None):
        arr = np.asarray(a)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.ascontiguousarray(arr)

    try:
        if p.lower().endswith(".laz"):
            try:
                with laspy.open(p, laz_backend=laspy.LazBackend.LazrsParallel) as f:
                    las = f.read()
            except Exception:
                with laspy.open(p, laz_backend=laspy.LazBackend.Lazrs) as f:
                    las = f.read()
        else:
            las = laspy.read(p)
    except Exception as e:
        raise SystemExit(f"[ERRO] Falha ao ler LAS/LAZ: {p}\n{e}")

    intensity = getattr(las, "intensity", None)
    if intensity is None:
        intensity = np.zeros_like(las.x)

    return {
        "x": to_np(las.x, np.float64),
        "y": to_np(las.y, np.float64),
        "z": to_np(las.z, np.float64),  # ja normalizado!
        "intensity": to_np(intensity, np.float32),
        "hdr": las.header,
    }


def parse_age_from_filename(las_path: Path) -> int | None:
    """
    Tenta extrair idade (meses) do padrao:
      REF_TILE_IDADE_denoised_thin_norm.laz
    Ex.: GNVIVI00580P18-002_210_70_denoised_thin_norm.laz -> 70
    """
    pat = re.compile(r"(.+)_(\d+)_(\d+)_denoised_thin_norm\.(laz|las)$", re.IGNORECASE)
    m = pat.match(las_path.name)
    if not m:
        return None
    return int(m.group(3))


# =====================================================================
# FEATURES LOCAIS
# =====================================================================

def compute_local_feats(X: np.ndarray,
                        z: np.ndarray,
                        intensity: np.ndarray,
                        k: int = KNN) -> np.ndarray:
    """
    Retorna features [N,5]:
      densidade, normal_z, var_z, rank, intensity_norm
    """
    n = len(X)
    if n < 3:
        # fallback simples
        dens = np.ones(n, dtype=np.float32)
        normals_z = np.ones(n, dtype=np.float32)
        var_z = np.zeros(n, dtype=np.float32)
        rank = np.zeros(n, dtype=np.float32)
        inten = intensity / (intensity.max() + 1e-6) if intensity.max() > 0 else intensity
        return np.stack([dens, normals_z, var_z, rank, inten.astype(np.float32)], axis=1).astype(np.float32)

    k_eff = int(min(k, n))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree").fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)

    dens = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-6)

    normals_z = np.zeros(n, dtype=np.float32)
    var_z = np.zeros(n, dtype=np.float32)

    # PCA por ponto (custa, mas ok)
    for i in range(n):
        P = X[idxs[i]]
        if len(P) > 2:
            C = np.cov(P.T)
            w, v = np.linalg.eigh(C)
            nvec = v[:, np.argmin(w)]
            normals_z[i] = float(abs(nvec[2]))
        var_z[i] = float(z[idxs[i]].var()) if len(idxs[i]) > 1 else 0.0

    rank = np.zeros(n, dtype=np.float32)
    for i in range(n):
        z_loc = z[idxs[i]]
        m = float(z_loc.max() - z_loc.min())
        rank[i] = 0.0 if m < 1e-6 else float((z[i] - z_loc.min()) / m)

    inten = intensity / (intensity.max() + 1e-6) if intensity.max() > 0 else intensity
    return np.stack([dens, normals_z, var_z, rank, inten.astype(np.float32)], axis=1).astype(np.float32)


# =====================================================================
# MODELO (igual ao seu, simplificado em z)
# =====================================================================

class GrowthDynamics(nn.Module):
    def __init__(self, feat_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        input_dim = 1 + 1 + feat_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.net[-1].bias.data = torch.tensor([0.1])

    def forward(self, t, state):
        z = state[..., :1]
        feats = state[..., 1:]

        B, N, _ = z.shape
        t_val = t.view(1, 1, 1).expand(B, N, 1)

        inp = torch.cat([z, t_val, feats], dim=-1)
        dz = F.softplus(self.net(inp))  # positivo

        height_factor = 1.0 / (1.0 + z / 20.0)
        dz = dz * height_factor

        dfeats = torch.zeros_like(feats)
        return torch.cat([dz, dfeats], dim=-1)


class NeuralODEForest(nn.Module):
    def __init__(self, feat_dim: int = 5, hidden_dim: int = 128, use_adjoint: bool = True):
        super().__init__()
        self.dynamics = GrowthDynamics(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.use_adjoint = use_adjoint
        self.ode_solver = odeint_adjoint if use_adjoint else odeint

    def forward(self, z, features, t_span):
        state0 = torch.cat([z, features], dim=-1)
        traj = self.ode_solver(
            self.dynamics,
            state0,
            t_span,
            method="dopri5",
            rtol=1e-3,
            atol=1e-4,
            options={"max_num_steps": 1000},
        )
        return traj[..., :1]


# =====================================================================
# APLICACAO
# =====================================================================

def infer(model_path: Path,
          in_las: Path,
          age_tgt: float,
          age_src: float | None,
          out_dir: Path,
          knn: int = KNN,
          min_height: float = MIN_HEIGHT,
          batch_pts: int = BATCH_PTS,
          n_steps: int = N_STEPS,
          device: str | None = None) -> Path:

    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    print(f"[INFO] Device: {dev}")

    # carregar checkpoint
    ckpt = torch.load(str(model_path), map_location=dev, weights_only=False)
    if "model" not in ckpt or "age_min" not in ckpt or "age_range" not in ckpt:
        raise SystemExit("[ERRO] Checkpoint invalido. Esperado chaves: model, age_min, age_range")

    age_min = float(ckpt["age_min"])
    age_range = float(ckpt["age_range"])

    model = NeuralODEForest(feat_dim=5, hidden_dim=HIDDEN_DIM, use_adjoint=USE_ADJOINT_IN_INFER).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # carregar nuvem
    P = load_las_points(in_las)

    # idade origem
    if age_src is None:
        age_src = parse_age_from_filename(in_las)
    if age_src is None:
        raise SystemExit("[ERRO] Nao consegui inferir a idade do arquivo. Passe --age_src <meses>.")

    age_src = float(age_src)
    age_tgt = float(age_tgt)

    if age_range <= 0:
        raise SystemExit("[ERRO] age_range do checkpoint invalido.")

    t0 = (age_src - age_min) / age_range
    t1 = (age_tgt - age_min) / age_range

    print(f"[INFO] Fonte: {in_las.name}")
    print(f"[INFO] Idade src = {age_src:.1f} meses | Idade tgt = {age_tgt:.1f} meses")
    print(f"[INFO] t0={t0:.4f} | t1={t1:.4f}")

    # filtrar pontos de vegetacao
    mask = P["z"] > float(min_height)
    x = P["x"][mask].astype(np.float32)
    y = P["y"][mask].astype(np.float32)
    z = P["z"][mask].astype(np.float32)
    inten = P["intensity"][mask].astype(np.float32)

    if len(z) == 0:
        raise SystemExit("[ERRO] Nenhum ponto com z > MIN_HEIGHT. Ajuste --min_height.")

    # features
    Xv = np.c_[x, y, z].astype(np.float32)
    print("[INFO] Computando features locais...")
    Fv = compute_local_feats(Xv, z, inten, k=int(min(knn, len(Xv))))

    # integrar em batches
    z_new = np.zeros_like(z, dtype=np.float32)
    t_span = torch.linspace(float(t0), float(t1), int(n_steps), device=dev)

    print("[INFO] Projetando...")
    with torch.no_grad():
        for s in tqdm(range(0, len(z), int(batch_pts)), desc="Infer"):
            e = min(s + int(batch_pts), len(z))
            z_batch = torch.from_numpy(z[s:e].reshape(-1, 1)).float().unsqueeze(0).to(dev)
            f_batch = torch.from_numpy(Fv[s:e]).float().unsqueeze(0).to(dev)

            z_traj = model(z_batch, f_batch, t_span)
            z_pred = z_traj[-1, 0, :, 0].detach().cpu().numpy().astype(np.float32)
            z_pred = np.maximum(z_pred, float(min_height))
            z_new[s:e] = z_pred

    dz = z_new - z
    print("[STATS] dz (m): "
          f"min={dz.min():.3f} | max={dz.max():.3f} | mediana={np.median(dz):.3f} | media={dz.mean():.3f}")

    # salvar LAS
    ref_hdr = P["hdr"]
    hdr = laspy.LasHeader(point_format=ref_hdr.point_format, version=ref_hdr.version)
    hdr.scales = ref_hdr.scales
    hdr.offsets = ref_hdr.offsets
    try:
        crs = ref_hdr.parse_crs()
        if crs is not None:
            hdr.add_crs(crs)
    except Exception:
        pass

    las_out = laspy.LasData(hdr)
    las_out.x = x
    las_out.y = y
    las_out.z = z_new

    # classificação simples por altura
    cls = np.where(z_new < 2.0, 3, np.where(z_new < 5.0, 4, 5)).astype(np.uint8)
    las_out.classification = cls

    # intensidade (reescala para uint16)
    vint = (inten / (inten.max() + 1e-6) * 100).astype(np.uint16)
    las_out.intensity = vint

    n = len(las_out.x)
    las_out.return_number = np.ones(n, dtype=np.uint8)
    las_out.number_of_returns = np.ones(n, dtype=np.uint8)

    out_path = out_dir / f"{in_las.stem}__proj_{int(age_tgt)}m.las"
    las_out.write(str(out_path))
    print(f"[OK] Salvo: {out_path}")

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Aplicacao do Neural ODE Forest (somente inferencia)")
    ap.add_argument("--model", required=True, type=str, help="Caminho do checkpoint .pth")
    ap.add_argument("--in_las", required=True, type=str, help="LAS/LAZ de entrada (z ja normalizado)")
    ap.add_argument("--age_tgt", required=True, type=float, help="Idade alvo (meses)")
    ap.add_argument("--age_src", required=False, type=float,
                    help="Idade da nuvem de entrada (meses). Se omitido, tenta extrair do nome do arquivo.")
    ap.add_argument("--out_dir", required=True, type=str, help="Diretorio de saida")
    ap.add_argument("--knn", default=KNN, type=int, help="KNN para features")
    ap.add_argument("--min_height", default=MIN_HEIGHT, type=float, help="Altura minima (m)")
    ap.add_argument("--batch_pts", default=BATCH_PTS, type=int, help="Pontos por batch")
    ap.add_argument("--n_steps", default=N_STEPS, type=int, help="Passos intermediarios do solver")
    ap.add_argument("--device", default=None, type=str, help="cuda | cpu | cuda:0 ... (opcional)")
    args = ap.parse_args()

    infer(
        model_path=Path(args.model),
        in_las=Path(args.in_las),
        age_tgt=args.age_tgt,
        age_src=args.age_src,
        out_dir=Path(args.out_dir),
        knn=args.knn,
        min_height=args.min_height,
        batch_pts=args.batch_pts,
        n_steps=args.n_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
