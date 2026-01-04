#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============== CONFIG ===================================
from pathlib import Path
import numpy as np
import numpy.random as npr
from tqdm import tqdm
import laspy, json, math

# Edite aqui:
DATA_DIR = Path(r"G:/PycharmProjects//Mestrado")
EPOCAS = [
    {"idade": 2.0,  "las": DATA_DIR/"RDBOBA00456P17-267_759_denoised_thin_norm_2019.laz"},
    {"idade": 5.0,  "las": DATA_DIR/"RDBOBA00456P17-267_759_denoised_thin_norm_2022.laz"},
    {"idade": 8.0,  "las": DATA_DIR/"RDBOBA00456P17-267_759_denoised_thin_norm_2025.laz"}
]
REF_INDEX = 'last'
ID_ALVO   = 10   # idade alvo (anos)
KNN = 16          # vizinhos para featurização
MAX_SAMP = 200000 # amostra máx. de pontos por época (controlar memória)
LR = 1e-3
EPOCHS = 20
BATCH_PTS = 32768 # pontos por batch (amostra) no treino
OUT_DIR = DATA_DIR/"out_flow"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _validate_and_sort_epocas(epocas):
    """Filtra apenas as épocas com arquivo existente e ordena por idade."""
    valid = []
    missing = []
    for e in epocas:
        p = Path(e["las"])
        if p.exists():
            valid.append({"idade": float(e["idade"]), "las": p})
        else:
            missing.append(str(p))
    if missing:
        print(f"[AVISO] Ignorando {len(missing)} arquivo(s) inexistente(s):")
        for m in missing[:10]:
            print("  -", m)
        if len(missing) > 10:
            print("  ...")
    if len(valid) < 2:
        raise SystemExit("[ERRO] São necessárias pelo menos 2 épocas válidas (.LAS) para treinar. "
                         "Atualize EPOCAS com caminhos corretos.")
    # ordena por idade ascendente
    valid = sorted(valid, key=lambda d: d["idade"])
    return valid

def _resolve_ref_index(epocas_validas, ref_index):
    """Garante que REF_INDEX aponte para uma época válida."""
    n = len(epocas_validas)
    if isinstance(ref_index, str) and ref_index.lower() == 'last':
        return n - 1
    if isinstance(ref_index, int):
        if -n <= ref_index < n:
            return ref_index % n
        else:
            print(f"[AVISO] REF_INDEX={ref_index} inválido para n={n}. Usando última época.")
            return n - 1
    print("[AVISO] REF_INDEX não definido/inesperado. Usando última época.")
    return n - 1

# =============== UTILS: DTM_ref via ground da referência ===
from scipy.spatial import cKDTree

class DTMRef:
    """DTM por kNN IDW sobre ground (class=2) da época de referência."""
    def __init__(self, xg, yg, zg, k=3):
        self.pts = np.c_[xg, yg]
        self.z   = zg
        self.kdt = cKDTree(self.pts)
        self.k = k
    def z_at(self, x, y):
        q = np.c_[x, y]
        d, idx = self.kdt.query(q, k=min(self.k, len(self.z)))
        if np.ndim(idx) == 0:
            return float(self.z[idx])
        zz = self.z[idx]
        w = 1.0 / (d + 1e-6)
        w /= w.sum(axis=-1, keepdims=True)
        return (zz * w).sum(axis=-1)

def load_las_points(las_path):
    import numpy as np
    import laspy
    from laspy import errors as laspy_errors

    p = str(las_path)
    try:
        if p.lower().endswith(".laz"):
            # força backend lazrs (paralelo -> normal)
            try:
                with laspy.open(p, laz_backend=laspy.LazBackend.LazrsParallel) as f:
                    las = f.read()
            except Exception:
                with laspy.open(p, laz_backend=laspy.LazBackend.Lazrs) as f:
                    las = f.read()
        else:
            las = laspy.read(p)
    except laspy_errors.LaspyException as e:
        raise SystemExit(
            f"[ERRO] Backend LAZ ausente. Instale: pip install 'laspy[lazrs]'\nArquivo: {p}\n{e}"
        )

    # ---- CONVERSÃO ROBUSTA p/ NumPy (escapa ScaledArrayView) ----
    def to_np(a, dtype=None):
        # np.asarray lida com ScaledArrayView; .copy() garante array independente
        arr = np.asarray(a)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.ascontiguousarray(arr)  # seguro para KDTree, torch, etc.

    x = to_np(las.x, dtype=np.float64)
    y = to_np(las.y, dtype=np.float64)
    z = to_np(las.z, dtype=np.float64)

    # Estes já são arrays normais, mas padronizamos:
    intensity      = to_np(getattr(las, "intensity", np.zeros_like(x, dtype=np.uint16)), dtype=np.float32)
    classification = to_np(getattr(las, "classification", np.ones_like(x, dtype=np.uint8)), dtype=np.uint8)

    return {
        "x": x, "y": y, "z": z,
        "intensity": intensity,
        "classification": classification,
        "hdr": las.header,
    }



def build_dtm_ref(epocas_validas, ref_idx):
    Pref = load_las_points(epocas_validas[ref_idx]["las"])
    is_ground = (Pref["classification"] == 2)
    if not is_ground.any():
        raise SystemExit("[ERRO] A época de referência não possui pontos class=2 (ground). "
                         "Classifique o solo ou escolha outra referência.")
    Gx, Gy, Gz = Pref["x"][is_ground], Pref["y"][is_ground], Pref["z"][is_ground]
    dtm = DTMRef(Gx, Gy, Gz, k=3)
    np.savez(OUT_DIR/"dtm_ref.npz", x=Gx, y=Gy, z=Gz)
    return dtm, Pref

def load_dtm_ref():
    d = np.load(OUT_DIR/"dtm_ref.npz")
    return DTMRef(d["x"], d["y"], d["z"], k=3)

# =============== FEATURIZAÇÃO LOCAL (kNN) ==================
from sklearn.neighbors import NearestNeighbors

def compute_local_feats(X, zprime, intensity, k=KNN):
    """
    X = [N,3] (x,y,z_abs); zprime = z - DTM(x,y); intensity=[N]
    Retorna features [N,F]: densidade local (~1/dist médio), normal(z), var z’, rank vertical, intensidade norm.
    """
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X)), algorithm='kd_tree').fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)
    # densidade aproximada
    dens = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-6)  # ignora dist=0 do próprio
    # normal local ~ autovetor de menor autovalor da covariância
    Feats = []
    normals_z = np.zeros(len(X), dtype=np.float32)
    var_zp    = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        P = X[idxs[i]]
        C = np.cov(P.T)
        w, v = np.linalg.eigh(C)
        n = v[:, np.argmin(w)]
        normals_z[i] = abs(n[2])  # componente vertical da normal
        var_zp[i] = zprime[idxs[i]].var() if len(idxs[i])>1 else 0.0
    # rank vertical (posição relativa no perfil local)
    rank = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        zp_loc = zprime[idxs[i]]
        m = zp_loc.max() - zp_loc.min()
        rank[i] = 0.0 if m<1e-6 else (zprime[i]-zp_loc.min())/m
    # intensidade normalizada
    if intensity.max() > 0:
        inten = intensity / intensity.max()
    else:
        inten = intensity
    F = np.stack([dens, normals_z, var_zp, rank, inten], axis=1).astype(np.float32)  # [N,5]
    return F

# =============== DATASET DE PARES (ta -> tb) ===============
def sample_points_epoch(P, dtm, max_samp=MAX_SAMP):
    """Filtra vegetação (não-ground), calcula z', e amostra até MAX_SAMP pontos."""
    is_ground = (P["classification"] == 2)
    keep = ~is_ground
    x,y,z = P["x"][keep], P["y"][keep], P["z"][keep]
    zref = dtm.z_at(x,y)
    zprime = z - zref
    # pontos úteis: acima de um limiar
    mask = (zprime > 0.3)  # 30 cm
    x,y,z = x[mask], y[mask], z[mask]
    zprime = zprime[mask]
    intensity = P["intensity"][keep][mask]
    # amostra
    N = len(x)
    if N == 0:
        return None
    if N > max_samp:
        sel = npr.choice(N, max_samp, replace=False)
        x,y,z,zprime,intensity = x[sel],y[sel],z[sel],zprime[sel],intensity[sel]
    X = np.c_[x,y,z]
    F = compute_local_feats(X, zprime, intensity, k=min(KNN, len(X)))
    return {"X": X.astype(np.float32), "zprime": zprime.astype(np.float32), "intensity": intensity.astype(np.float32), "F": F}

def make_training_pairs():
    epocas_validas = _validate_and_sort_epocas(EPOCAS)
    ref_idx = _resolve_ref_index(epocas_validas, REF_INDEX)

    print(f"[INFO] Épocas válidas (ordenadas):")
    for i,e in enumerate(epocas_validas):
        tag = " <- REF" if i == ref_idx else ""
        print(f"  [{i}] idade={e['idade']:.2f}  las={e['las']}{tag}")

    # DTM_ref
    if (OUT_DIR/"dtm_ref.npz").exists():
        dtm = load_dtm_ref()
        Pref = load_las_points(epocas_validas[ref_idx]["las"])
    else:
        dtm, Pref = build_dtm_ref(epocas_validas, ref_idx)

    # carrega épocas e monta pacotes amostrados (vegetação)
    epochs = [load_las_points(e["las"]) for e in epocas_validas]
    packs  = [sample_points_epoch(P, dtm) for P in epochs]
    ages   = [e["idade"] for e in epocas_validas]

    # pares consecutivos (t_i -> t_{i+1})
    pairs = []
    for i in range(len(packs)-1):
        if (packs[i] is None) or (packs[i+1] is None):
            print(f"[AVISO] Época {i} ou {i+1} sem amostra útil (após filtros). Ignorando par.")
            continue
        pairs.append({"src": packs[i], "t_src": ages[i],
                      "tgt": packs[i+1], "t_tgt": ages[i+1]})
    if not pairs:
        raise SystemExit("[ERRO] Nenhum par válido para treino (verifique filtros/alturas).")
    json.dump({"pairs": [(p['t_src'], p['t_tgt']) for p in pairs]},
              open(OUT_DIR/"pairs.json","w"))
    return pairs, epocas_validas, ref_idx

# =============== REDE: EdgeConv (DGCNN-like) ===============
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn_graph(x, k=8):
    # x: [B,N,C]; retorna indices [B,N,k]
    with torch.no_grad():
        B,N,C = x.shape
        # dist^2 = (x-x)^2 ~ usar truque: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        xx = (x**2).sum(-1, keepdim=True)            # [B,N,1]
        dist = xx + xx.transpose(1,2) - 2*torch.bmm(x, x.transpose(1,2))  # [B,N,N]
        idx = dist.topk(k, dim=-1, largest=False)[1] # menores distâncias
    return idx

class EdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=8):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch*2, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU()
        )
    def forward(self, x):  # x: [B,C,N]
        B,C,N = x.shape
        x_t = x.transpose(1,2)        # [B,N,C]
        idx = knn_graph(x_t, k=self.k)# [B,N,k]
        idx_exp = idx.unsqueeze(1).expand(-1,C,-1,-1) # [B,C,N,k]
        neighbors = torch.gather(x.unsqueeze(-1).expand(-1,-1,-1,self.k), 2, idx_exp) # [B,C,N,k]
        central = x.unsqueeze(-1).expand_as(neighbors)
        feat = torch.cat([central, neighbors - central], dim=1)  # [B,2C,N,k]
        out = self.mlp(feat)                                     # [B,out_ch,N,k]
        out = out.max(dim=-1)[0]                                 # [B,out_ch,N]
        return out

class FlowNet(nn.Module):
    """
    Entrada por ponto: [xyz_abs(3) + feats_locais(F) + z'(1) + cond(6)] -> fluxo v (3)
    """
    def __init__(self, f_extra=5, cond_dim=6, k=16):
        super().__init__()
        in_ch = 3 + f_extra + 1 + cond_dim  # xyz + F + z' + cond
        self.g1 = EdgeConv(in_ch, 64, k=k)
        self.g2 = EdgeConv(64, 128, k=k)
        self.g3 = EdgeConv(128, 128, k=k)
        self.head = nn.Sequential(
            nn.Conv1d(64+128+128, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 3, 1)  # vx, vy, vz (m/ano)
        )
    def forward(self, xyz, feats, zprime, cond):
        # xyz:[B,N,3], feats:[B,N,F], zprime:[B,N,1], cond:[B,cond_dim]
        B,N,_ = xyz.shape
        cond_rep = cond.unsqueeze(1).expand(-1,N,-1)    # [B,N,cond_dim]
        inp = torch.cat([xyz, feats, zprime, cond_rep], dim=-1).transpose(1,2)  # [B,C,N]
        f1 = self.g1(inp); f2 = self.g2(f1); f3 = self.g3(f2)
        f = torch.cat([f1,f2,f3], dim=1)
        v = self.head(f).transpose(1,2)  # [B,N,3]
        return v

# =============== CHAMFER e REGULARIZADORES =================
def chamfer_bidirectional(A, B):
    """
    A:[B,Na,3], B:[B,Nb,3]; retorna d^2(A->B)+d^2(B->A) em float32.
    Usa float64 internamente para estabilidade (coordenadas UTM grandes).
    """
    A64 = A.double()
    B64 = B.double()
    # cdist L2 e ao quadrado
    D = torch.cdist(A64, B64, p=2) ** 2  # [B,Na,Nb]
    da = D.min(dim=2).values  # [B,Na]
    db = D.min(dim=1).values  # [B,Nb]
    out = (da.mean() + db.mean()).clamp_min(0.0)
    return out.float()

def reg_forest(v, zprime_src, lambda_horiz=1e-3, lambda_vert_neg=5e-2, lambda_mag=1e-4):
    vxvy = (v[...,0]**2 + v[...,1]**2).mean()
    mask = (zprime_src[...,0] > 1.0).float()
    vz_neg = F.relu(-v[...,2]) * mask
    v_mag = (v**2).mean()
    return lambda_horiz*vxvy + lambda_vert_neg*vz_neg.mean() + lambda_mag*v_mag

# =============== LOOP DE TREINO ============================
def train_flow(
    pairs,
    dz_prior=0.5,           # m/ano (empurrão positivo suave no crescimento vertical)
    lambda_horiz=1e-3,      # regularização para drift horizontal
    lambda_vert_neg=5e-2,   # penalização forte para vz < 0 em dossel
    lambda_mag=1e-4,        # regularização de magnitude do vetor v
    grad_clip=1.0,          # clipping de gradiente (None/0 para desativar)
    normalize=False         # se True, normaliza XYZ no treino (use True também na inferência)
):
    import numpy as np
    import numpy.random as npr
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FlowNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)

    for ep in range(1, EPOCHS + 1):
        losses = []
        pbar = tqdm(pairs, desc=f"Epoch {ep}/{EPOCHS}")

        for pair in pbar:
            src, tgt = pair["src"], pair["tgt"]
            idade_src = pair["t_src"]
            idade_tgt = pair["t_tgt"]
            dt = float(idade_tgt - idade_src)
            if dt <= 0:
                continue

            # amostra aleatória
            Ns, Nt = len(src["X"]), len(tgt["X"])
            idx_s = npr.choice(Ns, min(BATCH_PTS, Ns), replace=False)
            idx_t = npr.choice(Nt, min(BATCH_PTS, Nt), replace=False)

            Xs_np = src["X"][idx_s].astype(np.float32)      # [N,3]
            Fs_np = src["F"][idx_s].astype(np.float32)      # [N,5]
            Zps_np = src["zprime"][idx_s].astype(np.float32)  # [N]
            Xt_np = tgt["X"][idx_t].astype(np.float32)      # [M,3]

            # normalização opcional (consistente com a inferência)
            if normalize:
                center = Xs_np.mean(axis=0, keepdims=True)            # [1,3]
                scale = max(1.0, Xs_np.std() * 10.0)
                Xs_in = (Xs_np - center) / scale
                Xt_in = (Xt_np - center) / scale
                dz_prior_used = dz_prior / scale  # prior em unidades normalizadas
            else:
                Xs_in = Xs_np
                Xt_in = Xt_np
                dz_prior_used = dz_prior

            # tensores
            Xs = torch.from_numpy(Xs_in).float().to(device)                  # [N,3]
            Fs = torch.from_numpy(Fs_np).float().to(device)                  # [N,5]
            Zps = torch.from_numpy(Zps_np).float().unsqueeze(-1).to(device)  # [N,1]
            Xt  = torch.from_numpy(Xt_in).float().to(device)                 # [M,3]
            cond = torch.tensor([[idade_src, dt, 0, 0, 0, 0]], dtype=torch.float32, device=device)

            # forward
            net.train()
            opt.zero_grad()
            v = net(Xs.unsqueeze(0), Fs.unsqueeze(0), Zps.unsqueeze(0), cond)  # [1,N,3] (m/ano no mesmo espaço de Xs_in)
            Xs_warp = Xs + v[0] * dt                                           # [N,3]

            # perdas
            loss_cd = chamfer_bidirectional(Xs_warp.unsqueeze(0), Xt.unsqueeze(0))
            # prior suave para crescimento vertical positivo
            prior_target = torch.full_like(v[0][..., 2], dz_prior_used)
            loss_prior = F.smooth_l1_loss(v[0][..., 2], prior_target)
            # regularizador florestal (Δz<0, drift horizontal, magnitude)
            loss_reg = reg_forest(v[0], Zps, lambda_horiz=lambda_horiz,
                                  lambda_vert_neg=lambda_vert_neg, lambda_mag=lambda_mag)

            loss = loss_cd + loss_reg + 1e-3 * loss_prior
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()

            lval = float(loss.detach().cpu())
            losses.append(lval)
            pbar.set_postfix(loss=f"{lval:.4f}", cd=f"{float(loss_cd.detach().cpu()):.4f}")

        print(f"Epoch {ep}/{EPOCHS} - loss: {np.mean(losses):.6f}")

    torch.save(net.state_dict(), OUT_DIR / "flow_model.pth")
    print("Modelo salvo:", OUT_DIR / "flow_model.pth")


# =============== INFERÊNCIA p/ IDADE ALVO ==================
def infer_to_age_target(idade_alvo, epocas_validas=None, ref_idx=None,
                        normalize=True,
                        vmax_xy=1.0,   # m/ano (horizontal)
                        vmax_z = 3.0   # m/ano (vertical)
                        ):
    """
    Projeta a vegetação da época de referência para a idade alvo e escreve um .LAS.
    - Ground (class=2) é copiado intacto.
    - Velocidades são CLAMPADAS a limites realistas (m/ano) para evitar 'teleporte'.
    - Header (point_format, version, scales, offsets, CRS) herda da referência.
    """
    import numpy as np
    import torch
    from tqdm import tqdm
    import laspy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- resolver épocas e referência (mesma lógica do treino) ---
    if epocas_validas is None:
        epocas_validas = _validate_and_sort_epocas(EPOCAS)
    if ref_idx is None:
        ref_idx = _resolve_ref_index(epocas_validas, REF_INDEX)
    ref_path = epocas_validas[ref_idx]["las"]
    idade_src = epocas_validas[ref_idx]["idade"]
    dt = float(idade_alvo - idade_src)

    # --- carregar rede ----
    net = FlowNet().to(device)
    net.load_state_dict(torch.load(OUT_DIR/"flow_model.pth", map_location=device))
    net.eval()

    # --- carregar DTM e LAS de referência ---
    dtm = load_dtm_ref()
    Pref = load_las_points(ref_path)
    is_ground = (Pref["classification"] == 2)

    # --- vegetação da referência (não-ground) ---
    keep = ~is_ground
    Xv = np.c_[Pref["x"][keep], Pref["y"][keep], Pref["z"][keep]].astype(np.float32)
    zref = dtm.z_at(Xv[:, 0], Xv[:, 1])
    Zp = (Xv[:, 2] - zref).astype(np.float32)
    inten = Pref["intensity"][keep].astype(np.float32)

    # features locais
    Fv = compute_local_feats(Xv, Zp, inten, k=min(KNN, len(Xv)))

    # --- normalização opcional (DESLIGADA por padrão para casar com o treino) ---
    if normalize:
        center = Xv.mean(axis=0, keepdims=True)
        scale  = max(1.0, Xv.std() * 10.0)
        Xv_net = (Xv - center) / scale
    else:
        center = np.zeros((1, 3), dtype=np.float32)
        scale  = 1.0
        Xv_net = Xv

    # condição (alimente SI/espac/… quando tiver)
    cond = torch.tensor([[idade_src, dt, 0, 0, 0, 0]],
                        dtype=torch.float32, device=device)

    # --- inferência em lotes + CLAMP de velocidade ---
    B = 65536
    Xnew = np.empty_like(Xv)
    with torch.no_grad():
        for s in tqdm(range(0, len(Xv), B), desc="Projetando vegetação"):
            e = min(s + B, len(Xv))
            t_xyz = torch.from_numpy(Xv_net[s:e]).float().unsqueeze(0).to(device)  # [1,n,3]
            t_F = torch.from_numpy(Fv[s:e]).float().unsqueeze(0).to(device)  # [1,n,F]
            t_zp = torch.from_numpy(Zp[s:e]).float().unsqueeze(0).unsqueeze(-1).to(device)  # [1,n,1]

            v = net(t_xyz, t_F, t_zp, cond)  # [1,n,3] m/ano

            # clamp em m/ano
            v[..., 0] = torch.clamp(v[..., 0], min=-vmax_xy, max=+vmax_xy)
            v[..., 1] = torch.clamp(v[..., 1], min=-vmax_xy, max=+vmax_xy)
            v[..., 2] = torch.clamp(v[..., 2], min=-vmax_z, max=+vmax_z)

            # LOG em m/ano (antes de qualquer conversão)
            if s == 0:
                vcpu_m = v[0].cpu().numpy()
                print(f"[INFO] v mediana (m/ano): "
                      f"vx={np.median(vcpu_m[:, 0]):.3f}, vy={np.median(vcpu_m[:, 1]):.3f}, vz={np.median(vcpu_m[:, 2]):.3f} "
                      f"| p95 vz={np.percentile(vcpu_m[:, 2], 95):.3f}")

            # >>> SE normalize=True, precisamos converter v para UNIDADES NORMALIZADAS antes de somar <<<
            if normalize:
                v = v / float(scale)  # agora v está em (unid. normalizadas)/ano

            Xwarp_net = t_xyz[0] + v[0] * dt  # [n,3] no mesmo espaço de t_xyz
            Xwarp = Xwarp_net.cpu().numpy()

            if normalize:
                Xwarp = Xwarp * scale + center  # volta para metros

            # trava acima do DTM
            zref_chunk = dtm.z_at(Xwarp[:, 0], Xwarp[:, 1])
            Xwarp[:, 2] = np.maximum(Xwarp[:, 2], zref_chunk + 0.10)  # 10 cm acima do solo

            Xnew[s:e] = Xwarp

    # --- sanity checks ---
    def bbox(arr):
        return float(arr.min()), float(arr.max())

    Gx, Gy, Gz = Pref["x"][is_ground], Pref["y"][is_ground], Pref["z"][is_ground]
    print("[SANITY] Gx:", bbox(Gx), "Gy:", bbox(Gy), "Gz:", bbox(Gz))
    print("[SANITY] Xnew:", bbox(Xnew[:, 0]), bbox(Xnew[:, 1]), bbox(Xnew[:, 2]))
    dx_med = np.median(np.abs(Xnew[:, 0] - np.median(Gx)))
    dy_med = np.median(np.abs(Xnew[:, 1] - np.median(Gy)))
    if dx_med > 100 or dy_med > 100:
        print(f"[AVISO] ΔXY mediano elevado (dx≈{dx_med:.2f} m, dy≈{dy_med:.2f} m). "
              f"Aumente o clamp (vmax_xy) ou verifique treino/condições.")

    # --- classes por altura relativa projetada ---
    zref_new = dtm.z_at(Xnew[:, 0], Xnew[:, 1])
    zp_new = Xnew[:, 2] - zref_new
    vcls = np.where(zp_new < 2.0, 3, np.where(zp_new < 5.0, 4, 5)).astype(np.uint8)

    # intensidade simples mantendo escala
    vint = (inten / (inten.max() + 1e-6) * 100).astype(np.uint16) if len(inten) > 0 else np.full(len(Xnew), 80, np.uint16)

    # --- escrever LAS com header da referência ---
    ref_hdr = Pref["hdr"]
    hdr = laspy.LasHeader(point_format=ref_hdr.point_format, version=ref_hdr.version)
    hdr.scales  = ref_hdr.scales
    hdr.offsets = ref_hdr.offsets
    try:
        crs = ref_hdr.parse_crs()
        if crs is not None:
            hdr.add_crs(crs)
    except Exception:
        pass

    las_out = laspy.LasData(hdr)
    las_out.x = np.concatenate([Gx, Xnew[:, 0]])
    las_out.y = np.concatenate([Gy, Xnew[:, 1]])
    las_out.z = np.concatenate([Gz, Xnew[:, 2]])

    Gint = Pref["intensity"][is_ground]
    Gcls = Pref["classification"][is_ground]
    las_out.intensity = np.concatenate([Gint.astype(np.uint16), vint])
    las_out.classification = np.concatenate([Gcls, vcls]).astype(np.uint8)

    n = len(las_out.x)
    las_out.return_number     = np.ones(n, dtype=np.uint8)
    las_out.number_of_returns = np.ones(n, dtype=np.uint8)

    out_path = OUT_DIR / f"projecao_idade_{idade_alvo:.1f}a.las"
    las_out.write(str(out_path))
    print("Gerado:", out_path)

def infer_to_age_raw(idade_alvo,
                     epocas_validas=None,
                     ref_idx=None,
                     normalize=True,
                     enforce_floor=True,
                     write_las=True,
                     # limites físicos (m/ano) – ajuste ao seu sítio:
                     xy_max_per_year=0.5,   # drift horizontal plausível
                     z_max_per_year =2.0,   # crescimento vertical plausível
                     # detecção robusta de outliers por percentil
                     pct_cap=99.0,
                     max_xy_from_bbox=100.0  # se XY fugir >100 m do bbox do ground, descarta
                     ):
    """
    Inferência SEM clamp "duro", mas com PODA/CLAMP ROBUSTO apenas para outliers.
    - Ground (class=2) copiado intacto.
    - Opcional enforce_floor: z >= DTM + 0.10 m.
    - Gera também uma versão 'denoised' com outliers removidos/clampeados.

    Retorna: dict com stats e caminhos dos .LAS.
    """
    import numpy as np
    import torch
    from tqdm import tqdm
    import laspy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- resolver épocas/ref ----------
    if epocas_validas is None:
        epocas_validas = _validate_and_sort_epocas(EPOCAS)
    if ref_idx is None:
        ref_idx = _resolve_ref_index(epocas_validas, REF_INDEX)
    ref_path = epocas_validas[ref_idx]["las"]
    idade_src = epocas_validas[ref_idx]["idade"]
    dt = float(idade_alvo - idade_src)

    # -------- carregar rede/DTM/ref ----------
    net = FlowNet().to(device)
    net.load_state_dict(torch.load(OUT_DIR / "flow_model.pth", map_location=device))
    net.eval()

    dtm = load_dtm_ref()
    Pref = load_las_points(ref_path)
    is_ground = (Pref["classification"] == 2)

    # vegetação ref
    keep = ~is_ground
    Xv = np.c_[Pref["x"][keep], Pref["y"][keep], Pref["z"][keep]].astype(np.float32)
    zref = dtm.z_at(Xv[:, 0], Xv[:, 1])
    Zp   = (Xv[:, 2] - zref).astype(np.float32)
    inten= Pref["intensity"][keep].astype(np.float32)

    Fv = compute_local_feats(Xv, Zp, inten, k=min(KNN, len(Xv)))

    # normalização opcional
    if normalize:
        center = Xv.mean(axis=0, keepdims=True)
        scale  = max(1.0, Xv.std() * 10.0)
        Xv_net = (Xv - center) / scale
    else:
        center = np.zeros((1,3), dtype=np.float32)
        scale  = 1.0
        Xv_net = Xv

    cond = torch.tensor([[idade_src, dt, 0, 0, 0, 0]],
                        dtype=torch.float32, device=device)

    # -------- inferência RAW ----------
    B = 65536
    Xnew_raw = np.empty_like(Xv)
    V_raw_samp = []

    with torch.no_grad():
        for s in tqdm(range(0, len(Xv), B), desc="Projetando vegetação (raw)"):
            e = min(s+B, len(Xv))
            t_xyz = torch.from_numpy(Xv_net[s:e]).float().unsqueeze(0).to(device)
            t_F   = torch.from_numpy(Fv[s:e]).float().unsqueeze(0).to(device)
            t_zp  = torch.from_numpy(Zp[s:e]).float().unsqueeze(0).unsqueeze(-1).to(device)

            v_raw = net(t_xyz, t_F, t_zp, cond)  # [1,n,3] m/ano
            if s == 0:
                vr = v_raw[0].cpu().numpy()
                print(f"[INFO raw] v mediana (m/ano): "
                      f"vx={np.median(vr[:, 0]):.3f}, vy={np.median(vr[:, 1]):.3f}, vz={np.median(vr[:, 2]):.3f} | "
                      f"p95 vz={np.percentile(vr[:, 2], 95):.3f}")
            V_raw_samp.append(v_raw[0].cpu().numpy()[::max(1, len(v_raw[0])//5000)])

            # >>> adequar unidades se normalize=True
            v_step = v_raw
            if normalize:
                v_step = v_raw / float(scale)

            Xwarp_net = t_xyz[0] + v_step[0] * dt
            Xwarp = Xwarp_net.cpu().numpy()
            if normalize:
                Xwarp = Xwarp * scale + center
            if enforce_floor:
                zref_chunk = dtm.z_at(Xwarp[:,0], Xwarp[:,1])
                Xwarp[:,2] = np.maximum(Xwarp[:,2], zref_chunk + 0.10)
            Xnew_raw[s:e] = Xwarp

    # -------- diagnóstico bruto ----------
    def bbox(arr): return float(arr.min()), float(arr.max())
    Gx, Gy, Gz = Pref["x"][is_ground], Pref["y"][is_ground], Pref["z"][is_ground]
    print("[SANITY] ground bbox X:", bbox(Gx), "Y:", bbox(Gy), "Z:", bbox(Gz))
    print("[SANITY] proj raw bbox X:", bbox(Xnew_raw[:,0]), "Y:", bbox(Xnew_raw[:,1]), "Z:", bbox(Xnew_raw[:,2]))

    d = Xnew_raw - Xv
    dx_med = np.median(np.abs(d[:,0])); dy_med = np.median(np.abs(d[:,1])); dz_med = np.median(d[:,2])
    print(f"[QC raw] mediana |Δx|={dx_med:.3f} m, |Δy|={dy_med:.3f} m, Δz={dz_med:.3f} m (dt={dt:.2f} a)")

    # -------- robust denoise + clamp só nos OUTLIERS ----------
    # 1) limites físicos por Δt
    lim_xy = xy_max_per_year * max(dt, 1e-6)
    lim_z  = z_max_per_year  * max(dt, 1e-6)

    # 2) limites por percentil (evita que 1% “puxe” demais)
    pxy = np.percentile(np.abs(d[:, :2]).reshape(-1), pct_cap)
    pz  = np.percentile(np.abs(d[:, 2]), pct_cap)
    lim_xy_rob = max(0.05, min(lim_xy, pxy))
    lim_z_rob  = max(0.10, min(lim_z , pz ))

    # 3) máscara de outliers por deslocamento
    mask_out = (np.abs(d[:,0])>lim_xy_rob) | (np.abs(d[:,1])>lim_xy_rob) | (np.abs(d[:,2])>lim_z_rob)

    # 4) máscara por “fuga do bbox” (XY muito longe do talhão)
    gxmin,gxmax = bbox(Gx); gymin,gymax = bbox(Gy)
    mask_far = (Xnew_raw[:,0] < gxmin - max_xy_from_bbox) | (Xnew_raw[:,0] > gxmax + max_xy_from_bbox) | \
               (Xnew_raw[:,1] < gymin - max_xy_from_bbox) | (Xnew_raw[:,1] > gymax + max_xy_from_bbox)

    outliers = np.nonzero(mask_out | mask_far)[0]
    keepers  = np.nonzero(~(mask_out | mask_far))[0]
    print(f"[DENOISE] removendo/clampeando {len(outliers)} de {len(Xv)} pontos ({100*len(outliers)/len(Xv):.2f}%).")

    # política: para outliers, CLAMPEAR Δ por componente aos limites robustos
    Xnew_den = Xnew_raw.copy()
    if len(outliers) > 0:
        d_clip = d.copy()
        d_clip[outliers,0] = np.clip(d_clip[outliers,0], -lim_xy_rob, +lim_xy_rob)
        d_clip[outliers,1] = np.clip(d_clip[outliers,1], -lim_xy_rob, +lim_xy_rob)
        d_clip[outliers,2] = np.clip(d_clip[outliers,2], -lim_z_rob , +lim_z_rob )
        Xnew_den = Xv + d_clip
        if enforce_floor:
            zref_den = dtm.z_at(Xnew_den[:,0], Xnew_den[:,1])
            Xnew_den[:,2] = np.maximum(Xnew_den[:,2], zref_den + 0.10)

    # -------- escrever LAS (raw e denoised) ----------
    ref_hdr = Pref["hdr"]
    def write_las(points_xyz, suffix):
        hdr = laspy.LasHeader(point_format=ref_hdr.point_format, version=ref_hdr.version)
        hdr.scales  = ref_hdr.scales
        hdr.offsets = ref_hdr.offsets
        try:
            crs = ref_hdr.parse_crs()
            if crs is not None:
                hdr.add_crs(crs)
        except Exception:
            pass
        las_out = laspy.LasData(hdr)
        las_out.x = np.concatenate([Gx, points_xyz[:,0]])
        las_out.y = np.concatenate([Gy, points_xyz[:,1]])
        las_out.z = np.concatenate([Gz, points_xyz[:,2]])
        vint = (inten/(inten.max()+1e-6)*100).astype(np.uint16) if len(inten)>0 else np.full(len(points_xyz),80,np.uint16)
        vcls = np.where((points_xyz[:,2]-dtm.z_at(points_xyz[:,0], points_xyz[:,1]))<2.0, 3,
                 np.where((points_xyz[:,2]-dtm.z_at(points_xyz[:,0], points_xyz[:,1]))<5.0, 4, 5)).astype(np.uint8)
        Gint = Pref["intensity"][is_ground]; Gcls = Pref["classification"][is_ground]
        las_out.intensity = np.concatenate([Gint.astype(np.uint16), vint])
        las_out.classification = np.concatenate([Gcls, vcls]).astype(np.uint8)
        n = len(las_out.x)
        las_out.return_number     = np.ones(n, dtype=np.uint8)
        las_out.number_of_returns = np.ones(n, dtype=np.uint8)
        out_path = OUT_DIR / f"projecao_idade_{idade_alvo:.1f}a_raw{suffix}.las"
        las_out.write(str(out_path))
        return out_path

    out_raw = out_den = None
    if write_las:
        out_raw = write_las(Xnew_raw, "")
        print("Gerado (raw):", out_raw)
        out_den = write_las(Xnew_den, "_denoised")
        print("Gerado (denoised):", out_den)

    # -------- stats finais ----------
    stats = {
        "dt_years": float(dt),
        "raw_bbox": {
            "x": bbox(Xnew_raw[:,0]), "y": bbox(Xnew_raw[:,1]), "z": bbox(Xnew_raw[:,2])
        },
        "den_bbox": {
            "x": bbox(Xnew_den[:,0]), "y": bbox(Xnew_den[:,1]), "z": bbox(Xnew_den[:,2])
        },
        "raw_median_disp": {
            "dx": float(dx_med), "dy": float(dy_med), "dz": float(dz_med)
        },
        "outlier_rate_pct": float(100*len(outliers)/max(1,len(Xv))),
        "robust_limits": {
            "lim_xy_rob": float(lim_xy_rob), "lim_z_rob": float(lim_z_rob),
            "bbox_guard": float(max_xy_from_bbox)
        }
    }
    return {"las_raw": out_raw, "las_denoised": out_den, "stats": stats}


# =============== MAIN ======================================
if __name__ == "__main__":
    print(">> Preparando pares de treino ...")
    pairs, epocas_validas, ref_idx = make_training_pairs()

    print(">> Treinando modelo de scene flow ...")
    train_flow(pairs)

    print(">> Inferindo projeção para idade alvo (com clamp) ...")
    infer_to_age_target(ID_ALVO, epocas_validas=epocas_validas, ref_idx=ref_idx, normalize=False)

    print(">> Inferindo projeção raw (sem clamp) para idade alvo ...")
    infer_to_age_raw(
        ID_ALVO, epocas_validas=epocas_validas, ref_idx=ref_idx,
        normalize=False, enforce_floor=True,
        xy_max_per_year=0.5, z_max_per_year=2.0,  # ajuste ao seu sítio
        pct_cap=99.0, max_xy_from_bbox=100.0
    )
