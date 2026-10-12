import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from model import MoEBatch
from torch_geometric.data import Batch as PyGBatch
import pandas as pd
import numpy as np
import nibabel as nib
import torch_scatter

def compute_graph_stats(g):
    x = g.x
    b = g.batch
    B = int(g.num_graphs)

    idx_diam = 1
    idx_dist_root = 4
    idx_dist_target = 6
    idx_align = 10
    idx_path = 12

    path_mask = (x[:, idx_path] > 0.5)

    # 1) 全图最小直径
    min_global_diam = torch_scatter.scatter_min(x[:, idx_diam], b, dim=0, dim_size=B)[0]

    # 2) 路径最小直径（按图；无路径的图填 0.1）
    path_diam = x[:, idx_diam]
    min_path_diam = torch_scatter.scatter_min(path_diam[path_mask], b[path_mask], dim=0, dim_size=B)[0] \
        if path_mask.any() else x.new_full((B,), float('inf'))
    min_path_diam = torch.where(torch.isfinite(min_path_diam), min_path_diam, x.new_full((B,), 0.1))

    # 3) 路径最小 dist_to_target（无路径填 1.0）
    path_dist = x[:, idx_dist_target]
    min_path_dist = torch_scatter.scatter_min(path_dist[path_mask], b[path_mask], dim=0, dim_size=B)[0] \
        if path_mask.any() else x.new_full((B,), float('inf'))
    min_path_dist = torch.where(torch.isfinite(min_path_dist), min_path_dist, x.new_full((B,), 1.0))

    # 4) 路径 mean alignment（无路径填 0）
    path_align = x[:, idx_align]
    mean_path_align = torch_scatter.scatter_mean(path_align[path_mask], b[path_mask], dim=0, dim_size=B) \
        if path_mask.any() else x.new_zeros((B,))

    # 5) path_ratio
    ones = torch.ones_like(b, dtype=torch.float32)
    total_nodes = torch_scatter.scatter_add(ones, b, dim=0, dim_size=B)
    path_nodes = torch_scatter.scatter_add(path_mask.float(), b, dim=0, dim_size=B)
    path_ratio = path_nodes / total_nodes.clamp_min(1)

    # 6) max dist_root
    max_dist_root = torch_scatter.scatter_max(x[:, idx_dist_root], b, dim=0, dim_size=B)[0]

    # 7) path_mean_diam
    mean_path_diam = torch_scatter.scatter_mean(path_diam[path_mask], b[path_mask], dim=0, dim_size=B) \
        if path_mask.any() else x.new_zeros((B,))

    stats = torch.stack([
        min_global_diam,
        min_path_diam,
        min_path_dist,
        mean_path_align,
        path_ratio,
        max_dist_root,
        mean_path_diam,
    ], dim=1)
    return stats


def collate_moe(samples: List[Dict[str, Any]]) -> MoEBatch:
    # samples[i] 是 dataset __getitem__ 返回的 dict
    # 约定字段：ct, graph, target_pos_mm, lobe_id, label (可选更多)
    fname = [s["fname"] for s in samples]
    ct_list = [s.get("ct", None) for s in samples]
    has_ct = all(x is not None for x in ct_list)
    ct = torch.stack(ct_list, dim=0) if has_ct else None

    graphs = [s["graph"] for s in samples]
    graph_batch = PyGBatch.from_data_list(graphs) if graphs[0] is not None else None

    target_pos_mm = torch.stack([s["target_pos_mm"] for s in samples], dim=0)
    lobe_id = torch.tensor([int(s["lobe_id"]) for s in samples], dtype=torch.long)
    label = torch.tensor([float(s["label"]) for s in samples], dtype=torch.float32)

    target_pos_norm = None
    if "target_pos_norm" in samples[0]:
        target_pos_norm = torch.stack([s["target_pos_norm"] for s in samples], dim=0)

    stats = compute_graph_stats(graph_batch)

    extra = {
        "stats": stats,
    }

    return MoEBatch(
        ct=ct,
        graph=graph_batch,
        target_pos_mm=target_pos_mm,
        target_pos_norm=target_pos_norm,
        lobe_id=lobe_id,
        label=label,
        fname=fname,
        extra=extra,
    )

def normalize_coords(pos_mm, lung, spacing):
    bbox = np.where(lung > 0)
    xmin, xmax = bbox[0].min(), bbox[0].max()
    ymin, ymax = bbox[1].min(), bbox[1].max()
    zmin, zmax = bbox[2].min(), bbox[2].max()
    sx, sy, sz = spacing
    lung_min_mm = np.array([xmin * sx, ymin * sy, zmin * sz])
    lung_extent_mm = np.array([(xmax - xmin + 1) * sx, (ymax - ymin + 1) * sy, (zmax - zmin + 1) * sz])
    pos_norm = (pos_mm - lung_min_mm) / lung_extent_mm
    pos_norm = torch.from_numpy(pos_norm).float()
    pos_norm = torch.clamp(pos_norm, 0.0, 1.0)
    return pos_norm

class BronchoDataset(Dataset):
    def __init__(self, df_path, ct_root, graph_root, lung_root, lobe_root, size):
        self.df = pd.read_csv(df_path)
        # self.df = df_path
        self.ct_root = ct_root
        self.graph_root = graph_root
        self.lung_root = lung_root
        self.lobe_root = lobe_root
        self.lobe_id_map = {
            "LUL": 1,
            "LLL": 2,
            "RUL": 3,
            "RML": 4,
            "RLL": 5,
        }
        self.size = size
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["filename"]
        y = int(row["label"])
        target_nodule_location = row["target_nodule_location"].split(';')[0].strip()
        lobe_id = self.lobe_id_map[target_nodule_location]-1
        center = row["center"][1:-1].split(' ')
        center = [int(float(x)) for x in center if x != '']
        center = center[::-1]
        target_pos = np.array(center)
        # 1) graph
        graph = torch.load(f"{self.graph_root}/{fname.replace('.nii.gz','.pt')}")
        
        # 2) ct patch / ct volume
        ct = torch.load(f"{self.ct_root}/{fname.replace('.nii.gz','.pt')}")

        # 3) lung
        lung = nib.load(f"{self.lung_root}/{fname}")
        spacing = lung.header.get_zooms()
        lung = lung.get_fdata()

        # 4) normalize target position
        target_pos_mm = target_pos * spacing
        target_pos_norm = normalize_coords(target_pos_mm, lung, spacing)
        target_pos_mm = torch.from_numpy(target_pos_mm).float()
        return {
            "ct": ct,  # tensor or None
            "graph": graph,
            "target_pos_mm": target_pos_mm,
            "target_pos_norm": target_pos_norm,
            "lobe_id": lobe_id,
            "label": y,
            "fname": fname,
        }
