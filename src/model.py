import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import torch
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

@dataclass
class MoEBatch:
    # --- image-like ---
    ct: Optional[torch.Tensor] = None          # [B, 1, D, H, W] float
    # --- graph-like ---
    graph: Optional[PyGBatch] = None           # PyG Batch
    # --- meta ---
    target_pos_mm: Optional[torch.Tensor] = None   # [B, 3]
    target_pos_norm: Optional[torch.Tensor] = None # [B, 3] optional
    lobe_id: Optional[torch.Tensor] = None         # [B] long
    label: Optional[torch.Tensor] = None           # [B] float or long
    fname: Optional[List[str]] = None              # [B] str
    extra: Optional[Dict[str, Any]] = None

class ExpertBase(nn.Module):
    def required_keys(self) -> List[str]:
        raise NotImplementedError

    def forward(self, batch: MoEBatch) -> torch.Tensor:
        """Return embedding: [B, D]"""
        raise NotImplementedError

class SEBlock3d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class EnhancedResBlock3d(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c)
        )
        self.se = SEBlock3d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_c)
            )

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        return F.relu(out + self.shortcut(x))

class GlobalCTExpert(ExpertBase):
    def __init__(self, out_dim=128):
        super().__init__()
        # 增加深度至 5 层，并引入 SE 注意力
        self.encoder = nn.Sequential(
            EnhancedResBlock3d(2, 32, stride=2),   # 64x64x16
            EnhancedResBlock3d(32, 64, stride=2),  # 32x32x8
            EnhancedResBlock3d(64, 128, stride=2), # 16x16x4
            EnhancedResBlock3d(128, 256, stride=2),# 8x8x2
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(256, out_dim)

    def required_keys(self): return ["ct"]

    def forward(self, batch: MoEBatch):
        feat = self.encoder(batch.ct).flatten(1)
        return self.fc(feat)

class LobeExpert(ExpertBase):
    def __init__(self, num_lobes=5, out_dim=128):
        super().__init__()
        self.emb = nn.Embedding(num_lobes, out_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def required_keys(self): return ["lobe_id"]

    def forward(self, batch: MoEBatch):
        e = self.emb(batch.lobe_id)  # [B,out_dim]
        return self.mlp(e)


class PathSequenceExpert(nn.Module):
    def __init__(self, in_dim=13, out_dim=128, hidden=128):
        """
        in_dim: 路由后的特征维度。
        14维原始特征中，排除索引 12 (on_target_path)，剩余 13 维。
        """
        super().__init__()
        # 强制特征路由索引 (排除 12: on_target_path) [cite: 2026-02-12]
        self.feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
        # self.feature_indices = [0, 1, 2, 5, 11, 13, 6]
        self.seqs = None
        self.lengths = None
        self.rnn = nn.GRU(
            input_size=len(self.feature_indices), 
            hidden_size=hidden, 
            num_layers=1,
            batch_first=True, 
            bidirectional=True
        )
        
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def extract_and_sort_path(self, x, batch):
        """
        从 PyG Batch 中提取路径序列并按解剖顺序排序 [cite: 2026-02-18]
        """
        # 1. 利用索引 12 获取路径掩码 [cite: 2026-02-12]
        path_mask = x[:, 12] > 0.5
        path_x = x[path_mask]
        path_batch = batch[path_mask]
        
        num_graphs = int(batch.max().item()) + 1
        sequences = []
        lengths = []

        for i in range(num_graphs):
            # 2. 提取当前病例的路径节点
            mask_i = (path_batch == i)
            case_x = path_x[mask_i]
            
            if case_x.size(0) > 0:
                # 3. 核心：按 generation (索引3) 升序排列，确保序列从根部(Carina)指向末端(Nodule) 
                _, sort_idx = torch.sort(case_x[:, 3])
                sorted_case_x = case_x[sort_idx]
                
                # 4. 特征路由：仅选择用于建模的物理/拓扑特征
                routed_x = sorted_case_x[:, self.feature_indices]
                sequences.append(routed_x)
                lengths.append(routed_x.size(0))
            else:
                # 安全占位：处理无路径节点的情况
                sequences.append(torch.zeros((1, len(self.feature_indices)), device=x.device))
                lengths.append(1)

        # 5. 填充为 [B, Max_Len, 13]
        padded_seqs = pad_sequence(sequences, batch_first=True)
        return padded_seqs, torch.tensor(lengths)

    def forward(self, batch):
        g = batch.graph
        # 执行序列提取与特征路由 
        seqs, lengths = self.extract_and_sort_path(g.x, g.batch)
        self.seqs = seqs
        self.lengths = lengths
        # 6. 打包序列以利用高效的 RNN 计算 (需确保 lengths 在 CPU) 
        packed = pack_padded_sequence(
            seqs, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 7. 通过双向 GRU 捕捉累积几何约束
        # h_n shape: [2, B, hidden] (num_layers * num_directions, batch, hidden)
        _, h_n = self.rnn(packed)
        
        # 8. 融合正向与反向的最后状态 (Final Hidden States) 
        # h_n[-2] 是正向序列末端，h_n[-1] 是反向序列末端 (即根部)
        fused_h = torch.cat([h_n[-2], h_n[-1]], dim=1) # [B, hidden * 2]
        
        # 9. 投影到统一的专家嵌入空间
        return self.proj(fused_h)


class LocationGating(nn.Module):
    def __init__(self, num_experts: int, lobe_vocab=6, pos_dim=3, stats_dim=7, hidden=128):
        super().__init__()
        # 肺叶解剖先验 Embedding
        self.lobe_emb = nn.Embedding(lobe_vocab, 16)
        
        # 输入维度 = 位置(3) + 肺叶(16) + 统计量(7) = 26
        input_dim = pos_dim + 16 + stats_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden), # 加入正则化防止过拟合
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_experts),
        )

    def forward(self, batch: MoEBatch):
        """
        从 MoEBatch 中提取位置、肺叶 ID 以及 Graph Stats 进行融合
        """
        # 1. 提取位置特征 (优先使用归一化坐标)
        pos = batch.target_pos_norm if batch.target_pos_norm is not None else batch.target_pos_mm
        
        # 2. 提取肺叶 Embedding
        lobe_e = self.lobe_emb(batch.lobe_id) # [B, 16]
        
        # 3. 核心改进：从 extra 中安全提取 graph_stats
        if batch.extra is not None and "stats" in batch.extra:
            stats = batch.extra["stats"] # [B, 7]
        else:
            # Fallback: 如果没有统计量，补零 (确保维度一致)
            stats = torch.zeros((pos.size(0), 7), device=pos.device)
        
        # 4. 特征拼接 (Feature Concat)
        # 组合 [位置, 肺叶先验, 图级路况摘要]
        z = torch.cat([pos, lobe_e, stats], dim=1) # [B, 26]
        # 5. 生成专家权重
        logits = self.net(z) # [B, E]
        w = F.softmax(logits, dim=1)
        
        return w, logits


class AaMoEModel(nn.Module):
    def __init__(self, experts: List[ExpertBase], embed_dim=128, num_lobes=6):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)

        self.gate = LocationGating(self.num_experts, lobe_vocab=num_lobes, pos_dim=3, stats_dim=7, hidden=128)

        # 融合后分类
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)  # binary logit
        )

    def compute_load_balancing_loss(self, w, num_experts):
        # w: [B, E] 门控权重
        # 计算各专家被选中的频率 f 和 概率均值 P
        # 理想状态是各专家被选中的概率均等
        f = torch.mean(w, dim=0)
        P = torch.mean(w, dim=0)
        return torch.sum(f * P) * num_experts # 越均匀 Loss 越小

    def compute_entropy_regularization(self, w):
        # w: [B, E]
        eps = 1e-8
        entropy = -torch.sum(w * torch.log(w + eps), dim=1)  # [B]
        return -entropy.mean()  # 负号 → 最大化熵

    def forward(self, batch: MoEBatch):
        # 1) experts
        expert_outs = []
        for ex in self.experts:
            # 可选：检查 required_keys 是否存在
            expert_outs.append(ex(batch))  # [B,D]
        E = torch.stack(expert_outs, dim=1)  # [B, num_experts, D]

        # 2) gating
        w, gate_logits = self.gate(batch)  # w: [B,E]

        # 3) fuse
        fused = torch.sum(E * w.unsqueeze(-1), dim=1)   # [B,D]

        # 4) predict
        logit = self.head(fused).squeeze(1)             # [B]

        # load_balancing_loss = self.compute_load_balancing_loss(w, self.num_experts)
        entropy_loss = self.compute_entropy_regularization(w)
        return {
            "logit": logit,
            "gate_w": w,
            "gate_logits": gate_logits,
            "expert_embed": E,
            "fused": fused,
            # "load_balancing_loss": load_balancing_loss,
            "entropy_loss": entropy_loss,
        }