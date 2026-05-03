"""
PathOmicDRP: Pathology-Omics Drug Response Predictor

Multi-modal deep learning framework integrating:
  - Genomic features (mutation binary + TMB)  [+ KD_GNN preprocessor]
  - Transcriptomic features (pathway-level scores)
  - Proteomic features (RPPA protein expression)  [+ KD_GNN preprocessor]
  - Histopathology features (UNI foundation model + ABMIL)

KD_GNN: per-modality GCN over genes/proteins as nodes, with an EMA
self-distillation teacher (despite the historical "KD" naming, the teacher
is an exponential-moving-average copy of the student rather than an
externally pretrained model — closer in spirit to BYOL-style self-distillation).

Fusion: Cross-attention between omics tokens and histology tokens
Output: Drug response prediction (IC50 regression / binary classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 0. Knowledge-Distilled GNN (EMA self-distillation) for genomic / proteomic
# ---------------------------------------------------------------------------

class OmicsGNN(nn.Module):
    """GCN-style encoder over omic features (each gene/protein is a node).

    Message passing uses a (provided) symmetrically-normalized adjacency.
    Default adjacency is fully connected with self-loops; callers can swap
    in a biologically-grounded graph (pathway/PPI/phosphorylation) via
    ``OmicsKDGNN.set_adjacency``.
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim: int = 1,
        hidden_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.node_embed = nn.Embedding(n_nodes, hidden_dim)

        self.gcn_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (B, n_nodes, in_dim)
            adj_norm:      (n_nodes, n_nodes) symmetrically normalized adjacency
        Returns:
            embeddings:    (B, n_nodes, hidden_dim)
        """
        h = self.input_proj(node_features)  # (B, n_nodes, hidden_dim)
        ids = torch.arange(self.n_nodes, device=node_features.device)
        h = h + self.node_embed(ids).unsqueeze(0)  # broadcast over batch

        for i in range(self.n_layers):
            # Message passing: agg[b,i,d] = sum_j adj_norm[i,j] * h[b,j,d]
            agg = torch.einsum('ij,bjd->bid', adj_norm, h)
            h_new = self.gcn_layers[i](agg)
            h_new = self.layer_norms[i](h_new)
            h_new = self.activation(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual

        return h


class OmicsKDGNN(nn.Module):
    """Student + EMA teacher GNN pair with self-distillation.

    The student is updated by gradient descent (regression loss + KD loss).
    The teacher is an exponential moving average of the student's weights
    and receives no gradient. KD loss is a 1 - cosine similarity between
    student and (detached) teacher per-node embeddings.
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim: int = 1,
        hidden_dim: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
        ema_decay: float = 0.99,
        adjacency: torch.Tensor = None,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay

        self.student = OmicsGNN(n_nodes, in_dim, hidden_dim, n_layers, dropout)
        self.teacher = OmicsGNN(n_nodes, in_dim, hidden_dim, n_layers, dropout)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Initialize teacher == student
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.data.copy_(ps.data)

        if adjacency is None:
            adjacency = torch.ones(n_nodes, n_nodes)
        self.register_buffer('adj_norm', self._normalize_adj(adjacency))

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalization with self-loops: D^-1/2 (A + I) D^-1/2."""
        adj = adj.float()
        eye = torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
        adj_self = ((adj + eye) > 0).float()
        deg = adj_self.sum(dim=1)
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        return adj_self * deg_inv_sqrt.unsqueeze(0) * deg_inv_sqrt.unsqueeze(1)

    def set_adjacency(self, adj: torch.Tensor) -> None:
        """Replace the adjacency buffer with a (binary) biological graph."""
        if tuple(adj.shape) != (self.n_nodes, self.n_nodes):
            raise ValueError(
                f"Expected adjacency shape ({self.n_nodes}, {self.n_nodes}), "
                f"got {tuple(adj.shape)}"
            )
        norm = self._normalize_adj(adj.to(self.adj_norm.device))
        self.adj_norm = norm.to(self.adj_norm.dtype)

    def train(self, mode: bool = True):
        """Override so the teacher is always in eval (no dropout, frozen BN)."""
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, decay: float = None) -> None:
        """EMA update: theta_teacher <- d * theta_teacher + (1 - d) * theta_student."""
        d = self.ema_decay if decay is None else decay
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(d).add_(ps.data, alpha=1.0 - d)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, n_nodes) or (B, n_nodes, in_dim) raw per-node feature(s)
        Returns:
            dict with
              - 'pooled':      (B, hidden_dim) mean-pooled student embedding
              - 'student_emb': (B, n_nodes, hidden_dim)
              - 'teacher_emb': (B, n_nodes, hidden_dim)  [only when training]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, n_nodes, 1)
        if x.size(-1) != self.in_dim:
            raise ValueError(
                f"OmicsKDGNN: expected feature dim {self.in_dim}, got {x.size(-1)}"
            )
        if x.size(1) != self.n_nodes:
            raise ValueError(
                f"OmicsKDGNN: expected n_nodes={self.n_nodes}, got {x.size(1)}"
            )

        student_emb = self.student(x, self.adj_norm)  # (B, n_nodes, hidden_dim)
        pooled = student_emb.mean(dim=1)              # (B, hidden_dim)

        out = {'pooled': pooled, 'student_emb': student_emb}
        if self.training:
            with torch.no_grad():
                teacher_emb = self.teacher(x, self.adj_norm)
            out['teacher_emb'] = teacher_emb
        return out

    @staticmethod
    def kd_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
        """1 - cosine similarity between per-node embeddings (teacher detached)."""
        s = F.normalize(student_emb, dim=-1)
        t = F.normalize(teacher_emb.detach(), dim=-1)
        return (1.0 - (s * t).sum(dim=-1)).mean()


# ---------------------------------------------------------------------------
# 1. Modality-Specific Encoders
# ---------------------------------------------------------------------------

class GenomicEncoder(nn.Module):
    """Encode binary mutation matrix + TMB into genomic tokens."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_tokens: int = 8, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Learnable token assignment: project to n_tokens
        self.token_proj = nn.Linear(hidden_dim, n_tokens)
        self.token_embed = nn.Linear(1, hidden_dim)
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) binary mutation + TMB features
        Returns:
            tokens: (B, n_tokens, hidden_dim) genomic tokens
        """
        h = self.projection(x)  # (B, hidden_dim)
        # Create tokens via learned decomposition
        weights = torch.softmax(self.token_proj(h), dim=-1)  # (B, n_tokens)
        tokens = weights.unsqueeze(-1) * h.unsqueeze(1)  # (B, n_tokens, hidden_dim)
        return tokens


class PathwayTokenizer(nn.Module):
    """Tokenize transcriptomic features at pathway level.

    Each pathway becomes a token, enabling biologically meaningful
    cross-attention with histology patches (SurvPath-inspired).
    """

    def __init__(self, n_pathways: int, genes_per_pathway: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_pathways = n_pathways
        self.pathway_encoder = nn.Sequential(
            nn.Linear(genes_per_pathway, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_pathways, genes_per_pathway) pathway-grouped expression
               OR (B, n_pathways) if already pathway scores (ssGSEA)
        Returns:
            tokens: (B, n_pathways, hidden_dim) pathway tokens
        """
        if x.dim() == 2:
            # Pathway scores: expand to (B, n_pathways, 1) then project
            x = x.unsqueeze(-1)
        tokens = self.pathway_encoder(x)  # (B, n_pathways, hidden_dim)
        tokens = self.layer_norm(tokens)
        return tokens


class ProteomicEncoder(nn.Module):
    """Encode RPPA protein/phosphoprotein expression into protein tokens."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_tokens: int = 16, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_proj = nn.Linear(hidden_dim, n_tokens)
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) RPPA protein expression
        Returns:
            tokens: (B, n_tokens, hidden_dim) protein tokens
        """
        h = self.projection(x)  # (B, hidden_dim)
        weights = torch.softmax(self.token_proj(h), dim=-1)  # (B, n_tokens)
        tokens = weights.unsqueeze(-1) * h.unsqueeze(1)  # (B, n_tokens, hidden_dim)
        return tokens


# ---------------------------------------------------------------------------
# 2. Histopathology Branch (ABMIL)
# ---------------------------------------------------------------------------

class ABMIL(nn.Module):
    """Attention-Based Multiple Instance Learning for WSI aggregation.

    Takes pre-extracted patch features (e.g., from UNI) and produces
    a slide-level embedding via gated attention.
    """

    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256, dropout: float = 0.1, n_tokens: int = 1):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim // 2, n_tokens)
        self.n_tokens = n_tokens

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple:
        """
        Args:
            x: (B, N_patches, feature_dim) patch features from UNI
            mask: (B, N_patches) boolean mask for valid patches
        Returns:
            slide_tokens: (B, n_tokens, hidden_dim) slide-level token(s)
            attention_weights: (B, n_tokens, N_patches) for interpretability
        """
        h = self.feature_proj(x)  # (B, N, hidden_dim)

        a_V = self.attention_V(h)  # (B, N, hidden_dim//2)
        a_U = self.attention_U(h)  # (B, N, hidden_dim//2)
        a = self.attention_W(a_V * a_U)  # (B, N, n_tokens)

        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        a = a.transpose(1, 2)  # (B, n_tokens, N)
        attention_weights = F.softmax(a, dim=-1)  # (B, n_tokens, N)

        slide_tokens = torch.bmm(attention_weights, h)  # (B, n_tokens, hidden_dim)
        return slide_tokens, attention_weights


# ---------------------------------------------------------------------------
# 3. Cross-Attention Fusion Module
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention between two sets of tokens."""

    def __init__(self, hidden_dim: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn_1to2 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_2to1 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1a = nn.LayerNorm(hidden_dim)
        self.norm1b = nn.LayerNorm(hidden_dim)
        self.norm2a = nn.LayerNorm(hidden_dim)
        self.norm2b = nn.LayerNorm(hidden_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens1: torch.Tensor, tokens2: torch.Tensor) -> tuple:
        """
        Args:
            tokens1: (B, N1, D) e.g., omics tokens
            tokens2: (B, N2, D) e.g., histology tokens
        Returns:
            updated_tokens1, updated_tokens2
        """
        # tokens1 attends to tokens2
        attn_out1, _ = self.cross_attn_1to2(
            query=tokens1, key=tokens2, value=tokens2
        )
        tokens1 = self.norm1a(tokens1 + attn_out1)
        tokens1 = self.norm1b(tokens1 + self.ffn1(tokens1))

        # tokens2 attends to tokens1
        attn_out2, _ = self.cross_attn_2to1(
            query=tokens2, key=tokens1, value=tokens1
        )
        tokens2 = self.norm2a(tokens2 + attn_out2)
        tokens2 = self.norm2b(tokens2 + self.ffn2(tokens2))

        return tokens1, tokens2


class MultiModalFusion(nn.Module):
    """Fuse omics tokens and histology tokens via cross-attention layers."""

    def __init__(self, hidden_dim: int = 256, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        # Self-attention over all tokens after cross-attention
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )

    def forward(self, omics_tokens: torch.Tensor, histo_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            omics_tokens: (B, N_omics, D) concatenated genomic + pathway + protein tokens
            histo_tokens: (B, N_histo, D) histology tokens (optional, None if missing)
        Returns:
            fused: (B, N_total, D) fused representation
        """
        if histo_tokens is not None:
            for layer in self.layers:
                omics_tokens, histo_tokens = layer(omics_tokens, histo_tokens)
            all_tokens = torch.cat([omics_tokens, histo_tokens], dim=1)
        else:
            all_tokens = omics_tokens

        fused = self.self_attn(all_tokens)
        return fused


# ---------------------------------------------------------------------------
# 4. Prediction Head
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """Drug response prediction from fused multi-modal representation."""

    def __init__(self, hidden_dim: int = 256, n_drugs: int = 1, task: str = 'regression', dropout: float = 0.2):
        super().__init__()
        self.task = task
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_drugs),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_tokens: (B, N, D) fused multi-modal tokens
        Returns:
            pred: (B, n_drugs) predicted IC50 or response probability
        """
        # Attention-weighted pooling over tokens
        attn_scores = self.pool_attn(fused_tokens).squeeze(-1)  # (B, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N)
        pooled = torch.bmm(attn_weights.unsqueeze(1), fused_tokens).squeeze(1)  # (B, D)

        pred = self.head(pooled)  # (B, n_drugs)
        if self.task == 'classification':
            pred = torch.sigmoid(pred)
        return pred


# ---------------------------------------------------------------------------
# 5. Full Model: PathOmicDRP
# ---------------------------------------------------------------------------

class PathOmicDRP(nn.Module):
    """
    PathOmicDRP: Multi-modal drug response prediction model.

    Integrates genomic (mutation), transcriptomic (pathway scores),
    proteomic (RPPA), and histopathology (UNI features) through
    cross-attention fusion.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        hidden_dim = config.get('hidden_dim', 256)
        dropout = config.get('dropout', 0.1)

        # --- KD_GNN preprocessor (genomic + proteomic) ---
        self.use_kd_gnn = config.get('use_kd_gnn', True)
        kd_hidden = config.get('kd_gnn_hidden', 32)
        kd_layers = config.get('kd_gnn_layers', 2)
        kd_dropout = config.get('kd_gnn_dropout', dropout)
        kd_in_dim_g = config.get('kd_gnn_in_dim_genomic', 1)
        kd_in_dim_p = config.get('kd_gnn_in_dim_proteomic', 1)
        kd_ema = config.get('kd_ema_decay', 0.99)
        self.lambda_kd = config.get('lambda_kd', 0.05)

        if self.use_kd_gnn:
            if config['genomic_dim'] % kd_in_dim_g != 0:
                raise ValueError(
                    "genomic_dim must be divisible by kd_gnn_in_dim_genomic "
                    f"(got {config['genomic_dim']} % {kd_in_dim_g})"
                )
            if config['proteomic_dim'] % kd_in_dim_p != 0:
                raise ValueError(
                    "proteomic_dim must be divisible by kd_gnn_in_dim_proteomic "
                    f"(got {config['proteomic_dim']} % {kd_in_dim_p})"
                )
            n_genomic_nodes = config['genomic_dim'] // kd_in_dim_g
            n_proteomic_nodes = config['proteomic_dim'] // kd_in_dim_p

            self.genomic_kd_gnn = OmicsKDGNN(
                n_nodes=n_genomic_nodes,
                in_dim=kd_in_dim_g,
                hidden_dim=kd_hidden,
                n_layers=kd_layers,
                dropout=kd_dropout,
                ema_decay=kd_ema,
            )
            self.proteomic_kd_gnn = OmicsKDGNN(
                n_nodes=n_proteomic_nodes,
                in_dim=kd_in_dim_p,
                hidden_dim=kd_hidden,
                n_layers=kd_layers,
                dropout=kd_dropout,
                ema_decay=kd_ema,
            )
            genomic_encoder_input_dim = kd_hidden
            proteomic_encoder_input_dim = kd_hidden
        else:
            self.genomic_kd_gnn = None
            self.proteomic_kd_gnn = None
            genomic_encoder_input_dim = config['genomic_dim']
            proteomic_encoder_input_dim = config['proteomic_dim']

        # --- Modality Encoders ---
        self.genomic_encoder = GenomicEncoder(
            input_dim=genomic_encoder_input_dim,
            hidden_dim=hidden_dim,
            n_tokens=config.get('genomic_tokens', 8),
            dropout=dropout,
        )
        self.pathway_tokenizer = PathwayTokenizer(
            n_pathways=config['n_pathways'],
            genes_per_pathway=1,  # ssGSEA scores: 1 value per pathway
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.proteomic_encoder = ProteomicEncoder(
            input_dim=proteomic_encoder_input_dim,
            hidden_dim=hidden_dim,
            n_tokens=config.get('proteomic_tokens', 16),
            dropout=dropout,
        )

        # Histology branch (optional — used when H&E features available)
        self.use_histology = config.get('use_histology', False)
        if self.use_histology:
            self.histology_encoder = ABMIL(
                feature_dim=config.get('histo_feature_dim', 1024),
                hidden_dim=hidden_dim,
                dropout=dropout,
                n_tokens=config.get('histo_tokens', 16),
            )

        # --- Fusion ---
        self.fusion = MultiModalFusion(
            hidden_dim=hidden_dim,
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_fusion_layers', 2),
            dropout=dropout,
        )

        # --- Prediction Head ---
        self.prediction_head = PredictionHead(
            hidden_dim=hidden_dim,
            n_drugs=config.get('n_drugs', 1),
            task=config.get('task', 'regression'),
            dropout=config.get('head_dropout', 0.2),
        )

        # --- Modality dropout for robustness ---
        self.modality_dropout_p = config.get('modality_dropout', 0.0)

    def _modality_dropout(self, tokens: torch.Tensor, name: str) -> torch.Tensor:
        """Randomly zero out entire modality tokens during training."""
        if self.training and self.modality_dropout_p > 0:
            if torch.rand(1).item() < self.modality_dropout_p:
                return torch.zeros_like(tokens)
        return tokens

    def forward(
        self,
        genomic: torch.Tensor,
        transcriptomic: torch.Tensor,
        proteomic: torch.Tensor,
        histology: torch.Tensor = None,
        histo_mask: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            genomic: (B, genomic_dim) mutation features (or (B, n_nodes, in_dim))
            transcriptomic: (B, n_pathways) pathway scores
            proteomic: (B, proteomic_dim) RPPA features (or (B, n_nodes, in_dim))
            histology: (B, N_patches, histo_feature_dim) UNI patch features (optional)
            histo_mask: (B, N_patches) valid patch mask (optional)
        Returns:
            dict with 'prediction', plus optionally 'kd_loss' (training only)
            and 'histo_attention'.
        """
        # --- KD_GNN preprocessing for genomic + proteomic ---
        kd_loss_total = None
        if self.use_kd_gnn:
            gen_kd = self.genomic_kd_gnn(genomic)
            prot_kd = self.proteomic_kd_gnn(proteomic)
            gen_input = gen_kd['pooled']    # (B, kd_hidden)
            prot_input = prot_kd['pooled']  # (B, kd_hidden)

            if 'teacher_emb' in gen_kd and 'teacher_emb' in prot_kd:
                kd_g = OmicsKDGNN.kd_loss(gen_kd['student_emb'], gen_kd['teacher_emb'])
                kd_p = OmicsKDGNN.kd_loss(prot_kd['student_emb'], prot_kd['teacher_emb'])
                kd_loss_total = kd_g + kd_p
        else:
            gen_input = genomic
            prot_input = proteomic

        # Encode each modality into tokens
        gen_tokens = self._modality_dropout(
            self.genomic_encoder(gen_input), 'genomic'
        )
        path_tokens = self._modality_dropout(
            self.pathway_tokenizer(transcriptomic), 'transcriptomic'
        )
        prot_tokens = self._modality_dropout(
            self.proteomic_encoder(prot_input), 'proteomic'
        )

        # Concatenate omics tokens
        omics_tokens = torch.cat([gen_tokens, path_tokens, prot_tokens], dim=1)

        # Histology branch
        histo_tokens = None
        histo_attn = None
        if self.use_histology and histology is not None:
            histo_tokens, histo_attn = self.histology_encoder(histology, histo_mask)
            histo_tokens = self._modality_dropout(histo_tokens, 'histology')

        # Cross-attention fusion
        fused = self.fusion(omics_tokens, histo_tokens)

        # Prediction
        prediction = self.prediction_head(fused)

        output = {'prediction': prediction}
        if histo_attn is not None:
            output['histo_attention'] = histo_attn
        if kd_loss_total is not None:
            output['kd_loss'] = kd_loss_total

        return output

    @torch.no_grad()
    def update_teacher(self, decay: float = None) -> None:
        """EMA update of KD_GNN teachers from the current student weights.

        Call this once per optimizer step (after backward + step).
        """
        if self.use_kd_gnn:
            self.genomic_kd_gnn.update_teacher(decay)
            self.proteomic_kd_gnn.update_teacher(decay)

    def set_genomic_adjacency(self, adj: torch.Tensor) -> None:
        """Inject a biological adjacency (pathway / signaling) for genomic GNN."""
        if self.use_kd_gnn:
            self.genomic_kd_gnn.set_adjacency(adj)

    def set_proteomic_adjacency(self, adj: torch.Tensor) -> None:
        """Inject a biological adjacency (PPI / phosphorylation) for proteomic GNN."""
        if self.use_kd_gnn:
            self.proteomic_kd_gnn.set_adjacency(adj)


# ---------------------------------------------------------------------------
# 6. Configuration helpers
# ---------------------------------------------------------------------------

def get_default_config(
    genomic_dim: int = 193,
    n_pathways: int = 50,
    proteomic_dim: int = 464,
    n_drugs: int = 1,
    use_histology: bool = False,
) -> dict:
    """Return default model configuration."""
    return {
        'genomic_dim': genomic_dim,
        'n_pathways': n_pathways,
        'proteomic_dim': proteomic_dim,
        'hidden_dim': 256,
        'dropout': 0.1,
        'head_dropout': 0.2,
        'genomic_tokens': 8,
        'proteomic_tokens': 16,
        'histo_tokens': 16,
        'histo_feature_dim': 1024,  # UNI ViT-L output dim
        'n_heads': 8,
        'n_fusion_layers': 2,
        'n_drugs': n_drugs,
        'task': 'regression',
        'use_histology': use_histology,
        'modality_dropout': 0.15,
        # --- KD_GNN settings (EMA self-distillation over gene/protein graphs) ---
        'use_kd_gnn': True,
        'kd_gnn_hidden': 32,
        'kd_gnn_layers': 2,
        'kd_gnn_dropout': 0.1,
        'kd_gnn_in_dim_genomic': 1,    # 1 if mutation-only; 2 for mutation + CNV
        'kd_gnn_in_dim_proteomic': 1,  # RPPA expression scalar per protein
        'kd_ema_decay': 0.99,
        'lambda_kd': 0.05,
    }


if __name__ == '__main__':
    # Quick test
    config = get_default_config(use_histology=True)
    model = PathOmicDRP(config)

    B = 4
    genomic = torch.randn(B, 193)
    transcriptomic = torch.randn(B, 50)
    proteomic = torch.randn(B, 464)
    histology = torch.randn(B, 100, 1024)  # 100 patches, UNI features

    model.train()
    output = model(genomic, transcriptomic, proteomic, histology)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Prediction shape: {output['prediction'].shape}")
    print(f"Histo attention shape: {output['histo_attention'].shape}")
    print(f"KD loss (train mode): {output.get('kd_loss')}")

    # Eval mode — KD loss should NOT be present
    model.eval()
    out_eval = model(genomic, transcriptomic, proteomic, histology)
    print(f"KD loss in eval mode: {out_eval.get('kd_loss')}  (should be None)")

    # Without histology, no KD_GNN
    config_no_histo = get_default_config(use_histology=False)
    config_no_histo['use_kd_gnn'] = False
    model2 = PathOmicDRP(config_no_histo)
    output2 = model2(genomic, transcriptomic, proteomic)
    print(f"\nOmics-only (no KD_GNN) parameters: {sum(p.numel() for p in model2.parameters()):,}")
    print(f"Omics-only prediction: {output2['prediction'].shape}")
