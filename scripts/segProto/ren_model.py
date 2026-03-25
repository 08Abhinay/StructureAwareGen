"""
Self-contained REN components for DINOv2 ViT-L/14 region token extraction.

Extracted from the REN repo (https://github.com/savyak2/ren):
  - model.py: PositionalEmbedding2D, AttentionLayer, MLPBlock,
              CrossAttentionBlock, RegionEncoder, TokenAggregator
  - task_utils.py: CenterPadding, upsample_features, group_predictions
  - ren.py: get_slic_points (→ SLICPrompter class)

This file is self-contained — no SAM2, OpenCLIP, or DINO ViT-B/8 imports.
Only loads DINOv2 ViT-L/14 via torch.hub when DINOv2Extractor is used.
"""

import os
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Utilities (from task_utils.py)
# ---------------------------------------------------------------------------

class CenterPadding(nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(
            self._get_pad(m) for m in x.shape[:1:-1]
        ))
        return F.pad(x, pads)


def upsample_features(image_features, new_h, new_w, padded_h, padded_w,
                       upsampling_method='bilinear'):
    if upsampling_method == 'bilinear':
        upsampled = F.interpolate(
            image_features, size=[padded_h, padded_w], mode='bilinear'
        )
        upsampled = T.CenterCrop((new_h, new_w))(upsampled)
    else:
        raise ValueError(f'{upsampling_method} is not a valid upsampling method.')
    return upsampled


def group_predictions(preds, similarity_threshold=0.9, min_component_size=3,
                      merge_small_groups=False):
    batch_size = preds.shape[0]
    features = F.normalize(preds.view(batch_size, -1), p=2, dim=1)
    max_chunk = 2048
    rows, cols, values = [], [], []
    for i in range(0, batch_size, max_chunk):
        end_i = min(i + max_chunk, batch_size)
        chunk_i = features[i:end_i]

        for j in range(i, batch_size, max_chunk):
            end_j = min(j + max_chunk, batch_size)
            chunk_j = features[j:end_j]
            similarities = torch.mm(chunk_i, chunk_j.t())
            sim_mask = similarities >= similarity_threshold
            chunk_rows, chunk_cols = sim_mask.nonzero(as_tuple=True)
            if i == j:
                valid_edges = chunk_rows != chunk_cols
                chunk_rows_valid = chunk_rows[valid_edges] + i
                chunk_cols_valid = chunk_cols[valid_edges] + j
            else:
                chunk_rows_valid = chunk_rows + i
                chunk_cols_valid = chunk_cols + j

            rows.extend(chunk_rows_valid.cpu().numpy())
            cols.extend(chunk_cols_valid.cpu().numpy())
            values.extend([1] * len(chunk_rows_valid))
            if i != j:
                rows.extend(chunk_cols_valid.cpu().numpy())
                cols.extend(chunk_rows_valid.cpu().numpy())
                values.extend([1] * len(chunk_rows_valid))

    adj_matrix = csr_matrix(
        (values, (rows, cols)), shape=(batch_size, batch_size)
    )
    n_components, labels = connected_components(
        csgraph=adj_matrix, directed=False, return_labels=True
    )
    groups = [[] for _ in range(n_components)]
    for idx, label in enumerate(labels):
        groups[label].append(idx)

    large_groups = [g for g in groups if len(g) >= min_component_size]
    if not merge_small_groups or min_component_size <= 1:
        return large_groups

    small_groups = [g for g in groups if len(g) < min_component_size]
    if not small_groups or not large_groups:
        return large_groups + small_groups

    large_centroids = []
    for group in large_groups:
        group_tensor = torch.tensor(group, device=features.device)
        group_features = features[group_tensor]
        large_centroids.append(torch.mean(group_features, dim=0))
    large_centroids_tensor = torch.stack(large_centroids, dim=0)
    large_centroids_normalized = F.normalize(large_centroids_tensor, p=2, dim=1)

    all_small_centroids = []
    for small_group in small_groups:
        group_tensor = torch.tensor(small_group, device=features.device)
        group_features = features[group_tensor]
        all_small_centroids.append(torch.mean(group_features, dim=0))

    if all_small_centroids:
        all_small_centroids_tensor = torch.stack(all_small_centroids, dim=0)
        all_small_centroids_normalized = F.normalize(
            all_small_centroids_tensor, p=2, dim=1
        )
        similarities = torch.mm(
            all_small_centroids_normalized, large_centroids_normalized.t()
        )
        best_matches = torch.argmax(similarities, dim=1).cpu().numpy()
        for idx, best_idx in enumerate(best_matches):
            large_groups[best_idx].extend(small_groups[idx])
    return large_groups


# ---------------------------------------------------------------------------
# DINOv2 Feature Extractor (only loads DINOv2 ViT-L/14)
# ---------------------------------------------------------------------------

class DINOv2Extractor:
    """
    Stripped-down feature extractor that loads ONLY DINOv2 ViT-L/14.
    Unlike the original FeatureExtractor, this does NOT load DINO ViT-B/8
    or OpenCLIP ViT-g/14, saving ~2GB+ of GPU memory.
    """

    @staticmethod
    def _patch_dinov2_for_py39():
        """
        The DINOv2 main branch uses Python 3.10+ union syntax (float | None).
        Patch cached hub files to add 'from __future__ import annotations'
        so they work under Python 3.9.
        """
        import sys
        if sys.version_info >= (3, 10):
            return  # no patching needed

        hub_dir = torch.hub.get_dir()
        dinov2_dir = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
        if not os.path.isdir(dinov2_dir):
            return

        import glob
        py_files = glob.glob(os.path.join(dinov2_dir, '**', '*.py'), recursive=True)
        future_import = 'from __future__ import annotations\n'
        for fpath in py_files:
            with open(fpath, 'r') as f:
                content = f.read()
            if future_import.strip() in content:
                continue
            # Only patch files that actually use union type syntax
            if ' | ' in content:
                with open(fpath, 'w') as f:
                    f.write(future_import + content)

    def __init__(self, device, torch_home=None):
        self.device = device
        if torch_home is not None:
            os.environ['TORCH_HOME'] = torch_home

        hub_dir = torch.hub.get_dir()
        repo_dir = os.path.join(hub_dir, 'facebookresearch_dinov2_main')

        # Ensure repo code is cached
        if not os.path.isdir(repo_dir):
            # Attempt a load which downloads the code; may fail on Py3.9
            try:
                torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14',
                               trust_repo=True)
            except (TypeError, Exception):
                pass  # code is now cached even if import failed

        # Patch type annotations for Python 3.9 compatibility
        self._patch_dinov2_for_py39()

        # Load from local cached directory
        self.model = torch.hub.load(
            repo_dir, 'dinov2_vitl14', source='local', trust_repo=True
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def extract(self, images, patch_length=14, layers=[23]):
        """
        Extract DINOv2 features (CLS token + feature maps).

        Args:
            images: [B, 3, H, W] raw tensor (0-1 range, NOT normalized).
            patch_length: Patch size (14 for DINOv2 ViT-L/14).
            layers: Which intermediate layer(s) to extract from.

        Returns:
            feature_maps: [B, C, H_patch, W_patch] — for RegionEncoder.
            cls_tokens:   [B, 1024] — DINOv2 CLS token.
        """
        import kornia
        transform = kornia.augmentation.AugmentationSequential(
            CenterPadding(multiple=patch_length),
            kornia.augmentation.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        )
        transformed_images = transform(images)

        feature_maps_list, cls_tokens_list = [], []
        with torch.inference_mode():
            for i in range(0, transformed_images.shape[0], 32):
                batch = transformed_images[i:i + 32].to(device=self.device)
                features_out = self.model.get_intermediate_layers(
                    batch, return_class_token=True, n=layers
                )[0]
                cls_token = features_out[1]      # [B, 1024]
                patch_feat = features_out[0]     # [B, N, 1024]
                cls_tokens_list.append(cls_token)

                B, _, C = patch_feat.size()
                H, W = batch.shape[2], batch.shape[3]
                pH = math.ceil(H / patch_length)
                pW = math.ceil(W / patch_length)
                fmap = patch_feat.permute(0, 2, 1).view(B, C, pH, pW)
                feature_maps_list.append(fmap)

        feature_maps = torch.cat(feature_maps_list, dim=0)
        cls_tokens = torch.cat(cls_tokens_list, dim=0)
        return feature_maps, cls_tokens


# ---------------------------------------------------------------------------
# SLIC Prompter (from ren.py get_slic_points)
# ---------------------------------------------------------------------------

class SLICPrompter:
    """
    Generate SLIC superpixel center-of-mass prompts for REN.
    Extracted from REN.get_slic_points() in ren.py.
    """

    def __init__(self, image_resolution=518):
        self.image_resolution = image_resolution

    def get_grid_points(self, grid_size):
        """Generate uniform grid prompts (fallback if fast_slic unavailable)."""
        x_coords = np.linspace(1, self.image_resolution - 2, grid_size, dtype=int)
        y_coords = np.linspace(1, self.image_resolution - 2, grid_size, dtype=int)
        return torch.tensor([(y, x) for y in y_coords for x in x_coords])

    def __call__(self, images, num_segments, use_slic=True):
        """
        Args:
            images: [B, 3, H, W] tensor (0-1 range, unnormalized).
            num_segments: Target number of superpixel segments.
            use_slic: If True, use fast_slic; otherwise use grid.

        Returns:
            prompts: list of [N_prompts, 2] tensors (y, x coords).
        """
        from scipy import ndimage as ndi

        if not use_slic:
            grid_size = int(math.sqrt(num_segments))
            grid_pts = self.get_grid_points(grid_size)
            return [grid_pts for _ in range(images.shape[0])]

        from fast_slic import Slic

        prompts = []
        for image in images:
            img_np = (image.permute(1, 2, 0).cpu().numpy().copy() * 255
                      ).astype(np.uint8)

            slic = Slic(num_components=num_segments, compactness=256)
            segments = slic.iterate(img_np)
            slic_segments = segments.max() + 1

            centers = np.array(ndi.center_of_mass(
                np.ones_like(segments), labels=segments,
                index=np.arange(slic_segments)
            ))
            centers = np.round(centers).astype(int)
            centers[:, 0] = np.clip(centers[:, 0], 0, segments.shape[0] - 1)
            centers[:, 1] = np.clip(centers[:, 1], 0, segments.shape[1] - 1)

            valid = (segments[centers[:, 0], centers[:, 1]]
                     == np.arange(slic_segments))
            if not np.all(valid):
                for seg_id in np.where(~valid)[0]:
                    mask = (segments == seg_id)
                    yx = np.argwhere(mask)
                    if len(yx) > 0:
                        centers[seg_id] = yx[len(yx) // 2]

            centers = torch.tensor(centers, dtype=torch.int64)

            # Pad if needed
            pad_len = num_segments - len(centers)
            if pad_len > 0:
                center_padding = torch.stack([centers[-1]] * pad_len)
                centers = torch.cat([centers, center_padding], dim=0)

            prompts.append(centers)
        return prompts


# ---------------------------------------------------------------------------
# Positional Embedding (from model.py)
# ---------------------------------------------------------------------------

class PositionalEmbedding2D(nn.Module):
    def __init__(self, embedding_dim=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        generator = torch.Generator()
        generator.manual_seed(42)
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, embedding_dim // 2), generator=generator),
        )

    def _pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Attention / Transformer blocks (from model.py)
# ---------------------------------------------------------------------------

class AttentionLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads=8, dropout=0.1,
                 use_bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        nn.init.kaiming_normal_(self.q_proj.weight, mode='fan_in',
                                nonlinearity='linear')
        nn.init.kaiming_normal_(self.k_proj.weight, mode='fan_in',
                                nonlinearity='linear')
        nn.init.kaiming_normal_(self.v_proj.weight, mode='fan_in',
                                nonlinearity='linear')
        if use_bias:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_in',
                                nonlinearity='linear')
        if use_bias:
            nn.init.zeros_(self.out_proj.bias)

        self.scale = (hidden_dim // num_heads) ** -0.5

    def forward(self, q, k, v, mask=None, project_values=True,
                attention_threshold=None):
        batch_size, q_len, _ = q.shape
        _, kv_len, _ = k.shape

        query = self.q_proj(q).view(
            batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        key = self.k_proj(k).view(
            batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        if project_values:
            value = self.v_proj(v).view(
                batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        else:
            value = v.view(
                batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        if attention_threshold is not None:
            max_attn_scores, _ = attn_scores.max(dim=-1, keepdim=True)
            thresholding_mask = attn_scores >= (
                attention_threshold * max_attn_scores)
            attn_scores = attn_scores.masked_fill(
                thresholding_mask == 0, -1e5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, value)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.hidden_dim)

        out = self.out_proj(attn_out)
        return out, attn_scores


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.linear1(x)
        z = self.gelu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        return z


class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, mlp_dim, num_heads,
                 dropout, use_bias):
        super().__init__()
        self.query_norm = nn.LayerNorm(q_dim)
        self.cross_attn = AttentionLayer(
            q_dim, kv_dim, hidden_dim, num_heads, dropout, use_bias)
        self.dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, context, mask=None, project_values=True):
        x = self.query_norm(query)
        x, attn_scores = self.cross_attn(
            q=x, k=context, v=context, mask=mask,
            project_values=project_values)
        x = self.dropout(x)
        x = x + query

        y = self.mlp_norm(x)
        y = self.mlp(y)
        out = self.out_norm(y) + x
        return out, attn_scores


# ---------------------------------------------------------------------------
# RegionEncoder (from model.py)
# ---------------------------------------------------------------------------

class RegionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['architecture']['hidden_dim']
        image_resolution = config['parameters']['image_resolution']
        upsample_features = config['parameters']['upsample_features']
        patch_size = config['pretrained']['patch_sizes'][0]

        position_embedder = PositionalEmbedding2D(hidden_dim)
        if upsample_features:
            self.location_embeddings = position_embedder(
                (image_resolution, image_resolution))
            self.feature_embeddings = self.location_embeddings.flatten(
                -2).permute(1, 0)
        else:
            self.location_embeddings = position_embedder(
                (image_resolution, image_resolution))
            self.feature_embeddings = position_embedder(
                (image_resolution // patch_size,
                 image_resolution // patch_size)
            ).flatten(-2).permute(1, 0)

        self.prompt_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.prompt_proj.weight, mode='fan_in',
                                nonlinearity='linear')

        self.decoder_layers = config['architecture']['decoder_layers']
        self.region_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                q_dim=hidden_dim,
                kv_dim=hidden_dim,
                hidden_dim=hidden_dim,
                mlp_dim=2 * hidden_dim,
                num_heads=config['architecture']['num_attention_heads'],
                dropout=0.1,
                use_bias=False,
            ) for _ in range(self.decoder_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.out_proj.weight, mode='fan_in',
                                nonlinearity='linear')

    def forward(self, feature_maps, grid_points):
        device = feature_maps.device
        feature_tokens = feature_maps.flatten(-2).permute(0, 2, 1)
        feat_emb = self.feature_embeddings.to(device)
        kv = feature_tokens + feat_emb[None]

        batch_size = feature_maps.shape[0]
        loc_emb = self.location_embeddings.to(device)
        prompt_embeddings = [
            torch.stack([loc_emb[:, pt[0], pt[1]] for pt in grid_points[i]])
            for i in range(batch_size)
        ]
        prompt_embeddings = torch.stack(prompt_embeddings).to(device)
        q = self.prompt_proj(prompt_embeddings)

        all_attn_scores = []
        for layer_idx, layer in enumerate(self.region_attention_layers):
            q = q + prompt_embeddings
            if layer_idx == self.decoder_layers - 1:
                pred_tokens, attn_scores = layer(
                    q, kv, project_values=False)
            else:
                q, attn_scores = layer(q, kv)
            all_attn_scores.append(attn_scores)

        proj_tokens = self.out_norm(pred_tokens)
        proj_tokens = self.out_proj(proj_tokens)
        return {
            'pred_tokens': pred_tokens,
            'proj_tokens': proj_tokens,
            'attn_scores': all_attn_scores,
        }


# ---------------------------------------------------------------------------
# TokenAggregator (from model.py)
# ---------------------------------------------------------------------------

class TokenAggregator(nn.Module):
    def __init__(self, merge_similarity=0.975):
        super().__init__()
        self.merge_similarity = merge_similarity

    def get_central_point(self, points):
        center = points.float().mean(dim=0, keepdim=True)
        dists = torch.norm(points.float() - center, dim=1)
        return points[dists.argmin()]

    def forward(self, pred_tokens, proj_tokens, attn_scores, grid_points):
        batch_size = attn_scores.size(0)
        results = {
            'aggregated_pred_tokens': [],
            'aggregated_proj_tokens': [],
            'aggregated_attn_scores': [],
            'aggregated_grid_points': [],
        }
        for batch_idx in range(batch_size):
            groups = group_predictions(
                pred_tokens[batch_idx], self.merge_similarity)
            b_pred = pred_tokens[batch_idx]
            b_proj = proj_tokens[batch_idx]
            b_attn = attn_scores[batch_idx]
            b_pts = grid_points[batch_idx].to(pred_tokens.device)

            new_pred, new_proj, new_attn, new_pts = [], [], [], []
            for group in groups:
                gt = torch.tensor(group, device=b_pred.device)
                new_pred.append(b_pred[gt].mean(dim=0))
                new_proj.append(b_proj[gt].mean(dim=0))
                new_attn.append(b_attn[:, gt].mean(dim=1))
                new_pts.append(self.get_central_point(b_pts[gt]))

            if new_pred:
                results['aggregated_pred_tokens'].append(
                    torch.stack(new_pred, dim=0))
                results['aggregated_proj_tokens'].append(
                    torch.stack(new_proj, dim=0))
                results['aggregated_attn_scores'].append(
                    torch.stack(new_attn, dim=1))
                results['aggregated_grid_points'].append(
                    torch.stack(new_pts, dim=0))
            else:
                # No groups found — return empty tensors with correct dims
                hidden = b_pred.shape[-1]
                n_heads = b_attn.shape[0]
                hw = b_attn.shape[-1]
                results['aggregated_pred_tokens'].append(
                    torch.zeros(0, hidden, device=b_pred.device))
                results['aggregated_proj_tokens'].append(
                    torch.zeros(0, hidden, device=b_pred.device))
                results['aggregated_attn_scores'].append(
                    torch.zeros(n_heads, 0, hw, device=b_attn.device))
                results['aggregated_grid_points'].append(
                    torch.zeros(0, 2, device=b_pred.device, dtype=torch.long))
        return results
