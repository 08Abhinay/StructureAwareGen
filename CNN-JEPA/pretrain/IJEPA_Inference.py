# /scratch/gilbreth/abelde/Thesis/CNN-JEPA/inference/IJEPA_Inference.py

import copy
import torch
from torch import nn
import torchvision.models as tv_models

# (If your original training used sparse/convnext modules, they are still imported 
#  so the checkpoint’s layer names match. If you never used sparse_encoder or convnext at all,
#  you can remove these imports.)
import models.sparse_encoder as sparse_encoder
import models.convnext  # registers any ConvNeXt variants, if used

class IJEPA_CNN_Inference(nn.Module):
    """
    A minimal inference‐only wrapper that reconstructs exactly the ResNet50‐based backbone
    used during IJEPA training, then loads only the 'backbone_momentum' (and optional
    'projection_head_momentum') weights from the Lightning checkpoint. No timm, no Lightly.
    """

    def __init__(self, cfg):
        super().__init__()

        # --- 1) Build the ResNet50 backbone exactly as Lightning did:
        # Lightning training code most likely did something akin to:
        #    self.backbone = timm.create_model("resnet50", pretrained=False, num_classes=0)
        #
        # We replicate that using torchvision.models.resnet50:
        #
        # torchvision's ResNet50 returns a final classifier head by default.
        # We want the “feature‐map” up to the last convolutional block (no FC).
        # So we take all layers except the final avgpool+fc. That yields a 2048×(H/32)×(W/32) tensor.
        full_resnet = tv_models.resnet50(pretrained=False)
        # Remove the final avgpool and fc: 
        #  └→ children() yields: 
        #      [Conv1, BN1, ReLU, MaxPool, layer1, layer2, layer3, layer4, avgpool, fc]
        #  We want everything up through `layer4`.
        modules = list(full_resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        # At this point, self.backbone(x) will output a (B, 2048, H/32, W/32) tensor.

        # --- 2) Create the “momentum” copy and freeze it:
        self.backbone_momentum = copy.deepcopy(self.backbone)
        for p in self.backbone_momentum.parameters():
            p.requires_grad = False

        # --- 3) If your training config used a projection head, rebuild it here exactly:
        #
        # In your Lightning, you did something like:
        #    if cfg.use_projection_head:
        #        proj_layers = [
        #            Conv2d(num_features, num_features, kernel_size=1, padding="same"),
        #            NormLayer(num_features),
        #            ReLU(),
        #        ]
        #
        # Under PyTorch 1.7, “padding='same'” is not supported, but for a 1×1 conv,
        # “same” means padding=0. So we rebuild it here using padding=0 and the same Norm.
        if cfg.use_projection_head:
            # Determine which normalization was used:
            if cfg.backbone.name.lower().startswith("resnet") or cfg.backbone.name.lower().startswith("wide_resnet"):
                norm_cls = nn.BatchNorm2d
            else:
                # (If you used ConvNeXt or some other backbone, you probably used LayerNorm2d from timm)
                from timm.layers import LayerNorm2d
                norm_cls = LayerNorm2d

            proj_layers = []
            proj_layers.append(
                nn.Conv2d(
                    in_channels=2048,
                    out_channels=2048,
                    kernel_size=1,
                    padding=0,  # “same” for 1×1 is padding=0
                )
            )
            proj_layers.append(norm_cls(2048))
            proj_layers.append(nn.ReLU(inplace=True))

            self.projection_head_momentum = nn.Sequential(*proj_layers)
            for p in self.projection_head_momentum.parameters():
                p.requires_grad = False
        else:
            self.projection_head_momentum = None

    def load_weights_from_checkpoint(self, ckpt_path):
        """
        1) torch.load(..., map_location="cuda") to get the checkpoint dict
        2) Filter only the keys that start with “backbone_momentum.” 
           and (if projection head exists) “projection_head_momentum.”
        3) Load those into this module with strict=False
        """
        checkpoint = torch.load(ckpt_path, map_location="cuda")
        state_dict = checkpoint["state_dict"]

        filtered = {}
        for key, val in state_dict.items():
            if key.startswith("backbone_momentum."):
                # We expect the key names in state_dict to match our submodule names exactly.
                # e.g. “backbone_momentum.0.weight”, “backbone_momentum.1.bias”, etc.
                filtered[key] = val

            if self.projection_head_momentum is not None and key.startswith("projection_head_momentum."):
                filtered[key] = val

        self.load_state_dict(filtered, strict=False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        1) Run x through backbone_momentum → (B, 2048, H/32, W/32)
        2) If projection_head_momentum exists, run through that
        3) Global‐avg‐pool to (B, 2048, 1, 1)
        4) Flatten to (B, 2048)
        """
        feats = self.backbone_momentum(x)  # → (B, 2048, h, w)
        if self.projection_head_momentum is not None:
            feats = self.projection_head_momentum(feats)  # → still (B, 2048, h, w)

        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1)  # → (B, 2048, 1, 1)
        return pooled.flatten(1)  # → (B, 2048)
