import torch
import torch.nn as nn
from rdm.pretrained_enc.moco_v3 import vits as moco_vits
from rdm.pretrained_enc.dino import vits as dino_vits
from rdm.pretrained_enc.ibot import vits as ibot_vits
from rdm.pretrained_enc.deit import vits as deit_vits
from rdm.pretrained_enc.ijepa import vision_transformer as ijepa_vits
from rdm.env_debug import print_env
print_env(__name__, globals())



def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

def load_pretrained_moco(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=True)
    return model


def load_pretrained_dino(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # rename dino pre-trained keys
    state_dict = checkpoint['teacher']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('backbone'):
            # remove prefix
            state_dict[k[len("backbone."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    del state_dict['head.last_layer.weight']
    model.load_state_dict(state_dict, strict=True)
    return model


def load_pretrained_ibot(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # rename ibot pre-trained keys
    state_dict = checkpoint['teacher']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('backbone'):
            # remove prefix
            state_dict[k[len("backbone."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    del state_dict['head.last_layer.weight_g']
    del state_dict['head.last_layer.weight_v']
    del state_dict['head.last_layer2.weight_g']
    del state_dict['head.last_layer2.weight_v']
    model.load_state_dict(state_dict, strict=True)
    return model


def load_pretrained_deit(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # load pre-trained model
    model.load_state_dict(checkpoint['model'])

    return model


def load_pretrained_mae(model, ckpt_path):
    raise NotImplementedError("MAE checkpoints are not wired in this port.")


def load_pretrained_ijepa(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("target_encoder", "encoder", "model", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("module.", "encoder.", "backbone."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned_state[new_k] = v

    model_state = model.state_dict()
    filtered_state = {}
    for k, v in cleaned_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_state[k] = v

    model.load_state_dict(filtered_state, strict=False)
    return model


def mocov3_vit_small(proj_dim, **kwargs):
    model = moco_vits.vit_small(**kwargs)
    hidden_dim = model.head.weight.shape[1]
    del model.head  # remove original head layer

    # projectors
    model.head = build_mlp(3, hidden_dim, 4096, proj_dim)
    return model


def mocov3_vit_base(proj_dim, **kwargs):
    model = moco_vits.vit_base(**kwargs)
    hidden_dim = model.head.weight.shape[1]
    del model.head  # remove original head layer

    # projectors
    model.head = build_mlp(3, hidden_dim, 4096, proj_dim)
    return model


def mocov3_vit_large(proj_dim, **kwargs):
    model = moco_vits.vit_large(**kwargs)
    hidden_dim = model.head.weight.shape[1]
    del model.head  # remove original head layer

    # projectors
    model.head = build_mlp(3, hidden_dim, 4096, proj_dim)
    return model


def dino_vit_base(proj_dim, **kwargs):
    model = dino_vits.vit_base()
    hidden_dim = model.head.weight.shape[1]
    del model.head  # remove original head layer

    model.head = dino_vits.DINOHead(in_dim=hidden_dim, bottleneck_dim=proj_dim)
    return model


def ibot_vit_base(proj_dim, **kwargs):
    model = ibot_vits.vit_base()
    hidden_dim = model.head.weight.shape[1]
    del model.head  # remove original head layer

    model.head = dino_vits.DINOHead(in_dim=hidden_dim, bottleneck_dim=proj_dim)
    return model


def deit_vit_base(proj_dim, **kwargs):
    model = deit_vits.deit_base_patch16_224()
    return model


def _ijepa_with_proj(model, proj_dim):
    hidden_dim = model.embed_dim
    model.head = build_mlp(3, hidden_dim, 4096, proj_dim)
    return model


def ijepa_vit_tiny(proj_dim, patch_size=16, **kwargs):
    model = ijepa_vits.vit_tiny(patch_size=patch_size, **kwargs)
    return _ijepa_with_proj(model, proj_dim)


def ijepa_vit_small(proj_dim, patch_size=16, **kwargs):
    model = ijepa_vits.vit_small(patch_size=patch_size, **kwargs)
    return _ijepa_with_proj(model, proj_dim)


def ijepa_vit_base(proj_dim, patch_size=16, **kwargs):
    model = ijepa_vits.vit_base(patch_size=patch_size, **kwargs)
    return _ijepa_with_proj(model, proj_dim)


def ijepa_vit_large(proj_dim, patch_size=16, **kwargs):
    model = ijepa_vits.vit_large(patch_size=patch_size, **kwargs)
    return _ijepa_with_proj(model, proj_dim)


def ijepa_vit_huge(proj_dim, patch_size=16, **kwargs):
    model = ijepa_vits.vit_huge(patch_size=patch_size, **kwargs)
    return _ijepa_with_proj(model, proj_dim)


def ijepa_vit_giant(proj_dim, patch_size=16, **kwargs):
    model = ijepa_vits.vit_giant(patch_size=patch_size, **kwargs)
    return _ijepa_with_proj(model, proj_dim)


def ijepa_vit_tiny_p14(proj_dim, **kwargs):
    return ijepa_vit_tiny(proj_dim, patch_size=14, **kwargs)


def ijepa_vit_small_p14(proj_dim, **kwargs):
    return ijepa_vit_small(proj_dim, patch_size=14, **kwargs)


def ijepa_vit_base_p14(proj_dim, **kwargs):
    return ijepa_vit_base(proj_dim, patch_size=14, **kwargs)


def ijepa_vit_large_p14(proj_dim, **kwargs):
    return ijepa_vit_large(proj_dim, patch_size=14, **kwargs)


def ijepa_vit_huge_p14(proj_dim, **kwargs):
    return ijepa_vit_huge(proj_dim, patch_size=14, **kwargs)


def ijepa_vit_giant_p14(proj_dim, **kwargs):
    return ijepa_vit_giant(proj_dim, patch_size=14, **kwargs)


def ijepa_vit_h14(proj_dim, **kwargs):
    return ijepa_vit_huge(proj_dim, patch_size=14, **kwargs)