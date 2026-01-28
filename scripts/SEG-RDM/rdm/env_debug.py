import sys
import types


def _get_version(obj):
    ver = getattr(obj, "__version__", None)
    if ver is not None:
        return ver
    version_fn = getattr(obj, "version", None)
    if callable(version_fn):
        try:
            return version_fn()
        except Exception:
            return None
    return None


def print_env(module_name, module_globals):
    print(f"[env] module={module_name}")
    print(f"[env] python={sys.version.replace(chr(10), ' ')}")
    for name, obj in sorted(module_globals.items()):
        if not isinstance(obj, types.ModuleType):
            continue
        mod_name = getattr(obj, "__name__", name)
        ver = _get_version(obj)
        if ver is None:
            continue
        print(f"[env] {mod_name}={ver}")

    # Optional CUDA info when torch is available in this module's globals.
    torch_mod = module_globals.get("torch")
    if isinstance(torch_mod, types.ModuleType):
        cuda_ver = getattr(getattr(torch_mod, "version", None), "cuda", None)
        if cuda_ver:
            print(f"[env] torch.cuda={cuda_ver}")
        cudnn = getattr(torch_mod.backends, "cudnn", None)
        if cudnn is not None:
            try:
                cudnn_ver = cudnn.version()
            except Exception:
                cudnn_ver = None
            if cudnn_ver is not None:
                print(f"[env] cudnn={cudnn_ver}")
