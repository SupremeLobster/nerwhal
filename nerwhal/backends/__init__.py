from importlib import import_module


def load(backend):
    if backend == "spacy":
        mod = ".spacy_backend"
        cls = "SpacyBackend"
    elif backend == "re":
        mod = ".re_backend"
        cls = "ReBackend"
    else:
        raise ValueError(f"Unknow backend type {backend}")

    backend_cls = getattr(import_module(mod, package="nerwhal.backends"), cls)
    return backend_cls
