import pickle
class _Stub:
    def __init__(self, *a, **k): self._init_args, self._init_kwargs = a, k
    def __setstate__(self, st): self._state = st
MISSING = set()
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            MISSING.add(f"{module}.{name}")
            return type(name, (_Stub,), {"__module__": module})
def load_pickle(path):
    with open(path, "rb") as fh:
        return SafeUnpickler(fh).load()
