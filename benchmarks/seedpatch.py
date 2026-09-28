"""Import after ../fdfinal.py: if FD_SEED is set, every FlashDeconv(...) uses
random_state=int(FD_SEED) (determinism check for scripts with a hard-coded seed)."""
import os

from flashdeconv.core.deconv import FlashDeconv

SEED = os.environ.get("FD_SEED")
if SEED is not None:
    _init = FlashDeconv.__init__

    def _seeded_init(self, *args, **kw):
        kw["random_state"] = int(SEED)
        _init(self, *args, **kw)

    FlashDeconv.__init__ = _seeded_init
    print(f"[seedpatch] random_state forced to {SEED}", flush=True)
