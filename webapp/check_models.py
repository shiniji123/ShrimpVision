from config import Config
from pathlib import Path

for key, meta in Config.MODELS.items():
    trained = Path(meta['trained_path']).exists()
    fallback = Path(meta['fallback_path']).exists()
    status = '[TRAINED]' if trained else ('[FALLBACK]' if fallback else '[MISSING]')
    print(f"{status}  {key:12s}  {meta['name']:12s}  [{meta['badge']:12s}]  {meta['speed_tier']}")
