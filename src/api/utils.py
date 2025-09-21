from functools import lru_cache
from datetime import datetime, time
import pytz

PARIS_TZ = pytz.timezone("Europe/Paris")

def paris_now():
    return datetime.now(PARIS_TZ)

def is_euronext_open(now=None):
    """
    Heures standard Euronext Paris (hors jours fériés):
    Ouverture ~ 09:00, Clôture ~ 17:30 (CET/CEST).
    """
    now = now or paris_now()
    wd = now.weekday()  # 0=Mon
    if wd >= 5:
        return False
    open_t = time(9, 0)
    close_t = time(17, 30)
    tt = now.time()
    return open_t <= tt <= close_t

@lru_cache(maxsize=128)
def ttl_cache_key(*args, **kwargs):
    return str(args) + str(kwargs)
