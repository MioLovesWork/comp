# 非比赛实验/调参 — dinov3 环境，勿在 work.ipynb 引用
from pathlib import Path
import sys

COMP = Path(__file__).resolve().parent.parent
TUNE = Path(__file__).resolve().parent


def setup_paths():
    """把 comp/ 与 tune/ 加入 sys.path，notebook 首格调用一次即可。"""
    for p in (COMP, TUNE):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return COMP, TUNE
