# 短跑：仅前 30 帧，快速看 LingBot 深度进 SplaTAM 的效果
from importlib.machinery import SourceFileLoader
from pathlib import Path

_base = SourceFileLoader(
    "_session_lingbot_base",
    str(Path(__file__).with_name("session_20260625_104051.py")),
).load_module()

config = _base.config
config["run_name"] = "session_20260625_104051_lingbot_short30_seed0"
config["data"]["num_frames"] = 30
config["eval_every"] = 30
config["report_global_progress_every"] = 30

config["save_checkpoints"] = True
config["checkpoint_interval"] = 100
