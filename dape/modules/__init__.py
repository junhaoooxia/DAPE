from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "dape.modules.GeneralConditioner",
    "params": {"emb_models": []},
}