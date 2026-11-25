"""Paper target numbers (arXiv:2603.17450v3) used to compare runs of the official code.

Values are copied from docs/PAPER_NOTES.md (main table = tables/perf_task2.tex).
Metric order everywhere: H@10, H@20, N@10, N@20.
Dataset keys use the official code's names: Beauty, Toys_and_Games, Sports, Clothing.
ML-1M is omitted because the official release cannot preprocess it.
"""

METRICS = ("H@10", "H@20", "N@10", "N@20")


def _m(*vals):
    return dict(zip(METRICS, vals))


# Checkpoint A = 5-core + join with metadata having title & image URL, before truncation.
DATA_STATS = {
    "Toys_and_Games": {"users": 19406, "items": 11842, "interactions": 166230, "avg_len": 8.6},
    "Beauty": {"users": 22363, "items": 12077, "interactions": 198064, "avg_len": 8.9},
    "Sports": {"users": 35597, "items": 18168, "interactions": 294045, "avg_len": 8.3},
    "Clothing": {"users": 39387, "items": 22445, "interactions": 270319, "avg_len": 6.9},
}

# paper_key -> dataset -> metrics
TARGETS = {
    # ID-only backbone
    "sasrec": {
        "Toys_and_Games": _m(0.0904, 0.1221, 0.0524, 0.0602),
        "Beauty": _m(0.0713, 0.1053, 0.0397, 0.0483),
        "Clothing": _m(0.0402, 0.0648, 0.0198, 0.0260),
        "Sports": _m(0.0428, 0.0648, 0.0240, 0.0294),
    },
    # VLM2Rec setting (no SR backbone, full ranking by sequence-item similarity)
    "vanilla": {
        "Toys_and_Games": _m(0.0843, 0.1197, 0.0481, 0.0570),
        "Beauty": _m(0.0674, 0.1054, 0.0351, 0.0442),
        "Clothing": _m(0.0384, 0.0610, 0.0189, 0.0245),
        "Sports": _m(0.0415, 0.0605, 0.0224, 0.0271),
    },
    "sft": {
        "Toys_and_Games": _m(0.1132, 0.1456, 0.0652, 0.0732),
        "Beauty": _m(0.0942, 0.1527, 0.0547, 0.0695),
        "Clothing": _m(0.0471, 0.0807, 0.0241, 0.0325),
        "Sports": _m(0.0506, 0.0817, 0.0279, 0.0358),
    },
    "vlm2rec": {
        "Toys_and_Games": _m(0.1254, 0.1682, 0.0720, 0.0823),
        "Beauty": _m(0.1161, 0.1732, 0.0641, 0.0784),
        "Clothing": _m(0.0553, 0.0875, 0.0290, 0.0370),
        "Sports": _m(0.0585, 0.0904, 0.0316, 0.0396),
    },
    # VLM2Rec_plug setting (SASRec backbone)
    "vanilla_plug": {
        "Toys_and_Games": _m(0.0841, 0.1178, 0.0491, 0.0577),
        "Beauty": _m(0.0685, 0.1031, 0.0374, 0.0463),
        "Clothing": _m(0.0365, 0.0596, 0.0183, 0.0240),
        "Sports": _m(0.0383, 0.0601, 0.0208, 0.0262),
    },
    "sft_plug": {
        "Toys_and_Games": _m(0.1058, 0.1461, 0.0598, 0.0699),
        "Beauty": _m(0.0922, 0.1349, 0.0515, 0.0622),
        "Clothing": _m(0.0499, 0.0755, 0.0260, 0.0325),
        "Sports": _m(0.0469, 0.0738, 0.0260, 0.0326),
    },
    "vlm2rec_plug": {
        "Toys_and_Games": _m(0.1169, 0.1588, 0.0684, 0.0786),
        "Beauty": _m(0.1080, 0.1511, 0.0618, 0.0728),
        "Clothing": _m(0.0540, 0.0823, 0.0276, 0.0345),
        "Sports": _m(0.0566, 0.0849, 0.0307, 0.0378),
    },
}

# Dropout test, VLM2Rec setting, N@20 — read off Fig. 1 (approximate, +-0.001). Beauty only.
DROPOUT_N20 = {
    "Beauty": {
        "vanilla": {"TEXT": 0.045, "VISION": 0.042, "FUSION": 0.0442},
        "sft": {"TEXT": 0.072, "VISION": 0.0645, "FUSION": 0.0695},
        "vlm2rec": {"TEXT": 0.078, "VISION": 0.077, "FUSION": 0.0784},
    }
}

PAPER_KEYS = tuple(TARGETS) + ("none",)
