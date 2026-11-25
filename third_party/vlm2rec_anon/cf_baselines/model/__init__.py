# Registry of downstream sequential-recommendation backbones.
# To add a new backbone (e.g. fmlp, gru4rec, diffurec), implement it as a
# subclass of model.abstract_model.AbstractModel and register it below.
from model.sasrec import SASRec

MODEL_DICT = {
    "sasrec": SASRec,
}
