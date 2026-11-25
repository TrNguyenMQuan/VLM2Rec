import datetime
import logging
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_logger(log_path, log_name="seqrec", mode="a"):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def get_local_time():
    return datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")


class EarlyStopping:
    """Early stop training when the monitored validation score stops improving."""

    def __init__(self, checkpoint_path, logger, patience=10, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger
        self.score_min = None

    def _is_worse(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self._save_checkpoint(score, model)
        elif self._is_worse(score):
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(score, model)
            self.counter = 0

    def _save_checkpoint(self, score, model):
        if self.verbose:
            self.logger.info("Validation score increased. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score
