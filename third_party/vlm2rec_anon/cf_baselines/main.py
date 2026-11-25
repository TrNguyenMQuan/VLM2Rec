import os
import time

import numpy as np
import torch

from dataset import get_dataloader, get_rating_matrix, get_seq_dic
from model import MODEL_DICT
from trainer import Trainer
from utils import EarlyStopping, get_local_time, set_logger, set_seed


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.train_name = (
        f"{args.emb_model}_{args.model_name}_{args.data_name}_{get_local_time()}"
    )
    folder_path = os.path.join(
        args.output_dir,
        f"{args.emb_model}_{args.model_name}_{args.data_name}/",
    )
    os.makedirs(folder_path, exist_ok=True)
    log_path = os.path.join(folder_path, args.train_name + ".log")
    logger = set_logger(log_path)
    set_seed(args.seed)

    if not hasattr(args, "checkpoint_path"):
        args.checkpoint_path = os.path.join(folder_path, args.train_name + ".pt")

    args.pretrained_item_emb_path = os.path.join(
        args.item_emb_root, args.data_name, args.emb_model
    )

    seq_dic = get_seq_dic(args)
    args.num_users = seq_dic["num_users"] + 1
    args.item_size = seq_dic["num_items"] + 1
    train_loader, eval_loader, test_loader = get_dataloader(args, seq_dic)

    model = MODEL_DICT[args.model_name.lower()](args=args)
    logger.info(str(args))
    trainer = Trainer(model, train_loader, eval_loader, test_loader, args, logger)

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(
        seq_dic, seq_dic["num_items"]
    )

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        logger.info(f"Load model from {args.checkpoint_path} for test!")
        scores, result_info = trainer.test(0)
    else:
        early_stopping = EarlyStopping(
            args.checkpoint_path,
            logger=logger,
            patience=args.patience,
            verbose=True,
        )
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("--------------- Test Score ---------------")
        time.sleep(2)
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)

    logger.info(args.train_name)
    logger.info(result_info)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_only", action="store_true")
    known_args, unknown_args = parser.parse_known_args()

    parsed = {}
    for i in range(0, len(unknown_args), 2):
        key = unknown_args[i].lstrip("--")
        value = unknown_args[i + 1]
        try:
            value = eval(value)
        except Exception:
            pass
        parsed[key] = value

    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

        def __str__(self):
            max_len = max(len(k) for k in self.__dict__)
            return "\n".join(
                f"{k.ljust(max_len)} : {v}" for k, v in sorted(self.__dict__.items())
            )

    main(Args(**parsed))
