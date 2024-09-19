import logging
import argparse
import random
from pathlib import Path
import pandas as pd
import os
import joblib
import torch
import numpy as np
import knodle.trainer.cosine.cosine
from knodle.trainer.auto_config import AutoConfig
from knodle.trainer.auto_trainer import AutoTrainer
from transformers import AdamW, AutoTokenizer
from examples.trainer.preprocessing import np_array_to_tensor_dataset, convert_text_to_transformer_input


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='trec', choices=['trec'])
    parser.add_argument('--data_home', type=str, default='')
    parser.add_argument('--bert_backbone', type=str, default='roberta-base',
                        choices=['bert-base-multilingual-cased', 'roberta-base'])

    parser.add_argument('--store_model', type=int, default=0, help='store model after training')
    parser.add_argument('--metric', type=str, default='accuracy', choices=['accuracy', 'f1_macro'])

    # preprocessing related
    parser.add_argument('--max_sen_len', type=int, default=128)

    # BERT-backbone settings related
    parser.add_argument('--bert_dropout_rate', type=float, default=0.1)

    # training related
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--train_eval_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=50, help='one batch is one step, '
                                                                  'eval_freq=n means eval at every n step')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--fast_eval', action='store_true',
                        help='use 10% of the test set for evaluation, to speed up the evaluation path')

    # cosine
    parser.add_argument('--T1', type=int, default=10)
    parser.add_argument('--T2', type=int, default=10)
    parser.add_argument('--T3', type=int, default=10)
    parser.add_argument('--teacher_label_type', type=str, default='soft', choices=['soft', 'hard'])
    parser.add_argument('--self_training_eps', type=float, default=0.1, help='threshold for confidence, '
                                                                                   'the xi in the paper')
    parser.add_argument('--self_training_power', type = float, default=2, help='power of pred score')
    parser.add_argument('--self_training_confreg', type = float, default=0, help = 'confidence smooth power, '
                                                                                     'the lambda in the paper')
    parser.add_argument('--self_training_contrastive_weight', type=float, default=1, help='contrastive learning weight')
    parser.add_argument('--distmetric', type=str, default="l2", choices=['cos', 'l2'],
                        help='distance type. Choices = [cos, l2]')

    # optimizer
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--exp_decay_rate', type=float, default=0.9998)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # hardware related
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda_device', type=str, default="0")
    parser.add_argument('--manualSeed', type=int, default=1234, help='random seed for reproducibility')
    parser.add_argument('--noisy_label_seed', type=int, default=1234, help='random seed for reproducibility')

    args = parser.parse_args()

    # for reproducibility
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.backends.cudnn.benchmark = False
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.deterministic = True

    # training code starts here

    processed_data_dir = Path(args.data_home) / args.dataset

    # TODO: we need sep=";" here, otherwise it will raise errors
    df_train = pd.read_csv(os.path.join(processed_data_dir, "df_train.csv"), sep=";")
    df_dev = pd.read_csv(os.path.join(processed_data_dir, "df_dev.csv"), sep=";")
    df_test = pd.read_csv(os.path.join(processed_data_dir, "df_test.csv"), sep=";")

    mapping_rules_labels_t = joblib.load(os.path.join(processed_data_dir, "mapping_rules_labels_t.lib"))

    train_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "train_rule_matches_z.lib"))
    dev_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "dev_rule_matches_z.lib"))
    test_rule_matches_z = joblib.load(os.path.join(processed_data_dir, "test_rule_matches_z.lib"))

    print(f"Train Z dimension: {train_rule_matches_z.shape}")
    print(f"Train avg. matches per sample: {train_rule_matches_z.sum() / train_rule_matches_z.shape[0]}")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_backbone)

    X_train = convert_text_to_transformer_input(tokenizer, df_train["sample"].tolist(), args.max_sen_len)
    X_dev = convert_text_to_transformer_input(tokenizer, df_dev["sample"].tolist(), args.max_sen_len)
    X_test = convert_text_to_transformer_input(tokenizer, df_test["sample"].tolist(), args.max_sen_len)

    y_dev = np_array_to_tensor_dataset(df_dev['label_id'].values)
    y_test = np_array_to_tensor_dataset(df_test['label_id'].values)


    custom_model_config = AutoConfig.create_config(
        name='cosine',
        args=args,
        optimizer=AdamW,
    )

    trainer = AutoTrainer(
        name="cosine",
        model=None,  # TODO: cosine knows how to create models.
        trainer_config=custom_model_config,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=X_train,
        rule_matches_z=train_rule_matches_z,
        dev_model_input_x=X_dev,
        dev_gold_labels_y=y_dev
    )

    trainer.train()
    trainer.test(X_test, y_test)


if __name__ == "__main__":
    main()
