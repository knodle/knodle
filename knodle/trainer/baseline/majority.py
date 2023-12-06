import logging
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import TensorDataset

from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.baseline.utils import accuracy_padded, SeqDataset
from knodle.trainer.trainer import BaseTrainer
from knodle.trainer.utils import log_section
from knodle.transformation.majority import input_to_majority_vote_input, seq_input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

logger = logging.getLogger(__name__)


@AutoTrainer.register('majority')
class MajorityVoteTrainer(BaseTrainer):
    """
    The baseline class implements a baseline model for labeling data with weak supervision.
        A simple majority vote is used for this purpose.
    """

    def __init__(self, **kwargs):
        if kwargs.get("trainer_config") is None:
            kwargs["trainer_config"] = MajorityConfig(optimizer=SGD, lr=0.001)
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)
        self._apply_rule_reduction()

        # initialise optimizer
        self.trainer_config.optimizer = self.initialise_optimizer()

        self.model_input_x, noisy_y_train, self.rule_matches_z = input_to_majority_vote_input(
            self.rule_matches_z, self.mapping_rules_labels_t, self.model_input_x,
            probability_threshold=self.trainer_config.probability_threshold,
            unmatched_strategy=self.trainer_config.unmatched_strategy,
            ties_strategy=self.trainer_config.ties_strategy,
            use_probabilistic_labels=self.trainer_config.use_probabilistic_labels,
            other_class_id=self.trainer_config.other_class_id,
            multi_label=self.trainer_config.multi_label,
            multi_label_threshold=self.trainer_config.multi_label_threshold
        )

        if self.trainer_config.use_probabilistic_labels:
            feature_label_dataset = input_labels_to_tensordataset(self.model_input_x, noisy_y_train, probs=True)
        else:
            feature_label_dataset = input_labels_to_tensordataset(self.model_input_x, noisy_y_train, probs=False)

        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self._train_loop(feature_label_dataloader)


@AutoTrainer.register('majority_seq')
class MajorityVoteSeqTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        if kwargs.get("trainer_config") is None:
            kwargs["trainer_config"] = MajorityConfig(optimizer=SGD, lr=0.001)
        super().__init__(**kwargs)

    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """
        self._load_train_params(model_input_x, rule_matches_z, dev_model_input_x, dev_gold_labels_y)

        # initialise optimizer
        self.trainer_config.optimizer = self.initialise_optimizer()

        # calculate labels
        noisy_y_train = seq_input_to_majority_vote_input(
            rule_matches_z, self.mapping_rules_labels_t, other_class_id=self.trainer_config.other_class_id
        ).argmax(axis=2)

        feature_label_dataset = input_labels_to_tensordataset(
            self.model_input_x, noisy_y_train, probs=False, dataset="sequence"
        )
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        self._train_loop(feature_label_dataloader)

    def _train_loop(self, feature_label_dataloader, **kwargs):

        gold_labels = self.dev_gold_labels_y.tensors[0].cpu().numpy()
        feature_label_dataset = input_labels_to_tensordataset(self.dev_model_input_x, gold_labels, dataset="sequence")
        valid_feature_label_dataloader = self._make_dataloader(feature_label_dataset, shuffle=False)

        print_val = True

        # Epochs
        for epoch in range(self.trainer_config.epochs):
            self.model.train()
            if print_val:
                avg_acc = 0
                for i, (tokens, labels_) in enumerate(valid_feature_label_dataloader):
                    preds_ = self.model(tokens)
                    preds = preds_.argmax(axis=2)
                    # labels = labels_.argmax(axis=2)
                    mask = (tokens != 0)
                    acc = accuracy_padded(predicted=preds, gold=labels_, mask=mask)
                    avg_acc = (avg_acc * i + acc) / (i + 1)
                print(avg_acc)
            for i, (tokens, labels) in enumerate(feature_label_dataloader):
                logits = self.model(tokens).permute(0, 2, 1)
                # loss = self.trainer_config.criterion(logits, labels)
                loss = self.calculate_loss(logits, labels)
                # Back prop.
                self.trainer_config.optimizer.zero_grad()
                loss.backward()
                if isinstance(self.trainer_config.grad_clipping, (int, float)):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.grad_clipping)
                self.trainer_config.optimizer.step()
            # adjust_learning_rate(optimizer, lr / (1 + (epoch + 1) * lr_decay))

        # log_section("Training starts", logger)
        #
        # self.model.to(self.trainer_config.device)
        #
        # for current_epoch in range(self.trainer_config.epochs):
        #     self.model.train()
        #     logger.info("Epoch: {}".format(current_epoch))
        #
        #     for num_batch, batch in enumerate(feature_label_dataloader):
        #         tokens, labels = self._load_batch(batch)
        #         tokens = tokens[0]
        #
        #         logits = self.model(tokens).permute(0, 2, 1)   # batchsize x #classes x #seqlength
        #         loss = self.calculate_loss(logits, labels)
        #         print(loss)
        #         self.trainer_config.optimizer.zero_grad()
        #         loss.backward()
        #
        #         if isinstance(self.trainer_config.grad_clipping, (int, float)):
        #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.grad_clipping)
        #
        #         self.trainer_config.optimizer.step()
        #
        #     # todo: add adjust_learning_rate
        #
        #     if self.dev_model_input_x is not None:
        #         avg_acc, dev_loss = self.test(self.dev_model_input_x, self.dev_gold_labels_y, loss_calculation=True)
        #         print(f"Epoch {current_epoch+1}/{self.trainer_config.epochs} \t Acc: {avg_acc} \t Loss: {dev_loss}")
        #         self.model.train()

    def test(
            self, features_dataset: TensorDataset, labels: TensorDataset, loss_calculation: bool = False
    ) -> Tuple[float, float]:
        # todo: add other metrics (= have metric as a parameter)
        gold_labels = labels.tensors[0].cpu().numpy()
        feature_label_dataset = input_labels_to_tensordataset(features_dataset, gold_labels, dataset="sequence")
        feature_label_dataloader = self._make_dataloader(feature_label_dataset, shuffle=False)

        self.model.eval()
        dev_loss, dev_acc, avg_acc = 0, 0, 0

        with torch.no_grad():
            for i, batch in enumerate(feature_label_dataloader):
                tokens, labels = self._load_batch(batch)
                tokens = tokens[0]     # batch_size x seqlength
                outputs = self.model(tokens).argmax(axis=2)
                mask = (tokens != 0)
                acc = accuracy_padded(predicted=outputs, gold=labels, mask=mask)
                avg_acc = (avg_acc * i + acc) / (i + 1)

                # self.trainer_config.optimizer.zero_grad()
                # outputs = self.model(tokens).permute(0, 2, 1)
                #
                # if loss_calculation:
                #     curr_dev_loss = self.calculate_loss(outputs, labels)
                #     dev_loss += float(curr_dev_loss)
                #
                # mask = (tokens != 0)               # batch_size x seqlength
                # acc = accuracy_padded(predicted=outputs.argmax(axis=1), gold=labels, mask=mask)
                # avg_acc = (avg_acc * i + float(acc)) / (i + 1)
        if dev_loss == 0:
            dev_loss = None
        return avg_acc, dev_loss

    def get_predicted_labels(self, features_dataset: TensorDataset) -> float:
        return self.model(features_dataset.tensors[0].to(torch.int64)).argmax(axis=2)

