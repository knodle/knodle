import logging

import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.trainer import Trainer
from knodle.trainer.utils.utils import log_section, accuracy_of_probs

logger = logging.getLogger(__name__)


class MajorityBertTrainer(Trainer):
    def train(self):
        """
        This function gets final labels with a majority vote approach and trains the provided model.
        """

        model_input_x, label_probs = input_to_majority_vote_input(
            self.model_input_x, self.rule_matches_z, self.mapping_rules_labels_t,
            filter_empty_z_rows=self.trainer_config.filter_non_labelled
        )

        feature_label_dataset = input_labels_to_tensordataset(model_input_x, label_probs)
        feature_label_dataloader = self._make_dataloader(feature_label_dataset)

        log_section("Training starts", logger)
        device = self.trainer_config.device
        self.model.train()

        for current_epoch in range(self.trainer_config.epochs):
            epoch_loss, epoch_acc = 0.0, 0.0
            logger.info("Epoch: {}".format(current_epoch))
            i = 0
            for input_ids_batch, attention_mask_batch, label_batch in tqdm(feature_label_dataloader):
                i = i + 1
                inputs = {
                    "input_ids": input_ids_batch.to(device),
                    "attention_mask": attention_mask_batch.to(device),
                }
                labels = label_batch.to(device)

                # forward pass
                self.trainer_config.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.trainer_config.criterion(outputs[0], labels)

                # backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.trainer_config.optimizer.step()
                acc = accuracy_of_probs(outputs[0], labels)

                epoch_loss += loss.detach()
                epoch_acc += acc.item()
                # if i > 0:
                #     break

            avg_loss = epoch_loss / len(feature_label_dataloader)
            avg_acc = epoch_acc / len(feature_label_dataloader)

            logger.info("Epoch loss: {}".format(avg_loss))
            logger.info("Epoch Accuracy: {}".format(avg_acc))

        del loss
        log_section("Training done", logger)

        self.model.eval()

    def test(self, features_dataset: TensorDataset, labels: TensorDataset):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        feature_label_dataloader = self._make_dataloader(
            TensorDataset(
                features_dataset.tensors[0], features_dataset.tensors[1],
                labels.tensors[0]
            ),
            shuffle=False
        )

        self.model.eval()
        predictions_list = []
        label_list = []
        with torch.no_grad():
            for input_ids_batch, attention_mask_batch, label_batch in tqdm(feature_label_dataloader):
                inputs = {
                    "input_ids": input_ids_batch.to(device),
                    "attention_mask": attention_mask_batch.to(device),
                }

                # forward pass
                self.trainer_config.optimizer.zero_grad()
                prediction_probs = self.model(**inputs)[0]
                predictions = np.argmax(prediction_probs.cpu().detach().numpy(), axis=-1)
                predictions_list.append(predictions)
                label_list.append(label_batch.cpu().detach().numpy())

        predictions = np.squeeze(np.hstack(predictions_list))
        gold_labels = np.squeeze(np.hstack(label_list))

        clf_report = classification_report(y_true=gold_labels, y_pred=predictions, output_dict=True)

        logger.info(clf_report)
        logger.info("Accuracy: {}, ".format(clf_report["accuracy"]))
        print(clf_report)
        print("Accuracy: {}, ".format(clf_report["accuracy"]))
        return clf_report
