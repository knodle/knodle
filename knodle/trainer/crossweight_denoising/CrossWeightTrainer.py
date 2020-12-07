import logging
from datetime import date

from knodle.model.BidirectionalLSTM.BidirectionalLSTM import BidirectionalLSTM
from knodle.model.LogisticRegression.LogisticRegressionModel import LogisticRegressionModel
from knodle.trainer.config.TrainerConfig import TrainerConfig
from knodle.trainer.crossweight_denoising import utils
from knodle.trainer.ds_model_trainer.ds_model_trainer import DsModelTrainer
import torch
import torch.nn as nn
from torch.nn import Module
import os
from knodle.trainer.config.CrossWeightDenoisingConfig import CrossWeightDenoisingConfig
from collections import defaultdict
import json
import numpy as np

# Set the seed for reproducibility
np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)

disable_cuda = True
device = None
if not disable_cuda and torch.cuda.is_available():
    print("Using GPU")
    device = torch.device('cuda')
else:
    print("Using CPU")
    device = torch.device('cpu')

logger = logging.getLogger(__name__)
GRADIENT_CLIPPING = 5


class CrossWeightTrainer(DsModelTrainer):
    def __init__(self,
                 model: Module,
                 denoising_config: CrossWeightDenoisingConfig = None,
                 trainer_config: TrainerConfig = None):
        super().__init__(model, denoising_config, trainer_config)     # , trainer_config

        # self.pattern2id, self.pattern2regex, self.relation2patterns, self.relations2ids = {}, {}, {}, {}

    def get_train_test_data(self, inputs_x, rule_matches_z, rules_shuffled_idx, labels, curr_fold):
        test_rules_idx = rules_shuffled_idx[curr_fold::self.denoising_config.crossweight_folds]
        train_rules_idx = [rules_shuffled_idx[x::self.denoising_config.crossweight_folds]
                           for x in range(self.denoising_config.crossweight_folds) if x != curr_fold]
        train_rules_idx = np.concatenate(train_rules_idx, axis=0)
        assert set(test_rules_idx).isdisjoint(set(train_rules_idx))

        # get train and test rules
        # train_rules = rule_assignments_t[train_rules_idx, :]
        # test_rules = rule_assignments_t[test_rules_idx, :]

        # select train and test samples according to the rules
        for rule_idx in train_rules_idx:
            train_samples = inputs_x[np.where(rule_matches_z[:, rule_idx] == 1)]        # todo: docheck it all!!!!
        for rule_idx in test_rules_idx:
            test_samples = inputs_x[np.where(rule_matches_z[:, rule_idx] == 1)]  # todo: docheck it all!!!!

        train_labels = labels[train_rules_idx, :]
        test_labels = labels[test_rules_idx, :]

        train_loader = utils.convert2tensor(train_samples, train_labels, self.denoising_config.batch_size)
        test_loader = utils.convert2tensor(test_samples, test_labels, self.denoising_config.batch_size)

        self.logger.info("Rules in training set:{}, rules in test set: {}".format(
            len(train_rules_idx), len(test_rules_idx)))

        return train_loader, test_loader

    def crossweight_train(self, word_emb_file, inputs_x, rule_matches_z, rule_assignments_t, path_to_weights_dir):
        self.logger.info("The trained sample weights will be saved to:{}".format(path_to_weights_dir))
        word2id, id2word, word_embedding_matrix = utils.vocab_and_vectors(word_emb_file, ['<PAD>', '<UNK>'])

        assert(rule_matches_z.shape[1] == rule_assignments_t.shape[0], "Check matrices dimensionality")
        labels = rule_matches_z.dot(rule_assignments_t)

        # initialise class and sample weights
        class_weights = torch.FloatTensor([1.0] + [2.0] * (self.denoising_config.output_classes - 1)).to(device=device)\
            if self.denoising_config.use_cross_class_weights is True \
            else torch.FloatTensor([1.0] * self.denoising_config.output_classes).to(device=device)
        sample_weights = np.array([self.denoising_config.samples_start_weights] * inputs_x.shape[0], dtype=np.float64)

        id2weights_to_update = defaultdict(int)

        # statistics
        relation_stat = defaultdict(lambda: defaultdict())
        rule_stat = defaultdict(lambda: defaultdict())

        for partition in range(self.denoising_config.crossweight_rounds):
            # get random indices
            rules_shuffled_idx = np.random.rand(rule_assignments_t.shape[0]).argsort()
            for curr_fold in range(self.denoising_config.crossweight_folds):
                train_loader, test_loader = self.get_train_test_data(inputs_x, rule_matches_z, rules_shuffled_idx,
                                                                     labels, curr_fold)
                model = BidirectionalLSTM(word_embedding_matrix.shape[0],
                                          word_embedding_matrix.shape[1],
                                          word_embedding_matrix,
                                          self.denoising_config.output_classes)
                criterion = self.denoising_config.criterion(weight=class_weights) \
                    if self.denoising_config.use_cross_class_weights else self.denoising_config.criterion
                optimizer = self.denoising_config.optimizer

                model.apply(utils.initialize_weights)
                model.train()

                for e in range(self.denoising_config.crossweight_epochs):

                    for tokens, labels in train_loader:
                        tokens, labels = tokens.to(device=device), labels.to(device=device)

                        model.zero_grad()
                        optimizer.zero_grad()  # theoretically they are equivalent, but ???
                        output = model(tokens)
                        loss = criterion(output, labels)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                        optimizer.step()
                model.eval()


    def denoise(self, inputs_x, rule_matches_z, rule_assignments_t, word_emb_file):

        path_to_res_dir = os.path.join(self.denoising_config.path_to_weights, str(date.today()))
        os.makedirs(path_to_res_dir, exist_ok=True)  # Create dir where weights will be saved

        # metrics_array = np.zeros((self.crossweight_rounds * classifier_rounds, 3))
        # best_p, best_r, best_f = 0, 0, 0

        for i in range(self.denoising_config.crossweight_rounds):

            self.logger.info("Crossweight Iteration {} out of {}".format(i, self.denoising_config.crossweight_rounds))
            path_to_weights_dir = os.path.join(path_to_res_dir, str(i))
            os.makedirs(path_to_weights_dir, exist_ok=True)

            self.crossweight_train(word_emb_file, inputs_x, rule_matches_z, rule_assignments_t, path_to_weights_dir)

    # def read_dicts(self):
    #     with open("preprocessing/pattern2id.json") as f:
    #         self.pattern2id = json.load(f)
    #     with open("preprocessing/pattern2regex.json") as f:
    #         self.pattern2regex = json.load(f)
    #     with open("preprocessing/relation2patterns.json") as f:
    #         self.relation2patterns = json.load(f)
    #     with open("preprocessing/relations2ids.json") as f:
    #         self.relations2ids = json.load(f)

    def train(self, x, z, t, word_emb_file):
        # todo: input t matrix in class (not parameter but mit init)
        # self.read_dicts()
        instance_weights = self.denoise(x, z, t, word_emb_file)
        pass


if __name__ == '__main__':
    CrossWeightTrainer(LogisticRegressionModel).denoise()

