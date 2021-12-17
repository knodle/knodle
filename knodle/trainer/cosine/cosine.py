import os
import logging
import copy
from torch.utils import data
from knodle.transformation.majority import z_t_matrices_to_majority_vote_labels
import torch.nn.functional as F
import torch
import numpy as np
from snorkel.utils import probs_to_preds
from sklearn.metrics import classification_report
from knodle.trainer.cosine.cosine_dataset import CosineDataset
from knodle.trainer.cosine.model import TextClassBert
from knodle.trainer.cosine.cosine_utils import ContrastiveLoss
from knodle.trainer.cosine.EarlyStopper import EarlyStopper
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Any, List, Optional, Union, Callable, Dict, Tuple
from transformers import AutoTokenizer

from torch.optim import SGD
from torch.utils.data import TensorDataset
import torch.nn as nn

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex

from knodle.transformation.majority import input_to_majority_vote_input
from knodle.transformation.torch_input import input_labels_to_tensordataset

from knodle.trainer.auto_trainer import AutoTrainer
from knodle.trainer.trainer import BaseTrainer
from knodle.trainer.cosine.config import CosineConfig
from knodle.trainer.utils.denoise import activate_neighbors

logger = logging.getLogger(__name__)


# This code (cosine trainer class) is adapted from the
# official code of the COSINE framework (https://github.com/yueyu1030/COSINE)

class MajorityVoting:
    def __init__(self, **kwargs):
        super().__init__()

    def predict_proba(self, weak_labels, weight=None, ABSTAIN=-1) -> np.ndarray:
        n_class = weak_labels.shape[1]
        L = weak_labels
        if weight is None:
            weight = np.ones_like(L)

        # n_class = dataset.num_classes
        n, m = L.shape
        Y_p = np.zeros((n, n_class))
        for i in range(n):
            counts = np.zeros(n_class)
            for j in range(m):
                if L[i, j] != ABSTAIN:
                    counts[L[i, j]] += 1 * weight[i, j]
            # Y_p[i, :] = np.where(counts == max(counts), 1, 0)
            if counts.sum() == 0:
                counts += 1
            Y_p[i, :] = counts
        Y_p /= Y_p.sum(axis=1, keepdims=True)

        return Y_p

    def predict(self, weak_label_mat, **kwargs) -> np.ndarray:


        proba = self.predict_proba(weak_label_mat, **kwargs)
        return probs_to_preds(probs=proba)

    def test(self, args, dataset, y_true=None, strict=True, **kwargs):
        if y_true is None:
            y_true = dataset.labels
        preds = self.predict(dataset, **kwargs)
        classification_score = classification_report(y_true, preds, output_dict=True)

        return classification_score


@AutoTrainer.register('cosine')
class CosineTrainer(BaseTrainer):
    def __init__(
            self,
            **kwargs
    ):
        self.trainer_config: CosineConfig = kwargs.get("trainer_config")
        self.model = None
        # if kwargs.get("trainer_config") is None:
        #     kwargs["trainer_config"] = CosineConfig()
        super().__init__(**kwargs)


    def process_dataset(self):
        # process train
        train_x = self.model_input_x
        # Either choose_random_label=False, other_class_id=-1
        # Or choose_random_label=True, other_class_id = None
        # majority_labels = z_t_matrices_to_majority_vote_labels(self.rule_matches_z, self.mapping_rules_labels_t,
        #                                                        choose_random_label=True, other_class_id=None)

        weak_label_mat = -np.ones(self.rule_matches_z.shape, dtype=np.int16)
        label_map_list = np.argmax(self.mapping_rules_labels_t, axis=1)

        for i in range(weak_label_mat.shape[0]):
            for j in range(weak_label_mat.shape[1]):
                weak_label_mat[i, j] = int(label_map_list[j]) if self.rule_matches_z[i, j] != 0.0 else -1

        self.majority_voter = MajorityVoting()
        train_y = self.majority_voter.predict(weak_label_mat)
        train_x_dict = {'input_ids': train_x.tensors[0], 'attention_mask': train_x.tensors[1]}
        train_set = CosineDataset(train_x_dict, train_y, weak_label_mat=weak_label_mat)

        # split to labeled and unlabeled
        l_set, ul_set = train_set.get_covered_subset()

        # create validation set
        dev_x_dict = {'input_ids': self.dev_model_input_x.tensors[0], 'attention_mask': self.dev_model_input_x.tensors[1]}
        val_set = CosineDataset(dev_x_dict, self.dev_gold_labels_y.tensors[0])

        return l_set, ul_set, val_set

    def concat_datasets(self, set1, set2):
        # assert set1.id2l == set2.id2l
        combined_dataset = CosineDataset(xs=set1.xs, labels=set1.labels, weak_label_mat=None)
        combined_dataset.xs = {k: torch.cat((v, set2.xs[k])) for k, v in combined_dataset.xs.items()}
        combined_dataset.labels = np.concatenate((combined_dataset.labels, set2.labels))
        return combined_dataset

    def get_batch(self, d_loader, d_iter):
        try:
            d_batch = next(d_iter)
        except StopIteration:
            d_iter = iter(d_loader)
            d_batch = next(d_iter)

        return d_batch, d_iter

    def needs_eval(self, config, global_step):
        if global_step % config.eval_freq == 0 and global_step != 0:
            return True
        else:
            return False

    def eval_model(self, config, logger, device,
                   eval_set_loader, model, fast_mode=False, verbose=False,  **kwargs):
        all_preds = []
        all_y = []
        model.eval()
        loss_sum = 0.0
        loss_fn = torch.nn.CrossEntropyLoss()

        num_batches = len(eval_set_loader)/10 if fast_mode else 0

        with torch.no_grad():
            for idx, eval_batch in enumerate(eval_set_loader):
                input_ids = eval_batch['input_ids'].to(device)
                attention_mask = eval_batch['attention_mask'].to(device)
                targets = eval_batch['labels'].to(device)

                y_logits = model(input_ids, attention_mask)['logits']
                loss_sum += loss_fn(y_logits, targets)
                y_preds = torch.max(y_logits.cpu(), 1)[1]
                all_preds.extend(y_preds.numpy())
                all_y.extend(list(targets.cpu()))

                if fast_mode and idx > num_batches:
                    break

            classification_score_dict = classification_report(all_y, np.array(all_preds).flatten(), output_dict=True)
            classification_score_str = classification_report(all_y, np.array(all_preds).flatten(), output_dict=False)

            if verbose:
                logger_prefix = kwargs['tag'] if 'tag' in kwargs else ''
                global_step = kwargs['global_step'] if 'global_step' in kwargs else -1
                self.logger.info(f'{logger_prefix} score at step {global_step}')
                self.logger.info(classification_score_str)

        return {'score_dict': classification_score_dict,
                'score_str': classification_score_str,
                'loss': loss_sum/(len(all_y))}

    def get_optimizer_grouped_parameters(self, config, model):
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        return optimizer_parameters


    def calc_loss(self, input, target, loss, device, thresh=0.95, soft=True, conf='max', confreg=0.1):
        softmax = nn.Softmax(dim=1)
        target = softmax(target.view(-1, target.shape[-1])).view(target.shape)

        if conf == 'max':
            weight = torch.max(target, dim=1).values
            # w = torch.FloatTensor([1 if x == True else 0 for x in weight > thresh]).to(device)
            w = torch.tensor(weight > thresh, dtype=torch.float)
        elif conf == 'entropy':
            weight = torch.sum(-torch.log(target + 1e-6) * target, dim=1)  # Entropy
            weight = 1 - weight / np.log(weight.size(-1))
            # w = torch.FloatTensor([1 if x == True else 0 for x in weight > thresh]).to(device)
            w = torch.tensor(weight > thresh, dtype=torch.float)
        else:
            raise ValueError(f'conf={conf} is unsupported')
        target = self.soft_frequency(target, probs=True, soft=soft)

        loss_batch = loss(input, target)

        lc = torch.sum(loss_batch * w.unsqueeze(1) * weight.unsqueeze(1))  # l_c loss in the paper

        n_classes_ = input.shape[-1]
        # Note this is l-=, i.e l = l - (...)
        lc -= confreg * (torch.sum(input * w.unsqueeze(1)) + np.log(n_classes_) * n_classes_)  # l_c + \lambda * R2 loss in the paper
        return lc  # which is L_c + \lambda * R_2 in the paper

    def contrastive_loss(self, input, feat, target, device, conf='none', thresh=0.1, distmetric='l2'):
        softmax = nn.Softmax(dim=1)
        target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        if conf == 'max':
            weight = torch.max(target, axis=1).values
            w = torch.tensor([i for i, x in enumerate(weight) if x > thresh], dtype=torch.long).to(device)
        elif conf == 'entropy':
            weight = torch.sum(-torch.log(target + 1e-6) * target, dim=1)
            weight = 1 - weight / np.log(weight.size(-1))
            w = torch.tensor([i for i, x in enumerate(weight) if x > thresh], dtype=torch.long).to(device)
        input_x = input[w] # prediction with high conf

        feat_x = feat[w]
        batch_size = input_x.size()[0]
        if batch_size == 0:
            return 0
        index = torch.randperm(batch_size).to(device)
        input_y = input_x[index, :]  # permutated version of input_x
        feat_y = feat_x[index, :]
        argmax_x = torch.argmax(input_x, dim=1)
        argmax_y = torch.argmax(input_y, dim=1)
        # agreement = torch.FloatTensor([1 if x == True else 0 for x in argmax_x == argmax_y]).to(device)
        agreement = torch.tensor(argmax_x == argmax_y, dtype=torch.float)

        criterion = ContrastiveLoss(margin=1.0, metric=distmetric)
        loss, dist_sq, dist = criterion(feat_x, feat_y, agreement)

        return loss

    def soft_frequency(self, logits, probs=False, soft=True):
        """
        Unsupervised Deep Embedding for Clustering Analysis
        https://arxiv.org/abs/1511.06335
        """
        power = self.trainer_config.self_training_power
        if not probs:
            softmax = nn.Softmax(dim=1)
            y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
        else:
            y = logits
        f = torch.sum(y, dim=0)
        t = y ** power / f
        # print('t', t)
        t = t + 1e-10
        p = t / torch.sum(t, dim=-1, keepdim=True)
        return p if soft else torch.argmax(p, dim=1)



    def phase1_training(self, l_set, dev_set):
        config = self.trainer_config
        device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
        num_classes = self.mapping_rules_labels_t.shape[1]
        model = TextClassBert(config.bert_backbone, config.bert_dropout_rate, num_classes)
        model = model.to(device)
        self.model = model
        T1 = config.T1
        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(config, model)
        optimizer = config.optimizer(optimizer_grouped_parameters, lr=config.lr)
        optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                              num_training_steps=T1)

        global_step = 0
        # if self.store_model_flag:
        #     early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
        #     if not os.path.exists(early_stopper_save_dir):
        #         os.makedirs(early_stopper_save_dir)
        # else:
        #     early_stopper_save_dir = None

        early_stopper = EarlyStopper(patience=20, delta=0, save_dir=None, verbose=False,
                                     trace_func=logger.info)
        ce_loss_fn = nn.CrossEntropyLoss()

        best_val_acc = -1

        l_loader = torch.utils.data.DataLoader(l_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
        l_iter = iter(l_loader)
        dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=config.eval_batch_size,
                                               shuffle=False,
                                               num_workers=0)

        # train the network
        for step in range(T1):

            l_batch, l_iter = self.get_batch(l_loader, l_iter)
            bs = len(l_batch['labels'])

            nl_input_ids = l_batch['input_ids'].to(device)
            nl_attention_mask = l_batch['attention_mask'].to(device)
            nl_labels = l_batch['labels'].to(device)

            model.train()
            model.zero_grad()
            outputs = model(nl_input_ids, nl_attention_mask)['logits']
            loss = ce_loss_fn(outputs, nl_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer_scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            batch_ce_loss = loss.item()
            if global_step % config.train_eval_freq == 0 and global_step != 0:
                logger.info(f'Step: {global_step}, ce_loss: {batch_ce_loss} ')

            # mi.investigate(self, model, t_loader, global_step, sw_test, device)  # investigate after

            if self.needs_eval(config, global_step):
                val_res = self.eval_model(config, logger, device, dev_loader, model, verbose=False)
                logger.info(f"val acc: {val_res['score_dict']['accuracy']}")
                early_stopper.register(val_res['loss'], model, optimizer)

            if global_step == T1 or early_stopper.early_stop:
                break

        model = TextClassBert(config.bert_backbone, config.bert_dropout_rate, num_classes)
        early_stopper_res = early_stopper.get_final_res()
        assert early_stopper_res is not None, "no model logged, try to train longer"
        early_stop_state_dict = early_stopper_res['es_best_model']
        model.load_state_dict(early_stop_state_dict)
        self.model = model

        return {"best_phase1_model": model}


    def phase2_training(self, init_teacher, l_set, ul_set, dev_set):
        num_classes = self.mapping_rules_labels_t.shape[1]
        config = self.trainer_config
        device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

        # student_model = TextClassBert(config.bert_backbone, config.bert_dropout_rate, num_classes)
        # student_model = student_model.to(device)
        student_model = init_teacher.to(device)
        self.model = student_model
        T2 = config.T2

        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(config, student_model)
        student_optimizer = config.optimizer(optimizer_grouped_parameters, lr=config.lr)
        student_optimizer_scheduler = get_linear_schedule_with_warmup(student_optimizer, num_warmup_steps=config.warmup_steps,
                                                              num_training_steps=T2)

        if ul_set is not None:
            tr_set_full = self.concat_datasets(l_set, ul_set)
        else:
            tr_set_full = l_set

        logger.info(f"[COSINE P2]: size of combined dataset: {len(tr_set_full)}")
        self_training_loss = nn.KLDivLoss(reduction='none') if config.soft else nn.CrossEntropyLoss(reduction='none')

        tr_loader = torch.utils.data.DataLoader(tr_set_full, batch_size=config.batch_size, shuffle=True, num_workers=0)
        tr_iter = iter(tr_loader)
        dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=config.eval_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        early_stopper = EarlyStopper(patience=20, delta=0, save_dir=None, verbose=False,
                                     trace_func=logger.info)

        global_step = 0
        selftrain_loss=0.0

        for step in range(T2):
            if global_step % config.T3 == 0:
                teacher_model = copy.deepcopy(student_model)  # .to("cuda")
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False

            student_model.train()
            tr_batch, tr_iter = self.get_batch(tr_loader, tr_iter)
            inputs = {
                'input_ids': tr_batch['input_ids'].to(device),
                'attention_mask': tr_batch['attention_mask'].to(device),
            }

            outputs = student_model(**inputs)
            outputs_pseudo = teacher_model(**inputs)

            logits = outputs['logits']
            log_soft_logits = torch.log(F.softmax(logits, dim=1))

            loss = self.calc_loss(input=log_soft_logits,
                                  target=outputs_pseudo['logits'],
                                  loss=self_training_loss,
                                  device=device,
                                  thresh=config.self_training_eps,
                                  soft=config.soft,
                                  conf='entropy',
                                  confreg=config.self_training_confreg)

            if config.self_training_contrastive_weight > 0:
                contrastive_loss = self.contrastive_loss(input=log_soft_logits,
                                                         feat=outputs_pseudo['cls_repr'],
                                                         target=outputs_pseudo['logits'],
                                                         device=device,
                                                         conf='entropy',
                                                         thresh=config.self_training_eps,
                                                         distmetric=config.distmetric)
                loss = loss + config.self_training_contrastive_weight * contrastive_loss


            selftrain_loss += loss
            loss.backward()

            student_optimizer.step()
            student_optimizer_scheduler.step()  # Update learning rate schedule
            student_model.zero_grad()
            teacher_model.zero_grad()
            global_step += 1

            logger.info(f"SelfTrain Step={step} Avg. Loss={selftrain_loss / global_step}")
            if self.needs_eval(config, global_step):
                val_res = self.eval_model(config, logger, device, dev_loader, student_model, verbose=False)
                logger.info(f"val acc: {val_res['score_dict']['accuracy']}")
                early_stopper.register(val_res['loss'], student_model, student_optimizer)

            if global_step == T2 or early_stopper.early_stop:
                break

        model = TextClassBert(config.bert_backbone, config.bert_dropout_rate, num_classes)
        early_stopper_res = early_stopper.get_final_res()
        assert early_stopper_res is not None, "no model logged, try to train longer"
        early_stop_state_dict = early_stopper_res['es_best_model']
        model.load_state_dict(early_stop_state_dict)
        self.model = model
        return {"best_phase2_model": model}


    def train(
            self,
            model_input_x: TensorDataset = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: TensorDataset = None, dev_gold_labels_y: TensorDataset = None
    ):

        l_set, ul_set, val_set = self.process_dataset()
        phase1_res = self.phase1_training(l_set, val_set)
        phase1_model = phase1_res["best_phase1_model"]
        final_res = self.phase2_training(phase1_model, l_set, ul_set, val_set)
        return final_res

    def test(
            self, features_dataset: TensorDataset, labels: TensorDataset, loss_calculation: bool = False
    ) -> Tuple[Dict, Union[float, None]]:
        config = self.trainer_config
        device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')
        test_x_dict = {'input_ids': features_dataset.tensors[0],
                      'attention_mask': features_dataset.tensors[1]}
        test_set = CosineDataset(test_x_dict, labels.tensors[0])

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.eval_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)

        test_model = self.model.to(device)
        test_res = self.eval_model(self.trainer_config, logger, device, test_loader, test_model, fast_mode=False)

        if config.metric == "accuracy":
            score = test_res['score_dict']['accuracy']
            logger.info(f"[cosine]: test accuracy: {score}")

        elif config.metric == "f1_macro":
            score = test_res['score_dict']['macro avg']['f1-score']
            logger.info(f"[cosine]: macro f1: {score}")
        else:
            raise ValueError(f"metric [{config.metric}] not supported")

        return score



