class LogisticRegressionModel(object):
    pass<img src="./knodle_logo.png" height="150"/>

### Knowledge infused deep learning framework

[![Python Version](https://img.shields.io/badge/python-3.7-red.svg)](https://www.python.org/downloads/release/python-360/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI](https://img.shields.io/pypi/v/knodle)
[![build status](https://github.com/knodle/knodle/workflows/lint_n_test/badge.svg)](https://github.com/knodle/knodle/actions?branch=style_guide)
## Installation

```
pip install knodle
```

## Usage

knodle offers various methods for denoising weak supervision sources and improve them. There are several methods available for denoising. Examples can be seen in the tutorials folder.

There are four mandatory inputs for knodle:

1. `model_input_x`: Your model features (e.g. TFIDF values) without any labels. Shape: n_instances x features
2. `mapping_rules_labels_t`: This matrix maps all weak rules to a label. Shape: n_rules x n_classes
3. `rule_matches_z`: This matrix shows all applied rules on your dataset. Shape: n_instances x n_rules
4. `model`: A PyTorch model which can take your provided `model_input_x` as input. Examples are in the [model folder](https://github.com/knodle/knodle/tree/develop/knodle/model/).

Example for training the baseline classifier:

```python
from knodle.model.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.baseline.majority import MajorityVoteTrainer
from knodle.data import get_imdb_dataset
from knodle.data.transformer import get_labels
from knodle.data.tfidf import get_tfidf_features

NUM_OUTPUT_CLASSES = 2

train_df, dev_df, test_df, z_train_rule_matches, train_rule_matches_z, mapping_rules_labels_t = \
        get_imdb_dataset()

train_tfidf, test_tfidf, dev_tfidf = get_tfidf_features(train_df, test_df, dev_df)
test_labels = get_labels(test_df)
dev_labels = get_labels(dev_df)

model = LogisticRegressionModel(train_tfidf.shape[1], NUM_OUTPUT_CLASSES)

trainer = MajorityVoteTrainer(
  model,
  mapping_rules_labels_t=mapping_rules_labels_t,
  model_input_x=train_tfidf,
  rule_matches_z=train_rule_matches_z,
  dev_model_input_x=dev_tfidf,
  dev_gold_labels_y=dev_labels
)

trainer.train()

trainer.test(test_tfidf, test_labels)
```

For seeing how the imdb dataset was created please have a look at the [dedicated tutorial](https://github.com/knodle/knodle/tree/develop/tutorials/ImdbDataset).

## Denoising Methods

There are several denoising methods available.

| Name                 | Module                                  | Description                                                                                                                                                                                                   |
| -------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline             | `knodle.trainer.baseline`               | This builds the baseline for all methods. No denoising takes place. The final label will be decided by using a simple majority vote approach and the provided model will be trained with these labels.        |
| kNN TFIDF Similarity | `knodle.trainer.knn_tfidf_similarities` | This method looks at the similarities in tfidf values of the sentences. Similar sentences will receive the same label matches of the rules. This counteracts the problem of missing rules for certain labels. |
| DSCrossWeigh         | `knodle.trainer.crossweigh`             | This method weighs the training samples basing on how reliable their labels are. The less reliable sentences (i.e. sentences, whose weak labels are possibly wrong) are detected using a CrossWeigh method, which is similar to k-fold cross-validation, and got reduced weights in further training. This counteracts the problem of wrongly classified sentences. |

## Tutorials

We also aimed at providing the users with basic tutorials that would explain how to use our framework. All of them are stored in [tutorials](https://github.com/knodle/knodle/tree/develop/tutorials/) folder and logically divided into two groups:
- tutorials that demonstrate how to prepare the input data for Knodle Framework...
    - ... on the example of a well-known ImdB dataset. A weakly supervised dataset is created by incorporating keywords as weak sources (LINK)
    - ... on the example of a TAC-based dataset in .conll format. A relation extraction dataset is created using entity pairs from Freebase as weak sources (LINK)
- tutorials how to work with Knodle Framework
    - ... on the example of AutoTrainer. This trainer is to be called when user wants to train a weak classifier, but has no intention to use any specific denoising method, but rather try all currently provided in Knodle. (LINK)
    - ... on the example of DSCrossWeighTrainer. With this trainer a weak classifier with DSCrossWeigh denoising method will be trained. (LINK)

## Compatibility

Currently the package is tested on Python 3.7. It is possible to add further versions. The CI/CD pipeline needs to be updated in that case.

## Structure

The structure of the code is as follows
```
knodle
├── knodle
│    ├── data
│    ├── evaluation
│    ├── trainer
│          ├── baseline
│          ├── cleanlab
│          ├── dscrossweigh_denoising
│          ├── knn_denoising
│          ├── snorkel
│          └── utils
│    ├── transformation
│    └── utils
├── tests
│    ├── data
│    ├── evaluation
│    ├── trainer
│          ├── baseline
│          ├── dscrossweigh
│          ├── snorkel
│          └── utils
│    └── transformation
└── examples
     ├── data_preprocessing
           ├── imdb_dataset
           └── tac_based_dataset
     └── training
           ├── AutoTrainer
           └── dscrossweigh

```

## License

Licensed under the [Apache 2.0 License](LICENSE).

## Contact

If you notices a problem in the code, you can report it by [submitting an issue](<https://github.com/knodle/kndole/issues/new>).

If you want to share your feedback with us or take part in the project, contact us via 
<knodle@cs.univie.ac.at>. 

And don't forget to follow [@knodle](https://twitter.com/knodle) on Twitter :)

## Authors

- [Anastasiia Sedova](https://github.com/agsedova)
- [Andreas Stephan](https://github.com/AndSt)
- [Marina Speranskaya](https://github.com/marina-sp) 
- [Alessandro Volpicella](https://github.com/AlessandroVol23)
- [Benjamin Roth](https://www.benjaminroth.net/)


## Acknowledgments

This research was funded by the WWTF though the project “Knowledge-infused Deep Learning for Natural Language Processing” (WWTF Vienna Research Group VRG19-008).
