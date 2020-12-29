<a href="https://www.github.com/knodle/knodle">
    <img src="./knodle_logo.png" alt="knodle logo" title="Knodle" align="right" height="100" />
</a>

# knodle

Knowledge infused deep learning framework

## Installation

`pip install knodle`

## Usage

knodle offers various methods for denoising weak supervision sources and improve them. There are several methods available for denoising. Examples can be seen in the tutorials folder.

There are four mandatory inputs for knodle:

1. `model_input_x`: Your model features (e.g. TFIDF values) without any labels. Shape: n_instances $\times$ 1
2. `mapping_rules_labels_t`: This matrix maps all weak rules to a label. Shape: n_rules $\times$ n_classes
3. `rule_matches_z`: This matrix shows all applied rules on your dataset. Shape: n_instances $\times$ n_rules
4. `model`: A PyTorch model which can take your provided `model_input_x` as input. Examples are in the [model folder](https://github.com/knodle/knodle/tree/develop/knodle/model/).

Example for training the baseline classifier:

```python
from knodle.model import LogisticRegressionModel
from knodle.trainer import TrainerConfig
from knodle.trainer.baseline.baseline import SimpleDsModelTrainer
from knodle.data import get_imdb_dataset()

OUTPUT_CLASSES = 2

model_input_x, mapping_rules_labels_t, rule_matches_z = get_imdb_dataset()

model = LogisticRegressionModel(model_input_x.shape[1], OUTPUT_CLASSES)

trainer = SimpleDsModelTrainer(
    model,
    mapping_rules_labels_t=mapping_rules_labels_t,
    model_input_x=train_dataset,
    rule_matches_z=train_rule_matches_z
)

trainer.train()

trainer.test(test_features=test_tfidf, test_labels=Tensor(y_test))
```

## Denoising Methods

There are several denoising methods available.

| Name                 | Module                                  | Description                                                                                                                                                                                                   |
| -------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline             | `knodle.trainer.baseline`               | This builds the baseline for all methods. No denoising takes place. The final label will be decided by using a simple majority vote approach and the provided model will be trained with these labels.        |
| kNN TFIDF Similarity | `knodle.trainer.knn_tfidf_similarities` | This method looks at the similarities in tfidf values of the sentences. Similar sentences will receive the same label matches of the rules. This counteracts the problem of missing rules for certain labels. |
| Crossweight          | `knodle.trainer.crossweight`            |                                                                                                                                                                                                               |

## Development

We follow the [git flow](https://gist.github.com/digitaljhelms/4287848) approach while developing this application. Basically we have three environments:

1. Develop: All features will be merged into develop and tested from the team.
2. Staging: After the develop environment was tested and the requested features were implemented a staging / release environment is prepared. On that environment all features should be tested by several end-users.
3. Main: This is the stage which automatically deploys to the the end-users.

Every new feature will be implemented with a feature branch wchih follows normally the naming convention of `feature/<ISSUE-NUMBER>`. All feature branches will be merged via a Pull Request after a code review. All default branches are locked for direct pushes.

### Add new requirements

Please add all requirements manually. If it isn't needed to have an explicit version just enter the package name (e.g. `pandas`) without any version.

## Compatibility

Currently the package will be tested on Python 3.7. It is possible to add further versions. The CI/CD pipeline needs to be updated in that case.

## Licence

TBC

## Authors

- Benjamin Roth
- Anastassia Sedova
- [Alessandro Volpicella](https://github.com/AlessandroVol23)
- Andreas Stephan
