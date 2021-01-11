<a href="https://www.github.com/knodle/knodle">
    <img src="./knodle_logo.png" alt="knodle logo" title="Knodle" align="right" height="100" />
</a>

# knodle

Knowledge infused deep learning framework

[![Python Version](https://img.shields.io/badge/python-3.7-red.svg)](https://www.python.org/downloads/release/python-360/)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation

`pip install knodle`

## Usage

knodle offers various methods for denoising weak supervision sources and improve them. There are several methods available for denoising. Examples can be seen in the tutorials folder.

There are four mandatory inputs for knodle:

1. `model_input_x`: Your model features (e.g. TFIDF values) without any labels. Shape: n_instances x features
2. `mapping_rules_labels_t`: This matrix maps all weak rules to a label. Shape: n_rules x n_classes
3. `rule_matches_z`: This matrix shows all applied rules on your dataset. Shape: n_instances x n_rules
4. `model`: A PyTorch model which can take your provided `model_input_x` as input. Examples are in the [model folder](https://github.com/knodle/knodle/tree/develop/knodle/model/).

Example for training the baseline classifier:

```python
from knodle.model import LogisticRegressionModel
from knodle.trainer import TrainerConfig
from knodle.trainer.baseline.baseline import SimpleDsModelTrainer
from knodle.data import get_imdb_dataset

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

For seeing how the imdb dataset was created please have a look at the [dedicated tutorial](https://github.com/knodle/knodle/tree/develop/tutorials/ImdbDataset).

## Denoising Methods

There are several denoising methods available.

| Name                 | Module                                  | Description                                                                                                                                                                                                   |
| -------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline             | `knodle.trainer.baseline`               | This builds the baseline for all methods. No denoising takes place. The final label will be decided by using a simple majority vote approach and the provided model will be trained with these labels.        |
| kNN TFIDF Similarity | `knodle.trainer.knn_tfidf_similarities` | This method looks at the similarities in tfidf values of the sentences. Similar sentences will receive the same label matches of the rules. This counteracts the problem of missing rules for certain labels. |
| CrossWeigh           | `knodle.trainer.crossweigh`             | This method weighs the training samples basing on how reliable their labels are. The less reliable sentences (i.e. sentences, whose weak labels are possibly wrong) are detected using a CrossWeigh method, which is similar to k-fold cross-validation, and got reduced weights in further training. This counteracts the problem of wrongly classified sentences. |

## Tutorials

The folder [tutorials](https://github.com/knodle/knodle/tree/develop/tutorials/) has different tutorials:

1. [IMDB Dataset Creation](https://github.com/knodle/knodle/tree/develop/tutorials/ImdbDataset): Shows how to create a weakly supervised dataset by incorporating keywords as weak sources.
2. [Relation Extraction Dataset](https://github.com/knodle/knodle/tree/develop/tutorials/RelationExtractionDataset): Shows the process of creating a dataset with the CONLL dataset.
3. [Baseline Training](https://github.com/knodle/knodle/tree/develop/tutorials/baseline_training_example): Shows the example process of training a baseline classifier.
4. [KNN Similarity Trainer](https://github.com/knodle/knodle/tree/develop/tutorials/knn_tfidf_similarity_example): Shows an example of how to use the denoising method of knn for training a weak classifier.

## Development

### Git Flow

We follow the [git flow](https://gist.github.com/digitaljhelms/4287848) approach while developing this application. Basically we have three environments:

1. Develop: All features will be merged into develop and tested from the team.
2. Staging: After the develop environment was tested and the requested features were implemented a staging / release environment is prepared. On that environment all features should be tested by several end-users.
3. Main: This is the stage which automatically deploys to the the end-users.

Every new feature will be implemented with a feature branch which follows normally the naming convention of `feature/<ISSUE-NUMBER>`. All feature branches will be merged via a Pull Request after a code review. All default branches are locked for direct pushes.

### Style Guide

We propose to follow all a same style guide within the code to ensure readability. The style guide follows heavily the [Pep 8 Style Guide](https://www.python.org/dev/peps/pep-0008/?). Google has a good example of a its [styleguide](https://google.github.io/styleguide/pyguide.html)

#### 1. Lint

We use `flake8` as a linter.

#### 2. Docstring

- Write docstrings.
- We need them to automatically create an API documentation.
- If it is completely obvious what is happening the docstring can be one-line. If the method is also private you can consider not writing docstrings.
- Use Google's docstring format. You can change it in your IDE (e.g. PyCharm, VsCode) to automatically insert the docstring.
- Contains:
  - Short description: What is the function doing
  - Arguments: All arguments provided
  - Raises: All exceptions
- Classes should have a docstring after the class declaration

#### 3. Imports

- Import from most generic (e.g. `os`, `sys`) to least generic (e.g. `knodle.trainer`)

#### 4. Naming

- Please use descriptive names. No 1 letter names and no abbreviations. If mathematical expression is needed for explanation append it to a descriptive name, e.g. `mapping_rules_labels_t`.
- Private functions start with an underscore:

```python
def _private_function(...):
    pass
```

##### File Names

Guidelines derived from [Guido](https://en.wikipedia.org/wiki/Guido_van_Rossum)'s Recommendations

<table rules="all" border="1" summary="Guidelines from Guido's Recommendations"
       cellspacing="2" cellpadding="2">

  <tr>
    <th>Type</th>
    <th>Public</th>
    <th>Internal</th>
  </tr>

  <tr>
    <td>Packages</td>
    <td><code>lower_with_under</code></td>
    <td></td>
  </tr>

  <tr>
    <td>Modules</td>
    <td><code>lower_with_under</code></td>
    <td><code>_lower_with_under</code></td>
  </tr>

  <tr>
    <td>Classes</td>
    <td><code>CapWords</code></td>
    <td><code>_CapWords</code></td>
  </tr>

  <tr>
    <td>Exceptions</td>
    <td><code>CapWords</code></td>
    <td></td>
  </tr>

  <tr>
    <td>Functions</td>
    <td><code>lower_with_under()</code></td>
    <td><code>_lower_with_under()</code></td>
  </tr>

  <tr>
    <td>Global/Class Constants</td>
    <td><code>CAPS_WITH_UNDER</code></td>
    <td><code>_CAPS_WITH_UNDER</code></td>
  </tr>

  <tr>
    <td>Global/Class Variables</td>
    <td><code>lower_with_under</code></td>
    <td><code>_lower_with_under</code></td>
  </tr>

  <tr>
    <td>Instance Variables</td>
    <td><code>lower_with_under</code></td>
    <td><code>_lower_with_under</code> (protected)</td>
  </tr>

  <tr>
    <td>Method Names</td>
    <td><code>lower_with_under()</code></td>
    <td><code>_lower_with_under()</code> (protected)</td>
  </tr>

  <tr>
    <td>Function/Method Parameters</td>
    <td><code>lower_with_under</code></td>
    <td></td>
  </tr>

  <tr>
    <td>Local Variables</td>
    <td><code>lower_with_under</code></td>
    <td></td>
  </tr>

</table>

#### 5. Function Length

- Use short and focused functions which should just do one thing
- Google's recommendation lies at about 40 LOC to think about breaking it apart into more functions

#### 6. Type Annotations

- Use [type annotations](https://www.python.org/dev/peps/pep-0484/) in every function

**BE CONSISTENT**. Review your code before submitting a PR and check if it consists with all other code. It'll be easier in the future for everybody to maintain :slightly_smiling_face:

### Add new requirements

Please add all requirements manually. If it isn't needed to have an explicit version just enter the package name (e.g. `pandas`) without any version.

## Compatibility

Currently the package will be tested on Python 3.7. It is possible to add further versions. The CI/CD pipeline needs to be updated in that case.

## License

Licensed under the [Apache 2.0 License](LICENSE).

## Authors

- [Benjamin Roth](https://www.benjaminroth.net/)
- [Anastasiia Sedova](https://github.com/agsedova)
- [Alessandro Volpicella](https://github.com/AlessandroVol23)
- Andreas Stephan
