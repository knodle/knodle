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

