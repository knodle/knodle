# knodle

Knowledge infused deep learning framework

## Development

We follow the [git flow](https://gist.github.com/digitaljhelms/4287848) approach while developing this application. Basically we have three environments:

1. Develop: All features will be merged into develop and tested from the team.
2. Staging: After the develop environment was tested and the requested features were implemented a staging / release environment is prepared. On that environment all features should be tested by several end-users.
3. Main: This is the stage which automatically deploys to the the end-users.

Every new feature will be implemented with a feature branch wchih follows normally the naming convention of `feature/<ISSUE-NUMBER>`. All feature branches will be merged via a Pull Request after a code review. All default branches are locked for direct pushes.

## Usage

## Requirements

There are two files.

1. `requirements.txt` is for the installation at the end-user's client.
2. `requirements_dev.txt` is for development purposes.

To test the framework locally `knodle` is installed locally with `pip install -e .`. This installs the package and all chanegs are automatially available. This is already included in the `requirements_dev.txt`.

### Add new requirements

Please add all requirements manually. If it isn't needed to have an explicit version just enter the package name (e.g. `pandas`) without any version.

## Compatibility

Currently the package will be tested on Python 3.8. It is possible to add further versions. The CI/CD pipeline needs to be updated in that case.

## Licence

TBC

## Authors
