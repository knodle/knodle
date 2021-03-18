import io
import os
import re
from typing import Dict

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())

with open("requirements.txt") as f:
    requirements = f.readlines()

test_requirements = ["pytest", "pytest-cov"]

# use external package to extract version of the latest release
# https://pypi.org/project/setuptools-git-versioning/

setup(
    name="knodle",
    version_config={
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}",
        "dirty_template": "{tag}.post{ccount}",  # during build always dirty, since untracked files are created
        "count_commits_from_version_file": False
    },
    setup_requires=['setuptools-git-versioning'],
    url="http://knodle.cc",
    project_urls={
        "github": "https://github.com/knodle/knodle",
        "Bug Tracker": "https://github.com/knodle/knodle/issues"
    },
    license="Apache 2.0",
    author="knodle",
    author_email="knodle@cs.univie.ac.at",
    description="Knowledge infused deep learning framework",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests"]),
    package_dir={'knodle': 'knodle'},
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require={"test": test_requirements},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
)
