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


VERSION: Dict[str, str] = {}
with open("knodle/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open('requirements.txt') as f:
    requirements = f.readlines()

test_requirements = ['pytest', 'pytest-cov']

setup(
    name="knodle",
    version="0.1.0",
    url="https://github.com/knodle/knodle",
    license="TBC",
    author="knodle",
    author_email="knodle@cs.univie.ac.at",
    description="Knowledge infused deep learning framework",
    long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require={'test': test_requirements},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
