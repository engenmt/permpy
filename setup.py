#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="permpy-engenmt",
    version="0.0.12",
    description="A package for dealing with permutation patterns.",
    author="Michael Engen and Cheyne Homberger and Jay Pantone",
    author_email="cheyne.homberger@gmail.com",
    maintainer="Michael Engen",
    maintainer_email="engenmt@gmail.com",
    url="https://github.com/cheyneh/permpy",
    keywords=["permutations", "patterns", "research", "enumeration"],
    classifiers=[],
    packages=[
        "permpy",
        "permpy.InsertionEncoding",
        "permpy.RestrictedContainer",
        "permpy.deprecated",
    ],
)
