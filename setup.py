#!/usr/bin/env python3

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="permpy",
    version="0.0.4",
    description="Permutation (patterns) workshop in Python",
    author="Michael Engen and Cheyne Homberger and Jay Pantone",
    author_email="cheyne.homberger@gmail.com",
    url="https://github.com/cheyneh/permpy",
    keywords=["permutations", "patterns", "research", "enumeration"],
    classifiers=[],
    packages=["permpy", "permpy.InsertionEncoding", "permpy.RestrictedContainer"],
)
