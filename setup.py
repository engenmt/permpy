#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="permpy",
    version="0.0.3",
    description="Permutation (patterns) workshop in Python",
    author="Michael Engen and Cheyne Homberger and Jay Pantone",
    author_email="cheyne.homberger@gmail.com",
    url="https://github.com/cheyneh/permpy",
    project_urls={
        "Source": "https://github.com/cheyneh/permpy",
        "Tracker": ("https://github.com/cheyneh/permpy" "/issues"),
    },
    packages=find_packages(),
    keywords=["permutations", "patterns", "research", "enumeration"],
    classifiers=[],
    install_requires=["matplotlib"],
)
