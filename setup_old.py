#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import getopt, sys

make_lite = False

options = "LV:"
long_options = ["lite", "version="]

try:
    # Parsing argument
    arguments, values = getopt.getopt(sys.argv[1:], options, long_options)

    print(arguments, values)
    # Checking each argument
    for argument, value in arguments:
        if argument in ("-L", "--lite"):
            make_lite = True
        elif argument in ("-V", "--version"):
            version = value

except getopt.error as err:
    # Output error and return with an error code.
    print(f"Error! {err=}")

print(f"{version=}, {make_lite=}")

if make_lite:
    name = "permpy-lite"
    author_email = "engenmt@gmail.com"
    packages = ["permpy"]
else:
    name = "permpy"
    author_email = "cheyne.homberger@gmail.com"
    packages = [
        "permpy",
        "permpy.InsertionEncoding",
        "permpy.RestrictedContainer",
        "permpy.deprecated",
    ]

setup(
    name=name,
    version=version,
    description="A package for dealing with permutation patterns.",
    author="Michael Engen and Cheyne Homberger and Jay Pantone",
    author_email=author_email,
    url="https://github.com/cheyneh/permpy",
    keywords=["permutations", "patterns", "research", "enumeration"],
    classifiers=[],
    packages=packages,
)
