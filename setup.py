#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='permpy',
      version='0.0.1',
      description='Permutation (patterns) workshop in Python',
      author='Cheyne Homberger',
      author_email='cheyne.homberger@gmail.com',
      url='https://github.com/cheyneh/permpy',
      packages=['permpy', 
                'permpy.InsertionEncoding', 
                'permpy.RestrictedContainer'],
     )
