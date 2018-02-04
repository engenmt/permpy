#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='permpy',
      version='0.0.2',
      description='Permutation (patterns) workshop in Python',
      author='Cheyne Homberger and Jay Pantone',
      author_email='cheyne.homberger@gmail.com',
      url='https://github.com/cheyneh/permpy',
      keywords = ['permutations', 'patterns', 'research',
          'enumeration'],
      classifiers=[],
      packages=['permpy', 
                'permpy.InsertionEncoding', 
                'permpy.RestrictedContainer'],
     )
