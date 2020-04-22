#!/usr/bin/env python
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='polychaos-learn',
    version='0.1.0',
    description='polynomial chaos bases function in scikit-learn',
    author='Ahmed H. Elsheikh',
    author_email='a.elsheikh@hw.ac.uk',
    url='https://github.com/ahmed-h-elsheikh/polychaos-learn',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
