#!/usr/bin/env python3

from setuptools import setup, Command, find_packages
import os
import platform
import shutil

modules = []
# open the file
with open('requirements.txt', 'r') as f:
    for module in f:
        modules.append(module.strip())

    if platform == "Linux":
        modules.append("evdev")

setup(
    name='symbiote',
    version='0.17.0',
    description='A command line harness to AI functions.',
    author='Wadih Khairallah',
    url='https://github.com/woodyk/symbiote',
    packages=find_packages(),
    install_requires=modules,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
    ],
    entry_points={
        'console_scripts': [
            'symbiote=symbiote.app:main',
        ],
    },
    python_requires='>=3.10, <3.13',
)
