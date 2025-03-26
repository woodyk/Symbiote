#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: setup.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-03-22 16:10:05
# Modified: 2025-03-23 14:50:03
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
    #python_requires='>=3.10, <3.13',
    python_requires='>=3.10',
)
