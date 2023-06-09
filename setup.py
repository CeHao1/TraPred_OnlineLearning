from setuptools import setup, find_packages

setup(
    name='ol_dataset',
    version='0.0.1',
    description='the dataset for online learning',
    author='Ce Hao',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'mindspore',
        'tqdm',
        'zlib',
        'pickle',
    ]
)