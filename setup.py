from setuptools import setup, find_packages

setup(
    name='AUTO1',
    version='0.1',
    packages=[
        'data',
        'model',
        'train'
        ],
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'torch', 
        'timm',
        'einops',
        'tensorboard'
    ],
)