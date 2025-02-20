# setup.py
from setuptools import setup, find_packages

setup(
    name='quadratic_ot',
    version='0.1.0',
    packages=find_packages(include=['core', 'finite_distributions', 'sinkhorn', 'visualizer']),
    install_requires=[], # Add dependencies here, if needed
)