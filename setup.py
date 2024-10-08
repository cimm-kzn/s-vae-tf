
import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='hyperspherical_vae',
    version='0.1.2',
    author='Nicola De Cao, Tim R. Davidson, Luca Falorsi',
    author_email='nicola.decao@gmail.com',
    description='Tensorflow implementation of Hyperspherical Variational Auto-Encoders',
    license='MIT',
    keywords='tensorflow vae variational-auto-encoder von-mises-fisher  machine-learning deep-learning manifold-learning',
    url='https://github.com/cimm-kzn/s-vae-tf',
    download_url='https://github.com/cimm-kzn/s-vae-tf',
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    install_requires=['numpy', 'tensorflow>=2.0.0', 'scipy'],
    packages=find_packages()
)
