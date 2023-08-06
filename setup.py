from setuptools import setup, find_packages

setup(
    name="NN_FAP",
    version="0.1",
    packages=find_packages(),
    install_requires=['matplotlib','numpy','scipy','tensorflow','scikit-learn','tqdm'],
)

