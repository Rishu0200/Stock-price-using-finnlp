from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Stock_Price_Prediction_FinBERT",
    version="0.1",
    author="Rishabh Anand",
    packages=find_packages(),
    install_requires = requirements,
)