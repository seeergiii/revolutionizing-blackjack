from setuptools import setup, find_packages

with open("requirements.txt", "r") as file:
    lines = file.readlines()

requirements = [each.strip() for each in lines]

setup(name="training", packages=["training"], install_requires=requirements)
