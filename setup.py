from setuptools import setup, find_packages

with open('requirements.txt', 'r') as file:
    lines = file.readlines()

requirements = [each.strip() for each in lines]

setup(name='blackjack', packages=find_packages(), install_requires=requirements)
