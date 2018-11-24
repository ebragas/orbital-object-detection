from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = []

setup(
    name="flowers_model",
    version="0.1",
    author="Eric Bragas",
    author_email="ericbragas412@gmail.com",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Image Classification CNN on Google ML Engine",
    requires=[],
)
