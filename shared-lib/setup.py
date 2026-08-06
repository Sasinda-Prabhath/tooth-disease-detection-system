from setuptools import find_packages, setup

setup(
    name="dental-common",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.0",
        "pillow>=10.4.0",
        "pydicom>=2.4.4",
        "pydantic>=2.8.0",
    ],
)
