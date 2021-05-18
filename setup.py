from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="root-tomography",
    version="0.3",
    packages=["root_tomography"],
    url="https://github.com/PQCLab/pyRootTomography",
    author="Boris Bantysh",
    author_email="bbantysh60000@gmail.com",
    description="Python library for the root approach quantum tomography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy"
    ],
)
