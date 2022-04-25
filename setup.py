import setuptools

setuptools.setup(
    name="dgnn",
    version="0.0.1",
    description="A framework for dynamic graph neural networks",
    url="https://github.com/yuchenzhong/dynamic-graph-neural-networks",
    packages=setuptools.find_packages(exclude=("tests")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)
