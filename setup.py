import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stepwise_formula",
    version="0.0.4",
    author="Gabriela Plantie",
    author_email="glplantie@gmail.com",
    description="Stepwise Formula",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabriela-plantie/stepwise_formula",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.1.2',
        'statsmodels>=0.12.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)