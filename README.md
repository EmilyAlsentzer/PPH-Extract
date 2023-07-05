# PPH-extract

## Overview
This repository contains the code for the paper "Zero-shot Interpretable Phenotyping of Postpartum Hemorrhage Using Large Language Models". While we unfortunately cannot release the data used in the paper because it contains protected health information, we release the code base, which includes all prompts we use in our experiments.

## Installation

### Download the repo
First, download our github repository via the following:

```
git clone https://github.com/EmilyAlsentzer/PPH-Extract
cd PPH-Extract
```

### Environment Setup

To create a conda environment containing all of the required packages, ensure that conda is installed and run the following:
```
conda env create -f environment.yml
conda activate pph-extract
```

### Install the Repo

After the conda environment is created and activated, install the github repo with the following:

```
pip install -e .
```

### Set up the config file
Go to `config.py` and set the project directory (`PROJECT_DIR`) to be the path where you will store the data and model outputs for the project.

## Usage