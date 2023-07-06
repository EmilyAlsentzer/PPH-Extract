# PPH-Extract

## Overview
This repository contains the code for the paper ["Zero-shot Interpretable Phenotyping of Postpartum Hemorrhage Using Large Language Models"]([url](https://www.medrxiv.org/content/10.1101/2023.05.31.23290753v1)). While we unfortunately cannot release the data used in the paper because it contains protected health information, we release the code base, which includes all prompts we use in our experiments.

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

### Perform zero-shot prediction

You can generate predictions for notes using the script at `model/zero_shot_predictions.py`. 

a) If you do **not** have annotations for your notes, `cd` into the `model` folder and run as follows:
```
 accelerate launch \
  --main_process_port PORT_NUMBER \
  --config_file bf_16.yaml  \
  zero_shot_predictions.py  \
  --label LABEL \
  --batch_size BATCH_SIZE \
  --max_length MAX_LEN \
  --prompt_desc PROMPT \
  --unlabelled \
  --unlab_filename FILENAME_OF_UNLABELED_DATA

```

where `PORT_NUMBER`,  `BATCH_SIZE`, and `MAX_LEN` are parameters, `LABEL` is one of the labels in config.binary_labels or config.ie_labels, `PROMPT` is the type of prompt used (all prompts can be found in the `get_prompt` function), and  `FILENAME_OF_UNLABELED_DATA` is the filename of the csv containing the unlabelled clinical notes found in the config.UNLABELLED_DATA_DIR directory. This uses Huggingface's Accelerate to fit more data into memory at once.

b) if you do have annotations for your notes,  `cd` into the `model` folder and run as follows:

```
 accelerate launch \
  --main_process_port PORT_NUMBER \
  --config_file bf_16.yaml  \
  zero_shot_predictions.py  \
  --label LABEL \
  --batch_size BATCH_SIZE \
  --max_length MAX_LEN \
  --prompt_desc PROMPT \
  --annotations_filename FILENAME_OF_ANNOTATIONS_CSV \
  --all_notes_filename FILENAME_OF_ALL_NOTES_CSV \
  --labeled FILENAME_OF_LABELED_DATA
```

where `PORT_NUMBER`, `LABEL`, `BATCH_SIZE`, `MAX_LEN`, and `PROMPT` are as described above, `FILENAME_OF_ANNOTATIONS_CSV` is the csv containing the annotations for all notes (processed using `process_annotator_output.py`), `FILENAME_OF_ALL_NOTES_CSV` is the csv containing all notes, and `FILENAME_OF_LABELED_DATA` is the csv containing the labelled notes we're trying to evaluate.
