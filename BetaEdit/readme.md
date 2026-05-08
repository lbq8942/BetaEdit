# FastEdit: Low-Rank Regularization Model Editing

## Overview

This repository contains the implementation of **MEMIT-F**, a low-rank variant of the MEMIT algorithm for effective, efficient and scalable model editing.

## Data and Code Structure

- **Data**: Provided in the `data/` directory.  
- **Core implementation**: Located in `algs/memitf/`.

## Quick Start

To run sequential edits using MEMIT-F on the CounterFact dataset with 10,000 edits:

```bash
python main.py
```

This command will:

- Automatically edit **LLaMA-3-8B** using MEMIT-F, with 10000 sequential edits on the Counterfact dataset.
- Save the edited model in the `cache/` directory

## Customization

### Specify a Different Model

You can select a different pre-trained model via the `llms` argument:


```bash
python main.py llms=gpt2-xl
```
Supported models include:

- `gpt2-xl`
- `gpt-j-6b`
- `llama3-8b`

(See the `llms/` directory for the full list of available models.)

### Adjust the Number of Edits

To run with a custom number of sequential edits (e.g., 2,000):

```bash
python main.py num_edits=2000
```

## Evaluation
### Post-Editing Evaluation
After editing, evaluate the model on the same set of edits:

```bash
python main.py test_only=True
```

Results will be saved in the `results/` directory.

GLUE Evaluation
To evaluate on the Stanford Sentiment Treebank (SST-2), a downstream GLUE task:

```bash
python main.py test_only=True glue_eval=True
```
This provides an additional test of edit locality.




