# BetaEdit: Null-Space Constrained Sequential Model Editing

## Overview

This repository contains the implementation of **BetaEdit**, featuring two mechanisms: first, the knowledge leakage induced by the pseudo null space is penalized with $\lambda_1$, second, the projection matrix **P** is refreshed every $\tau$ edits to maximize the profit of history-aware update.

**Paper accepted to IJCAI 2026.** 🎉

## Data and Code Structure

- **Data**: Provided in the `data/` directory.  
- **Core implementation**: Located in `algs/betaedit/`.

## Quick Start

To run sequential edits using BetaEdit on the CounterFact dataset with 10,000 edits:

```bash
python main.py algs=betaedit llms=gpt2-xl data=multi_counterfact_20877 num_edits=10000 bs=1 eval_every=10000 save_name=justtest
```

This command will:

- Edit **gpt2-xl** using BetaEdit, with 10000 sequential edits (batch size=1) on the CounterFact dataset.
- After 10000 edits, the program automatically evaluates the edited model (eval_every=10000). If set to 5000, it evaluates after 5000 and 10000 respectively.
- Evaluation results will be saved in `../result/multi_counterfact_20877/gpt2-xl/betaedit-justtest`.

## Customization

### Specify a Different Model

You can select a different pre-trained model via the `llms` argument:

```bash
llms=llama3-8b
```

Supported models include:

- `gpt2-xl`
- `gpt-j-6b`
- `llama3-8b`

(See the `llms/` directory for the full list of available models.)

### Adjust the Number of Edits

To run with a custom number of sequential edits (e.g., 2,000):

```bash
num_edits=2000
```
### Use different hyperparameters

We have two hyperparameters, the knowledge leakeage penalty coefficient $\lambda_1$ and the period $\tau$ to refresh the projection matrix. To change the hyperparameters, use the following command:

```bash
algs.lambda1=5000 algs.tau=500
```

### GLUE Evaluation

Use the `glue_eval=True` parameter:

```bash
glue_eval=True
```

## Citation

If you find this work helpful, please cite our paper:

```
BetaEdit: Null-Space Constrained Sequential Model Editing. IJCAI 2026.
```

