# BetaEdit: Null-Space Constrained Sequential Model Editing



**2026/05/01 Paper accepted to IJCAI 2026.** 🎉


## Overview

This repository contains the implementation of **BetaEdit**, featuring two mechanisms: first, the knowledge leakage induced by the pseudo null space is penalized with $\lambda_1$ (set to 3000), second, the projection matrix $P_t$ is refreshed every $\tau$ (set to 1000) edits to maximize the profit of history-aware update. BetaEdit remains effective even after **10000** sequential edits.

The update rule of BetaEdit is,

$$
\boxed{
\Delta_t = \mathbf{R}_t \mathbf{K}_t^\top \mathbf{P}_t \left( \lambda_2 \mathbf{I} + \left( \mathbf{K}_t \mathbf{K}_t^\top + \lambda_1 \mathbf{K}_0 \mathbf{K}_0^\top + \sum_{i=1}^{t-1} \mathbf{K}_i \mathbf{K}_i^\top \right)\mathbf{P}_t \right)^{-1}
}
$$

While AlphaEdit runs as follows,

$$
\boxed{
\Delta_t = \mathbf{R}_t \mathbf{K}_t^\top \mathbf{P}_0 \left( \lambda_2 \mathbf{I} + \left( \mathbf{K}_t \mathbf{K}_t^\top + \sum_{i=1}^{t-1} \mathbf{K}_i \mathbf{K}_i^\top \right)\mathbf{P}_0 \right)^{-1}
}
$$


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
### Use Different Hyperparameters

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
@inproceedings{ijcai2026p508,
  title     = {BetaEdit: Null-Space Constrained Sequential Model Editing},
  author    = {Liu, Bingqing and Liu, Wei and Li, Yuhua},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Diego Calvanese},
  pages     = {4563--4571},
  year      = {2026},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2026/508},
  url       = {https://doi.org/10.24963/ijcai.2026/508},
}

```

