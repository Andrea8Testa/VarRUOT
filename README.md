# VarRUOT

Thank you for reviewing our manuscript.

## Environment Setup

We recommend using Conda for environment management. You can create a new environment as follows:

```bash
conda env create -f environment.yml
conda activate VarRUOT
```

VarRUOT has been tested on Linux systems with CUDA available and on Windows 11.

## Running Training Scripts

All training operations are performed in `train.py`. First, specify the dataset. For example, if you want to use the three-gene simulation dataset, use:

```python
dataset = SimulationDataset()
```

Next, define a network to approximate $\lambda(\mathbf{x},t)$ based on the dimensions of the dataset:

```python
multipliermodel = MultiplierModel(data_dim=2)
```

Create an instance of the `VarRUOT` class. If you want to use the standard WFR metric, use:

```python
my_ruot = VarRUOT_WFR(dataset, multiplyer_model=multipliermodel)
```

If you prefer to use the modified metric, use:

```python
my_ruot = VarRUOT_modified(dataset, multiplyer_model=multipliermodel)
```

Specify the path for saving the model and start training:

```python
train(my_ruot, save_path="mouse_easy")
```

The model will be saved in the `checkpoints` folder. To visualize the model's learning trajectory, use:

```python
my_ruot.visualize2dresult()
```

If you want to see the growth values of the data points, use:

```python
my_ruot.visualizedatagrowth()
```

## Datasets

Below are the names and dimensions of several datasets:

- Three-Gene Simulation Dataset: `SimulationDataset()`, 2 dimensions
- EMT Dataset: `EmtDataset`, 10 dimensions
- Mouse Hematopoiesis (2D): `MouseDataset()`, 2 dimensions
- Pancreatic $\beta$-Cell Differentiation Dataset: `VeresDataset()`, 30 dimensions
- Mouse Hematopoiesis (50D): `Dim50Dataset()`, 50 dimensions
- 50-Dimensional Gaussian Dataset: `Gaussian50Dataset()`, 50 dimensions
- 100-Dimensional Gaussian Dataset: `Gaussian100Dataset()`, 100 dimensions

All datasets can be trained using the process described above.
