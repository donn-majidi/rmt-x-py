# rmt-x-py
RMT Tooolbox in Python

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Modules](#modules)
- [Requirements](#requirements)
- [References](#references)
- [License](#license)

## Overview
This repository contains a collection of Python modules for the study and simulation analysis of common random matrix ensembles, including Gauss-Wigner and Wishart-Laguerre ensembles.

## Installation
```bash
git clone https://github.com/donn-majidi/rmt-x-py.git
cd rmt-x-py
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

See [Requirements](#requirements) for the list of required packages.
## Modules

### `class GW_Ensemble`
```python
class GW_Ensemble(endog: np.ndarray | None = None,
                  beta: int | None = None)
```
Class handle for Gauss-Wigner ensembles. This class handles spectral analysis and Monte Carlo simulations for Gauss-Wigner ensembles. Supported ensembles are the GOE, GUE and GSE, identified with their corresponding Dyson index -- 1,2 and 4, respectively.

#### Parameters
- `endog`: 2-Dimensional array of an instance of the Gauss-Wigner ensemble. Has to be a self-dual matrix to qualify as an instance of the Gauss-Wigner ensemble. This parameter is optional.
- `beta`: Integer number determining the Dyson index of the ensemble. Has to be chosen among (1,2,4).

#### Properties
- `evals`: Eigenvalues of the endog matrix. Available only if endog is passed.
- `evals_sim`: Simulated eigenvalues of the ensemble. Available once the method simulate() is called.

#### Methods
```python
simulate(ndim: int,
         beta: int,
         nsim: int,
         seed: int | None = 1776)
```
This function simulates random matrix ensembles from the Gaussian Orthogonal, Unitary, and Symplectic ensembles and calculates their eigenvalues.

Parameters:
- `ndim`: Dimension of the random matrix ensemble to be simulated.
- `beta`: Dyson index of the random matrix ensemble to be simulated.
- `nsim`: Number of simulations.
- `seed`: Random seed to feed to numpy default_rng().

```python
plot_spectrum(ax: plt.Axes | None = None,
              **kwargs)
```
This function plots the empirical spectral density of the ensembles. It optionally accepts a plt.Axes object as input to use as the canvas.

Parameters:
- `ax`: plt.Axes object on which to draw the plot. If omitted, a new figure is created.
- `**kwargs`: Other keyword arguments to pass to matplotlib.

Returns:
- `plt.Figure`: The handle to the figure.

```python
plot_scree(ax: plt.Axes | None = None,
           **kwargs)
```
This function graphs the scree plots of the sorted eigenvalues of the endogenous and the simulated matrices. If the endogenous matrix is not set, it only graphs the scree plot of the eigenvalues of the simulated matrix.

Parameters:
- `ax`: plt.Axes object on which to draw the plot. If omitted, a new figure is created.
- `**kwargs`: Other keyword arguments to pass to matplotlib.

Returns:
- `plt.Figure`: The handle to the figure.

### `class WL_Ensemble`
```python
class WL_Ensemble(endog: np.ndarray,
                  beta: int)
```
Wishart-Laguerre Ensemble
This class handles spectral analysis and Monte Carlo simulations for
Wishart-Laguerre ensembles. Supported ensembles are the LOE, LUE and LSE, identified
with their corresponding Dyson index -- 1,2 and 4, respectively. The class has to be
instantiated with a self-dual matrix (real symmetric, complex Hermitian, or quaternionic self-dual)
serving as the poulation covariance matrix and its corresponding Dyson index.
>[!NOTE]
  > The complex component of the Population covariance matrix will be discarded for the Unitary and Symplectic ensembles. This is because numpy cannot handle complex-valued covariance matrices.

#### Parameters
- `endog`: 2-Dimensional array of an instance of the Wigner-Laguerre ensemble serving as the model Population covariance matrix. Has to be a self-dual positive definite matrix.
- `beta`: Integer determining the Dyson index of the ensemble. Has to be chosen among (1,2,4).

#### Properties
- `evals`: Eigenvalues of the endog matrix.
- `sample_cov`: Simulated sample covariance matrix. Available once the method simulate() is called.
- `evals_sample`: Eigenvalues of the sample covariance matrix. Available once the method simulate() is called.
- `lw_cov`: Ledoit-Wolf Linear Shrinkage estimator. Available only for the Laguerre Orthogonal Ensembles and once the method simulate() is called.
- `evals_lw`: Eigenvalues of the Ledoit-Wolf Linear Shrinkage estimator. Available only for the Laguerre Orthogonal Ensembles and once the method simulate() is called.
- `shrinkage_lw`: Shrinkage intensity of the Ledoit-Wolf Linear Shrinkage estimator. Available only for the Laguerre Orthogonal Ensembles and once the method simulate() is called.

#### Methods
```python
simulate(nsim: int,
         seed: int | None = 1776)
```
Parameters:
- `nsim`: Number of simulations. This will be used as the sample size.
- `seed`: Random seed to feed to numpy default_rng().

```python
plot_spectrum(plot_type: str | None = 'density',
              ax: plt.Axes | None = None,
              **kwargs)
```
This function plots the empirical spectral density of the ensembles. If beta is 1, and ax is passed, make sure that it is 2D, for the second panel is required to plot the spectral density of the Ledoit-Wolf estimator.
Parameters:
- `plot_type`: String character to choose the plot type. Choices are: 'density' and 'histogram'. Defaults to 'density'.
- `ax`: plt.Axes object on which to draw the plot. If omitted, a new figure is created.
- `**kwargs`: Other keyword arguments to pass to matplotlib.

Returns:
- `plt.Figure`: The handle to the figure.

```python
plot_scree(ax: plt.Axes | None = None,
           **kwargs)
```
This function graphs the scree plots of the ordered eigenvalues of the Population and Sample covariance matrices. For the Laguerre Orthogonal Ensembles it also graphs the scree plot of the eigenvalues of the Ledoit-Wolf Linear Shrinkage estimator.

Parameters:
- `ax`: plt.Axes object on which to draw the plot. If omitted, a new figure is created.
- `**kwargs`: Other keyword arguments to pass to matplotlib.

Returns:
- `plt.Figure`: The handle to the figure.

```python
plot_heatmap(cov_matrix: str | None = None,
             ax: plt.Axes | None = None,
             **kwargs)
```
This function plots and compares the heatmaps of the Population covariance matrix and the sample covariance matrix or the Ledoit-Wolf Linear Shrinkage estimator. Comparison with the Ledoit-Wolf Linear Shrinkage estimator is only available for the Laguerre Orthogonal Ensembles (LOE).
        
If ax is passed it has to be 2D for the heatmap for each covariance matrix is plotted in a separate panel.

Parameters:
- `cov_matrix`: String character to choose which covariance matrix to compare with the Population covariance matrix. Choices are: 'sample_cov' for the sample covariance matrix and 'lw_cov' for the Ledoit-Wolf LS estimator. 'lw_cov' is only available for the Laguerre Orthogonal Ensembles. Defaults to 'sample_cov'.
- `ax`: plt.Axes object on which to draw the plot. If omitted, a new figure is created.
- `**kwargs`: Other keyword arguments to pass to matplotlib.

Returns:
- `plt.Figure`: The handle to the figure.

## Requirements
- [`numpy>=2.3.0`](https://numpy.org/)
- [`pandas>=2.3.0`](https://pandas.pydata.org/)
- [`matplotlib>=3.10.0`](https://matplotlib.org/)
- [`seaborn>=0.13.0`](https://seaborn.pydata.org/)
- [`scikit-learn>=1.8.0`](https://scikit-learn.org/)
- [`statsmodels>=0.14.0`](https://www.statsmodels.org/)
## References
- Ledoit, O., & Wolf, M., (2004), A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis*, 88, 365–411.
- Livan, G., Novaes, M., & Vivo, P. (2018). Introduction to Random Matrices: Theory and Practice. *SpringerBriefs in Mathematical Physics* (Vol. 26). Springer, Cham. 10.1007/978-3-319-70885-0
## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
