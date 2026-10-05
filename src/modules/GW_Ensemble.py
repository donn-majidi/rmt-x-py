import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from statsmodels.tools.validation import (string_like,
                                          array_like,
                                          bool_like,
                                          float_like,
                                          int_like,
                                          )

plt.rc('figure', figsize=(16,6))
plt.rc('savefig', dpi=90)
plt.rc('font', family='sans-serif')
plt.rc('font', size=14)
sns.set_style('darkgrid')

#### Wigner's Semicirle law
sc_density = lambda x: 1/np.pi * np.sqrt( 2 - x**2 )

class GW_Ensemble:
    
    '''
    Gauss-Wigner Ensemble
    This class handles spectral analysis and Monte Carlo simulations for
    Gauss-Wigner ensembles. Supported ensembles are the GOE, GUE and GSE, identified
    with their corresponding Dyson index -- 1,2 and 4, respectively.
    '''
    
    def __init__(self, endog: np.ndarray | None = None, beta: int | None = None):
        
        self._endog = endog
        self._beta = int_like(beta, 'beta', optional=True)
        self._ndim = None
        self.evals = None
        
        self._beta_sim = None
        self.evals_sim = None
        
        if self._endog is not None and not isinstance(self._endog, np.ndarray):
            raise ValueError('endog matrix has to be a 2D numpy array.')
        
        if self._endog is not None:
            if np.any(self._endog != self._endog.conj().T):
                raise ValueError('Input matrix is not self-dual, therefore it '
                                 'cannot belong to the Gauss-Wigner ensemble.')
            if self._beta is None:
                raise ValueError('If endog is passed, beta must be passed too.')
                
            ## !NOTE: Quaternionic self-dual matrices are 2N by 2N matrices by
            ##        consturction, whereas the dimension of the system is N.
            if self._beta == 4:
                self._ndim = self._endog.shape[0]/2
            else:
                self._ndim = self._endog.shape[0]
            self._compute_eig()
            
        if self._beta is not None:
            if self._beta not in (1,2,4):
                raise ValueError('The only supported values for the Dyson index '
                                 'are (1,2,4).')
        
        
    def __str__(self):
        string = 'Class handle for Gauss-Wigner Random Matrix Ensembles.\n'
        if self._beta is not None:
            if self._beta == 1:
                string += 'Class is initiated for the GOE.\n'
            elif self._beta == 2:
                string += 'Class is initiated for the GUE.\n'
            else:
                string += 'Class is initiated for the GSE.\n'
                
            string += '(ndim = ' + str(self._ndim) + ' -- '
            string += 'beta = ' + str(self._beta) + ')'
                
        return string
    
    def __repr__(self):
        string = self.__str__()
        string = string[:-1]
        string += ', id: ' + hex(id(self)) + ')'
        return string
    
    def simulate(self, ndim: int, beta: int, nsim: int | None = 1000,
                 seed: int | None = 1776):
        
        '''
        This function simulates random matrix ensembles from the Gaussian Orthogonal,
        Unitary, and Symplectic ensembles and calculates their eigenvalues.
        '''
        
        ndim = int_like(ndim, 'ndim')
        beta = int_like(beta, 'beta')
        nsim = int_like(nsim, 'nsim')
        seed = int_like(seed, 'seed')
        
        endog_ndim = self._ndim
        endog_beta = self._beta
        
        if ndim <= 0:
            raise ValueError('ndim has to be greater than zero.')
        
        if nsim < 0:
            raise ValueError('nsim has to be greater than zero.')
        
        if beta not in (1,2,4):
            raise ValueError('Dyson index has to be one of (1,2,4).')
            
        if endog_ndim is not None and endog_ndim != ndim:
            warnings.warn('ndim is different from the dimension of the endog matrix.',
                          category=Warning, stacklevel=2)
            
        if endog_beta is not None and endog_beta != beta:
            warnings.warn('beta is different from the Dyson index of the endog matrix.',
                          category=Warning, stacklevel=2)
            
        evals = np.array([])
        rng = np.random.default_rng(seed=seed)
        
        match beta:
            case 1:
                for i in range(nsim):
                    H = rng.standard_normal(size=[ndim,ndim])
                    GOM = (H + H.T)/2
                    _evals = np.linalg.eigvalsh(GOM)
                    evals = np.append(evals, _evals)
            
            case 2:
                for i in range(nsim):
                    A = rng.standard_normal(size=[ndim,ndim])
                    B = rng.standard_normal(size=[ndim,ndim])
                    H = A + 1j * B
                    GUM = (H + H.conj().T)/2
                    _evals = np.linalg.eigvalsh(GUM)
                    evals = np.append(evals, _evals)
                    
            case 4:
                for i in range(nsim):
                    A1 = rng.standard_normal(size=[ndim, ndim])
                    A2 = rng.standard_normal(size=[ndim, ndim])
                    B1 = rng.standard_normal(size=[ndim, ndim])
                    B2 = rng.standard_normal(size=[ndim, ndim])
                    
                    A = A1 + 1j * A2
                    B = B1 + 1j * B2
                    
                    H = np.block([[A, B], [-B.conj(), A]])
                    GSM = (H + H.conj().T)/2
                    _evals = np.linalg.eigvalsh(GSM)
                    _evals = np.sort(_evals)
                    ## Quaternionic self-dual matrices have 2N eigenvalues
                    ## N of which are unique. This is a small trick to only
                    ## get the unique eigenvalues.
                    ## Note that np.unique doesn't work because repeated
                    ## eigenvalues are not exactly the same due to rounding errors.
                    mask = []
                    mask.append(True)   # The mask for the first element.
                                        # Set to true because the iteration
                                        # begins from this element.
                    for j in range(1, len(_evals)):
                        if np.round(_evals[j], 5) == np.round(_evals[j-1], 5):
                            mask.append(False)
                        else:
                            mask.append(True)
                    _evals = _evals[mask]   # These are now the unique eigenvalues.
                    evals = np.append(evals, _evals)
                    
            case _:
                raise NotImplementedError()
                
        ## Rescale eigenvalues x -> sqrt(N * beta)x
        evals /= np.sqrt(beta * ndim)
        ## Sort eigenvalues from largest to smallest
        indx = np.argsort(evals)
        indx = indx[::-1]
        evals = evals[indx]
        
        self.evals_sim = evals
        self._beta_sim = beta
        
    def plot_spectrum(self, ax: plt.Axes | None = None, **kwargs):
        
        '''
        This function plots the empirical spectral density of the ensembles.
        It optionally accepts a plt.Axes object as input to use as the canvas.
        '''
        
        evals = self.evals
        evals_sim = self.evals_sim
        beta = self._beta
        beta_sim = self._beta_sim
        
        if evals is None and evals_sim is None:
            raise Exception('Nothing to plot. Either an input endogenous matrix has to be provided '
                            'or an ensemble simulated via the simulate() mehtod.')
            
        if ax is not None:
            canvas = ax
        
        else:
            fig, canvas = plt.subplots(**kwargs)
        
        ## Wigner's semicircle law over the scaled eigenvalue grid [-2,2]
        x_grid = np.linspace(-np.sqrt(2),np.sqrt(2),500)
        rho = sc_density(x_grid)
        
        ## Due to rounding error, the value of rho at the boundaries is undefined
        ## manually set to zero
        rho[0], rho[-1] = 0, 0
        
        sns.lineplot(x=x_grid, y=rho, ax=canvas, color = 'coral',
                     label="Wigner's Semicircle Law")
        canvas.set_ylabel('Density')
        canvas.set_xlabel('$x$')
        canvas.set_title('Average Spectral Density')
        
        if evals is not None:
            max_nbins = int(min(50, evals.shape[0]/10))
            nbins = max(10, max_nbins)
            
            e_density, e_edges = np.histogram(evals, bins=nbins, density=True)
            
            ## Find the center of the bins
            e_grid = (e_edges[1:] + e_edges[:-1])/2
            
            ## plot the empirical density of the eigenvalues
            sns.scatterplot(x=e_grid, y=e_density, ax=canvas, marker = 'X', color='darkmagenta',
                            label=f'Empirical Spectral Density of the endog Matrix - $\\beta_0 = {beta}$')
            
        if evals_sim is not None:
            max_nbins = int(min(50, evals_sim.shape[0]/10))
            nbins = max(10, max_nbins)
            
            e_density, e_edges = np.histogram(evals_sim, bins=nbins, density=True)
            e_grid = (e_edges[1:] + e_edges[:-1])/2
            
            sns.scatterplot(x=e_grid, y=e_density, ax=canvas, label=f'Simulated Empirical Spectral Density - $\\beta_s = {beta_sim}$')
        
        return canvas
    
    def plot_scree(self, ax: plt.Axes | None = None, **kwargs):
        
        '''
        This function graphs the scree plots of the sorted eigenvalues of th endogenous 
        and the simulated matrices. If the endogenous matrix is not set, it only graphs
        the scree plot of the eigenvalues of the simulated matrix.
        '''
        
        evals = self.evals
        evals_sim = self.evals_sim
        beta = self._beta
        beta_sim = self._beta_sim
        
        if evals is None and evals_sim is None:
            raise Exception('Nothing to plot. Either an input endogenous matrix has to be provided '
                            'or an ensemble simulated via the simulate() mehtod.')
            
        if ax is not None:
            canvas = ax
        
        else:
            fig, canvas = plt.subplots(**kwargs)
            
        if evals is not None:
            sns.scatterplot(evals, ax=canvas, marker = 'X', color='darkslategrey',
                            alpha = 0.5, linewidth  = 0.8,
                            label=f'Eigenvalues of the Endogenous Matrix - $\\beta_0 = {beta}$')
            
        if evals_sim is not None:
            sns.scatterplot(evals_sim, ax=canvas, alpha = 0.5, linewidth = 0.8,
                            label=f'Simulated Eigenvalues - $\\beta_s = {beta_sim}$')
            
        canvas.set_ylabel('Sorted Eigenvalues')
        canvas.set_xlabel('$x$')
        canvas.set_title('Scree Plot of Sorted Eigenvalues')
        
        return canvas
    
    def _compute_eig(self):
        
        endog = self._endog
        ndim = self._ndim
        beta = self._beta
        
        _evals = np.linalg.eigvalsh(endog)
        ## For quaternionic self-dual matrices perform the following routine
        ## to keep only the unique eigenvalues
        if beta == 4:
            mask = []
            mask.append(True)
            for j in range(1, len(_evals)):
                if np.round(_evals[j], 5) == np.round(_evals[j-1], 5):
                    mask.append(False)
                else:
                    mask.append(True)
            _evals = _evals[mask]   # These are now the unique eigenvalues.
        
        ## Scale the eigenvalues by sqrt(N * beta)
        _evals /= np.sqrt(ndim * beta)
        
        ## Sort eigenvalues from largest to smallest
        _evals = _evals[::-1]
        self.evals = _evals
        
            
        
