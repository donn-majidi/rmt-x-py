import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.covariance import ledoit_wolf

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

#### Marcenko-Pastur law
## mp_density = lambda x,xmin,xmax: 1/(2*np.pi*x) * np.sqrt( (x-xmin) * (xmax-x) )
mp_density = lambda x,c,sigma,xmax,xmin: ( np.sqrt( (xmax - x) * (x - xmin) ) /
                                              (2 * np.pi * sigma**2 * c * x) )

class WL_Ensemble:
    
    '''
    Wishart-Laguerre Ensemble
    This class handles spectral analysis and Monte Carlo simulations for
    Wishart-Laguerre ensembles. Supported ensembles are the LOE, LUE and LSE, identified
    with their corresponding Dyson index -- 1,2 and 4, respectively. The class has to be
    instantiated with a self-dual matrix (real symmetric, complex Hermitian, or quaternionic self-dual)
    serving as the poulation covariance matrix and its corresponding Dyson index.
    !NOTE: The complex component of the Population covariance matrix will be discarded
           for the Unitary and Symplectic ensembles. This is because numpy cannot handle
           complex-valued covariance matrices.
    '''
    
    def __init__(self, endog: np.ndarray, beta: int):
        
        self._endog = array_like(endog, 'endog', ndim=2)
        self._beta = int_like(beta, 'beta')
        self._ndim = self._endog.shape[0]
        self._nsim = None
        
        if np.any(self._endog != self._endog.conj().T):
            raise ValueError('Input matrix is not self-dual, therefore it '
                             'cannot belong to the Wishart-Laguerre ensemble.')
            
        if not np.all(np.linalg.eigvalsh(self._endog) > 0):
            raise ValueError('Input matrix is not positive definite.')
            
        self.evals = None

        self.sample_cov = None
        self.evals_sample = None

        self.lw_cov = None
        self.evals_lw = None
        self.shrinkage_lw = None

            
        self._compute_eig()
            
    def __str__(self):
        string = 'Class handle for Wishart-Laguerre Random Matrix Ensembles.\n'
        if self._beta is not None:
            if self._beta == 1:
                string += 'Class is initiated for the LOE.\n'
            elif self._beta == 2:
                string += 'Class is initiated for the LUE.\n'
            else:
                string += 'Class is initiated for the LSE.\n'
                
            string += '(ndim = ' + str(self._ndim) + ' -- '
            string += 'beta = ' + str(self._beta) + ')'
                
        return string
    
    def __repr__(self):
        string = self.__str__()
        string = string[:-1]
        string += ', id: ' + hex(id(self)) + ')'
        return string
    
    def simulate(self, nsim: int, seed: int | None = 1776):
        
        '''
        This fucntion simulates sample data from the multivariate Normal distribution
        with mean zero and the Population covariance matrix. It then computes the sample
        covariance matrix and its eigenvalues. For the Laguerre Orthogonal Ensemble it also
        computes the Ledoit-Wolf Linear Shrinkage estimator, its corresponding eigenvalues
        and the shrinkage intensity.
        '''
        
        nsim = int_like(nsim, 'nsim')
        seed = int_like(seed, 'seed')
        
        endog = self._endog
        ndim = self._ndim
        beta = self._beta
        
        if nsim <= 0:
            raise ValueError('nsim has to be greater than zero.')
            
        if nsim < ndim:
            warnings.warn('nsim is smaller than the dimension of the population '
                          'covariance matrix. The sample covariance matrix is singular.',
                          category=Warning, stacklevel=2)
        
        if nsim == ndim:
            raise ValueError('nsim cannot be equal to ndim.')
        
        _evals = np.array([])
        _evals_lw = None
        shrinkage = None
        mu = np.zeros(shape=ndim)
        rng = np.random.default_rng(seed=seed)
            
        match beta:
            case 1:
                ## !NOTE: In this routine the data matrix has dimension m by n
                ##        where m is the number of variables - i.e., the dimension
                ##        of the covariance matrix and n is the number of observations.
                X = rng.multivariate_normal(mean=mu, cov=endog, size=nsim).T
                S = X @ X.T / nsim
                _evals = np.linalg.eigvalsh(S)
                LW, shrinkage = ledoit_wolf(X.T, assume_centered=True)
                _evals_lw = np.linalg.eigvalsh(LW)
                
            case 2:
                X1 = rng.multivariate_normal(mean=mu, cov=endog, size=nsim).T
                X2 = rng.multivariate_normal(mean=mu, cov=endog, size=nsim).T
                X = X1 + 1j * X2
                S = X @ X.conj().T / nsim
                _evals = np.linalg.eigvalsh(S)
                
            case 4:
                X11 = rng.multivariate_normal(mean=mu, cov=endog, size=nsim).T
                X12 = rng.multivariate_normal(mean=mu, cov=endog, size=nsim).T
                X21 = rng.multivariate_normal(mean=mu, cov=endog, size=nsim).T
                X22 = rng.multivariate_normal(mean=mu, cov=endog, size=nsim).T
            
                X1 = X11 + 1j * X12
                X2 = X21 + 1j * X22
                
                X = np.block( [ [X1, X2], [-X2.conj(), X1.conj()] ] )
                S = X @ X.conj().T / nsim
                _evals = np.linalg.eigvalsh(S)
                
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

            case _:
                raise NotImplementedError()
                
        ## Rescale eigenvalues x -> (beta)x
        _evals /= (beta)
                
        ## Sort eigenvalues from largest to smallest
        _evals = _evals[::-1]
        ## If ndim > nsim, excatly ndim-nsim of the eigenvalues of the sample
        ## covariance matrix are zero. These are, however, not exactly zero in 
        ## the _evals array becasue of rounding error, but they are very small.
        ## Therefore, we manually set them to zero:
        if ndim > nsim:
            diff = ndim - nsim
            _evals[-diff:] = 0
        
        self.sample_cov = S
        self.evals_sample = _evals
        self._nsim = nsim
        
        if _evals_lw is not None:
            _evals_lw *= nsim / (ndim * beta)
            _evals_lw = _evals_lw[::-1]
            self.lw_cov = LW
            self.evals_lw = _evals_lw
            self.shrinkage_lw = shrinkage
        
    def _compute_eig(self):
        
        endog = self._endog
        ## beta = self._beta
        ## ndim = self._ndim
        
        _evals = np.linalg.eigvalsh(endog)
        
        '''
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
        
        ## Rescale eigenvluaes x -> (beta)x
        _evals /= (beta)
        '''
        
        ## Sort eigenvalues from largest to smallest
        _evals = _evals[::-1]
        self.evals =_evals
        
    def plot_spectrum(self, plot_type: str | None = 'density',
                      ax: plt.Axes | None = None, **kwargs):
        
        '''
        This function plots the empirical spectral density of the ensembles.
        If beta is 1, and ax is passed, make sure that it is 2D, for the second
        panel is required to plot the spectral density of the Ledoit-Wolf estimator.
        '''
        
        plot_type = string_like(plot_type, 'plot_type')
        endog = self._endog
        evals = self.evals
        evals_sample = self.evals_sample
        ndim = self._ndim
        nsim = self._nsim
        beta = self._beta
        
        evals_lw = self.evals_lw
        shrinkage = self.shrinkage_lw
        
        if evals_sample is None:
            raise Exception('Nothing to plot. First simulate sample data '
                            'calling the simulate() method.')
        
        if ax is not None:
            if beta == 1:
                canvas = ax[0]
                canvas_lw = ax[1]
            else:
                canvas = ax
        elif ax is None and beta != 1:
            fig, canvas = plt.subplots()
        else:
            fig, ax = plt.subplots(ncols=2, figsize=(22,8), **kwargs)
            canvas = ax[0]
            canvas_lw = ax[1]
            
        c = ndim / nsim
        ## Compute the scale factor for the MP Law
        ## !NOTE: This is only exact if the population covariance matrix is
        ##        spherical and homoskedastic, i.e., it is a multiple of the identity.
        sigma_hat = np.sqrt(np.trace(endog)/ndim)
        xmax = sigma_hat**2 * (1 + np.sqrt(c))**2 
        xmin = sigma_hat**2 * (1 - np.sqrt(c))**2 
        
        ## Marcenko-Pastur Law over the eigenvalue grid
        x_grid = np.linspace(xmin, xmax, 500)
        rho = mp_density(x_grid, c, sigma_hat, xmax, xmin)
        
        sns.lineplot(x=x_grid, y=rho, ax=canvas, color='coral',
                     label=f'Marcenko-Pastur Law - C = {c:.2f}')
        canvas.set_ylabel('Density')
        canvas.set_xlabel('$x$')
        canvas.set_title('Average Spectral Density')
        
        ## Compute max number of bins
        max_nbins = int(min(50,ndim/10))
        nbins = max(10, max_nbins)
        
        ## Get histogram of sample eigenvalues
        if ndim < nsim:
            e_density, e_edges = np.histogram(evals_sample, bins=nbins, density=True)
            e_grid = (e_edges[1:] + e_edges[:-1])/2
            if plot_type == 'density':
                sns.scatterplot(x=e_grid, y=e_density, ax=canvas, marker='X', color='mediumblue',
                            label='Sample Covariance Matrix')
            else:
                sns.histplot(evals_sample, bins=nbins, ax=canvas, stat='density',
                         label='Sample Covariance Matrix')
        else:
            e_density, e_edges = np.histogram(evals_sample[:nsim], bins=nbins, density=True)
            e_grid = (e_edges[1:] + e_edges[:-1])/2
            ## Scale the histogram such that its integral over all values sums to 1/c
            e_density *= 1/c
            ## Point mass 1-1/c at x=0
            e_grid = np.append(np.array([0]), e_grid)
            e_density = np.append(np.array([1-1/c]), e_density)
            if plot_type == 'density':
                sns.scatterplot(x=e_grid, y=e_density, ax=canvas, marker='X', color='mediumblue',
                            label='Sample Covariance Matrix')
            else:
                sns.histplot(evals_sample[:nsim], bins=nbins, ax=canvas, stat='density',
                         label='Sample Covariance Matrix')
                canvas.axvline(x=0, ymax=1/plt.ylim()[1]*(1-1/c), label='Mass '
                               f'{1-1/c:.2f} at zero.')
                
            for patch in canvas.patches:
                patch.set_height(patch.get_height() * 1/c)
            
        canvas.legend()
          
        if beta == 1:
            canvas_lw.set_ylabel('Density')
            canvas_lw.set_xlabel('$x$')
            canvas_lw.set_title('Average Spectral Density')
        
            if evals_lw is not None:
                e_density, e_edges = np.histogram(evals_lw, bins=nbins, density=True)
                e_grid = (e_edges[1:] + e_edges[:-1])/2
                if plot_type == 'density':
                    sns.scatterplot(x=e_grid, y=e_density, ax=canvas_lw, marker='X', color='mediumblue',
                                    label=f'Ledoit-Wolf Estimator - Shrinkage: {shrinkage:.2f}')
                    if np.std(evals) > 0:
                        sns.rugplot(evals, ax=canvas_lw, color='darkslategrey',
                                    label='Population Covariance Matrix')
                else:
                    sns.histplot(evals_lw, bins=nbins, ax=canvas_lw, stat='density',
                                 label=f'Ledoit-Wolf Estimator - Shrinkage: {shrinkage:.2f}')
                    if np.std(evals) > 0:
                        sns.rugplot(evals, ax=canvas_lw, color='darkslategrey',
                                    label='Population Covariance Matrix')
        
            canvas_lw.legend()

        if beta == 1:
            return canvas, canvas_lw
        else:
            return canvas
        
    def plot_scree(self, ax: plt.Axes | None = None, **kwargs):
        
        '''
        This function graphs the scree plots of the ordered eigenvalues of the 
        Population and Sample covariance matrices. For the Laguerre Orthogonal Ensemble
        it also graphs the scree plot of the eigenvalues of the Ledoit-Wolf Linear Shrinkage
        estimator.
        '''
        
        evals = self.evals
        evals_sample = self.evals_sample
        evals_lw = self.evals_lw
        shrinkage_lw = self.shrinkage_lw
    
        if evals_sample is None:
            raise Exception('Nothing to plot. First simulate sample data '
                            'calling the simulate() method.')
        
        if ax is not None:
            canvas = ax
        else:
            fig, canvas = plt.subplots(**kwargs)
            
        canvas.set_ylabel('Ordered Eigenvalues')
        canvas.set_xlabel('$x$')
        canvas.set_title('Scree Plot')
        
        sns.scatterplot(evals, alpha=0.5, linewidth=0.8, ax=canvas,
                        label = 'Population eigenvalues')
        sns.scatterplot(evals_sample, alpha=0.5, linewidth=0.8, ax=canvas,
                        label = 'Sample eigenvalues')
        
        if evals_lw is not None:
            sns.scatterplot(evals_lw, alpha=0.5, linewidth=0.8, ax=canvas,
                            label= f'LW estimator eigenvalues - Shrinkage Intensity = {shrinkage_lw:.2f}')
            
        canvas.legend()
        
        return canvas

    def plot_heatmap(self, cov_matrix: str | None = None,
                     ax: plt.Axes | None = None, **kwargs):
        
        '''
        This function plots and compares the heatmaps of the Population covariance matrix
        and the sample covariance matrix or the Ledoit-Wolf Linear Shrinkage estimator.
        Comparison with the Ledoit-Wolf Linear Shrinkage estimator is only available for
        the Laguerre Orthogonal Ensemble (LOE).
        
        If ax is passed it has to be 2D for the heatmap for each covariance matrix
        is plotted in a separate panel.
        '''
        
        cov_matrix = string_like(cov_matrix, 'cov_matrix', optional=True)
        if cov_matrix is None:
            cov_matrix = 'sample_cov'
            
        beta = self._beta
        endog = self._endog
        sample_cov = self.sample_cov
        lw_cov = self.lw_cov
        shrinkage_lw = self.shrinkage_lw

        if cov_matrix not in ('sample_cov', 'lw_cov'):
            raise ValueError("cov_matrix has to be chosen from 'sample cov' or 'LW cov'")
                    
        if beta != 1 and cov_matrix == 'LW cov':
            raise NotImplementedError('LW cov is only available for the LOE')
            
        if ax is not None:
            canvas = ax
        else:
            fig, canvas = plt.subplots(ncols=2, figsize=(22,8), **kwargs)
        
        canvas[0].set_title('Heatmap of the Population Covariance Matrix')
        sns.heatmap(endog, ax=canvas[0])
        
        if cov_matrix == 'sample_cov':
            canvas[1].set_title('Heatmap of the Sample Covariance Matrix')
            sns.heatmap(sample_cov, ax=canvas[1])
        else:
            canvas[1].set_title('Heatmap of the Ledoit-Wolf LS Estimator\n'
                                f'Shrinkage Intensity = {shrinkage_lw:.2f}')
            sns.heatmap(lw_cov, ax=canvas[1])
            
        return canvas
    
            
     