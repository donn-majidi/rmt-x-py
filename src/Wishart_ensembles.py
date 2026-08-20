import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('figure', figsize=(16,6))
plt.rc('lines', linewidth=1.5)
sns.set_style('darkgrid')

## Define the Marcenko-Pastur density function
mp_density = lambda x,xmin,xmax: 1/(2*np.pi*x) * np.sqrt( (x-xmin) * (xmax-x) )

N = 60
M = 80

c = N/M
xmin = (1 - 1/np.sqrt(c))**2
xmax = (1 + 1/np.sqrt(c))**2
x = np.linspace(xmin,xmax,500)

rho = mp_density(x,xmin,xmax)

def wishart_ensembles(ndim1: int, ndim2: int, beta: int, simul_count: int):
    
    if ndim1 >= ndim2:
        raise ValueError('First dimension must be strictly smaller than the second dimension.')

    evals = np.array([])

    match beta:
        case 1:
            for i in range(0,simul_count):
                H = np.random.standard_normal([ndim1,ndim2])
                W = H @ H.T
                _evals = np.linalg.eigvalsh(W)
                evals = np.append(evals, _evals)
                
        case 2:
            for i in range(0,simul_count):
                H = np.random.standard_normal([ndim1,ndim2]) + 1j * np.random.standard_normal([ndim1, ndim2])
                W = H @ H.conj().T
                _evals = np.linalg.eigvalsh(W)
                evals = np.append(evals, _evals)
                
        case 4:
            for i in range(0,simul_count):
                A = np.random.standard_normal([ndim1,ndim2]) + 1j * np.random.standard_normal([ndim1,ndim2])
                B = np.random.standard_normal([ndim1,ndim2]) + 1j * np.random.standard_normal([ndim1,ndim2])
                H = np.block( [ [A,B], [-B.conj(), A.conj()] ] )
                W = H @ H.conj().T
                _evals = np.linalg.eigvalsh(W)
                ## A little work around to obtain the unique eigenvalues
                mask = []
                mask.append(True) ## The mask of the first element is always True
                for j in range(1, len(_evals)):
                    if np.round(_evals[j], 5) == np.round(_evals[j-1], 5):
                        mask.append(False)
                    else:
                        mask.append(True)
                _evals = _evals[mask]   ## These are the unique eigenvalues
                evals = np.append(evals, _evals)
                
        case _:
            raise ValueError('Dyson index is invalid.')
                
    ## Scale the eigenvalues
    evals *= 1/(beta * ndim1)
    
    ## Compute the Marcenko-Pastur density
    c = ndim1/ndim2
    xmin = ( 1 - 1/np.sqrt(c) )**2
    xmax = ( 1 + 1/np.sqrt(c) )**2
    x = np.linspace(xmin,xmax,500)
    rho = mp_density(x, xmin, xmax)
    
    fig, ax = plt.subplots()
    ax.hist(evals, bins='auto', alpha=0.75, density=True, label=f'Empirical Density - $\\beta$ = {beta}')
    ax.plot(x, rho, color='darkorange', linewidth = 2, label=f'Marcenko-Pastur Density - $C$ = {c}')
    ax.set_xlim(0,xmax+1)
    ax.legend()
    ax.set_title('Empirical vs. Asymptotic Density')
    plt.show()