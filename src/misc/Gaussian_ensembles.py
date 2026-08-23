import numpy as np
import matplotlib.pyplot as plt

def gaussian_ensembles(dim: int, simul_count: int,
        beta: list[int] | None = None):
        # --- Input validation ---
        if beta is None:
            beta = [1,2,4]
        elif type(beta) is int:
            beta = [beta]
        elif type(beta) is list:
            beta = beta
        else:
            raise ValueError(f"Unsupported data type for beta: {beta}. Must be either an integer or a list.")
        unsupported = set(beta) - {1, 2, 4}
        if unsupported:
            raise ValueError(f"Unsupported Dyson index: {unsupported}. Use 1, 2 or 4.")
        
        evals_1 = np.array([])
        evals_2 = np.array([])
        evals_4 = np.array([])
        
        for i in range(simul_count):
            M1 = np.random.normal(0,1,[dim,dim])
            M2 = np.random.normal(0,1,[dim,dim])
            M3 = np.random.normal(0,1,[dim,dim])
            M4 = np.random.normal(0,1,[dim,dim])
            
            GOM = (M1 + M1.T)/2
            
            GUM = ((M1 + M2 * 1j) + (M1 + M2 * 1j).conj().T)/2
            
            A = M1 + M2 * 1j
            B = M3 + M4 * 1j
            M = np.block([[A, B],[-np.conj(B), np.conj(A)]])
            GSM = (M + M.conj().T)/2
            
            _evals_1 = np.linalg.eigvalsh(GOM)
            _evals_2 = np.linalg.eigvalsh(GUM)
            _evals_4 = np.linalg.eigvalsh(GSM)
            
            evals_1 = np.append(evals_1, _evals_1)
            evals_2 = np.append(evals_2, _evals_2)
            evals_4 = np.append(evals_4, _evals_4)
            
        return evals_1, evals_2, evals_4
     
def gaussian_ensembles_plot(ax: plt.Axes, dim: int, beta: int,
                            simul_count: int):
        # --- Input validation ---
        if beta is None:
            raise ValueError("Please provide a value for beta.")
        unsupported = set([beta]) - {1, 2, 4}
        if unsupported:
            raise ValueError(f"Unsupported value for beta: {unsupported}. Must be 1, 2 or 4.")
            
        evals = np.array([])
            
        match beta:
            case 1:
                for i in range(simul_count):
                    M = np.random.standard_normal([dim,dim])
                    GOM = (M + M.T)/2
                    _evals = np.linalg.eigvalsh(GOM)
                    evals = np.append(evals, _evals)
            
            case 2:
                for i in range(simul_count):
                    M1 = np.random.standard_normal([dim,dim])
                    M2 = np.random.standard_normal([dim,dim])
                    GUM = ( (M1 + M2 * 1j) + (M1 + M2 * 1j).conj().T )/2
                    _evals = np.linalg.eigvalsh(GUM)
                    evals = np.append(evals, _evals)
            
            case 4:
                for i in range(simul_count):
                    M1 = np.random.standard_normal([dim,dim])
                    M2 = np.random.standard_normal([dim,dim])
                    M3 = np.random.standard_normal([dim,dim])
                    M4 = np.random.standard_normal([dim,dim])
                    
                    A = M1 + M2 * 1j
                    B = M3 + M4 * 1j
                    M = np.block([[A, B], [-np.conj(B), np.conj(A)]])
                    GSM = (M + M.conj().T)/2
                    _evals = np.linalg.eigvalsh(GSM)
                    evals = np.append(evals, _evals)
                    
            case _:
                raise ValueError("No valid input for the Dyson index provided.")
                
        out = ax.hist(evals, bins='auto')
        return out





