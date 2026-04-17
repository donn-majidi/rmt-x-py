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
        
z_1, z_2, z_4 = gaussian_ensembles(6, 50000)
 
        
fig, ax = plt.subplots(3,1)
ax[0].hist(z_1, bins='auto')
ax[1].hist(z_2, bins='auto')
ax[2].hist(z_4, bins='auto')
plt.show()







