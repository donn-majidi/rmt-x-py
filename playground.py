import numpy as np
import matplotlib.pyplot as plt

beta = [1,2,4]
dim = 8
simul_count = 50000

evals = np.array([])

for i in range(simul_count):
    M = np.random.normal(0,1,[dim,dim])
    M = (M + M.T)/2
    _evals = np.linalg.eigvals(M)
    evals = np.append(evals, _evals)
    
fig, ax = plt.subplots()
ax.hist(evals, bins='auto')
plt.show()
