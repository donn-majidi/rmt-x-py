import matplotlib.pyplot as plt
from src.Gaussian_ensembles import gaussian_ensembles_plot
from src.Wishart_ensembles import wishart_ensembles

fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize = (18,12))

gaussian_ensembles_plot(ax1, 20, 1, 1000)
gaussian_ensembles_plot(ax2, 20, 2, 1000)
gaussian_ensembles_plot(ax4, 20, 4, 1000)

plt.show()

beta = [1,2,4]
N = 8
M = N + 7
simul_count = 1000

for i in beta:
    wishart_ensembles(N, M, i, simul_count)

