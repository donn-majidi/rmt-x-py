import matplotlib.pyplot as plt
from src.Gaussian_ensembles import gaussian_ensembles_plot

fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize = (18,12))

gaussian_ensembles_plot(ax1, 8, 1, 50000)
gaussian_ensembles_plot(ax2, 8, 2, 50000)
gaussian_ensembles_plot(ax4, 8, 4, 50000)

plt.show()