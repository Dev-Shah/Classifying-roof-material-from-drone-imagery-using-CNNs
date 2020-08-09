from matplotlib import pyplot
from math import pi
from math import cos
from math import floor


def cosine_annealing(n_epochs, n_cycles, lrate_max):
    series = []
    for epoch in range(n_epochs):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        series.append(lrate_max/2 * (cos(cos_inner) + 1))
    # print(series)
    ratio = n_epochs/n_cycles
    points = [ratio*x-1 for x in range(1,n_cycles+1)]
    # print(points)
    pyplot.figure(figsize=(10,6))
    pyplot.plot(series)
    pyplot.xlabel("Epoch",fontsize=15)
    pyplot.ylabel("Learning Rate",fontsize=15)
    pyplot.title("Cyclical Learning Rate using Cosine Annealing",fontsize=20)
    pyplot.plot(points,[min(series)+1e-4]*n_cycles,'x',mew=10, ms=5)
    pyplot.savefig('../CLR.png')

    return series
