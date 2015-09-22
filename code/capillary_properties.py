from __future__ import division, print_function, with_statement
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import vgm


def compute_cblock_conductance(G, slength=300, nsamples=100, omin=300, omax=1000):
    """Computes the conductance of a cubical capillary network in x,y,z
    direction. Repeats the computation for nsamples randomly chosen cubes and
    returns the maximum, mean, and median of the results.
    INPUT: G: VascularGraph.
           slength: The side-length of the cube.
           nsamples: The number of samples to be acquired.
           omin: The minimum x,y,z value allowed as the origin of the cube.
           omax: The maximum x,y,z value allowed as the origin of the cube.
    OUTPUT: The mean, maximum, and median of the cube conductance in x,y,z
            direction.
    """
    conductance = [[], [], []]
    for sample in xrange(nsamples):
        origin = (np.random.rand(1, 3)[0] * (omax - omin) + omin).tolist()
        sg = vgm.fuzzy_block_subgraph(G, 3 * [slength], origin, False, dRange=[0, 8])
        vgm.add_conductance(sg, 'a')
        direction = np.array([[1, 0, 0],  [0, 1, 0],  [0, 0, 1]])
        for i in range(3):
            c = vgm.fuzzy_block_conductance(sg, (-1 * direction[i]).tolist(),
                                            direction[i].tolist())
            conductance[i].append(c)
    maxConductance = np.amax(np.array(conductance), 1)
    meanConductance = np.mean(np.array(conductance), 1)
    medianConductance = np.median(np.array(conductance), 1)
    return meanConductance, maxConductance, medianConductance


def capillary_length():
    """Reads amira spatialGraph files and creates normalized histograms of the 
       linear and tortuous lengths of their capillaries.
    """
    directory = vgm.uigetdir('Select data directory')
    files = glob.glob1(directory,'*')
    files.remove('capillary_length.py')
    dirnames = ['linLength', 'tortuousLength']
    for d in dirnames:
        if not os.path.exists(d):
            os.mkdir(d)

    for f in files:
        try:
            g = vgm.read_amira_spatialGraph_v2(os.path.join(directory, f))
        except:
            continue
        linLengths = [np.linalg.norm(g.vs[e.source]['r'] - g.vs[e.target]['r']) for e in g.es(diameter_le=8.)]
        vgm.hist_pmf(linLengths, np.linspace(0,500,101))
        plt.savefig(os.path.join('linLength', f + '.png'))
        tortuousLengths = g.es(diameter_le=8.)['length']
        vgm.hist_pmf(tortuousLengths, np.linspace(0,500,101))
        plt.savefig(os.path.join('tortuousLength', f + '.png'))
