from __future__ import division, print_function
import numpy as np
import vgm

def create_network(n=10., l_capillary=60., d_capillary=5., d_noncapillary=25., 
                   includeAV=True):
    """Create a vessel network containing a 2D honeycomb capillary bed, and 
    (optionally) a feeding artery and draining vein.
    INPUT: n: Number of hexagons the capillary bed should contain in each
              spatial direction. This determines the overall size of the 
              network.
           l_capillary: The length of a capillary segment.
           d_capillary: The diameter of a capillary segment.
           d_noncapillary: The diameter of artery and vein.
           includeAV: Include non-capillary vessels? (boolean)
    OUTPUT: nw: Vessel network.
    """

    # Create Vertices:
    #=================
    y_offset = l_capillary * np.cos(np.pi / 6)
    x_offset = l_capillary * np.sin(np.pi / 6)
    y_step = 2 * y_offset
    x_step = np.tile([l_capillary, l_capillary + 2 * x_offset], np.ceil(n/2))

    y = np.arange(0, (n+1)*y_step, y_step).tolist()
    x = [x_offset]; x.extend(np.cumsum(x_step).tolist() + x_offset)

    nx = len(x)
    ny = len(y)

    x.extend((0.,))
    x.extend((xx + l_capillary + x_offset for xx in x[:-2]))
    y.extend((yy + y_offset for yy in y[:]))
    r = []
    for i in xrange(nx):
        for j in xrange(ny):
            r.append((x[i], y[j]))
    x = [xx + l_capillary + x_offset for xx in x[:-1]]
    x.insert(0, 0.)
    y = [yy + y_offset for yy in y[:]]
    for i in xrange(nx):
        for j in xrange(ny):
            r.append((x[i], y[j]))
    xmax = max([rr[0] for rr in r])
    ymax = max([rr[1] for rr in r])
    if includeAV:
        r.append((-l_capillary, 0.))
        artery_vid = np.size(r, 0)
        r.append((-l_capillary, ymax))
        vein_vid = np.size(r, 0)
    r = np.array(r)
    r[:,0] = r[:,0] + l_capillary
    r = zip(r[:,1], r[:,0], [0. for rr in r])

    # Create Edges:
    #==============
    edgelist = [];
    # Capillary edges:
    for i in xrange(int(np.floor(nx/2))):
        sis = 1 + i * 2 * ny;
        sit = sis + ny;
        edgelist.extend(zip(range(sis, sis+ny-1), range(sit, sit+ny-1)))
    edgelist.extend([(e[0] + (ny*(nx+1)-1), e[1] + (ny*(nx+1)-1)) 
                    for e in edgelist[:]])
    for i in xrange(nx):
        sis = i * ny;
        sit = sis + nx * ny;
        edgelist.extend(zip(range(sis, sis+ny), range(sit, sit+ny)))
        edgelist.extend(zip(range(sis+1, sis+ny), range(sit, sit+ny-1)))
    N = nx*ny;
    capillary_eids = range(len(edgelist))
    # Artery edges:
    if includeAV:
        edgelist.extend(zip(range(0, N-ny, ny), range(ny, N, ny)))
        edgelist.append((0, 2*N))
        artery_eids = range(capillary_eids[-1]+1, len(edgelist))
    else:
        edgelist.extend(zip(range(0, N-ny, 2*ny), range(ny, N, 2*ny)))
    # Vein edges:
    if includeAV:
        edgelist.extend(zip(range(N-1+ny, 2*N-ny, ny), range(N+2*ny-1, 2*N, ny)))
        edgelist.append((N+ny-1, 2*N+1))
        vein_eids = range(artery_eids[-1]+1, len(edgelist))
    else:
        edgelist.extend(zip(range(N-1+2*ny, 2*N, ny*2), range(N+3*ny-1, 2*N+ny, ny*2)))
    # Create neighbors and adjacency lists:
    G = vgm.VascularGraph(edgelist)
    r = [(x[0], 0.0, x[1]) for x in r]
    G.vs['r'] = np.array(r)
    if includeAV:
        G['capillary_eids'] = capillary_eids
        G['artery_eids'] = artery_eids
        G['vein_eids'] = vein_eids
        G.es[capillary_eids]['diameter'] = [d_capillary for e in capillary_eids]
        G.es[artery_eids]['diameter'] = [d_noncapillary for e in artery_eids]
        G.es[vein_eids]['diameter'] = [d_noncapillary for e in vein_eids]
        G.es[capillary_eids]['kind'] = ['c' for e in capillary_eids]
        G.es[artery_eids]['kind'] = ['a' for e in artery_eids]
        G.es[vein_eids]['kind'] = ['v' for e in vein_eids]
        G.es['length'] = [l_capillary for e in G.es]
        G['av'] = [2*N]
        G['vv'] = [2*N+1]
    else:
        G['av'] = [ny*(nx-1)]
        G['vv'] = [ny*(nx+1) - 1]
        G.es['diameter'] = [d_capillary] * G.ecount()
        G.es['length'] = [l_capillary] * G.ecount()
        G.es['kind'] = ['c'] * G.ecount()
    return G

