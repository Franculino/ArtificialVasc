from __future__ import division, print_function, with_statement
import copy
import numpy as np
import matplotlib.pyplot as plt
import vgm


def lattice(lengths, spacing, origin=None, **kwargs):
    """Constructs a VascularGraph with lattice structure (2D or 3D).
    INPUT: lengths: The x,y,z length of the domain as list.
           spacing: The spacing between vertices in x,y,z direction as list.
           origin: The origin of the lattice as list. If not provided, the
                   origin will be set to [0.0, 0.0, 0.0].
           **kwargs:
           diameter: The diameter assigned to the lattice edges.
           conductance: The conductance assigned to the lattice edges.
           volume: The volume assigned to the lattice edges.
    OUTPUT: G: VascularGraph with lattice structure.        
    """
    dim = [int(np.ceil(z[0] / z[1])) + 1 for z in zip(lengths, spacing)]
    G = vgm.VascularGraph.Lattice(dim, nei=1, directed=False, 
                                  mutual=True, circular=False)
    if origin is None:
        origin = [0.0 for l in lengths]

    if len(dim) == 2:
        dim.append(1)
        lengths.append(0.0)
        spacing.append(0.0)
        origin.append(0.0)

    r = []
    for z in xrange(dim[2]):
        for y in xrange(dim[1]):
            for x in xrange(dim[0]):
                r.append(np.array([origin[0] + x * spacing[0],
                                   origin[1] + y * spacing[1],
                                   origin[2] + z * spacing[2]]))
    G.vs['r'] = r

    keynames = ['diameter', 'conductance', 'volume']
    for key in keynames:
        if kwargs.has_key(key):
            G.es[key] = np.ones(G.ecount()) * kwargs[key]
 
    return G


def honeycomb(xlim=(0, 100), ylim=(0, 100), zlim=(0, 50), l=10.):
    """Create a 3D honeycomb capillary network. Each vertex in the network
    (except those at the edges of the domain) has a coordination number of
    three. The general shape of the network is as follows: the xy-planes
    consist of a two-dimensional, regular honeycomb network. In z-direction,
    the xy honeycomb planes are stacked on top of each other. Now each edge
    is divided in two equal parts by an additional vertex. It is at these 
    vertices that the xy-planes are connected with each other. Half the 
    edges of each hexagon connect with the layer above, the other half with
    the layer below.
    INPUT: xlim: Limits of the network in x-direction.
           ylim: Limits of the network in y-direction.
           zlim: Limits of the network in z-direction.
           l: Hexagon side-length.
    OUTPUT: G: VascularGraph of 3D-honeycomb structure.       
    """
    
    # The 3D-Honeycomb network is build as follows:
    # Zig-zag line in x-direction
    # Replicate zig-zag line along y
    # Connect lines to form a xy-honeycomb plane
    # Replicate plane along z
    # Connect planes to form a 3D-honeycomb network
    
    
    lx = 0.5 * l * np.cos(np.deg2rad(30)) # node distance in x-direction
    ly = 1.5 * l # distance of lines
    #lz = 0.5 * l # distance of planes
    lz = 0.25 * l # distance of planes
    nx = int(np.floor((xlim[1] - xlim[0]) / lx) + 1)
    ny = int(np.floor((ylim[1] - ylim[0]) / ly) + 1)
    nz = int(np.floor((zlim[1] - zlim[0]) / lz) + 1)
    
    # Create xy-honeycomb plane:
    edgelist = []
    r = []
    rc = []
    for iy in xrange(ny):
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.ones(nx) * ylim[0] + ly * iy
        if np.mod(iy, 2) == 0:
            y[0::4] = y[0::4] - 0.25 * l 
            y[2::4] = y[2::4] + 0.25 * l
            cx = x[2::4]
            cy = y[2::4] + 0.5 * l
            rowVertices = xrange(iy*nx+2, (iy+1)*nx, 4)
            nextRowVertices = xrange((iy+1)*nx+2, (iy+2)*nx, 4)
        else:
            y[0::4] = y[0::4] + 0.25 * l 
            y[2::4] = y[2::4] - 0.25 * l      
            cx = x[0::4]
            cy = y[0::4] + 0.5 * l
            rowVertices = xrange(iy*nx, (iy+1)*nx, 4)
            nextRowVertices = xrange((iy+1)*nx, (iy+2)*nx, 4)            
                
        interRowVertices = xrange(nx*ny + len(rc), nx*ny + len(rc) + len(rowVertices))
        edgelist.extend(zip(rowVertices, interRowVertices))

        if iy+1 < ny:
            edgelist.extend(zip(nextRowVertices, interRowVertices))
        cz = np.zeros(nx) * zlim[0]     
        z = np.ones(nx) * zlim[0]
        r.extend(zip(x,y,z))
        rc.extend(zip(cx,cy,cz))
        edgelist.extend(zip(xrange(nx*iy, nx*(iy+1)-1), xrange(nx*iy+1, nx*(iy+1))))

    r.extend(rc)
    origR = np.array(r)
    origEdgelist = copy.deepcopy(edgelist)
    
    # Replicate plane along z and make appropriate connections to create 3D network:
    for iz in xrange(1, nz):
        layerVertices = []
        niv = 0 # number of inter-row vertices
        for iy in xrange(ny):
            if np.mod(iz, 2) == 0:
                nrv = len(xrange(iy*nx+2, (iy+1)*nx, 4)) # number of row vertices connected to inter-row vertices
                layerVertices.extend(xrange(len(origR)*iz+iy*nx+1, len(origR)*iz+(iy+1)*nx, 4)) # row vertices
                layerVertices.extend(xrange(len(origR)*iz+nx*ny+niv+1, len(origR)*iz+nx*ny+niv+nrv, 2)) # inter-row vertices
                niv = niv + len(np.linspace(xlim[0], xlim[1], nx)[2::4])                     
            else:
                nrv = len(xrange(iy*nx, (iy+1)*nx, 4))
                layerVertices.extend(xrange(len(origR)*iz+iy*nx+3, len(origR)*iz+(iy+1)*nx, 4))
                if iz+1 < nz:
                    layerVertices.extend(xrange(len(origR)*iz+nx*ny+niv, len(origR)*iz+nx*ny+niv+nrv, 2))
                    niv = niv + len(np.linspace(xlim[0], xlim[1], nx)[0::4])
                            
        previousLayerVertices = [v-len(origR) for v in layerVertices]
        edgelist.extend(zip(layerVertices, previousLayerVertices))    	                
        edgelist.extend([(x[0] + iz*len(origR), x[1] + iz*len(origR)) for x in  origEdgelist])
        newR = copy.deepcopy(origR)
        newR[:, 2] = iz * lz
        r.extend(map(tuple, newR))
    
    # Use edgelist and coordinate array to create VascularGraph:    
    G = vgm.VascularGraph(edgelist)
    G.vs['r'] = np.array(r)
    G.es['length'] = [np.linalg.norm(G.vs[e.source]['r'] - G.vs[e.target]['r']) for e in G.es]
    G.es['diameter']=[4]*G.ecount()
        
    return G
    

def honeycomb_2d(sideLength=10, size=20):
    """Creates a 2D honeycomb capillary network.
    INPUT: sideLength: The side length of a hexagon of the honeycomb network.
           size: The size of the network as multiples of hexagons
    OUTPUT: Honeycomb network as VascularGraph
    """
    ff = np.cos(np.deg2rad(30.))
    G = honeycomb((0, size*2*ff*sideLength), 
                  (0, size*2*ff*sideLength), 
                  (0,0), sideLength)
    maxima = np.max(G.vs['r'], axis=0)
    G.delete_vertices([v.index for v in G.vs if v['r'][1] == maxima[1]])
    minima = np.min(G.vs['r'], axis=0)
    maxima = np.max(G.vs['r'], axis=0)
    blacklist = [v.index for v in G.vs if np.allclose(v['r'][1], minima[1])]
    blacklist.extend([v.index for v in G.vs 
                      if np.allclose(v['r'][1], maxima[1])])
    tmplist = sorted([(v['r'][1], v.index) for v in G.vs 
                      if np.allclose(v['r'][0], minima[0])])
    tmplist = [x[1] for x in tmplist]
    blacklist.extend(np.setdiff1d(tmplist, tmplist[2::3]).tolist())
    tmplist = sorted([(v['r'][1], v.index) for v in G.vs 
                      if np.allclose(v['r'][0], maxima[0])])
    tmplist = [x[1] for x in tmplist]
    blacklist.extend(np.setdiff1d(tmplist, tmplist[2::3]).tolist())

    G.delete_order_two_vertices(blacklist=blacklist)
    vgm.write_vtp(G, 'G1.vtp', False)
    
    return G
    
    
def perturb_network_v2(G, meanL, stdL):
    """Perturbs the vertex coordinates and replaces the straight connections
    by splines, such that the resulting network closely matches the desired
    statistics.
    INPUT: G: VascularGraph to be perturbed.
           meanL: The mean of the desired length distribution.
           stdL: The standard deviation of the desired length distribution.
    OUTPUT: None, G is modified in-place.       
    """
    # TODO: either complete or delete this function.
    newL = np.random.normal(meanL, stdL, [G.ecount()])
    

def randomized_honeycomb(Gxtm, xlim=(0, 100), ylim=(0, 100), zlim=(0, 50)):
    """Creates a randomized honeycomb capillary network.  Initially, a regular
    honeycomb network is created, with a linear distance between vertices
    corresponding to the mean linear vertex distance of the provided srXTM
    dataset.  Then, the vertex locations are randomly perturbed, conserving the
    mean of the linear vertex distance. Finally, tortuous lengths and diameters
    are assigned based on the statistics of the srXTM dataset.
    INPUT: Gxtm: VascularGraph obtained from srXTM data.
           xlim: Limits of the network in x-direction.
           ylim: Limits of the network in y-direction.
           zlim: Limits of the network in z-direction.
    OUTPUT: Hc: VascularGraph of randomized 3D-honeycomb structure.       
    """
    
    linDistances = [np.linalg.norm(Gxtm.vs[e.source]['r'] - Gxtm.vs[e.target]['r']) for e in Gxtm.es]
    empiricalShrinkageFactor = np.round(45.47 / 26.53)
    linDistance = np.mean(linDistances) * empiricalShrinkageFactor 
    Hc = honeycomb(xlim, ylim, zlim, linDistance)

    print('srXTM mean linear distance: %.2f\n' % (linDistance / empiricalShrinkageFactor))
    lPerturb = linDistance * 0.5
    Hc.vs['r'] = Hc.vs['r'] + np.random.rand(Hc.vcount(), 3) * lPerturb - lPerturb / 2.
    linDistances = [np.linalg.norm(Hc.vs[e.source]['r'] - Hc.vs[e.target]['r']) for e in Hc.es]
    print('honeycomb mean linear distance: %.2f\n' % (np.mean(linDistances)))
    
    ld = sorted(zip(Gxtm.es(diameter_le=7)['length'], Gxtm.es(diameter_le=7)['diameter']))
    l = np.array(ld)[:,0]
    llower = np.searchsorted(l, linDistances)
    #random choices where the choice is larger than x. And x is the index of the length which is the corresponding in ld
    randChoices = np.array([np.random.randint(x, len(ld)) for x in llower])
    print('len rand choices: %i, ecount: %i\n' % (len(randChoices), Hc.ecount()))
    ld = np.array(ld)[randChoices]
    Hc.es['length'] = ld[:,0]
    Hc.es['diameter'] = ld[:,1]
    
    return Hc

