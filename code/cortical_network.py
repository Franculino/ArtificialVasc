from __future__ import division, print_function, with_statement
import numpy as np
import scipy as sp
from scipy import finfo
from scipy.spatial import kdtree
import os
import sys
from sys import stdout

from vascular_tree_db import extend_from_db
import capillary_bed
import implant_srxtm
import vgm
import control
import time


def construct_pial_network(sf=5, epperc=15, epmode='distributed'):
    """Builds a VascularGraph representing the pial network.
    INPUT: sf: Scaling Factor by which to multiply vertex positions and
               diameters (e.g. to convert units of pixels to microns).
           epperc: Percentage of offshoots that should be en-passent. These are
                   either chosen randomly or such, that the penetrating vessels
                   are distributed as homogeneously as possible.
           epmode: Mode by which en-passent vessels are chosen. This can be
                   either 'random' or 'distributed'
    OUTPUT: G: VascularGraph.
    Note that this function is specifically tuned to the csv files and would
    need to change if the csv files change. Moreover, it relies on input from 
    control.py
    """
    # Get settings from control file:
    basedir = control.basedir
    pialDB = control.pialDB
    av = control.av
    vv = control.vv

    # Read csv files and scale from mm to micron:
    G = vgm.read_csv(os.path.join(pialDB, 'vertices.csv'),
                     os.path.join(pialDB, 'edges.csv'))
    del G.vs['nkind']
    G.vs['r'] = [r * sf for r in G.vs['r']]
    G.es['diameter'] = [d * sf for d in G.es['diameter']]
    G.es['length'] = [np.linalg.norm(G.vs[e.source]['r'] - 
                                     G.vs[e.target]['r'])
                      for e in G.es]
    G.vs['isAv'] = [0] * G.vcount()
    G.vs['isVv'] = [0] * G.vcount()
    # add 'av' and 'vv' from control dict
    G.vs(av)['isAv'] = [1] * len(av)
    G.vs(vv)['isVv'] = [1] * len(vv)

    G.delete_selfloops()
    G.add_points(100)
    G.delete_order_two_vertices()
    minnp = 3

    if epmode == 'distributed':
        # Find arterial and venous components: 
        #Find clusters = group of connected nodes
        co = G.components(mode='weak')
        #all vertices belonging to arterlial network and venous network, respectively
        aComponent = []
        vComponent = []
        for c in co:
            #everything which is not in av => vComponent
            if any(map(lambda x: x in c, av)):
                aComponent.extend(c)
            else:
                vComponent.extend(c)

        G.vs['degree'] = G.degree()
        G.es['nPoints'] = [len(x) for x in G.es['points']]
        G.es['midpoint'] = [np.mean(G.vs[e.tuple]['r'], axis=0) for e in G.es]
        
        epVertices = []
        for component, avvv in zip((aComponent, vComponent), (av, vv)):
            #All edges belong to the component
            edges = G.get_vertex_edges(G.vs(component).indices, 'all', True)
	    #edges where nPoints >= minnp
            edges = G.es(edges, nPoints_ge=minnp).indices
            #vertices of the component with only one adjacent edge
            dovertices = G.vs(component, degree_eq=1).indices
            # Compute number of en-passent offshoots:
            nep = int(epperc / (100 - epperc) * (len(dovertices) - len(avvv)))
            rlist = G.vs(dovertices)['r']
            epEdges = []
            for i in xrange(nep):
                Kdt = kdtree.KDTree(rlist, leafsize=10)
                distances = [(Kdt.query(G.es[e]['midpoint'])[0], e) for e in edges]
		#edge which is farest away
                epEdge = sorted(distances, reverse=True)[0][1]
                edges.remove(epEdge)
                epEdges.append(epEdge)
                rlist.append(G.es[epEdge]['midpoint'])
            newVertices = range(G.vcount(), G.vcount()+len(epEdges))
            component.extend(newVertices)
            epVertices.extend(newVertices)
            for edge in epEdges:
                G.split_edge(edge, int(G.es[edge]['nPoints']/2.), False)
            G.delete_edges(epEdges)
	    #Prevent new vertices from belonging to in and outlet vertices
            G.vs(newVertices)['isVv']=[0]*len(newVertices)
            G.vs(newVertices)['isAv']=[0]*len(newVertices)
        del G.es['midpoint']

    elif epmode == 'random':
        # Compute number of en-passent offshoots:
        nep = epperc / (1 - epperc) * \
              (len(np.nonzero(np.array(G.degree()) == 1)[0]) - len(av) - len(vv))
        G.es['nPoints'] = [len(x) for x in G.es['points']]
        epEdges = np.random.permutation(G.es(nPoints_gt=minnp).indices)[:nep].tolist()
        epVertices = range(G.vcount(), G.vcount()+len(epEdges))
        for edge in epEdges:
            G.split_edge(edge, int(G.es[edge]['nPoints']/2.), False)
        G.delete_edges(epEdges)

        # Find arterial and venous components:
        co = G.components(mode='weak')
        aComponent = []
        vComponent = []
        for c in co:
            if any(map(lambda x: x in c, av)):
                aComponent.extend(c)
            else:
                vComponent.extend(c)

    G['av'] = G.vs(isAv_eq=1).indices
    G['vv'] = G.vs(isVv_eq=1).indices
    del G.vs['isAv']
    del G.vs['isVv']
    del G.es['nPoints']
    #all penetrating vessels
    pv = G.vs(_degree_eq=1).indices
    #if in or outflow -> remove from penetrating vessels
    for prop in ['av', 'vv']:
        for x in G[prop]:
            try:
                pv.remove(x)
            except:
                pass
    #add en passent penetrating vessels
    pv.extend(epVertices)
    G.vs['degree']=G.degree()
    remove=[]
    for i in pv:
        if G.vs['degree'][i] > 2:
            remove.append(i)

    print('Need to be removed')
    print(remove)
    for i in remove:
        pv.remove(i)

    #List of penetratiting arteries (apv) and venules (vpv)
    G['apv'] = [v for v in pv if v in aComponent]
    G['vpv'] = [v for v in pv if v not in aComponent]
    G.vs['degree'] = G.degree()

    #vertex characteristic pial arterty or pial venule
    G.vs[aComponent]['kind'] = ['pa' for v in aComponent]
    G.vs[vComponent]['kind'] = ['pv' for v in vComponent]
    G.vs['nkind'] = [0] * G.vcount()
    #0 = pial artery, 1 = pial venule, 2 = en passent penetrating
    G.vs[aComponent]['nkind'] = [0 for v in aComponent]
    G.vs[vComponent]['nkind'] = [1 for v in vComponent]
    G.vs[epVertices]['nkind'] = [2 for v in epVertices]
    
    # Set in- and outflow pressure boundary conditions.
    # Arterial pressure 60 mmHg, venous pressure 10 mmHg, according to a
    # Hypertesion paper by Harper and Bohlen 1984.
    G.vs[G['av'][0]]['pBC'] = 60 * vgm.scaling_factor_du('mmHg', G['defaultUnits'])
    for v in G['vv']:
        G.vs[v]['pBC'] = 10 * vgm.scaling_factor_du('mmHg', G['defaultUnits'])

    vgm.write_pkl(G,'pial.pkl')
    vgm.write_vtp(G,'pial.vtp',False)
    return G
    
    
def construct_cortical_network(G, gXtm, originXtm=None, insertXtm=False, 
                               invivo=True,BC=[60,10,0.2],species='rat',cb='standard'):
    """Builds a VascularGraph that includes the pial vessels, penetrating
    arterioles and draining veins, as well as the capillary bed.
    INPUT: G: VascularGraph of the pial network. This should be created using
              the function construct_pial_network().
           gXTM: VascularGraph of srXTM data. This is implanted at location
                 originXtm of the cortical network.
           originXtm: The origin (x,y) of the cylindrical srXTM in the cortical
                      network, i.e. the location of its rotational axis. This
                      will be the network's center of mass, if not provided.
           sf: Scaling Factor by which to multiply vertex positions and
               diameters (e.g. to convert units of pixels to microns).
           epperc: Percentage of offshoots that should be en-passent. These are
                   chosen randomly.
           BC: list with pBC at inlet, pBC at outlet, inflow tube hematocrit 
           species: species that should be used for the database of the penetrating vessels
           cb: type of capillarybed: standard, randomized
    OUTPUT: G: VascularGraph.
    Note that this function relies on input from control.py
    """
    
    # Get settings from control file:
    basedir = control.basedir
    treeDB = control.treeDB

    eps = finfo(float).eps * 1e4

    # Add offshoots and remove parts with z<0:
    print('Adding offshoots (%i arterial and %i venous)...' % 
          (len(G['apv']), len(G['vpv'])))
    t0 = time.time()
    vcount = G.vcount()
    G.vs['apv']=[0]*G.vcount()
    G.vs['vpv']=[0]*G.vcount()
    for i in G['apv']:
        G.vs[i]['apv']=1
    for i in G['vpv']:
        G.vs[i]['vpv']=1

    G.vs['x'] = [r[0] for r in G.vs['r']]
    G.vs['y'] = [r[1] for r in G.vs['r']]

    xMin=np.min(G.vs['x'])
    xMax=np.max(G.vs['x'])
    yMin=np.min(G.vs['y'])
    yMax=np.max(G.vs['y'])

    stdout.flush()

    #add arterial penetratiing trees
    G.vs['degree'] = G.degree()
    print('Number of Components: Only pial')
    print(len(G.components()))
    print('Should be KEPT!')
    components=len(G.components())
    if species == 'rat':
        extend_from_db(G, G['apv'], os.path.join(treeDB, 'arteryDB'))
    elif species == 'mouse':
        extend_from_db(G, G['apv'], os.path.join(treeDB, 'arteryDBmouse'))
    else:
        print('Species not available')
    print('Number of Components: Artery trees added')
    print(len(G.components()))
    if len(G.components()) > components:
        print('ERROR 1')
        print('More components than expected')
        vgm.write_pkl(G,'ERROR_Output.pkl')
    #Assign kind
    G.vs[range(vcount, G.vcount())]['kind'] = ['a' for v in xrange(vcount, G.vcount())]            
    for i in G['apv']:
        if 'a' not in G.vs[G.neighbors(i)]['kind']:
            print('ERROR in adding penetrating arterioles')
            print(i)
    G.vs['degree']=G.degree()
    vcount = G.vcount()
    #add venous penetrating trees
    if species == 'rat':
        extend_from_db(G, G['vpv'], os.path.join(treeDB, 'veinDB'))
    elif species == 'mouse':
        extend_from_db(G, G['vpv'], os.path.join(treeDB, 'veinDBmouse'))
    else:
        print('Species not available')
    print('Number of Components: Vein trees added')
    print(len(G.components()))
    if len(G.components()) > components:
        print('ERROR 2')
        print('More components than expected')
        vgm.write_pkl(G,'ERROR_Output.pkl')
    #Assign kind
    G.vs[range(vcount, G.vcount())]['kind'] = ['v' for v in xrange(vcount, G.vcount())]           
    G.vs['degree']=G.degree()
    for i in G['vpv']:
        if 'v' not in G.vs[G.neighbors(i)]['kind']:
            print('ERROR in adding ascending venules')
            print(i)
    #adding trees lead to some new dead ends at the z=0 level, in contrast to the penetrating vessels their kind 
    #is either 'a' or 'v' but not 'pa' and not 'pv'
    #Remove capillaries (d < 7 mum) in pial vessels and penetrating vessels
    print(str(len(G.es(diameter_le=7)))+' Edges have a diameter which is smaller than 7mum and are therefor removed')
    G.delete_edges(G.es(diameter_le=7))
    G.vs['degree']=G.degree()
    print('Number of Components: Capillaries deleted')
    print(len(G.components())) 
    if len(G.components()) > components:
        print('ERROR')
        print('More components than expected')
        print(components)
        print(len(G.components()))
        for i in len(G.components()):
            print(len(G.components()[i]))
    #Remove vertices where z < 0 (would be abolve pial surface)
    G.vs['z'] = [r[2] for r in G.vs['r']]
    deleteList=G.vs(z_lt=-eps).indices
    print('Number of vertices where z<0')
    print(len(deleteList))
    deleteList.sort(reverse=True)
    for i in range(len(deleteList)):
        G.delete_vertices(deleteList[i])
    print('Number of components: z lt 0 deleted')
    print(len(G.components()))
    if len(G.components()) > components:
        print('ERROR 3')
        print('More components than expected')
        vgm.write_pkl(G,'ERROR_Output.pkl')
    G.vs['degree']=G.degree()
    G.delete_vertices(G.vs(degree_eq=0))
    print('Number of components: degree 0 deleted')
    print(len(G.components()))
    if len(G.components()) > components:
        print('ERROR')
        print('More components than expected')
    #Delete len=0 edges
    G.es['length'] = [np.linalg.norm(G.vs[e.source]['r'] - G.vs[e.target]['r']) for e in G.es]
    G.delete_edges(G.es(length_eq=0).indices)
    print('...done. Time taken: %.1f min' % ((time.time()-t0)/60.))
    G.vs['degree']=G.degree()

    #Assign nkind to vertex
    for i in range(G.vcount()):
        if G.vs['kind'][i]=='pa':
            G.vs[i]['nkind']=0
        elif G.vs['kind'][i]=='a':
            G.vs[i]['nkind']=2
        elif G.vs['kind'][i]=='pv':
            G.vs[i]['nkind']=1
        elif G.vs['kind'][i]=='v':
            G.vs[i]['nkind']=3

    G.vs['x'] = [r[0] for r in G.vs['r']]
    G.vs['y'] = [r[1] for r in G.vs['r']]
    print('Number of components before deleting trees, that extend the pial structure')
    print(len(G.components()))

    G.delete_vertices(G.vs(kind_eq='a',x_lt=xMin).indices)
    G.delete_vertices(G.vs(kind_eq='v',x_lt=xMin).indices)
    G.delete_vertices(G.vs(kind_eq='a',x_gt=xMax).indices)
    G.delete_vertices(G.vs(kind_eq='v',x_gt=xMax).indices)
    G.delete_vertices(G.vs(kind_eq='a',y_lt=yMin).indices)
    G.delete_vertices(G.vs(kind_eq='v',y_lt=yMin).indices)
    G.delete_vertices(G.vs(kind_eq='a',y_gt=yMax).indices)
    G.delete_vertices(G.vs(kind_eq='v',y_gt=yMax).indices)

    print('Number of components after deleting trees, that extend the pial structure')
    print(len(G.components()))
    while len(G.components()) > components:
        print(len(G.components()))
        G.delete_vertices(G.components()[len(G.components())-1])

    print('Correct number of components')
    print(len(G.components()))

    vgm.write_pkl(G, 'stage1.pkl')
    vgm.write_vtp(G, 'stage1.vtp',False)
    print('number of capillaries')
    print(len(G.vs(kind_eq='c')))
    stdout.flush()

    # Add capillary bed as honeycomb network:
    # Add volume and conductance before adding capillary grid
    print('Adding capillary grid...')
    numNCVertices = G.vcount()
    vcount=G.vcount()
    ecount=G.ecount()
    t0 = time.time()
    rMin = np.min(G.vs['r'], axis=0)
    rMax = np.max((np.max(G.vs['r'], axis=0), 
                   (np.max(gXtm.vs['r'], axis=0)-np.min(gXtm.vs['r'], axis=0))), axis=0)

    print('Dimensions of capillary bed')
    print(rMin)
    print(rMax)
    if cb == 'standard':
        #honeycomb = capillary_bed.honeycomb((rMin[0], rMax[0]), (rMin[1], rMax[1]), (rMin[2], 1.1*rMax[2]),65)
        honeycomb = capillary_bed.honeycomb((rMin[0], rMax[0]), (rMin[1], rMax[1]), (rMin[2], 1.1*rMax[2]),32)
    elif cb == 'randomized':
        honeycomb = capillary_bed.randomized_honeycomb(gXtm, (rMin[0], rMax[0]), (rMin[1], rMax[1]), (rMin[2], 1.1*rMax[2]))
    else:
        print('Type of capillary bed is not specified')

    print('CHECK DEGREE HONEYCOMB')
    print(max(honeycomb.degree()))
    honeycomb.vs['degree']=honeycomb.degree()
    print('Degree 1 vertices in honeycomb')
    print(len(honeycomb.vs(degree_eq=1)))
    #Honeycomb network is moved such that it finishes at the same level as the pial+penetrating-network
    rMinHC= np.min(honeycomb.vs['r'], axis=0)
    offset=np.array([0.,0.,rMinHC[2]*(-1)])
    print('')
    print('Honeycomb Network is shifted in z-direction by')
    print(offset)
    print('such that zmin of Honeycomb Network = 0')
    vgm.shift(honeycomb,offset
)
    #honeycomb network and pial+penetrating network are put together
    G.disjoint_union_attribute_preserving(honeycomb, 'omit', False)
    G.vs[range(vcount, G.vcount())]['kind'] = ['c' for v in xrange(vcount, G.vcount())]
    G.vs[range(vcount, G.vcount())]['nkind'] = [4 for v in xrange(vcount, G.vcount())]
    G.vs['capGrid']=np.zeros(G.vcount())
    G.vs[range(vcount,G.vcount())]['capGrid'] = [1]*(G.vcount()-vcount)
    G.es['capGrid']=np.zeros(G.ecount())
    G.es[range(ecount,G.ecount())]['capGrid'] = [1]*(G.ecount()-ecount)
    vcount = G.vcount()
    vgm.write_pkl(G, 'stage2.pkl')
    stdout.flush()

    # Connect offshoots to capillary grid to penetrating vessels:
    print('Connecting offshoots to capillary grid...')
    t0 = time.time()
    G.vs['degree'] = G.degree()
    G.vs['z'] = [r[2] for r in G.vs['r']]
    #cv vertices of penetrating trees with dead end --> to be connected to capillary bed
    cv = G.vs(z_ge=0.0-eps, degree_eq=1,capGrid_eq=0).indices
    #In and outlet of pial vessels should be preserved
    for i in G['vv']:
        if i in cv:
            cv.remove(i)
        else:
            print(G.vs[i]['r'])
    for i in G['av']:
        if i in cv:
            cv.remove(i)
        else:
            print(G.vs[i]['r'])

    #Capillary bed should not be attached to pial vessels
    for i in cv:
        if G.vs[i]['kind'] == 'pa':
            print('ERROR pial artery is to be connected to the capillary bed')
        if G.vs[i]['kind'] == 'pv':
            print('ERROR pial vein is to be connected to the capillary bed')

    Kdt = kdtree.KDTree(honeycomb.vs['r'], leafsize=10)
    print('Number of Dead Ends of pentrating vessels to be connected to the capillary bed')
    print(len(cv))
    stdout.flush()
    #Compute possible Connections for vertices. Maximum degree = 4
    posConnections=[]
    G.vs['degree']=G.degree()
    G.vs['posConnections']=[3]*G.vcount()
    deg4=G.vs(degree_ge=4).indices
    G.vs[deg4]['posConnections']=[0]*len(deg4)
    deg3=G.vs(degree_eq=3).indices
    G.vs[deg3]['posConnections']=[1]*len(deg3)
    deg2=G.vs(degree_eq=2).indices
    G.vs[deg2]['posConnections']=[2]*len(deg2)
    print('posConnections assigned')
    stdout.flush()
    print('Total number of possible connections')
    posConnections=np.sum(G.vs[numNCVertices:G.vcount()-1]['posConnections'])
    print(posConnections)
    stdout.flush()
    #Start connection process
    for v in cv:
        diameter = G.es[G.adjacent(v)[0]]['diameter']
        newVertexFound=0
        count= 1
        while newVertexFound != 1:
            #start with 5 possible nearest neighbors
            nearestN=Kdt.query(G.vs[v]['r'],k=10*count)
            for i in range((count-1)*10,count*10):
                newVertex=int(nearestN[1][i])
                if G.vs['posConnections'][newVertex+numNCVertices]  == 0:
                    print('No connection possible')
                else:
                   G.vs[newVertex+numNCVertices]['posConnections'] = int(np.round(G.vs[newVertex+numNCVertices]['posConnections'] - 1))
                   newVertexFound = 1
                   break
            count += 1
        stdout.flush()
        nn = newVertex
        #Distance between analyzed endppoint and closest point in honeycomb network (a minimum length of 4 micron is preserved) 
        length = max(4.,np.linalg.norm(G.vs[v]['r'] - G.vs[numNCVertices + nn]['r']))
	#add edge between the points
        G.add_edges((v, numNCVertices + nn))
        G.es[G.ecount()-1]['diameter'] = diameter
        G.es[G.ecount()-1]['length'] = length
    print('...done. Time taken: %.1f min' % ((time.time()-t0)/60.))
    G.vs['degree'] = G.degree()
    if max(G.degree()) > 4:
        print('ERROR Maximum degree > 4 after connecting penetrating vessels to the capillary bed')
        print(max(G.degree()))

    # Remove degree 1 vertices:
    # (Deg1 vertices of artificial capillary bed, should be located at the borders of the artificial capillary bed)
    print('Number of degree 1 vertices')
    G.vs['degree']=G.degree()
    print(len(G.vs(degree_eq=1)))
    deg1=G.vs(degree_eq=1).indices
    G.vs['av']=[0]*G.vcount()
    G.vs['vv']=[0]*G.vcount()
    for i in G['av']:
        G.vs[i]['av']=1
        deg1.remove(i)
    for i in G['vv']:
        G.vs[i]['vv']=1
        deg1.remove(i)
    while len(deg1) > len(G['av'])+len(G['vv']):
        G.delete_vertices(deg1)
        G.vs['degree']=G.degree()
        deg1=G.vs(degree_eq=1).indices
        for i in G['av']:
            deg1.remove(i)
        for i in G['vv']:
            deg1.remove(i)
    G['av']=G.vs(av_eq=1).indices 
    G['vv']=G.vs(vv_eq=1).indices 
    print('Number of degree 1 vertices')
    G.vs['degree']=G.degree()
    print(len(G.vs(degree_eq=1)))
    deg1=G.vs(degree_eq=1).indices

    vgm.write_vtp(G, 'stage3.vtp', False)
    vgm.write_pkl(G, 'stage3.pkl')
    stdout.flush()

    # Set conductance of all vessels:
    print('Adding conductance...')
    t0 = time.time()
    aindices = G.vs(kind='pa').indices
    aindices.extend(G.vs(kind='a').indices)
    vindices = G.vs(kind='pv').indices
    vindices.extend(G.vs(kind='v').indices)
    cindices = G.vs(kind='c').indices
    vgm.add_conductance(G, 'a', invivo,edges=G.get_vertex_edges(aindices))
    vgm.add_conductance(G, 'v', invivo,edges=G.get_vertex_edges(vindices))
    vgm.add_conductance(G, 'a', invivo,edges=G.get_vertex_edges(cindices))
    print('...done. Time taken: %.1f min' % ((time.time()-t0)/60.))
    stdout.flush()

    # Insert srXTM sample at originXtm:
    print('Embedding SRXTM...')
    t0 = time.time()
    if insertXtm:
        G = implant_srxtm.implant_srxtm(G, gXtm, originXtm)    
    print('...done. Time taken: %.1f min' % ((time.time()-t0)/60.))
    G['av']=G.vs(av_eq=1).indices
    G['vv']=G.vs(vv_eq=1).indices
    vgm.write_vtp(G, 'stage4.vtp', False)
    vgm.write_pkl(G, 'stage4.pkl')
    print('stage4 written')
    stdout.flush()
    
    # Delete obsolete graph properties:
    for vProperty in ['z', 'degree', 'nkind', 'epID', 'maxD']:
        if vProperty in G.vs.attribute_names():
            del G.vs[vProperty]
    for eProperty in ['diameter_orig', 'diameter_change', 'cost','depth']:
        if eProperty in G.es.attribute_names():
            del G.es[eProperty]
    for gPropery in ['attachmentVertex', 'sampleName', 'distanceToBorder',
                     'avZOffset']:
        if gPropery in G.attributes():
            del G[gPropery]
    
    # Add a numerical version of vessel kind to facilitate paraview display:
    nkind = {'pa':0, 'pv':1, 'a':2, 'v':3, 'c':4, 'n':5}
    for v in G.vs:
        v['nkind'] = nkind[v['kind']]
    G.vs['degree']=G.degree()
    print('CHECK DEGREE 7')
    print(max(G.vs['degree']))

    #1. If effective Diameter exists it is assigned as diameter
    if 'effDiam' in G.es.attribute_names():
        diamEff=G.es(effDiam_ne=None).indices
        print('effective diameter available')
        print(len(diamEff))
        for i in diamEff:
            G.es[i]['diameter']=G.es[i]['effDiam']

    #2. check if there are degree 0 vertices 
    G.vs['degree']=G.degree()
    deg0=G.vs(degree_eq=0).indices
    print('Degree 0 vertices?')
    print(len(deg0))
    G.delete_vertices(deg0)
    G.vs['degree']=G.degree()

    #Reassign av and vv and apv and vpv
    G['av']=G.vs(av_eq=1).indices
    G['vv']=G.vs(vv_eq=1).indices
    G['apv']=G.vs(apv_eq=1).indices
    G['vpv']=G.vs(vpv_eq=1).indices

    #Eliminate small diameters in capGrid
    diam0capGrid=G.es(diameter_lt=1.0,capGrid_eq=1.).indices
    G.es[diam0capGrid]['diameter']=[1]*len(diam0capGrid)

    #Recheck smalles diameter
    diam0=G.es(diameter_lt=1.0).indices
    if len(diam0) > 0:
        print('ERROR no small diameters should exist')

    #Recheck for loops at deadEnd vertices
    print('check for loops')
    G.es['length2'] = [np.linalg.norm(G.vs[e.source]['r'] -G.vs[e.target]['r']) for e in G.es]
    len0=G.es(length2_eq=0).indices
    print('Len0 Edges are deleted')
    print(len(len0))
    G['infoDeadEndLoops']=len(len0)
    for i in len0:
        e=G.es[i]
        if e.tuple[0] != e.tuple[1]:
            print('WARNING')
    
    G.delete_edges(len0)
    G.vs['degree']=G.degree()
    del(G.es['length2'])

    #Delete new deg1s
    print('Deleting the loops can lead to more degree 1 vertices')
    G.vs['degree']=G.degree()
    deg1=G.vs(degree_eq=1).indices
    print(len(deg1))
    while len(deg1) > len(G['av'])+len(G['vv']):
        print('')
        print(len(deg1))
        av=G.vs(av_eq=1).indices
        vv=G.vs(vv_eq=1).indices
        for i in av:
            if i in deg1:
                deg1.remove(i)
        print(len(deg1))
        for i in vv:
            if i in deg1:
                deg1.remove(i)
        print(len(deg1))
        G.delete_vertices(deg1)
        G.vs['degree']=G.degree()
        deg1=G.vs(degree_eq=1).indices
        stdout.flush()
    
    print('All newly created Dead Ends have ben elimanted')
    G.vs['degree']=G.degree()
    print(len(G.vs(degree_eq=1)))


    #Recheck the degrees of the graph
    G.vs['degree']=G.degree()
    deg1=G.vs(degree_eq=1).indices
    print('Degree 1 vertices')
    print(len(deg1))
    for i in deg1:
        if i not in G['av'] and i not in G['vv']:
            print('ERROR deg1 vertex not in av and neither in vv')

    #Recheck max and min degree
    print('min degree')
    print(min(G.degree()))
    if min(G.degree()) < 1:
        print('ERROR in min degree')
    if np.max(G.vs['degree']) > 4:
        print('ERROR in Maximum degree')
        print(np.max(G.vs['degree']))

    #Reassign av and vv and apv and vpv
    G['av']=G.vs(av_eq=1).indices
    G['vv']=G.vs(vv_eq=1).indices
    G['apv']=G.vs(apv_eq=1).indices
    G['vpv']=G.vs(vpv_eq=1).indices

    # 6. Diameter Srxtm vessels is adjusted such, that at least one RBC fits in
    #    every vessel
    if 'isSrxtm' in G.es.attribute_names():
        P=vgm.Physiology()
        vrbc=P.rbc_volume('mouse')
        G.es['minDist'] = [vrbc / (np.pi * e['diameter']**2 / 4) for e in G.es]
        maxMinDist=max(G.es['minDist'])
        countDiamChanged=0
        diamNew=[]
        edge=[]
        probEdges=G.es(length_lt=1*maxMinDist,isSrxtm_eq=1).indices
        count=0
        for i in probEdges:
            if 1*G.es['minDist'][i] > G.es['length'][i]:
                G.es[i]['diameter']=np.ceil(100*np.sqrt(4*2*vrbc/(np.pi*G.es['length'][i])))/100.
                countDiamChanged += 1
                diamNew.append(np.ceil(100*np.sqrt(4*2*vrbc/(np.pi*G.es['length'][i])))/100.)
                edge.append(i)
            count += 1
    
        print('vessel volume has been increased in')
        print(countDiamChanged)
        G.es['minDist'] = [vrbc / (np.pi * e['diameter']**2 / 4) for e in G.es]
        stdout.flush()
    
    # 7. Length is added to edges which do not have a length yet
    noLength=G.es(length_eq=None).indices
    print('Edges with no length')
    print(len(noLength))
    for i in noLength:
       e=G.es[i]
       G.es[i]['length']=np.linalg.norm(G.vs[e.source]['r']-G.vs[e.target]['r'])
    
    noLength=G.es(length_eq=None).indices
    print('Edges with no length')
    print(len(noLength))

    # 3. Assign pressure BCs
    print('Assign pressure BCs')
    for i in G['av']:
        G.vs[i]['pBC']=BC[0]
        print(G.vs[i]['pBC'])
    
    for i in G['vv']:
        G.vs[i]['pBC']=BC[1]
        print(G.vs[i]['pBC'])
    
    print('Pressure BCs assigned')

    # 4. Assign tube hematocrit boundary conditions
    print('assign inlet tube hematocrits')
    for i in G['av']:
        G.es[G.adjacent(i)[0]]['httBC']=BC[2]

    #Recheck BCs (pBC,rBC, httBC)
    for i in G['av']:
        print('')
        print(G.vs['pBC'][i])
        print(G.vs['rBC'][i])
        print(G.es[G.adjacent(i)]['httBC'])
        if G.vs['pBC'][i] == None and G.vs['rBC'][i] == None:
            print('ERROR in av boundary condition')
        if G.es[G.adjacent(i)]['httBC'] == None:
            print('ERROR in av boundary condition: httBC')
    
    for i in G['vv']:
        print('')
        print(G.vs['pBC'][i])
        print(G.vs['rBC'][i])
        if G.vs['pBC'][i] == None and G.vs['rBC'][i] == None:
            print('ERROR in vv boundary condition')
    
    stdout.flush()

    G.vs[G.vs(apv_eq=None).indices]['apv']=[0]*len(G.vs(apv_eq=None))
    G.vs[G.vs(vpv_eq=None).indices]['vpv']=[0]*len(G.vs(vpv_eq=None))

    vgm.write_pkl(G, 'stage5.pkl')
    vgm.write_vtp(G, 'stage5.vtp',False)
    stdout.flush()
    return G
