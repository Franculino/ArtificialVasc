from __future__ import division
import glob
import numpy as np
import os
from scipy.spatial import kdtree
from copy import deepcopy
import os
import sys
import vgm
import multiprocessing as mp
import control
from sys import stdout

basedir = control.basedir
pjoin = os.path.join
sys.path.append(pjoin(basedir, 'Programming/Python/preprocessing/code'))
sys.path.append(pjoin(basedir, 'Programming/Python/'))
from vesselProperties import strahler    

def generate_db():
    """Generates a database of vascular trees, both arterial and venous.
    INPUT: None
    OUTPUT: Arterial and venous tree database written to disk.
    Note that this function uses hardcoded directories and filenames!
    """

    ### START INPUT -------------------------------------------------------
    dataDir = pjoin(basedir, 'Data/2010/srXTM_processed')
    outDir = pjoin(basedir, 'Data/2010/vascularTrees')
    vtpDir = pjoin(basedir, 'Data/2010/vascularTrees/vtp')
    amDir = pjoin(basedir, 'Data/2010/vascularTrees/am')
    fileEnding = '_rr.pkl'
    parallel = False
    ### END INPUT ---------------------------------------------------------
    
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    sampleDirs = glob.glob1(dataDir, '*')
    sampleDirs = [sd for sd in sampleDirs if os.path.isdir(os.path.join(dataDir, sd))]
    
    Processes = []
    for sd in sampleDirs:
        print(sd)
        G = vgm.read_pkl(os.path.join(dataDir, sd, sd+fileEnding))
        vertices = []
        if ('av' in G.attributes()) and ('vv' in G.attributes()):
            vertices.append(G['av'])
            vertices.append(G['vv'])
        else:
            for vtype in ['arteries', 'veins']:
                coords = vgm.read_landmarks(os.path.join(dataDir, sd, vtype+'_'+sd))
                vs = vgm.vertices_from_coordinates(G, coords)
                if max(vs[1]) > 0.0:
                    print('WARNING: landmark file %s faulty!' % vtype+'_'+sd)
                vertices.append(vs[0])
        if parallel:       
            P1 = mp.Process(target=add_to_db, args=(G, vertices[0], 7., os.path.join(outDir, 'arteryDB')), 
                            kwargs={'sampleName':sd, 'vtpDir':vtpDir+'_artery','amDir':amDir+'_artery'})
            P2 = mp.Process(target=add_to_db, args=(G, vertices[1], 7., os.path.join(outDir, 'veinDB')), 
                            kwargs={'sampleName':sd, 'vtpDir':vtpDir+'_vein','amDir':amDir+'_vein'})
            P1.start()
            P2.start()
            P1.join()
            P2.join()
        else:                         
            add_to_db(G, vertices[0], 7., os.path.join(outDir, 'arteryDB'),
                      sampleName=sd, vtpDir=vtpDir+'_artery',amDir=amDir+'_artery')
            add_to_db(G, vertices[1], 7., os.path.join(outDir, 'veinDB'),
                      sampleName=sd, vtpDir=vtpDir+'_vein', amDir=amDir+'_vein')


def vascular_tree_subgraph(G, vertex, dThreshold, 
                           enforceSingleRoot=True, nce=None, **kwargs):
    """Extracts a tree subgraph from a VascularGraph, given a root vertex and a
    threshold diameter that determines the non-capillary / capillary border.
    INPUT: G: VascularGraph.
           vertex: The root vertex at which the tree extraction is started.
           dThreshold: The diameter below which vessels are considered as
                       capillaries.
           enforceSingleRoot: Should other roots be deleted if they exist?
                              (Boolean.)
           nce: Indices of non capillary endpoints, i.e. other roots. If not
                provided, the function will try to find them.
           kwargs**:
           The keyword parameters can be used to refine the search for other
           roots.    
           rDegree: The maximum degree of a root. 
           rDiameter: The minimum diameter of a root.
           rZ: The maximal cortical depth of a root.
    OUTPUT: Vascular tree subgraph, holding the additional vertex properties
            z (cortical depth), diameter (maximum diameter of adjacent edges),
            epID (0 if not an endpoint, 1 if regular endpoint, 2 if 
            anastomosis)
    """
    sc = G.get_tree_subcomponent(vertex, dThreshold)
    vascularTree = G.subgraph(sc)
    vascularTree.delete_edges(vascularTree.es(diameter_le=dThreshold).indices)

    if enforceSingleRoot and len(sc) > 1: # more than 1 vertex
        vascularTree.vs['epID'] = [0 if len(vascularTree.neighbors(v)) > 1 
                                 else 1 for v in xrange(vascularTree.vcount())]                                   
        vascularTree.vs['maxD'] = \
          [max(vascularTree.es[vascularTree.adjacent(v)]['diameter']) 
           for v in xrange(vascularTree.vcount())]
        vascularTree.vs['originalDegree'] = np.array(G.degree())[np.array(sc)]
        vascularTree.vs['z'] = [r[2] for r in vascularTree.vs['r']]
        if nce is None:
            searchParameters = [1, 14, 400]
            for i, kw in enumerate(['rDegree', 'rDiameter', 'rZ']):
                if kw in kwargs.keys():
                    searchParameters[i] = kwargs[kw]
            nce = vascularTree.vs(originalDegree_le=searchParameters[0], 
                                  maxD_ge=searchParameters[1], 
                                  z_le=searchParameters[2]).indices
        del vascularTree.vs['originalDegree']
        
        for j, v in enumerate(vascularTree.vs['r']):
            if all(v == G.vs[vertex]['r']):
                attachmentVertex = j
                break
    
        if attachmentVertex in nce:
            nce.remove(attachmentVertex)
        if len(nce) > 0:
            vascularTree.es['weight'] = vascularTree.es['length'] / \
                                        np.array(vascularTree.es['diameter'])**4
            distance = vascularTree.shortest_paths(nce, weights='weight', mode='all')
            minDist = np.amin(distance, 0)
            reference = vascularTree.shortest_paths(attachmentVertex, weights='weight', mode='all')[0]
            del vascularTree.es['weight']
            delete = np.nonzero(minDist < reference)[0]
            vascularTree.delete_vertices(map(int, delete)) # conversion from numpy int
            endpoints = vascularTree.get_endpoints()
            vascularTree.vs[endpoints]['epID'] = \
                                         [1 if vascularTree.vs[ep]['epID'] == 1
                                          else 2 for ep in endpoints]

    # Find the index of the rootVertex in the vascularTree and add it as a
    # graph attribute (required for artificial graph construction from the
    # database): 
    for j, v in enumerate(vascularTree.vs['r']):
        if all(v == G.vs[vertex]['r']):
            vascularTree['attachmentVertex'] = j
            break
    return vascularTree 



def add_to_db(G, rootVertices, dThreshold, dbDirectory, 
              esr=True, eMin=1, **kwargs):
    """Adds specific vascular trees of a given VascularGraph to a database.
    Performs advanced tree extraction checking for multiple roots and taking 
    only those vertices 'closest' to the provided rootVertices.
    Several information properties are also added to the tree:
    distanceToBorder - the minimal distance of the tree's attachment vertex to
                       the srXTM sample border.
    sampleName - the name of the srXTM sample from which the tree is extracted.
    avZOffset - the distance in z-direction between the attachment vertex and
                the vertex with minimal cortical depth.  
    INPUT: G: VascularGraph.
           rootVertices: The root vertices of the vascular tree to be added.
           dThreshold: The diameter threshold below or equal to which edges are
                       to be ignored (required for tree extraction).
           dbDirectory: The directory of the database as string. If this 
                        directory does not exist, it is created.
           esr: Enforce that all trees have a single root only? (Boolean.)             
           eMin: The minimum number of edges that a tree needs to consist of, in
                 order to be admitted to the database.
           **kwargs:
           vtpDir: The name of the directory in which to save a paraview
                   vtp-file representation of the tree.
           amDir: The name of the directory in which to save an amira am-file
                  representation of the tree.
           sampleName: The name of the srXTM sample from which the tree is
                       taken. (Added to the tree's dictionary.)            
    OUTPUT: None (pickled VascularGraphs written to disk.)                        
    """
    # Create database directory, if nonexistant:
    if not os.path.exists(dbDirectory):
        os.mkdir(dbDirectory)
    
    # Determine number of vascular trees already present in the database:
    dbCount = len(glob.glob1(dbDirectory, '*.pkl'))
    
    # Loop through the rootVertices, extract the corresponding trees and add
    # them to the database:

    discardedTreeCount = 0
    for i, vertex in enumerate(rootVertices):
        vascularTree = vascular_tree_subgraph(G, vertex, dThreshold, esr)
        # Reject tree, if it consists of less than eMin edges:
        if vascularTree.ecount() < eMin:
            discardedTreeCount += 1
            continue

        # Add sample name to the tree's dictionary:
        if kwargs.has_key('sampleName'):
            vascularTree['sampleName'] = kwargs['sampleName']

        # Add the distance attachmentVertex - sample border to the tree's dict:
        shape = vascularTree.shape()
        av = vascularTree.vs[vascularTree['attachmentVertex']]['r'][:2]
        if shape == 'cylinder':
            radius, center = G.radius_and_center()
            vascularTree['distanceToBorder'] = np.linalg.norm(av - center)
        elif shape == 'cuboid':
            minXY = np.min(G.vs['r'], 0)[:2]
            maxXY = np.max(G.vs['r'], 0)[:2]
            vascularTree['distanceToBorder'] = np.min([np.min(av - minXY), np.min(maxXY - av)])

       # Align order zero element with z-axis:    
       # strahler.modified_strahler_order(vascularTree, 0.99)
       # TreeEl = strahler.assign_elements(vascularTree)
       # av = vascularTree['attachmentVertex']
       # avElms = []
       # edgeIndices = []
       # for elm in xrange(TreeEl.vcount()):
       #     if av in TreeEl.vs[elm]['vertices']:
       #         avElms.append((TreeEl.vs[elm]['order'], TreeEl.vs[elm]['edges']))
       # edgeIndices = sorted(avElms, reverse=True)[0][1]
       # points = np.concatenate([[vascularTree.vs[e.source]['r'], 
       #                           vascularTree.vs[e.target]['r']] 
       #                           for e in vascularTree.es[edgeIndices]]) # just node points            
       # z = points[:,2]
       # orderOfIndices = np.argsort(z) # increasing z values
       # direction1 = vgm.g_math.pca(points[orderOfIndices])
       # if np.rad2deg(np.arccos(np.dot(direction1, np.array([0,0,1]))/np.linalg.norm(direction1))) > 90:
       #     direction1 = -direction1
       # direction2 = [0.,0.,1.]
       # vgm.rotate_using_two_vectors(vascularTree, direction1, direction2)

        # Add z_{attachmentVertex} - z_{minimum} to the tree's dict:
        vascularTree['avZOffset'] = vascularTree.vs[vascularTree['attachmentVertex']]['r'][2] - \
                                    np.min(vascularTree.vs['r'], 0)[2]

        # Add vascularTree to database:
        vgm.write_pkl(vascularTree, 
                      os.path.join(dbDirectory, str(dbCount + i - discardedTreeCount) + '.pkl'))
        
        # Write vtp and / or am files if required: 
        for kw, ke, fn in zip(['vtpDir', 'amDir'], ['.vtp', '.am'], 
                              [vgm.write_vtp, vgm.write_amira_mesh_ascii]):
            if kwargs.has_key(kw):
                if not os.path.exists(kwargs[kw]):
                    os.mkdir(kwargs[kw])
                fn(vascularTree, os.path.join(kwargs[kw], str(dbCount + i - discardedTreeCount) + ke))
            


def extend_from_db(G, attachmentVertices, dbDirectory):
    """Extends a VascularGraph at given attachment vertices with vascular trees
    chosen randomly from a tree database.
    INPUT: G: VascularGraph
           attachmentVertices: Vertices at which the vascular trees are to be
                               attached.
           dbDirectory: The directory of the database as string.
    OUTPUT: None (G is modified in-place.)                           
    """
    # Get names of database files:
    dbFiles = glob.glob1(dbDirectory, '*.pkl')
    # Determine number of vascular trees present in the database:
    dbCount = len(dbFiles)
    
    # At each attachment vertex, extend the VascularGraph with a vascular tree
    # from the database:
    Gtrees = vgm.VascularGraph(0)
    treeno = 1
    G.vs['degree']=G.degree()
    for vertex in attachmentVertices:
        #print('Tree %i of %i' % (treeno, len(attachmentVertices))); treeno += 1
        i = np.random.random_integers(0,dbCount-1,1)[0]
        vascularTree = vgm.read_pkl(os.path.join(dbDirectory, dbFiles[i]))
        offset = G.vs[vertex]['r'] - \
                 vascularTree.vs[vascularTree['attachmentVertex']]['r']
        vgm.shift(vascularTree, offset)
        Gtrees.disjoint_union_attribute_preserving(vascularTree, 'omit')
        #G.union_attribute_preserving(vascularTree)
    if G.vs['apv'][vertex] == 1:
        kindAttr='pa'
    elif G.vs['vpv'][vertex] == 1:
        kindAttr='pv'
    else:
        kindAttr='n'
    stdout.flush()
    G.union_attribute_preserving(Gtrees, 'omit',kind=kindAttr)
