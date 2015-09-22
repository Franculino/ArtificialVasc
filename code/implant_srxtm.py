from __future__ import division, print_function, with_statement
import copy
import numpy as np
from scipy.spatial import kdtree
from scipy import finfo
from sys import stdout 
from copy import deepcopy
import vgm

def implant_srxtm_cube(Ga, Gd, origin=None, crWidth=150):
    """Implants a cubic srXTM sample in an artificial vascular network
    consisting of pial vessels, arterioles, venoles and capillaries. At the 
    site of insertion, artificial vessels are removed to make space for the 
    srXTM sample. At the border between artificial and data networks,
    connections are made between the respective loose ends.
    INPUT: Ga: VascularGraph of the artificial vasculature.
           Gd: VascularGraph of the srXTM data. The graph is expected to have
               the attributes 'av' and 'vv' that denote the indices of 
               endpoints of the large penetrating arteries and veins 
               respectively.
           origin: The two-dimensional origin in the xy plane, where the srXTM
                   sample should be inserted. This will be the network's 
                   center of mass, if not provided.
           crWidth: Width of connection region. After computing the center and
                    radius of the SRXTM sample, loose ends of the SRXTM at
                    radius - 2*crWidth are connected to loose ends of the
                    artificial network at radius + 2*crWidth.
    OUTPUT: Ga: Modified input Ga, with Gd inserted at origin.       
    """
    
    # Create Physiology object with appropriate default units:
    P = vgm.Physiology(Ga['defaultUnits'])
    eps = finfo(float).eps * 1e4


    return Ga
    
    
def implant_srxtm(Ga, Gd, origin=None, crWidth=150):
    """Implants a cylindrical srXTM sample in an artificial vascular network
    consisting of pial vessels, arterioles, venoles and capillaries. At the 
    site of insertion, artificial vessels are removed to make space for the 
    srXTM sample. At the border between artificial and data networks,
    connections are made between the respective loose ends.
    INPUT: Ga: VascularGraph of the artificial vasculature.
           Gd: VascularGraph of the srXTM data. The graph is expected to have
               the attributes 'av' and 'vv' that denote the indices of 
               endpoints of the large penetrating arteries and veins 
               respectively.
           origin: The two-dimensional origin in the xy plane, where the srXTM
                   sample should be inserted. This will be the network's 
                   center of mass, if not provided.
           crWidth: Width of connection region. After computing the center and
                    radius of the SRXTM sample, loose ends of the SRXTM at
                    radius - 2*crWidth are connected to loose ends of the
                    artificial network at radius + 2*crWidth.
    OUTPUT: Ga: Modified input Ga, with Gd inserted at origin.       
    """
    
    # Create Physiology object with appropriate default units:
    P = vgm.Physiology(Ga['defaultUnits'])
    eps = finfo(float).eps * 1e4

    # Define exceptions:
    class ShapeException(Exception):
        def __init__(self):
            return
        def __str__(self):
            return 'The shape of the srXTM sample must be cylindrical!'
            
    # Assert that the srXTM sample is cylindrical:        
    if Gd.shape() != 'cylinder':
        raise ShapeException            

    # Compute the network's center of mass as origin, if origin is not provided:
    if origin is None:
        origin = np.mean(Ga.vs['r'], axis=0)[:2]
    print('origin')
    print(origin)
    print('Original number of components')
    print(len(Ga.components()))
    stdout.flush()
    print('Kind Check')
    print(np.unique(Ga.vs['kind']))

    # Remove all vertices of Ga which are within Gd's radius of 'origin' and
    # at a depth less than the maximum depth of Gd. Leave the pial vessels
    # of Ga untouched:
    Ga.vs['distanceFromOrigin'] = [np.linalg.norm(r[:2] - origin) for r in Ga.vs['r']]
    Ga.vs['isPial'] = [0 if k not in ['pa', 'pv'] else 1 for k in Ga.vs['kind']]
    Ga.vs['z'] = [r[2] for r in Ga.vs['r']]
    radius, center = Gd.radius_and_center()
    area, length = Gd.cross_sectional_area_and_length(shape='cylinder')
    Ga.delete_vertices(Ga.vs(distanceFromOrigin_le=radius, z_le=length, isPial_eq=0))
    print('Kind Check 1 1')
    print(np.unique(Ga.vs['kind']))
    print('Center deleted')

    # Remove all non-capillaries that have lost their root:
    Ga.vs['degree'] = Ga.degree()
    #pial endpoints are located at the pial level in the area, where the realistic network is implanted
    pialEndpoints = [v.index for v in Ga.vs(degree_eq=1) if v['kind'] in ['pa', 'pv']]
    if len(pialEndpoints) > 0:
        print('pialEndpoints')
        print(pialEndpoints)
        print(len(pialEndpoints))
        Gc = copy.deepcopy(Ga)
        Gc.vs['index'] = xrange(Gc.vcount())
        del(Gc.vs['degree'])
        del(Gc.vs['distanceFromOrigin'])
        del(Gc.vs['nkind'])
        del(Gc.vs['pBC'])
        del(Gc.vs['z'])
        del(Gc.vs['isPial'])
        del(Gc.es['diameter'])
        del(Gc.es['diameter_orig'])
        del(Gc.es['diameters'])
        del(Gc.es['lengths'])
        del(Gc.es['diameter_change'])
        del(Gc.es['conductance'])
        del(Gc.es['length'])
        del(Gc.es['cost'])
        del(Gc.es['points'])
        #Gc consists of pial vessels and penetrating trees
        Gc.delete_vertices(Gc.vs(kind_eq='c').indices)
        print('Capillaries deleted')
        print('Number of components')
        print(len(Gc.components()))
        del(Gc.vs['kind'])
        #add random vertex 
        Gc.add_vertices(1)
        Gc.vs[Gc.vcount()-1]['index']=100000000000000
        Gc.vs[Gc.vcount()-1]['r']=[-500.,-500.,0.]
        pialEndpoints2 = []
        #pialEndpoints2 = pialEndpoints in Gc graph
        for i in pialEndpoints:
           pialEndpoints2.append(Gc.vs(index_eq=i).indices[0])
        #connect all pial endpoints (=in and outflows at pial level and deg1 pial vertices resulting from hole at implant position) to that random vertices
        Gc.add_edges([(v, Gc.vcount()-1) for v in pialEndpoints2])
        print('Number of components 2 --> unconnected penetrating trees')
        print(len(Gc.components()))
        #deletes connected component
        print('Kind Check 1 2')
        print(np.unique(Ga.vs['kind']))
        Gc.delete_vertices(Gc.subcomponent(Gc.vcount()-1))
        Ga.delete_vertices(Gc.vs['index'])
        #unconnected capillaries still present, but no more unconnected penetrating trees
    else:
        print('Reduce components')
        print(len(Ga.components()))
        while len(Ga.components()) > 1:
            delVertices=Ga.components()[len(Ga.components())-1]
            print(len(delVertices))
            Ga.delete_vertices(Ga.components()[len(Ga.components())-1])
            print('number of components')
            print(len(Ga.components()))
    print('Kind Check 1 3')
    print(np.unique(Ga.vs['kind']))

    #Delete unconnected capillaries
    print('Number of components 3')
    print(len(Ga.components()))
    components=Ga.components()
    delVertices=[]
    stdout.flush()
    for i in range(1,len(components)):
        delVertices = delVertices + components[i]
    if len(delVertices) > len(components[0]):
        print('ERROR --> main component deleted')
    Ga.delete_vertices(delVertices)
    print('Number of components 4')
    print(len(Ga.components()))
    
    # Insert Gd into Ga (insert realistic network into artificial network):
    print('CHECK DEGREE SRXTM')
    print(max(Gd.degree()))
    origin = np.append(origin, 0.)
    center = np.append(center, min([r[2] for r in Gd.vs['r']]))
    shift = (origin-center).tolist()
    print('shift realistic networ by')
    print(shift)
    vgm.shift(Gd, shift)
    print('Components gXTM')
    print(len(Gd.components()))
    print('Components trees and capillaries')
    print(len(Ga.components()))
    initialVcount = Ga.vcount()    
    initialEcount = Ga.ecount()
    Ga.union_attribute_preserving(Gd)
    print('Put together')
    print(len(Ga.components()))
    Ga.vs['isSrxtm'] = [0 if x < initialVcount else 1 for x in xrange(Ga.vcount())] # artificial network: 0, srXTM: 1
    Ga.es['isSrxtm'] = [0 if x < initialEcount else 1 for x in xrange(Ga.ecount())] # artificial network: 0, srXTM: 1
    stdout.flush()

    # Connect loose ends of Gd (srXTM to artificial capillary bed):
    # only dead ends in a given radius are connected
    origin = origin[:2]
    Ga.vs(isSrxtm_eq=1)['distanceFromOrigin'] = [np.linalg.norm(r[:2] - origin) for r in Ga.vs(isSrxtm_eq=1)['r']]
    #All capillary vertices which are at max 2.0*radius away from the center of the srXTM
    #possible capillary attachment vertices
    capillaryVertices = Ga.vs(isSrxtm_eq=0, distanceFromOrigin_le=radius*2.0, kind_eq='c', degree_le=3)
    Kdt = kdtree.KDTree(capillaryVertices['r'], leafsize=10)
    #all srxtm vertices in radius
    print('Connect Implant to capillary bed')
    looseEndsI = Ga.vs(distanceFromOrigin_le=radius+2*crWidth, distanceFromOrigin_ge=radius-2*crWidth, 
                       degree_eq=1, isSrxtm_eq=1).indices # srxtm capillaries and non-capillaries
    print('Remove av and vv of Srxtm from the looseEndsI list')
    print(len(looseEndsI))
    for i in Gd['av']:
        vIndex = int(np.nonzero(np.all(Gd.vs[i]['r'] == Ga.vs['r'], axis=1))[0][0])
        if vIndex in looseEndsI:
            looseEndsI.remove(vIndex)
    print(len(looseEndsI))
    for i in Gd['vv']:
        vIndex = int(np.nonzero(np.all(Gd.vs[i]['r'] == Ga.vs['r'], axis=1))[0][0])
        if vIndex in looseEndsI:
            looseEndsI.remove(vIndex)
    print(len(looseEndsI))
    print('Non capillaries artificial network')
    looseEndsI_2=Ga.vs(distanceFromOrigin_le=radius+2*crWidth, distanceFromOrigin_ge=radius-2*crWidth, 
                      degree_eq=1, isSrxtm_eq=0, kind_ne='c').indices # artificial network non-capillaries
    print(len(looseEndsI_2))
    dummy1=Ga.vs(distanceFromOrigin_le=radius+2*crWidth, distanceFromOrigin_ge=radius-2*crWidth, 
                      degree_eq=1, isSrxtm_eq=0, kind_eq='pa').indices
    dummy2=Ga.vs(distanceFromOrigin_le=radius+2*crWidth, distanceFromOrigin_ge=radius-2*crWidth, 
                      degree_eq=1, isSrxtm_eq=0, kind_eq='pv').indices
    print('Remove dummy1 (pa) from loosEndsI_2')
    print(len(looseEndsI_2))
    print(len(dummy1))
    print('Kind Check 3')
    print(np.unique(Ga.vs['kind']))
    for i in dummy1:
        if i in looseEndsI_2:
            looseEndsI_2.remove(i)
    print(len(looseEndsI_2))
    print('Remove dummy2 (pv) from loosEndsI_2')
    for i in dummy2:
        if i in looseEndsI_2:
            looseEndsI_2.remove(i)
    print(len(dummy2))
    print(len(looseEndsI_2))
    print('Exten looseEndsI')
    print(len(looseEndsI))
    looseEndsI.extend(looseEndsI_2)
    print(len(looseEndsI))
    print('Check loosEnds 2')
    looseEndesII=Ga.vs(distanceFromOrigin_le=radius+2*crWidth, distanceFromOrigin_ge=radius-2*crWidth,
                      degree_eq=1, isSrxtm_eq=0, kind_ne='c').indices
    for i in looseEndesII:
        print(Ga.vs[i]['kind'])
    print('len(looseEndsI)')
    print(len(looseEndsI))
    print('Degree one artificial')
    Ga.vs['degree']=Ga.degree()
    print(len(Ga.vs(isSrxtm_eq=0, distanceFromOrigin_le=radius*2.0, kind_eq='c',degree_le=3)))
    posConnections=[]
    for i in Ga.vs['degree']:
        if i >= 3:
            posConnections.append(0)
        elif i == 3:
            posConnections.append(1)
        elif i == 2:
            posConnections.append(2)
        else:
            posConnections.append(3)
    Ga.vs['posConnections']=posConnections
    posConnections=np.sum(Ga.vs[Ga.vs(isSrxtm_eq=0, distanceFromOrigin_le=radius*2.0, kind_eq='c', degree_le=3).indices]['posConnections'])
    print('Possible Connections')
    print(posConnections)
    looseEnds = Ga.vs(looseEndsI)
    newEdges = []
    vertex_withSrxtm=[]
    stdout.flush()
    print('Kind Check 4')
    print(np.unique(Ga.vs['kind']))
    for le in looseEnds:
        print('')
        print(le.index)
        stdout.flush()
        newVertexFound=0
        count= 1
        while newVertexFound != 1:
            #start with 5 possible nearest neighbors
            if count*5 > len(capillaryVertices):
                print('WARNING last chance to find a connecting vertex')
                print(count*5)
            nearestN=Kdt.query(le['r'],k=5*count)
            for i in range((count-1)*5,count*5):
                newVertex=capillaryVertices[int(nearestN[1][i])].index
            #newVertex=capillaryVertices[int(Kdt.query(le['r'])[1])].index
                if Ga.vs['posConnections'][newVertex]  == 0:
                    print('No connection possible')
                else:
                   Ga.vs[newVertex]['posConnections'] = Ga.vs[newVertex]['posConnections'] - 1
                   newVertexFound = 1
                   break
            count += 1
        print('CONNECTION FOUND')
        print(le.index)
        newEdges.append((le.index, newVertex))
        vertex_withSrxtm.append(le.index)
    Ga.add_edges(newEdges)
    Ga.vs['degree']=Ga.degree()
    print('CHECK DEGREE 5')
    print(max(Ga.degree()))
    #TODO think of fixed diameter and fixed length
    diameter = 7.
    length = 50.
    conductance = P.conductance(diameter, length, P.dynamic_blood_viscosity(diameter, 'a'))
    Ga.es[-len(newEdges):]['diameter'] = [diameter for e in newEdges]
    Ga.es[-len(newEdges):]['conductance'] = [conductance for e in newEdges]
    Ga.es['newEdge'] = [0 if x < Ga.ecount()-len(newEdges) else 1 for x in xrange(Ga.ecount())]
    print('Kind Check 4')
    print(np.unique(Ga.vs['kind']))
    # Connect endpoints of large penetrating vessels to pial vessels:
    #'av' - arterial inflows, 'pv' -venule outflow
    print('Standard Network av')
    print(Ga['av'])
    print(len(Ga['av']))
    print('Standard Network vv')
    print(Ga['vv'])
    print(len(Ga['vv']))
    print('Implant Network av')
    print(Gd['av'])
    print(len(Gd['av']))
    print('Implant Network vv')
    print(Gd['vv'])
    print(len(Gd['av']))
    print(zip((Gd['av'], Gd['vv']), ('pa', 'pv')))
    Ga['vWithSRXTM']=vertex_withSrxtm
    posConnections=[]
    stdout.flush()
    for i in Ga.vs['degree']:
        if i > 2:
            posConnections.append(0)
        elif i == 2:
            posConnections.append(1)
        else:
            posConnections.append(2)
    Ga.vs['posConnections']=posConnections
    print(np.unique(Ga.vs['kind']))
    for avv, pk in zip((Gd['av'], Gd['vv']), ('pa', 'pv')):
        print('CHECK')
        print(len(avv))
        print(pk)
	#alle pial vessels (either arteries or venules)
        pialIndices = Ga.vs(kind_eq=pk).indices
        print(pialIndices)
        #KDTree of all pial vessels (either arteries or venules) of main graph
        Kdt = kdtree.KDTree(Ga.vs(pialIndices)['r'], leafsize=10)
        newEdges = []
        length = []
        diameter = []
        conductance = []
        srxtmAV = []
        vs=[]
        degree0=[]
        for v in avv:
            #to find the vertex number in the new graph
            vIndex = int(np.nonzero(np.all(Gd.vs[v]['r'] == Ga.vs['r'], axis=1))[0][0])
            #corresponding vertexNumber of Inflow/Outflow vertices of srxtm 
            srxtmAV.append(vIndex)
            degree0.append(Ga.vs[vIndex]['degree'])
            vs.append(v)
	    #Closes vertex of pial vertices
            newVertexFound = 0
            count = 1
            while newVertexFound != 1:
                #pIndex = pialIndices[int(Kdt.query(Ga.vs[vIndex]['r'])[1])]
                nearestN = Kdt.query(Ga.vs[vIndex]['r'],k=5*count)
                for i in range((count-1)*5,count*5):
                    pIndex=pialIndices[int(nearestN[1][i])]
                    if Ga.vs['posConnections'][pIndex]  == 0 or pIndex in vertex_withSrxtm:
                        print('No connection possible')
                    else:
                       print('posConnections')
                       print(Ga.vs['posConnections'][pIndex])
                       Ga.vs[pIndex]['posConnections'] = Ga.vs[pIndex]['posConnections'] - 1
                       newVertexFound = 1
                       print(Ga.vs['posConnections'][pIndex])
                       break
                count += 1
            print('CONNECTION FOUND')
            print(v)
            newEdges.append((vIndex, pIndex))
            length.append(np.linalg.norm(Ga.vs[vIndex]['r'] - Ga.vs[pIndex]['r']))
            #diameter is set to the maximum diameter of the adjacent edges of the pial vertex
            diameter.append(max(Ga.es[Ga.adjacent(pIndex)]['diameter']))
            conductance.append(P.conductance(diameter[-1], length[-1], P.dynamic_blood_viscosity(diameter[-1], pk[1])))

        #inflow/outflow vertices of srxtm
        Ga['srxtm_' + pk[1]] = srxtmAV
        Ga.add_edges(newEdges)
        Ga.vs['degree']=Ga.degree()
        Ga.es[-len(newEdges):]['diameter'] = diameter
        Ga.es[-len(newEdges):]['conductance'] = conductance
	#newly introduced edges a/v
        Ga.es['newEdge_'+pk] = [0 if x < Ga.ecount()-len(newEdges) else 1 for x in xrange(Ga.ecount())]

    noneEdges=Ga.es(newEdge_pa_eq=None).indices
    Ga.es[noneEdges]['newEdge_pa']=[0]*len(noneEdges)
    Ga.vs['srxtm_a']=[0]*Ga.vcount()
    Ga.vs['srxtm_v']=[0]*Ga.vcount()
    for i in Ga['srxtm_a']:
        Ga.vs[i]['srxtm_a']=1
    for i in Ga['srxtm_v']:
        Ga.vs[i]['srxtm_v']=1
    del(Ga['srxtm_a'])
    del(Ga['srxtm_v'])

    Ga.vs['degree']=Ga.degree()
    print('CHECK DEGREE 6')
    print(max(Ga.degree()))

    # Remove Dead Ends
    # Dead Ends around the implant
    print('Delete dead ends around implant')
    deg1=Ga.vs(degree_eq=1,distanceFromOrigin_gt=radius+2*crWidth,distanceFromOrigin_lt=radius-2*crWidth,isSrxtm_eq=1).indices
    for i in deg1:
        if Ga.vs[i]['av'] == 1.0:
            deg1.remove(i)
        elif Ga.vs[i]['vv'] == 1.0:
            deg1.remove(i)

    print('avs and vvs')
    print(len(Ga.vs(av_eq=1)))
    print(len(Ga.vs(vv_eq=1)))

    while len(deg1) > 0:
        print('')
        print(len(deg1))
        Ga.delete_vertices(deg1)
        Ga.vs['degree']=Ga.degree()
        print('avs and vvs')
        print(len(Ga.vs(av_eq=1)))
        print(len(Ga.vs(vv_eq=1)))
        deg1=Ga.vs(degree_eq=1,distanceFromOrigin_gt=radius+2*crWidth,distanceFromOrigin_lt=radius-2*crWidth,isSrxtm_eq=1).indices
        for i in deg1:
            if Ga.vs[i]['av'] == 1.0:
                deg1.remove(i)
            elif Ga.vs[i]['vv'] == 1.0:
                deg1.remove(i)

    # Delete obsolete vertex properties:    
    del Ga.vs['degree']        
    stdout.flush()

    #Delte av and vv of SRXTM
    av=Ga.vs(av_eq=1).indices
    print('len av')
    print(len(av))
    for i in av:
        if Ga.vs['isSrxtm'][i]:
            Ga.vs[i]['av']=0

    av=Ga.vs(av_eq=1).indices
    print('len av')
    print(len(av))
    Ga['av']=av
    vv=Ga.vs(vv_eq=1).indices
    print('len vv')
    print(len(vv))
    for i in vv:
        if Ga.vs['isSrxtm'][i]:
            Ga.vs[i]['vv']=0

    vv=Ga.vs(vv_eq=1).indices
    print('len vv')
    print(len(vv))
    Ga['vv']=vv

    #There might be deadEnd Vertices of the artificial capillary bed where the SRXTM was implemented
    #Those vertices are delted
    #First assign capGrid = 0 to SRXTM
    capGridNone=Ga.vs(capGrid_eq=None).indices
    Ga.vs[capGridNone]['capGrid']=[0]*len(capGridNone)
    Ga.vs['degree'] = Ga.degree()
    print('Check if deg1 are only located at in- and outlet and at borders of capillary grid')
    deg1=Ga.vs(degree_eq=1).indices
    print(len(deg1))
    for i in deg1:
        if Ga.vs[i]['capGrid']  == 0:
            if Ga.vs['av'] == 0 and Ga.vs['vv'] == 0:
                print('ERROR')
                print(i)
                print(Ga.vs[i]['isSrxtm'])
                print(Ga.vs[i]['capGrid'])

    while len(deg1) > len(Ga['av'])+len(Ga['vv']):
        print('')
        print(len(deg1))
        av=Ga.vs(av_eq=1).indices
        vv=Ga.vs(vv_eq=1).indices
        for i in av:
            if i in deg1:
                deg1.remove(i)
        print(len(deg1))
        for i in vv:
            if i in deg1:
                deg1.remove(i)
        print(len(deg1))
        Ga.delete_vertices(deg1)
        Ga.vs['degree']=Ga.degree()
        deg1=Ga.vs(degree_eq=1).indices

    del Ga.vs['distanceFromOrigin']
    print('All newly created Dead Ends have ben elimanted')
    Ga.vs['degree']=Ga.degree()
    Ga['av']=Ga.vs(av_eq=1).indices
    Ga['vv']=Ga.vs(vv_eq=1).indices
    print(len(Ga.vs(degree_eq=1)))
    print(len(Ga['av']))
    print(len(Ga['vv']))
    stdout.flush()

    return Ga
    
    
        
        

    
    
