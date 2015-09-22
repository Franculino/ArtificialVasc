from __future__ import division, print_function
import os
import sys
import string
import glob
import vgm

def export(input_dir, output_dir):
    """Exports pickled iGraph files to GraphML format.
    INPUT: input_dir: The directory where the pickled iGraph files reside.
           output_dir: The directory in which to store the GraphML files.
    OUTPUT: GraphML files written to disk.
    """
    files = glob.glob1(input_dir, '*.pkl')
    if os.path.isdir(output_dir):
        ans = raw_input('Directory %s exists. Overwrite files? [y/n]:  ' % output_dir)
        if ans[0] == 'y':
            gml_files = glob.glob(os.path.join(output_dir, '*'))
            map(os.remove, gml_files)
        else:
            return
    else:
        os.mkdir(output_dir)

    for f in files:
        outname = f[:-4] + '.graphml'
        G = vgm.read_pkl(os.path.join(input_dir,f))
        # Delete unnecessary vertex properties:
        v_attributes = G.vs.attribute_names() 
        for va in v_attributes:
            if va not in ['r']:
                del G.vs[va]
        # Delete unnecessary edge properties:
        e_attributes = G.es.attribute_names() 
        for ea in e_attributes:
            if ea not in ['diameter', 'diameters', 'length', 'lengths', 'points']:
                del G.es[ea]

        # Delete unnecessary graph properties:
        g_attributes = G.attributes()
        for ga in g_attributes:
            if ga not in ['attachmentVertex', 'sampleName', 'distanceToBorder', 'defaultUnits', 'avZOffset']:
                del G[ga]

        vgm.write_graphml(G, os.path.join(output_dir, outname))

if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    export(input_dir, output_dir)
        
