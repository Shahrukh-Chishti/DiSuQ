import numpy
import numpy as np

## upper diagonal matrix un/compression

def arrayMatrix(arr):
    N = int((np.sqrt(1+8*len(arr))-1)/2) #calculate dimension of matrices from number of upper triagonal entries
    matrix = np.zeros((N,N))
    matrix[np.triu_indices(N,k=0)] = arr
    matrix = np.maximum(matrix, matrix.transpose())
    return matrix

def matrixArray(matrix):
    arr = numpy.triu(matrix)
    import ipdb;ipdb.set_trace()
    return arr

## de/coagulating circuit parameters

def circuitWrap(C,J,L):
    Circuit = np.concatenate([C,J,L])
    return Circuit

def circuitUnwrap(Circuit):
    N = int(len(Circuit)/2)
    C = Circuit[:N]
    J = Circuit[N:]
    #L = Circuit[2*N:]
    return C,J

## C,J,L are connectivity matrices

def defineComponentBounds(circuit_bounds,N):
    bounds = [(circuit_bounds['C']['low'],circuit_bounds['C']['high'])]*N
    bounds += [(circuit_bounds['J']['low'],circuit_bounds['J']['high'])]*N
    if 'L' in circuit_bounds:
        bounds += [(circuit_bounds['L']['low'],circuit_bounds['L']['high'])]*N
    return bounds

## Design Random Circuit

def designRandomCircuit(c_specs,j_specs,l_specs=None,phiOffs_specs=None):

    mask      = np.ones(j_specs['dimension'])
    indices   = np.arange(len(mask))
    np.random.shuffle(indices)
    mask[indices[:j_specs['dimension']-j_specs['keep_num']]] *= 0.

    # Draw junctions
    junctions  = np.random.uniform(j_specs['low'], j_specs['high'], j_specs['dimension'])
    junctions *= mask

    # Draw capacitances
    capacities  = np.random.uniform(c_specs['low'], c_specs['high'], c_specs['dimension'])
    capacities  = capacities * (np.random.uniform(0., 1., c_specs['dimension']) < c_specs['keep_prob'])
    # Case of 4-node circuit: add zero at forbidden connection 2-4
    #if c_specs['dimension']==9:
    #    capacities = np.insert(capacities, 6, 0)
    #capacities += junctions * CJFACTOR

    # Draw inductances
    if l_specs != None:
        inductances  = np.random.uniform(l_specs['low'], l_specs['high'], l_specs['dimension'])
        inductances  = inductances * (np.random.uniform(0., 1., l_specs['dimension']) < l_specs['keep_prob'])
        # Case of 4-node circuit: add zero at forbidden connection 2-4
        if l_specs['dimension']==9:
            inductances = np.insert(inductances, 6, 0)
        inductances[junctions > 0] = 0.
    else:
        inductances = None

    # draw flux offsets for loops
    if phiOffs_specs is not None:
        phiOffs = np.random.choice(phiOffs_specs['values'], phiOffs_specs['dimension'])
    else:
        phiOffs = None

    circuit = {'junctions': junctions, 'capacities': capacities, 'inductances': inductances, 'phiOffs': phiOffs}
    return circuit
