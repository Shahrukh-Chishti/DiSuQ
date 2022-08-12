from numpy import array,concatenate,unique
from numpy import argwhere,isin,argmin,copy,around
from numpy.linalg import det
from numpy.random import randint,rand
from torch import roll,zeros_like,sum,zeros
from torch import complex128 as complex,column_stack as stack
from time import perf_counter,sleep

def sgn(sigma):
    # cycle tracing
    N = len(sigma)
    signature = +1
    track = list(sigma.keys())
    while len(track)>0:
        count = 0
        index = track[0]
        while index in sigma and index in track:
            track.remove(index)
            index = sigma[index]
            count += 1

        signature *= (-1)**(count-1)
    return signature

def increment(index,sigma):
    row,col = index
    sigma[row] = col
    return sigma

def minor(M,index):
    row,col = index
    M = M[M[:,0]!=row]
    M = M[M[:,1]!=col]
    return M

def attacking(M):
    if len(M)==1:
        return False
    return True

def choices(M):
    index,distribution = unique(M[:,0],return_counts=True)
    mini = index[argmin(distribution)]
    return M[M[:,0]==mini]

def distributionNull(M,N):
    rows = unique(M[:,0])
    cols = unique(M[:,1])
    if N > len(rows) or N > len(cols) or N==0:
        return True
    return False

def characteristicEdit(index,m,coeffs):
    if index[0]==index[1]: # diagonal
        return m*coeffs - roll(coeffs,1)
    return m*coeffs

def product_poly(coeffs,M,data,sigma):
    values = []#zeros(len(M),len(coeffs),dtype=complex)
    #for ind,index in enumerate(M):
    for index in M:
        m = data[tuple(index.tolist())]
        #values[ind] += characteristicEdit(index,m,coeffs)
        values.append(characteristicEdit(index,m,coeffs))
        increment(index,sigma)
    values = stack(values)
    return sum(values,axis=0)*sgn(sigma)

def charPoly(coeffs,M,N,data,stats,sigma=dict()):
    if distributionNull(M,N):
        stats['terminal'] += 1
        return zeros_like(coeffs)
    elif attacking(M):
        values = []#zeros(len(M),len(coeffs),dtype=complex)
        for ind,index in enumerate(choices(M)):
            m = data[tuple(index)]
            coeffs = characteristicEdit(index,m,coeffs.clone())
            #values[ind] += charPoly(coeffs,minor(M,index),N-1,data,stats,increment(index,sigma.copy()))
            values.append(charPoly(coeffs,minor(M,index),N-1,data,stats,increment(index,sigma.copy())))
        values = stack(values)
        return sum(values,axis=0)
    else:
        stats['leaf'] += 1
        return product_poly(coeffs,M,data,sigma.copy())
