# physiology_utilities.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
import cvxpy as cp
from scipy.optimize import linprog

def computeFlows(A, V,scalar=1):
    n = len(V)
    K = cp.Variable((n, n))
    constraints = []
    for ii in range(n):
        inflows = 0
        outflows = 0
        for jj in range(n):
            inflows+=K[jj,ii]*V[jj]
            outflows+=K[ii,jj]*V[ii]
            if ii != jj:
                if A[ii, jj] == 0:
                    constraints.append(K[ii, jj] == 0)
                else:
                    constraints.append(K[ii, jj] >= scalar)
        constraints.append(inflows-outflows==0)
                    
    # Objective: minimize 1-norm (fast)
    objective = cp.Minimize(cp.norm(K,1))
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    K = K.value
    K[np.abs(K)<=1e-6]=0
    return K

# pack and unpack functions to make the ODE function a bit more extensible and transparent
def dictToVec(dict,speciesList,compartmentList):
    newVector = []
    for species in speciesList:
        for compartment in compartmentList:
            newVector.append(dict[species][compartment])

    return newVector

def vecToDict(vec,speciesList,compartmentList):
    newDict = {}
    if np.all(vec == 0): vec = np.zeros(len(speciesList)*len(compartmentList))
    vectorInd = -1
    
    for species in speciesList:
        newDict[species] = {}
        for compartment in compartmentList:
            vectorInd +=1
            newDict[species][compartment] = vec[vectorInd]
    return newDict

def dictToGraph(kdict,compartmentList):
    A = np.zeros((len(compartmentList),len(compartmentList)))
    ii = -1
    for compR in compartmentList:
        ii+=1
        jj = -1
        for compC in compartmentList:
            jj +=1
            if kdict[compR][compC]:
                A[ii,jj] = kdict[compR][compC]
    return A

def graphToDict(A,compartmentList):
    if np.all(A==0):
        A = np.zeros((len(compartmentList),len(compartmentList)))
    ii = -1
    kdict = {}
    for compR in compartmentList:
        ii+=1
        kdict[compR] = {}
        jj = -1
        for compC in compartmentList:
            jj +=1
            kdict[compR][compC] = A[ii,jj]
    return kdict
