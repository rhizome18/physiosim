import copy
import numpy as np
from physiology_utilities import computeFlows, dictToVec, vecToDict, dictToGraph, graphToDict

def odemasterfxn(t, state, speciesList,compartmentList, k, externalK, externalW, growth,carrierDict):
    # Unpack the state variables using vecToDict
    stateDict = vecToDict(state,speciesList,compartmentList)
    # dynamics: call the modular flowDynamics function for each species
    diffDict = vecToDict(0,speciesList,compartmentList)
    # TODO: fix code sloppiness so that carriers don't need to run ahead
    # Re-order so that we don't need two compartment-list loops here
    netflows = {}
    for sp in speciesList:
        x = stateDict[sp]
        dspecies = {}
        netflows[sp] = {}
        if not carrierDict[sp]:
            for cpt in compartmentList:
                insystem_inflows = 0
                insystem_outflows = 0
                for key in k[cpt].keys(): # for each compartment
                    # this compartment to someplace
                    insystem_outflows+=k[cpt][key]*x[cpt]
                    # someplace to this compartment 
                    insystem_inflows+=k[key][cpt]*x[key]
                netflows[sp][cpt] = insystem_inflows-insystem_outflows
                dspecies[cpt] = growth[sp][cpt]*x[cpt]+netflows[sp][cpt]+externalK[sp][cpt](t,x)*x[cpt]+externalW[sp][cpt](t,x)
        else:
            carriersp = carrierDict[sp]
            netflows[sp] = netflows[carriersp]
            for cpt in compartmentList:
                # TODO: make this cleaner
                cV = stateDict[carriersp][cpt]
                dspecies[cpt] = (growth[sp][cpt]*x[cpt]*cV+netflows[sp][cpt]*cV+externalK[sp][cpt](t,x)*x[cpt]*cV+externalW[sp][cpt](t,x)*cV)/cV
        diffDict[sp] = dspecies
    return dictToVec(diffDict,speciesList,compartmentList)
