import copy
import numpy as np
from physiology_utilities import computeFlows, dictToVec, vecToDict, dictToGraph, graphToDict

def odemasterfxn(t, state, speciesList,compartmentList, k, externalK, growth,carrierDict):
    # Unpack the state variables using vecToDict
    stateDict = vecToDict(state,speciesList,compartmentList)
    # dynamics: call the modular flowDynamics function for each species
    diffDict = vecToDict(0,speciesList,compartmentList)
    # TODO: fix code sloppiness so that carriers don't need to run ahead
    # Re-order so that we don't need two compartment-list loops here
    netflows = {}
    for sp in speciesList:
        xsp = stateDict[sp]
        x = stateDict
        dspecies = {}
        netflows[sp] = {}
        for cpt in compartmentList:
            insystem_inflows = 0
            insystem_outflows = 0
            for key in k[cpt].keys(): # for each compartment
                # this compartment to someplace
                new_out = k[cpt][key]*xsp[cpt]
                new_in = k[key][cpt]*xsp[key]
                carrierKey = carrierDict[sp]
                while carrierKey:
                    new_out*=stateDict[carrierKey][cpt]
                    new_in*=stateDict[carrierKey][key]
                    carrierKey = carrierDict[carrierKey]
                insystem_outflows+=new_out
                insystem_inflows+=new_in
            netflows[sp][cpt] = insystem_inflows-insystem_outflows
            dspecies[cpt] = growth[sp][cpt]*xsp[cpt]+netflows[sp][cpt]+externalK[sp][cpt](t,x)
            diffDict[sp] = dspecies
    return dictToVec(diffDict,speciesList,compartmentList)
