import matplotlib.pyplot as plt
from random import randint
import numpy as np
import collections
import networkx as nx
import sys
import os
import time

def construct_graph(tab_index, anonymized_degree):
    graph = nx.Graph()
    if sum(anonymized_degree) % 2 == 1:
        return None

    while True:
        if not all(di >= 0 for di in anonymized_degree):
            return None
        if all(di == 0 for di in anonymized_degree):
            return graph
        v = np.random.choice((np.where(np.array(anonymized_degree) > 0))[0])
        dv = anonymized_degree[v]
        anonymized_degree[v] = 0
        for index in np.argsort(anonymized_degree)[-dv:][::-1]:
            if index == v:
                return None
            if not graph.has_edge(tab_index[v], tab_index[index]):
                graph.add_edge(tab_index[v], tab_index[index])
                anonymized_degree[index] = anonymized_degree[index] - 1


def calcualte_cost(original_list, modified_list):
    cost = 0
    for i in range(len(modified_list)):
        cost = cost + abs(modified_list[i] - original_list[i])
    return cost

def plotGraph(perfRatioOptimised):
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        k = perfRatioOptimised.keys()
        R = perfRatioOptimised.values()
        X = np.arange(len(perfRatioOptimised.keys()))
        plt.xlabel("Degree K")
        plt.ylabel("Performance Ratio")
        ax.bar(k,R)
        plt.show()
        ax.bar(k,R)
        plt.show()  

def put_in_same_group(sequence):
    newSequence = []
    for _ in range(len(sequence)):
        newSequence.append(sequence[0])
    cost = calcualte_cost(sequence, newSequence)
    return cost, newSequence


def dynamic_programing_graph_anonymization(k, degreeSequence, initialPosition=None):
    numberOfNodes = len(degreeSequence)
    anonymizedDegreeSequences = []
    anonymizationCosts = []
    if k > numberOfNodes:
        raise Exception("Value of K can not be more than number of Nodes")

    elif k > numberOfNodes / 2:
        return put_in_same_group(degreeSequence)

    else:
        for chunk in range( k, numberOfNodes - k+1):
            toAnonymize = degreeSequence[0: chunk]
            inSameGroup = degreeSequence[chunk: numberOfNodes]
            costAnonymize, sequenceAnonymized = dynamic_programing_graph_anonymization(k, toAnonymize)
            costSameGroup, sameGroupSquence = put_in_same_group(inSameGroup)
            chunkCost = costAnonymize + costSameGroup
            anonymizedChunk = sequenceAnonymized+sameGroupSquence
            anonymizationCosts.append(chunkCost)
            anonymizedDegreeSequences.append(anonymizedChunk)
        
        return min(anonymizationCosts), anonymizedDegreeSequences[anonymizationCosts.index(min(anonymizationCosts))]

def dynamic_programing_graph_anonymization_opt(k, degreeSequence, initialPosition=None):

    numberOfNodes = len(degreeSequence)
    anonymizedDegreeSequences = []
    anonymizationCosts = []
    if k > numberOfNodes:
        raise Exception("Value of K can not be more than number of Nodes")

    elif k > numberOfNodes / 2:
        return put_in_same_group(degreeSequence)

    else:
        for chunk in range( max(k , (numberOfNodes + 1) - 2 * k + 1) , numberOfNodes - k+1):
            toAnonymize = degreeSequence[0: chunk]
            inSameGroup = degreeSequence[chunk: numberOfNodes]
            costAnonymize, sequenceAnonymized = dynamic_programing_graph_anonymization_opt(k, toAnonymize)
            costSameGroup, sameGroupSquence = put_in_same_group(inSameGroup)
            chunkCost = costAnonymize + costSameGroup
            anonymizedChunk = sequenceAnonymized+sameGroupSquence
            anonymizationCosts.append(chunkCost)
            anonymizedDegreeSequences.append(anonymizedChunk)
        
        return min(anonymizationCosts), anonymizedDegreeSequences[anonymizationCosts.index(min(anonymizationCosts))]

def compute_I(d):
    d_i = d[0]
    i_value = 0
    for d_j in d:
        i_value += (d_i - d_j)
    return i_value


def c_merge(d, d1, k):
    c_merge_cost = d1 - d[k] + compute_I(d[k+1:min(len(d), 2*k)])
    return c_merge_cost


def c_new(d, k):
    t = d[k:min(len(d), 2*k-1)]
    c_new_cost = compute_I(t)
    return c_new_cost


def greedy_rec_algorithm(array_degrees_greedy, k_degree, pos_init, extension):
    if pos_init + extension >= len(array_degrees_greedy) - 1:
        for i in range(pos_init, len(array_degrees_greedy)):
            array_degrees_greedy[i] = array_degrees_greedy[pos_init]
        return array_degrees

    else:
        d1 = array_degrees_greedy[pos_init]
        c_merge_cost = c_merge(array_degrees_greedy, d1, pos_init + extension)
        c_new_cost = c_new(d, pos_init + extension)
        if c_merge_cost > c_new_cost:
            for i in range(pos_init, pos_init + extension):
                array_degrees[i] = d1
            greedy_rec_algorithm(array_degrees_greedy, k_degree, pos_init + extension, k_degree)
        else:
            greedy_rec_algorithm(array_degrees_greedy, k_degree, pos_init, extension + 1)

if __name__ == "__main__":
        timeArr = []
        file_graph = sys.argv[1]
        perfRatio = {}
        perfRatioOptimised = {}
        G = nx.Graph()
        G = nx.read_gml(file_graph)    
        d = [x[1] for x in G.degree()]#[10,1,2,3,1,2,3,10,9,9,9,9,8,8,8,6,6,4,4,5]
        print(len(d))
        array_index = np.argsort(d)[::-1]
        array_degrees =  np.sort(d)[ ::-1 ]
        degreeSumOriginal = sum(array_degrees)
        print('Original Degrees',array_degrees)

        for k_degree in range (3,20):
            print('K = ',k_degree)

            start = time.time()
            cost, array_degrees_DP = dynamic_programing_graph_anonymization(k_degree, np.sort(d)[ ::-1 ])
            print('time',time.time() - start)
            timeArr.append(time.time() - start)
            degreeSumDynamic = sum(array_degrees_DP)
            print(array_degrees_DP, degreeSumDynamic)

            start = time.time()
            cost, array_degrees_DPO = dynamic_programing_graph_anonymization_opt(k_degree, np.sort(d)[ ::-1 ])
            print('time',time.time() - start)
            timeArr.append(time.time() - start)
            degreeSumDynamicOptimised = sum(array_degrees_DPO) 
            print(array_degrees_DPO, degreeSumDynamicOptimised)

            array_degrees_greedy = array_degrees
            start = time.time()
            greedy_rec_algorithm(np.sort(d)[ ::-1 ], k_degree, 0, k_degree)
            print('time',time.time() - start)
            timeArr.append(time.time() - start)
            degreeSumGreedy = sum(array_degrees_greedy)
            print(array_degrees_greedy, degreeSumGreedy)
            
            graph_DP = construct_graph(array_index , array_degrees_DP)
            if graph_DP is not None:
                        print("Nodes in DPO :{}".format(nx.nodes(graph_DP)))
                        print("Edges in DPO :{}".format(nx.edges(graph_DP)))
            else:
                        print("Cant construct a Graph for Optimized Dynamic Alg")

            graph_DP_optimized = construct_graph(array_index , array_degrees_DPO)
            if graph_DP_optimized is not None:
                        print("Nodes in DPO :{}".format(nx.nodes(graph_DP_optimized)))
                        print("Edges in DPO :{}".format(nx.edges(graph_DP_optimized)))
            else:
                        print("Cant construct a Graph for Optimized Dynamic Alg")
                        
            graph_greedy = construct_graph(array_index, array_degrees_greedy)
            if graph_greedy is not None:
                        print("Nodes in Greedy :{}".format(nx.nodes(graph_greedy)))
                        print("Edges in Greedy :{}".format(nx.edges(graph_greedy)))
            else:
                        print("Cant construct a Graph for Greedy")

            if degreeSumGreedy != 0 and degreeSumDynamic != 0:
                try:
                    perfRatio[k_degree] = (degreeSumGreedy-degreeSumOriginal)/(degreeSumDynamic-degreeSumOriginal)
                except:
                    perfRatio[k_degree] = 0
            if  degreeSumGreedy != 0 and degreeSumGreedy != 0:
                try:
                    perfRatioOptimised[k_degree] = (degreeSumGreedy-degreeSumOriginal)/(degreeSumDynamicOptimised-degreeSumOriginal)
                except:
                    perfRatioOptimised[k_degree] = 0
            
        
        print(perfRatio, perfRatioOptimised)

        plotGraph(perfRatio) 
        plotGraph(perfRatioOptimised) 
        
        


       