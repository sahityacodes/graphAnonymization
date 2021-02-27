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


def greedy_rec_algorithm(array_degrees, k_degree, pos_init, extension):
    if pos_init + extension >= len(array_degrees) - 1:
        for i in range(pos_init, len(array_degrees)):
            array_degrees[i] = array_degrees[pos_init]
        return array_degrees

    else:
        d1 = array_degrees[pos_init]
        c_merge_cost = c_merge(array_degrees, d1, pos_init + extension)
        c_new_cost = c_new(d, pos_init + extension)
        if c_merge_cost > c_new_cost:
            for i in range(pos_init, pos_init + extension):
                array_degrees[i] = d1
            greedy_rec_algorithm(array_degrees, k_degree, pos_init + extension, k_degree)
        else:
            greedy_rec_algorithm(array_degrees, k_degree, pos_init, extension + 1)

if __name__ == "__main__":
        k_degree = int(sys.argv[1])
        file_graph = sys.argv[2]
        G = nx.Graph()
        G = nx.read_gml(file_graph)
        d = [x[1] for x in G.degree()]
        array_index = np.argsort(d)[::-1]
        array_degrees =  np.sort(d)[ ::-1 ]
        degreeSumOriginal = sum(array_degrees)
        start = time.time()
        cost, array_degrees_DP = dynamic_programing_graph_anonymization(k_degree, np.sort(d)[ ::-1 ])
        print('time',time.time() - start)
        degreeSumDynamic = sum(array_degrees_DP)
        print(array_degrees_DP, degreeSumDynamic)
        start = time.time()
        cost, array_degrees_DPO = dynamic_programing_graph_anonymization_opt(k_degree, np.sort(d)[ ::-1 ])
        print('time',time.time() - start)
        degreeSumDynamicOptimised = sum(array_degrees_DPO) 
        print(array_degrees_DP, degreeSumDynamicOptimised)
        array_degrees_greedy = array_degrees
        greedy_rec_algorithm(array_degrees_greedy, k_degree, 0, k_degree)
        degreeSumGreedy = sum(array_degrees_greedy)

        graph_DP = construct_graph(array_index , array_degrees_DP)
        if graph_DP is not None:
                    print("Average Clustering:{}".format(nx.average_clustering(graph_DP)))
        else:
                    print("Cant construct a Graph for Dynamic Alg")

        graph_DP_optimized = construct_graph(array_index , array_degrees_DPO)
        if graph_DP_optimized is not None:
                    print("Average Clustering:{}".format(nx.average_clustering(graph_DP_optimized)))
        else:
                    print("Cant construct a Graph for Optimized Dynamic Alg")
                    
        graph_greedy = construct_graph(array_index, array_degrees_greedy)
        if graph_greedy is not None:
                    print("Average Clustering:{}".format(nx.average_clustering(graph_greedy)))
        else:
                    print("Cant construct a Graph for Greedy")

        if degreeSumGreedy != 0 and degreeSumDynamic != 0:
          print('Performance Ratio of Dynamic Alg R =',str((degreeSumGreedy-degreeSumOriginal)/(degreeSumDynamic-degreeSumOriginal)))
        if degreeSumDynamicOptimised != 0 and degreeSumGreedy != 0:
          print('Performance Ratio of Optimized Dynamic Alg R1 =',str((degreeSumGreedy-degreeSumOriginal)/(degreeSumDynamicOptimised-degreeSumOriginal)))
