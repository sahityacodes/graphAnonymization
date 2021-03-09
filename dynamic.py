def calculateCost(original_list, modified_list):
    cost = 0 
    for original, modified in zip(original_list, modified_list):
        cost += abs(modified - original)
    return cost

def putInSameGroup(sequence):
    anonymized_sequence = [max(sequence)]*len(sequence)
    cost = calculateCost(sequence, anonymized_sequence)
    return cost, anonymized_sequence


# Dynemic programing algorithm with and wothout Optimization
def dpGraphAnonymization(k, degree_sequence, optimization = False):

    number_of_nodes = len(degree_sequence)
    cost_vs_degree_sequences = dict()
    
    if k > number_of_nodes or k <= 0:
        raise Exception("Value of K can not be more than the number of Nodes or less than 0")

    elif k == 1:
        return 0, degree_sequence

    elif k > number_of_nodes / 2:
        return putInSameGroup(degree_sequence)

    else:
        cost_vs_degree_sequences.clear()
        upper_range = number_of_nodes - k+1
        if optimization == False :
            lower_range = k
        
        else: 
            lower_range = max(k, (number_of_nodes - 2*k + 1))
        
        for chunk in range(lower_range, upper_range):
            to_anonymize = degree_sequence[0: chunk]
            in_same_group = degree_sequence[chunk: number_of_nodes]
            cost_anonymize, sequence_anonymized = dpGraphAnonymization(k, to_anonymize, optimization)
            cost_same_group, same_group_squence = putInSameGroup(in_same_group)
            chunk_cost = cost_anonymize + cost_same_group
            anonymized_chunk = sequence_anonymized + same_group_squence
            cost_vs_degree_sequences[chunk_cost] = anonymized_chunk
            #print((chunk, number_of_nodes), anonymized_chunk)
        
        return min(cost_vs_degree_sequences.items(), key = lambda x: x[0])


# Dynamic programing algorithm with Memorization
def memorizedDPGraphAnonymization(k, degree_sequence, optimization = False, cache=dict()):

    if not cache:
        cache = dict()

    number_of_nodes = len(degree_sequence)
    cost_vs_degree_sequences = dict()
    if k > number_of_nodes or k <= 0:
        raise Exception("Value of K can not be more than the number of Nodes or less than 0")
    
    elif k == 1:
        return 0, degree_sequence

    elif k > number_of_nodes / 2:
        return putInSameGroup(degree_sequence)

    else:
        cost_vs_degree_sequences.clear()
        upper_range = number_of_nodes - k+1
        if optimization == False :
            lower_range = k
        
        else: 
            lower_range = max(k, (number_of_nodes - 2*k + 1))
        
        for chunk in range(lower_range, upper_range):
            to_anonymize = degree_sequence[0: chunk]
            in_same_group = degree_sequence[chunk: number_of_nodes]
            if (chunk, number_of_nodes) in cache.keys():
                record_found = cache.get((chunk, number_of_nodes))
                chunk_cost, anonymized_chunk = record_found[0], record_found[1]

            else: 
                cost_anonymize, sequence_anonymized = memorizedDPGraphAnonymization(k, to_anonymize, optimization, cache)
                cost_same_group, same_group_squence = putInSameGroup(in_same_group)
                chunk_cost = cost_anonymize + cost_same_group
                anonymized_chunk = sequence_anonymized + same_group_squence
                cache[(chunk, number_of_nodes)] = (chunk_cost, anonymized_chunk)
            
            cost_vs_degree_sequences[chunk_cost] = anonymized_chunk

        return min(cost_vs_degree_sequences.items(), key = lambda x: x[0])

    