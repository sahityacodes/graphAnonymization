def c_merge(d , d1 , k):
    res = d1 - d[ k ] + compute_I(d[ k + 1: min(len(d) , 2 * k) ])
    return res


def c_new(d , k):
    t = d[ k:min(len(d) , 2 * k - 1) ]
    res = compute_I(t)
    return res


def compute_I(d):
    d_i = d[ 0 ]
    res = 0
    for d_j in d:
        res += d_i - d_j
    return res

def greedy_rec_algorithm(array_degrees , k_degree , pos_init , extension):
    # complete this function
    modified_degrees = list(map(lambda x: x, array_degrees))
    #print(modified_degrees)
    if pos_init + extension >= len(modified_degrees) - 1:
        for i in range(pos_init , len(modified_degrees)):
            modified_degrees[ i ] = modified_degrees[ pos_init ]
            return modified_degrees
    else:
        d1 = modified_degrees[ pos_init ]
        c_merge_cost = c_merge(modified_degrees , d1 , pos_init + extension)
        c_new_cost = c_new(array_degrees , pos_init + extension)

        if c_merge_cost > c_new_cost:
            for i in range(pos_init , pos_init + extension):
                modified_degrees[ i ] = d1
            return greedy_rec_algorithm(modified_degrees , k_degree , pos_init + extension , k_degree)
        else:
            return greedy_rec_algorithm(modified_degrees , k_degree , pos_init , extension + 1)
