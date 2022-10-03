import copy
import itertools
import time
import math
import numpy as np

def pipe_dp(L, cost_e, cost_c, k, B):
    # Generate all possible max length
    possible = [0]
    
    for i in range(1, L+1):
        ptr = 0
        while ptr + i <= L:
            possible.append(sum(cost_e[ptr:ptr+i]))
            ptr += 1
    
    possible = sorted(list(set(possible)))
    # print(possible)
    # trace will be a 3D list
    trace = []
    for i in range(L):
        outer = []
        for j in range(k):
            inner = []
            for m in range(len(possible)):
                inner.append(([],np.infty))
            outer.append(inner)
        trace.append(outer)
    
    # i: layer id, starting from 0
    # j: number of cut (=GPU-1)
    for i in range(L):
        for j in range(k):
            for m in range(len(possible)):
                if i+1 <= j: # invalid
                    pass
                else:
                    if j == 0: # base case: 0 cut
                        cur_sum = sum(cost_e[:i+1])
                        assert cur_sum in possible
                        trace[i][j][m] = ([i+1], (B-1) * max(0, cur_sum - possible[m]))
                    else:
                        cost_best = np.infty
                        S_best = []
                        for cut in range(j-1, i):
                            cur_sum = sum(cost_e[cut+1:i+1])
                            assert cur_sum in possible
                            S, cost_ = trace[cut][j-1][possible.index(max(cur_sum, possible[m]))]
                            #print(S, cost_)
                            cost_ += (B-1) * max(0, cur_sum - possible[m])
                            cost_ += cost_c[cut][j-1]
                            if cost_ < cost_best:
                                cost_best = cost_
                                S_ = copy.deepcopy(S)
                                S_.append(i-cut)
                                S_best = S_
                        trace[i][j][m] = (S_best, cost_best)
                        
    for i in range(L):
        for j in range(k):
            pass
            #print(i, j, trace[i][j])
    return trace[L-1][k-1][0]

def brute_force(L, cost_e, cost_c, k, B):
    best_S = []
    best_cost = np.infty
    ptr_done = [0] * (k-1)
    possible = list(itertools.combinations(range(L-1), k-1))
    for p in possible:
        p = list(p)
        p.append(L-1)
        lens = [sum(cost_e[:p[0]+1])]
        s = [p[0]+1]
        for i in range(len(p)-1):
            lens.append(sum(cost_e[p[i]+1:p[i+1]+1]))
            s.append(p[i+1]-p[i])     
        max_l = max(lens)
        cost = (B-1) * max_l
        for i in range(k-1):
            cost += cost_c[p[i]][i]
        if cost < best_cost:
            best_cost = cost
            best_S = s
    return best_S, best_cost

def uniform_split(L, cost_e, cost_c, k, B):
    per_stage = L // k
    
    s = [per_stage] * (k-1)
    s.append(L-sum(s))
    p = [s[0]-1]
    for i in range(1, k):
        p.append(p[i-1] + s[i])
    lens = [sum(cost_e[:p[0]+1])]
    for i in range(len(s)-1):
        lens.append(sum(cost_e[p[i]+1:p[i+1]+1]))
    max_l = max(lens)
    cost = (B-1) * max_l
    for i in range(k-1):
        cost += cost_c[p[i]][i]
    return s, cost
