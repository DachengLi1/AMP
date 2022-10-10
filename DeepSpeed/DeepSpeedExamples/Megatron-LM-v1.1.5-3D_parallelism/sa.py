import copy
import time
import os
from collections import defaultdict

import torch
import numpy as np

from amp_utils import factor

class SymDict(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key if key[0] < key[1] else (key[1],key[0]))

    def __setitem__(self, key, value):
        dict.__setitem__(self, key if key[0] < key[1] else (key[1],key[0]), value)

    def __delitem__(self, key):
        return dict.__delitem__(self, key if key[0] < key[1] else (key[1],key[0]))

    def __contains__(self, key):
        return dict.__contains__(self, key if key[0] < key[1] else (key[1],key[0]))

class candidate():
    def __init__(self, h, w, mbs, conf, partition, rank_map):
        self.h = h
        self.w = w
        self.mbs = mbs
        self.conf = conf
        self.partition = partition
        self.rank_map = rank_map

    def __eq__(self, obj):
        #for k, v in self.__dict__.items():
        #    if v != obj.__dict__[k]:
        #        equal = False
        if self.h != obj.h:
            return False

        if self.w != obj.w:
            return False

        if self.mbs != obj.mbs:
            return False

        if (self.conf != obj.conf).any():
            return False

        if self.partition != obj.partition:
            return False

        if self.rank_map != obj.rank_map:
            return False
        
        return True

    def __repr__(self):
        return self.__dict__.__repr__()


def draw_fill(puzzle, patternLength, patternWidth, start, count, solList):
    count += 1
    puzzleLength, puzzleWidth = puzzle.shape
    patternNum = (puzzleWidth*puzzleLength)/(patternWidth*patternLength)
    
    horizonal = False
    if start[0] + patternLength <= puzzleLength and start[1] + patternWidth <= puzzleWidth:
        horizonal = True
        #if (puzzle[start[0]:start[0]+patternLength, start[1]:start[1]+patternWidth] != 0).any():
        for i in range(start[0], start[0]+patternLength):
             for j in range(start[1], start[1]+patternWidth):
                 if puzzle[i][j] != 0:
                     horizonal = False
    if horizonal:
        newPuzzle = copy.deepcopy(puzzle)
        for i in range(start[0], start[0]+patternLength):
            for j in range(start[1], start[1]+patternWidth):
                newPuzzle[i][j] = count
        if count == patternNum:
            solList.append(newPuzzle)
            return
        for i in range(start[0], puzzleLength):
            for j in range(0, puzzleWidth):
                if newPuzzle[i][j] == 0:
                    newStart = (i, j)
                    break
            else:
                continue
            break
        draw_fill(newPuzzle, patternLength, patternWidth, newStart, count, solList)

    vertical = False
    if patternLength != patternWidth and start[0]+patternWidth <= puzzleLength and start[1]+patternLength <= puzzleWidth:
        vertical = True
        for i in range(start[0], start[0]+patternWidth):
            for j in range(start[1], start[1]+patternLength):
                if puzzle[i][j] != 0:
                    vertical = False
    if vertical:
        newPuzzle = copy.deepcopy(puzzle)
        for i in range(start[0], start[0]+patternWidth):
            for j in range(start[1], start[1]+patternLength):
                newPuzzle[i][j] = count
        if count == patternNum:
            solList.append(newPuzzle)
            return
        for i in range(start[0], puzzleLength):
            for j in range(0, puzzleWidth):
                if newPuzzle[i][j] == 0:
                    newStart = (i, j)
                    break
            else:
                continue
            break
        draw_fill(newPuzzle, patternLength, patternWidth, newStart, count, solList)

def backtrack(puzzleLength, puzzleWidth, patternLength, patternWidth, domino):
    if domino:
        patternNum = (puzzleWidth*puzzleLength)/(patternWidth*patternLength)
        solList = []
        if patternNum%1 == 0:
            inputPuzzle = np.zeros((puzzleLength, puzzleWidth))
            draw_fill(inputPuzzle, patternLength, patternWidth, (0, 0), 0, solList)
    #solList = np.asarray(solList).reshape((puzzleLength, puzzleWidth))
    else:
        solList = [ds_config(puzzleLength, puzzleWidth, patternLength, patternWidth)]
    return solList

def cool_down(iter, max_iter, init_temp):
    return init_temp * (1 - iter / max_iter)

def generate_initial(M, N, gbs, threads, domino=True):
    h_w_list = []
    micro_bs_list = [1] * threads
   
    if threads==2: 
        h_w_list.append((M, 1))
        h_w_list.append((1, N))
    elif threads==3: 
        h_w_list.append((M, N))
        h_w_list.append((M, 1))
        h_w_list.append((1, N))
    else:
        # debug mode
        assert threads == 1
        #h_w_list.append((M,1))
        h_w_list.append((M,4))

    known = dict()

    configs = []
    for (h, w) in h_w_list:
        solution = backtrack(M, N, h, w, domino)
        solution_mbs = []
        
        assert (M * N) % (w * h) == 0, "new config not feasible"
        assert gbs % w == 0

        remain = gbs // w
        
        assert len(solution) > 0
        for sol in solution:

            solution_mbs.append((sol, []))
            for r in factor(remain):
                solution_mbs[-1][1].append(r)
 
        config_idx = np.random.choice(list(range(len(solution_mbs))))
        config = solution_mbs[config_idx][0]
        configs.append(config)
        
        solution_mbs[config_idx][1].pop(0)
       
        if len(solution_mbs[config_idx][1]) == 0:
            solution_mbs.pop(config_idx)

        known[(h, w)] = solution_mbs
    
  #  print("known debug", known)
    return h_w_list, micro_bs_list, configs, known
    
def neighbor(cur, gbs, known, M, N, timeout = 1, verbose=True, domino=False):
    h, w, mbs = cur
    assert (M * N) % w == 0 and (M * N) % h == 0, "world size not divisible"
    
    feasible_index = [0, 1]    
    time_s = time.time()
    
    while len(feasible_index) > 0:
        index = np.random.choice(feasible_index, size=1)[0]
        if index == 0:
            valid = []
            
            for i in factor((M*N//w)):
                if i > M: continue # don't set mp too high
                if (i, w) in known.keys():
                    solution_mbs = known[(i, w)]
                else:
                    solution = backtrack(M, N, i, w, domino)
                    solution_mbs = []

                    assert gbs % w == 0

                    remain = gbs // w
                     
                    for sol in solution:
                        solution_mbs.append((sol, []))
                        for r in factor(remain):
                            solution_mbs[-1][1].append(r)

                    known[(i, w)] = solution_mbs

                if len(solution_mbs) > 0:
                    valid.append(i)

            if len(valid) == 0:
                feasible_index.pop(feasible_index.index(0))
                continue
                
            new_h = np.random.choice(valid, size=1)[0]

            # TODO (low)
            new_config_idx = np.random.choice(list(range(len(known[(new_h, w)]))))
            
            ret = known[(new_h, w)][new_config_idx][0]
            
            assert (M * N) % (new_h * w) == 0, "new config not feasible"
            
            new_mbs = np.random.choice(known[(new_h, w)][new_config_idx][1])
           
            new_mbs_idx = known[(new_h, w)][new_config_idx][1].index(new_mbs)
            known[(new_h, w)][new_config_idx][1].pop(new_mbs_idx)

            if len(known[(new_h, w)][new_config_idx][1]) == 0:
                known[(new_h, w)].pop(new_config_idx)
            
            return new_h, w, new_mbs, ret

        else:
            valid = []
            for i in factor((M*N//h)): 
                if gbs % i != 0: continue

                if (h, i) in known.keys():
                    solution_mbs = known[(h, i)]
                else:
                    solution = backtrack(M, N, h, i, domino)

                    assert gbs % i == 0

                    remain = gbs // i
                    solution_mbs = []

                    for sol in solution:
                        solution_mbs.append((sol, []))
                        for r in factor(remain):
                            solution_mbs[-1][1].append(r)

                    known[(h, i)] = solution_mbs
                
                if len(solution_mbs) > 0:
                    valid.append(i)

            if len(valid) == 0:
                feasible_index.pop(feasible_index.index(1))
                continue

            new_w = np.random.choice(valid, size=1)[0]
            new_config_idx = np.random.choice(list(range(len(known[(h, new_w)]))))
            ret = known[(h, new_w)][new_config_idx][0]
            
            assert (M * N) % (new_w * h) == 0, "new config not feasible"
            
            assert gbs % new_w == 0

            new_mbs = np.random.choice(known[(h, new_w)][new_config_idx][1])
            
            new_mbs_idx = known[(h, new_w)][new_config_idx][1].index(new_mbs)
            known[(h, new_w)][new_config_idx][1].pop(new_mbs_idx)
            if len(known[(h, new_w)][new_config_idx][1]) == 0:
                known[(h, new_w)].pop(new_config_idx)

            return h, new_w, new_mbs, ret
    return None

def ds_config(M, N, m, n):
    sol = np.zeros((N, N))
    for i in range(N):
        for j in range(M):
            r = i*M + j
            p = r // (m*n) + 1
            sol[j][i] = p
    rank_map = defaultdict(list)
    pp = np.max(sol)
    rank_counter =  np.zeros(int(pp))
    for j in range(N):
        for k in range(M):
        # TODO: bad code here, config counts from 1
            cur_pp = int(sol[k][j] - 1)
            rank_map[j].append((rank_counter[cur_pp] + cur_pp * m * n))
            rank_counter[cur_pp] += 1    

    for i in range(N):
        assert rank_map[i] == list(range(i*M, (i+1)*M))
    return sol

def megatron_strategy(M, N, gbs, known):
    if known is None:
        known = defaultdict(list)
        ele_count = 0
        for h in factor(M): # mp
            assert M*N % h == 0
            remain = M*N // h
            for w in factor(remain): # dp
                assert gbs % w == 0
                for mbs in factor(gbs // w):
                    ele_count += 1
                    known[mbs].append((h, w))
        print(f"total possible megatron {ele_count}")
    if len(known.keys()) == 0:
        return None

    print(known)
    mbs = list(known.keys())[0]
    (h, w) = known[mbs].pop(0)
    if len(known[mbs]) == 0:
       known.pop(mbs, None)

    return h, w, mbs, known


def amp_no_placement_strategy(M, N, gbs, known):
    if known is None:
        known = defaultdict(list)
        ele_count = 0
        for h in factor(M): # mp
            assert M*N % h == 0
            remain = M*N // h
            for w in factor(remain): # dp
                assert gbs % w == 0
                for mbs in factor(gbs // w):
                    ele_count += 1
                    known[mbs].append((h, w))
        print(f"total possible amp candidates without placment: {ele_count}")
    if len(known.keys()) == 0:
        return None

    #print(known)
    mbs = list(known.keys())[0]
    (h, w) = known[mbs].pop(0)
    if len(known[mbs]) == 0:
       known.pop(mbs, None)

    return h, w, mbs, known

def random_strategy(M, N, gbs, known, domino=True):
    if known is None:
        known = dict()
    
        # generate all possible (h, w)
        h_w_list = []
        print(factor(M), N)
        for i in factor(M):
            for j in factor(N):
                assert M*N % (i*j) == 0
                h_w_list.append((i, j))

        print(f"All possible h_w {h_w_list}")
        
        tot_sol = 0

        for (h, w) in h_w_list:
            solution = backtrack(M, N, h, w, domino)
            solution_mbs = []
        
            assert (M * N) % (w * h) == 0, "new config not feasible"
            assert gbs % w == 0

            remain = gbs // w
        
            assert len(solution) > 0
            for sol in solution:
                assert gbs & w == 0
                max_bs = gbs // w
            
                solution_mbs.append((sol, []))
                for i in factor(max_bs):
                    tot_sol += 1
                    solution_mbs[-1][1].append(i)
 
            known[(h, w)] = solution_mbs

        print(f"RS space is {tot_sol} large!")
    
    # randomly sample one
    deg_id = np.random.choice(list(range(len(known.keys()))))
    deg = list(known.keys())[deg_id]

    h, w = deg

    value = known[deg]

    tup_id = np.random.choice(list(range(len(value))))
    tup = value[tup_id]

    config = tup[0]

    mbs_id = np.random.choice(list(range(len(tup[1]))))
    mbs = tup[1].pop(mbs_id)

    if len(tup[1]) == 0:
        value.pop(tup_id)
        if len(value) == 0:
            known.pop(deg, None)
    
    # convert to rank_map
    rank_map = defaultdict(list)
    pp = np.max(config)
    rank_counter =  np.zeros(int(pp))
    for j in range(N):
        for k in range(M):
        # TODO: bad code here, config counts from 1
            cur_pp = int(config[k][j] - 1)
            rank_map[j].append(int((rank_counter[cur_pp] + cur_pp * h * w)))
            rank_counter[cur_pp] += 1    
    return  h, w, mbs, rank_map, known

