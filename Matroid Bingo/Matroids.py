#!/usr/bin/env python3
#Graphic Matroid module
#PG, 12.04.2025

import itertools
import random
 

class Matroid():
    def __init__(self, ground_list, circuits):
        self.ground_list = ground_list
        self.circuits = tuple(map(frozenset, circuits))

    @property
    def size(self):
        return len(self.ground_list)
    
    @property
    def circuit_count(self):
        return len(self.circuits)
    
    def __str__(self):
        return f"({self.ground_list}, {self.circuits})"
    
    @classmethod
    def brute_generate(cls, size, circuit_count, method="r", special=None, isomorphisms=True, max_iterations = 100000):

        ground_list = range(1, size+1)
        #circuits = [frozenset() for i in range(circuit_count)]

        if type(special) == int and special <= size:
            viable_sets = map(frozenset, itertools.combinations(ground_list, special))

        elif type(special) == list:
            viable_sets = map(
            frozenset,
            itertools.chain.from_iterable(itertools.combinations(ground_list, k) for k in special)
            )
            
        else:
            viable_sets = map(
            frozenset,
            itertools.chain.from_iterable(itertools.combinations(ground_list, k) for k in range(1, size + 1))
            )

        sized_sets = itertools.combinations(list(viable_sets), circuit_count)
        print("Generated Viable Decks")
        match method:
            case "r":
                sized_sets = list(sized_sets)
                iterations = 0
                for i in range(max_iterations):
                    iterations +=1
                    r = random.randint(0, len(sized_sets)-1)
                    if Matroid.is_circuit_set(sized_sets[r]):
                        circuits = sized_sets[r]
                        break
                    else:
                        del sized_sets[r]
                    if iterations == max_iterations:
                        print("Maximum generation iterations reached")
            case "m":
                circuits_set = [s for s in sized_sets if Matroid.is_circuit_set(s)]
                print("Found All Circuits")
                if isomorphisms == False:
                    print("Removing Isomorphisms")
                    iso_set = set()
                    for p in itertools.permutations(ground_list):              
                        for i in range(len(circuits_set)):
                            c1 = circuits_set[i]
                            permuted = [set(p[i-1] for i in circuit) for circuit in c1]
                            for j in range(i+1, len(circuits_set)):
                                c2 = circuits_set[j]            
                                if set(map(frozenset, permuted)) == set(map(frozenset, c2)):
                                    iso_set.add(c2)
                    circuits_set = set(circuits_set) - iso_set
                                
                print("Circuit sets:")
                for s in circuits_set:
                    print(tuple(map(set, tuple(s))))
                circuits = eval(str(input("choose a circuit set")))
            case "":
                NotImplemented
            case _:
                raise Exception("Unknown generation method")
            
        return cls(ground_list, circuits)
    
    @classmethod
    def generate(cls, size, circuit_count, method="r"):

        ground_list = [i for i in range(1, size+1)] 
        circuits = [frozenset() for i in range(circuit_count)]

        match method:
            case "r":
                NotImplemented
            case "m":
                NotImplemented
            case "":
                NotImplemented
            case _:
                raise Exception("Unknown generation method")
            
        return cls(ground_list, circuits)
     
    @staticmethod
    def is_circuit_set(circuits_set):
        circuit_count = len(circuits_set)

        for i in range(circuit_count):
            c_i = circuits_set[i]
            for j in range(i + 1, circuit_count):
                c_j = circuits_set[j]
                if c_i.issubset(c_j) or c_j.issubset(c_i):
                    return False         
        # for i in range(circuit_count):
        #     if any(circuits_set[i].issubset(s) for s in circuits_set[:i]+circuits_set[i+1:]):         
        #         return False
        for i in range(circuit_count):
            c1 = circuits_set[i]
            for j in range(i+1, circuit_count):
                c2 = circuits_set[j]
                common = c1 & c2
                if not common:
                    continue
                union = c1 | c2
                for x in common:
                    union_minus_x = union - {x}
                    if not any(c3 <= union_minus_x for k, c3 in enumerate(circuits_set) if k != i and k != j):
                        return False
                    # found = False
                    # for c3 in circuits_set:
                    #     if c3.issubset(union_minus_x):
                    #         found = True
                    #         break
                    # if not found:
                    #     return False
                        
        return True
    
    @staticmethod
    def is_isomorphic(ground_set, c1, c2):
        for p in itertools.permutations(ground_set):
            permuted = [set(p[i-1] for i in circuit) for circuit in c1] 
            if set(map(frozenset, permuted)) == set(map(frozenset, c2)):
                return True

        return False

if __name__ == "__main__":
    M = Matroid.brute_generate(5, 5)
    print(M.circuit_count)
    print(M.size)
    print(M)

    M = Matroid.brute_generate(5, 5, method="m", special=4)
    print(M.circuit_count)
    print(M.size)
    print(M)

    M = Matroid.brute_generate(5, 5, method="m", special=[2,3,4,5])
    print(M.circuit_count)
    print(M.size)
    print(M)

    M = Matroid.brute_generate(5, 5, method="m", isomorphisms=False)
    print(M.circuit_count)
    print(M.size)
    print(M)
