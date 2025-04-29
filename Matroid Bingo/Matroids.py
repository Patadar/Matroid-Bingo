#!/usr/bin/env python3
# Graphic Matroid module
# P, 29.04.2025


# --- Imports and TODO ---


from __future__ import annotations
from typing import Optional, Iterable, Any
import itertools
import random
import time
from math import comb, ceil
import multiprocessing
import numpy as np
from numba import njit
from tqdm import tqdm


# TODO:
# Check isomorphism algorithm
# check generation algorithm
# Better parallel processing
# Random method
# Check Untitled-2.py for any tricks
# Add save generated circuits to file and read file


# --- Debug Utilities ---


def timer() -> float:
    """Returns current high-resolution timestamp for timing."""
    return time.perf_counter()


def timed(
    start: float, 
    message: str,
) -> None:
    """Prints elapsed time since `start` timestamp."""
    print(f"{message} {time.perf_counter() - start}")


def no_time1() -> None:
    """Placeholder: Does nothing."""
    pass


def no_time2(
    start: Any,
    message: Any,
) -> None:
    """Placeholder: Does nothing."""
    pass


# def _filter_chunk_timed(chunk: Iterable[tuple[int, ...]]) -> list:
#     """
#     Main circuit set batch filter, with timing

#     Args:
#         chunk: Iterable object containing batch of circuit sets

#     Returns:
#         List of bitmask circuit sets
#     """
#     start = time.perf_counter()
#     chunk = (np.array(cs, dtype=np.int64) for cs in chunk)
#     chunk = [cs for cs in chunk if is_circuit_set(cs)]
#     print(time.perf_counter()-start)
#     return chunk


# --- Bitmask Utilities ---


@njit
def set_to_bitmask(circuit: set[int]) -> int:
    """
    Convert a circuit to a bitmask representation.

    Args:
        circuit: Set of integers from 1 to size.

    Returns:
        Bitmask where bit at index i is set if (i+1) is in s.
    """
    mask = 0
    for elem in circuit:
        mask |= 1 << (elem - 1)    # Adds bit to mask in position corresponding to element index
    return mask


@njit
def bitmask_to_set(
    mask: int, 
    ground_size: int,
) -> set[int]:
    """
    Convert a bitmask circuit representation to a set.

    Args:
        mask: Bitmask representation.
        ground_size: Total number of ground elements.

    Returns:
        Set of integers corresponding to 1-bits in mask.
    """
    circuit: set[int] = set()
    for i in range(ground_size):
        if mask & (1 << i):       # If Bitmask corresponds to True for element (i+1) in ground set with index i
            circuit.add(i + 1)    # If so add that corresponding element to the circuit set
    return circuit


@njit
def _lsb_index(lsb: int) -> int:
    """
    Return index of the least significant set bit in x.

    Args:
        lsb: The least significant 1 bit of some number

    Returns:
        index of lsb
    """
    # Works by finding bit length of lsb
    idx = 0
    while lsb > 1:
        lsb >>= 1
        idx += 1
    return idx


# --- Circuit Utilities ---


@njit()
def is_circuit_set(circuits: np.ndarray) -> bool:
    """
    Validate that circuits satisfy all axioms.

    Args:
        circuits: 1D array of bitmask circuit representations.

    Returns:
        True if valid circuit set; False otherwise.
    """
    n = circuits.shape[0]
  
    for i in range(n):
        c1 = circuits[i]
        for j in range(i + 1, n):
            c2 = circuits[j]
            common = c1 & c2

            if common == 0:
                continue
            
            # Elimination axiom check
            if common == c1:
                return False
            if common == c2:
                return False
            
            # Exchange axiom check
            union = c1 | c2
            while common:
                lsb = common & -common        # Isolate a shared element
                common ^= lsb                 # Remove that element from common
                union_minus = union & ~lsb    # Remove chosen shared element from union
                found = False
                for k in range(n):
                    if k != i:
                        if k != j: 
                            if (circuits[k] & union_minus) == circuits[k]:
                                found = True
                                break
                if not found:
                    return False
    return True


def _filter_chunk(chunk: Iterable[tuple[int, ...]]) -> list:
    """
    Main circuit set batch filter

    Args:
        chunk: Iterable object containing batch of circuit sets

    Returns:
        List of bitmask circuit sets
    """
    chunk = (np.array(c, dtype=np.int64) for c in chunk)
    return [cs for cs in chunk if is_circuit_set(cs)]


@njit
def are_isomorphic_masks(
    circuits1: np.ndarray, 
    circuits2: np.ndarray, 
    permutations: np.ndarray,
) -> bool:
    """
    Check if two circuit bitmask representation sets are isomorphic under any permutation.

    Args:
        circuits1: 1D array of masks for the first set.
        circuits2: 1D array of masks for the second set.
        permutations: 2D array of shape (p, ground_size) of permutations.

    Returns:
        True if circuits1 can be permuted to match circuits2; False otherwise.
    """
    for i in range(permutations.shape[0]):
        perm = permutations[i]
        mapped = np.empty_like(circuits1)

        for c in range(circuits1.shape[0]):
            circuit = circuits1[c]
            mapped_circuit = 0

            while circuit:
                lsb = circuit & -circuit 
                circuit ^= lsb
                pos = _lsb_index(lsb)
                mapped_circuit |= 1 << perm[pos]
            mapped[c] = mapped_circuit

        if np.array_equal(np.sort(mapped), np.sort(circuits2)):
            return True
    return False


@njit(parallel=True)
def remove_isomorphic_masks(
    all_circuits: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    """
    Remove duplicate circuit sets up to isomorphism.

    Args:
        all_circuits: 2D array; each row is one set of masks.
        permutations: 2D permutation array.

    Returns:
        2D array of unique circuit sets.
    """
    n = all_circuits.shape[0]

    keep = np.ones(n, dtype=np.uint8)
    for i in range(n):
        if keep[i] == 0:
            continue
        for j in range(i + 1, n):
            if keep[j] and are_isomorphic_masks(all_circuits[i], all_circuits[j], permutations):
                keep[j] = 0

    # collect representatives
    count = np.sum(keep)
    unique_sets = np.empty((count, all_circuits.shape[1]), dtype=all_circuits.dtype)
    index = 0
    for i in range(n):
        if keep[i]:
            unique_sets[index, :] = all_circuits[i]
            index += 1
    return unique_sets


class Matroid:
    """
    Matroid with internal bitmask representation and set-based interface.
    """
    def __init__(
        self, 
        ground_size: int, 
        circuits: tuple[set[int], ...],
    ) -> None:
        """
        Initialize from tuple of circuit sets.

        Args:
            ground_size: Number of elements in ground set.
            circuits: Tuple of sets, each a circuit.
        """
        self._ground_size = ground_size
        self._circuits = np.array(
            [set_to_bitmask(c) for c in circuits],
            dtype=np.int64,
        )
        self._cached_sets: Optional[tuple[set[int], ...]] = None


    def __str__(self) -> str:
        """Readable representation."""
        return (
            f"Ground set: {self.ground_set}, "
            f"Circuits: {self.circuits}"
        )
    

    def __eq__(
        self,
        b: Matroid
    ) -> bool:
        """
        Equality comparison for Matroid: two matroids are equal if their circuit sets match.

        Args:
            b: Another Matroid instance to compare against.

        Returns:
            True if both matroids have identical circuits (as sets), False otherwise.
        """
        return set(map(frozenset, self.circuits)) == set(map(frozenset, b.circuits))
    

    def __iter__(self) -> Iterable[set[int]]:
        """Iterate over circuits as Python sets, one at a time."""
        for i in self.circuits:
            yield i

    
    def __getitem__(
        self, 
        index,
    ) -> set[int]:
        """Get circuit at index: `index`"""
        return self.circuits[index]
    

    def index(
        self, 
        circuit: set
    ) -> int:
        """Get index of a circuit"""
        return self.circuits.index(circuit)


    @property
    def ground_size(self) -> int:
        """Size of the ground set."""
        return self._ground_size


    @property
    def circuit_count(self) -> int:
        """Number of circuits."""
        return self._circuits.size


    @property
    def circuits(self) -> tuple[set[int], ...]:
        """
        Return circuits as tuple of Python sets.
        """
        if self._cached_sets is None:
            self._cached_sets = tuple(
                bitmask_to_set(m, self._ground_size) for m in self._circuits
            )
        return self._cached_sets
    

    @property
    def ground_set(self) -> set[int]:
        """Return ground set as Python set."""
        return set(range(1, self.ground_size + 1))


    @classmethod
    def generate(
        cls,
        ground_size: int,
        circuit_count: int,
        method: str = "m",
        special: int | list[int] | None = None,
        isomorphisms: bool = False,
        batch_size: int = 100_000,
        max_iterations: int = 100_000,
        interactive: bool = True,
        show_time: bool = True,
        #timing_mode: int = 0, docstring: timing_mode: 0 or 1, only if show_time = True, 1 shows bonus timing data.
    ) -> Matroid:
        """
        Generate a Matroid by brute force.

        Methods:
        - 'r': randomly choose a circuit set.
        - 'm': exhaustive search with pruning and isomorphism removal

        Args:
            ground_size: Size of ground set.
            circuit_count: Number of circuits.
            method: "r", or "m".
            special: Restrict circuit sizes.
            isomorphisms: If isomorphisms should be kept.
            max_iterations: Max iterations for random mode.
            interactive: Prompt user in exhaustive mode.
            show_time: show time performance of individual sections.

        Returns:
            (1. Matroid Instance or 2. Set of circuits): 
            1. if method == "r" or if method == "m" and if interactive == True

            2. if method == "m" and if interactive == False
        """
        print()
        if show_time:
            timer_obj = timer
            timed_obj = timed
            message = "Extracted all circuits in:"
            filter_chunk = _filter_chunk
            # if timing_mode == 1:
            #     filter_chunk = _filter_chunk_timed
            #     message = "Extracted all circuits in (Including printing bonus time info):"
            # else:
            #     filter_chunk = _filter_chunk
            #     message = "Extracted all circuits in:"
        else:
            timer_obj = no_time1
            timed_obj = no_time2
            filter_chunk = _filter_chunk
            message = None

        chosen_circuits: tuple[int]

        if method == "r":
            chosen_circuits = Matroid.generate_random(
                ground_size=ground_size, 
                circuit_count=circuit_count,
                special = special,
                max_iterations=max_iterations,
                interactive=interactive,
            )
  
        elif method == "m":
            # build pool of viable sets
            ground_set = range(1, ground_size + 1)
            if isinstance(special, int):
                viable_circuits = itertools.combinations(ground_set, special)
                if show_time:
                    count = comb(ground_size, special)
                    total = comb(count, circuit_count)
            elif isinstance(special, list):
                viable_circuits = itertools.chain.from_iterable(
                    itertools.combinations(ground_set, k) 
                    for k in special
                )
                if show_time:
                    count = sum(comb(ground_size, k) for k in special)
                    total = comb(count, circuit_count)
            else:
                viable_circuits = itertools.chain.from_iterable(
                    itertools.combinations(ground_set, k)
                    for k in ground_set
                )
                if show_time:
                    count = sum(comb(ground_size, k) for k in ground_set)
                    total = comb(count, circuit_count)
            viable_masks = [set_to_bitmask(set(c)) for c in viable_circuits]
            viable_sets = itertools.combinations(viable_masks, circuit_count)

            if interactive:
                print("Generated viable circuit sets")

            #Get circuit sets from viable sets
            start = timer_obj()
            with multiprocessing.Pool() as pool:
                if show_time:
                    results = tqdm(
                        pool.imap_unordered(filter_chunk, itertools.batched(viable_sets, batch_size)),
                        total=ceil(total/batch_size), desc="Processing"
                    )
                else:
                    results = pool.imap_unordered(filter_chunk, itertools.batched(viable_sets, batch_size))
                circuit_sets = []
                for batch in results:
                    circuit_sets.extend(batch)
            timed_obj(start, message)

            if interactive and not show_time:
                print("Found all circuit sets")

            if isomorphisms == False:
                start = timer_obj()
                # Group by sorted circuit sizes to prune comparisons
                groups: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
                for tup in circuit_sets:
                    inv = tuple(sorted(m.bit_count() for m in tup))  # popcounts
                    groups.setdefault(inv, []).append(tup)

                # Precompute all permutations once
                permutations = np.array(
                    list(itertools.permutations(range(ground_size))),
                    dtype=np.int64,
                )

                # Within each group, remove isomorphisms
                unique_tup: list[tuple[int, ...]] = []
                for group in groups.values():
                    masks_arr = np.array(group, dtype=np.int64)
                    uniq = remove_isomorphic_masks(masks_arr, permutations)
                    for i in range(uniq.shape[0]):
                        unique_tup.append(tuple(uniq[i]))

                timed_obj(start, "Removed isomorphisms in:")
                circuit_sets = unique_tup

            print()
            print(f"Found {len(circuit_sets)} circuit sets")
            print()

            if interactive:
                print(f"All circuit sets:")
                for i, cs in enumerate(circuit_sets):
                    print(i, tuple(bitmask_to_set(c, ground_size) for c in cs))
                print()
                choice = int(input("Select index: "))
                chosen_circuits = circuit_sets[choice]
            else:
                return tuple([tuple([bitmask_to_set(c, ground_size) for c in cs]) for cs in circuit_sets]), set(ground_set)
        else:
            raise ValueError(f"Unknown method {method}")

        circuits = tuple(bitmask_to_set(c, ground_size) for c in chosen_circuits)
        return cls(ground_size, circuits)
    
    
    @staticmethod
    def generate_random(        
        ground_size: int,
        circuit_count: int,
        special: int | list[int] | None = None,
        max_iterations: int = 100_000,
        interactive: bool = True,
    ) -> tuple[int]:
        """
        Generate a random circuit.

        Args:
            ground_size: Size of ground set.
            circuit_count: Number of circuits.
            method: "r", or "m".
            special: Restrict circuit sizes.
            max_iterations: Max iterations for random search.
            interactive: Inform user of current state of search.

        Returns:
            A random circuit set of bitmask representations
        """
        NotImplemented

    @staticmethod
    def is_isomorphic(
        ground_size: int,
        c1: Matroid,
        c2: Matroid,
    ) -> bool:
        """Returns True if c1 and c2 are isomorphic"""
        ground_set = range(1, ground_size+1)
        for p in itertools.permutations(ground_set):
            permuted = [set(p[i-1] for i in circuit) for circuit in c1] 
            if set(map(frozenset, permuted)) == set(map(frozenset, c2)):
                return True

        return False


if __name__ == "__main__":
    #print(Matroid.generate(7, 6, method="r"))
    print(Matroid.generate(7, 6, method="m", show_time=True))
    
