#!/usr/bin/env python3
# Matroid Bingo Module
# P, 29.04.2025


# --- Imports and TODO ---


import itertools
import matplotlib.pyplot as plt
import numpy as np
from Matroids import Matroid
import math


# TODO:
# Better documentation
# Better graphs and prints
# Refactor class


# --- Main Class ---


class Bingo():
    """Matroid Bingo interface"""
    def __init__(
        self, 
        matroid: Matroid,
    ) -> None:
        """Initialise using Matroid object"""
        self.distribution_ran = False
        self.brute_ran = False
        self.deck = matroid

    def play(self):
        """Play a random game of Bingo"""
        done = False
        nums = set()
        while not done:
            rng = np.random.default_rng()
            nums.add(rng.integers(1, high = self.deck.ground_size+1))

            for card in self.deck:
                if card <= nums:
                    winner = self.deck.index(card)
                    done = True
        return winner
    

    def card_probability(
        self, 
        card_index: int,
    ) -> float:
        """Calculate the probability of a card winning Bingo in a specific Deck"""
        cards = self.deck[:card_index] + self.deck[card_index+1:]
        card = self.deck[card_index]
        unions = []
        for r in range(1, self.deck.circuit_count):
            for combo in itertools.combinations(cards, r):
                union_set = set().union(*combo).union(card)
                unions.append((len(combo), len(union_set)))

        partial = 1/len(card)
        for i in unions:
            partial += (-1)**(i[0])/(i[1])

        probability = len(card)*partial
        print(f"C{card_index+1}: {probability}")
        return probability

    def deck_probability(
            self, 
            graph=True,
    ) -> None:
        """Calculates the probability of winning of each card in the deck, if graph == True, shows a graph overview"""
        probabilities = []

        for i in range(self.deck.circuit_count):
            probabilities.append(self.card_probability(i))

        if graph:
            plt.scatter([f"C{i}" for i in range(1, self.deck.circuit_count+1)], probabilities, label="probabilities")
            plt.xlabel("Cards")
            plt.ylabel("value")
            plt.title("Win Probability (Change formatting, and line fit)")
            plt.legend()
            plt.show()

    def card_distribution(
        self, 
        card_index: int,
    ) -> list[float]:
        """Calculate the probability distribution of a card"""
        cards = self.deck[:card_index] + self.deck[card_index+1:]
        card = self.deck[card_index]
        unions = []

        for r in range(1, self.deck.circuit_count):
            for combo in itertools.combinations(cards, r):
                union_set = set().union(*combo).union(card)
                unions.append((len(combo), len(union_set)))

        distribution = []
        n = self.deck.ground_size
        for i in range(1, n+1):
            if i < len(card):
                distribution.append(0)
            else:
                partial = math.comb(n-len(card), i-len(card)) / math.comb(n-1, i-1)
                for u in unions:
                    if u[1]<=i:
                        partial += (-1)**(u[0]) * (math.comb(n-u[1], i-u[1]) / math.comb(n-1, i-1))
                distribution.append(len(card) * partial / n)
        print(sum(distribution))
        return distribution
    
    def deck_distribution(self) -> None:
        """discrete probability distribution of each card in deck"""
        distributions = []
        rounds = range(1, self.deck.ground_size+1)

        for i in range(self.deck.circuit_count):
            distributions.append(self.card_distribution(i))

        for i in range(self.deck.circuit_count):
            plt.plot(rounds, distributions[i], label=f"C{i+1}")
        plt.xlabel("Round")
        plt.ylabel("Probability")
        plt.title("Win Probability Distribution (Change formatting, and line fit)")
        plt.legend()
        plt.show()

    def deck_cumulative(self) -> None:
        """Cumulative probability distribution of each card in the deck"""
        distributions = []
        cumulative = []
        rounds = range(1, self.deck.ground_size+1)

        for i in range(self.deck.circuit_count):
            distributions.append(self.card_distribution(i))

        for i in range(self.deck.circuit_count):
            cumulative.append([sum(distributions[i][:r+1]) for r in range(self.deck.ground_size)])

        for i in range(self.deck.circuit_count):
            plt.plot(rounds, cumulative[i], label=f"C{i+1}")
        plt.xlabel("Round")
        plt.ylabel("Cummulative Probability")
        plt.title("Win Cummulative Probability Distribution (Change formatting, and line fit)")
        plt.legend()
        plt.show()

    def brute_probability(self):
        """Calculates the probability of winning of each card in the deck by playing every single possible game"""
        self.brute_ran = True
        games = itertools.permutations(self.deck.ground_set)
        self.iterations = 0
        self.win_list = [0 for i in range(self.deck.circuit_count)]
        for i in games:
            self.iterations +=1
            game = set()
            for num in i:
                game.add(num)
                for c in range(self.deck.circuit_count):
                    if set(self.deck[c]) <= game:
                        self.win_list[c] +=1
                        break
                else:
                    continue     
                break

    def distribution(self,
        iterations: int = 100,
    ) -> None:
        """Plays iteration number of random games using Bingo.play"""
        self.iterations = iterations
        self.distribution_ran = True
        self.win_list = [0 for i in range(self.deck.circuit_count)]
        
        for iter in range(self.iterations):
            self.win_list[self.play()] += 1

    def distribution_results(self):
        """Shows graph overview of games played"""
        if self.distribution_ran == False:
            if self.brute_ran == False:
                self.brute_probability()
            
        plt.plot([f"C{i}" for i in range(1, self.deck.circuit_count+1)], self.win_list, label="temp")
        plt.xlabel("Cards")
        plt.ylabel("Wins")
        plt.title("Wins (Change formatting, and line fit)")
        plt.legend()
        plt.show()
        
        #counts, edges = np.histogram(self.win_list, bins=len(set(self.win_list)))
        counts = np.histogram(self.win_list, bins=range(0,max(self.win_list)+2))[0]
        #plt.stairs(counts, range(len(set(self.win_list))+1), fill=True, label="histogram of wins")
        plt.stairs(counts, range(max(self.win_list)+2), fill=True, label="histogram of wins")
        #plt.xticks([0.5+i for i in range(len(set(self.win_list)))], [w for w in sorted(set(self.win_list))], rotation=45)
        plt.xticks([0.5+i for i in range(max(self.win_list)+1)], [None if w not in self.win_list else w for w in range(max(self.win_list)+1)], rotation=90)
        plt.xlabel("Wins")
        plt.ylabel("# of Cards")
        plt.title("Win Distribution")
        plt.legend()
        plt.show()

    def probability_results(self):
        """Shows probability on graph"""
        if self.distribution_ran == False:
            if self.brute_ran == False:
                self.brute_probability()

        plt.scatter([f"C{i}" for i in range(1, self.deck.circuit_count+1)], np.array(self.win_list)/self.iterations, label="Probabilities")
        av = 1/self.deck.circuit_count
        s=0
        for w in self.win_list:
            s += (w/self.iterations-av)**2
        sd = np.sqrt(s/(self.iterations))
        plt.plot([f"C{i}" for i in range(1, self.deck.circuit_count+1)], [av for i in range(1, self.deck.circuit_count+1)], "--k", label="Average Probability")
        plt.plot([f"C{i}" for i in range(1, self.deck.circuit_count+1)], [av+sd for i in range(1, self.deck.circuit_count+1)], ":k", label="Standard Deviation")
        plt.plot([f"C{i}" for i in range(1, self.deck.circuit_count+1)], [av-sd for i in range(1, self.deck.circuit_count+1)], ":k")
        plt.xlabel("Cards")
        plt.ylabel("value")
        plt.title("Win Probability (Change formatting, and line fit)")
        plt.legend()
        plt.show()
        
    @staticmethod
    def all_decks(
        ground_size: int, 
        circuit_count: int, 
        method: str = "m", 
        special: int | list[int] | None =None, 
        isomorphisms: bool = False, 
        batch_size: int = 100_000, 
        max_iterations: int = 100_000, 
        show_time: bool = True, 
        graph: bool = True
    ) -> None:
        """
        Generate all possible decks, find probabilities of each card in each deck, show graphic/print overview.

        Args:
            See Matroid.generate
            graph: If deck.probability should also show a graphic overview of probabilities
        """
        c_set, ground = Matroid.generate(
            ground_size=ground_size,
            circuit_count=circuit_count, 
            method=method, special=special, 
            isomorphisms=isomorphisms, 
            batch_size=batch_size, 
            max_iterations = max_iterations, 
            interactive=False, 
            show_time=show_time
            )
        
        print(f"Ground set: {set(ground)}")
        for c in c_set:
            print()
            game = Bingo(Matroid(len(ground), tuple(c)))
            print(f"Deck: {game.deck.circuits}")
            game.deck_probability(graph=graph)
            game.deck_distribution()
            game.deck_cumulative()

## Multidimensional bingo???/ abstraction of matroids

if __name__ == "__main__":
    # game1 = Bingo(Matroid(7, [{1, 2, 3, 4},{6, 7, 2, 3}, {4, 7, 5}, {1, 5, 6}, {1, 4, 7, 6}, {4, 5, 6, 2, 3}, {1, 5, 7, 2, 3}]))
    # game1.distribution(iterations=1000)
    # game1.distribution_results()
    # game1.probability_results()
    game2 = Bingo(Matroid(11, ({1,2,3,4,5},{6,7,8,9},{6,7,11},{7,8,10},{8,9,11},{9,10,6})))
    game2.deck_probability()
    game2.deck_distribution()
    game2.deck_cumulative()

    # game2 = Bingo(Matroid(17, ({1,2,3,4,5,6},{7,8,9, 10, 11}, {7,12,13},{8,14,15},{9,16,17},{8,9,10,11,12,13},{7,9,10,11,14,15},{7,8,10,11,16,17}, {10,11,12,13,14,15,16,17},{10,11,7,14,15,16,17},{10,11,12,13,8,16,17},{10,11,12,13,14,15,9})))
    # game2.deck_probability()

    # game2 = Bingo(Matroid(7, ({1,2},{3,4,5}, {5,6,7}, {3,4, 6, 7})))
    # game2.deck_probability()

    # game2 = Bingo(Matroid(15, ({1,2,3,4},{5,6,7},{5,8,9},{6,10,11},{7,12,13},{8,9,10,11,12,13},{5,6,12,13},{6,7,8,9},{5,7,10,11},{10,11,8,9,7},{10,11,12,13,5},{8,9,12,13,6},{5,14,15},{14,15,8,9},{14,15,6,12,13},{14,15,7,10,11},{10,11,8,9,7},{10,11,12,13,14,15},{8,9,12,13,6})))
    # game2.deck_probability()


    # game3 = Bingo(Matroid.generate(6, 6, method="m", isomorphisms=False))
    # print(game3.deck)
    # game3.distribution_results()
    # game3.probability_results()
    # game3.deck_probability()
    # game3.deck_distribution
    # game3.deck_cumulative()

    Bingo.all_decks(7, 5, special=[3,4,5,7], graph=False)

    Bingo.all_decks(6, 6, graph=False)

    # game4 = Bingo(Matroid.brute_generate(5, 5))
    # print(game4.deck)
    # game4.distribution(iterations=1000)
    # game4.distribution_results()
    # game4.probability_results()
