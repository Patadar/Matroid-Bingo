#!/usr/bin/env python3
#Matroid Bingo Module
#PG, 09.04.2025

import matplotlib.pyplot as plt
import numpy as np
#from collections import Counter
from Matroids import Matroid


class Bingo():
    def __init__(self, matroid):
        self.distribution_ran = False
        self.deck = matroid.circuits
        self.deck_size = matroid.circuit_count
        self.card_num = matroid.size

    def play(self):
        done = False
        nums = set()
        while not done:
            rng = np.random.default_rng()
            nums.add(rng.integers(1, high = self.card_num+1))

            for card in self.deck:
                if card <= nums:
                    winner = self.deck.index(card)
                    done = True
        return winner

    def distribution(self, iterations = 100):
        self.iterations = iterations
        self.distribution_ran = True
        self.win_list = [0 for i in range(self.deck_size)]
        
        for iter in range(self.iterations):
            self.win_list[self.play()] += 1

    def distribution_results(self):
        if self.distribution_ran == False:
            self.distribution()
            
        plt.plot([f"C{i}" for i in range(1, self.deck_size+1)], self.win_list, label="temp")
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
        if self.distribution_ran == False:
            self.distribution()

        plt.scatter([f"C{i}" for i in range(1, self.deck_size+1)], np.array(self.win_list)/self.iterations, label="Probabilities")
        av = 1/self.deck_size
        s=0
        for w in self.win_list:
            s += (w/self.iterations-av)**2
        sd = np.sqrt(s/(self.iterations))
        plt.plot([f"C{i}" for i in range(1, self.deck_size+1)], [av for i in range(1, self.deck_size+1)], "--k", label="Average Probability")
        plt.plot([f"C{i}" for i in range(1, self.deck_size+1)], [av+sd for i in range(1, self.deck_size+1)], ":k", label="Standard Deviation")
        plt.plot([f"C{i}" for i in range(1, self.deck_size+1)], [av-sd for i in range(1, self.deck_size+1)], ":k")
        plt.xlabel("Cards")
        plt.ylabel("value")
        plt.title("Win Probability (Change formatting, and line fit)")
        plt.legend()
        plt.show()
        

## Multidimensional bingo???/ abstraction of matroids

if __name__ == "__main__":
    #game = Bingo(300, 300)

    #game.set_deck(*[{i+1} for i in range(300)])
    #game.distribution(iterations=10000)
    #game.distribution_results()
    #game.probability_results()

    #del game.deck
    
    #game.generate_deck()
    #game.distribution(iterations=10000)
    #game.distribution_results()
    #game.probability_results()

    game2 = Bingo(Matroid({1, 2, 3, 4, 5, 6, 7}, [{1, 2, 3, 4},{1, 5, 6},{1, 5, 7}, {1, 5, 6, 7}, {2, 3, 4, 5, 6, 7}]))
    game2.distribution(iterations=1000)
    game2.distribution_results()
    game2.probability_results()

    game3 = Bingo(Matroid.brute_generate(5, 5, method="m", isomorphisms=False))
    print(game3.deck)
    game3.distribution(iterations=1000)
    game3.distribution_results()
    game3.probability_results()

    game4 = Bingo(Matroid.brute_generate(5, 5))
    print(game4.deck)
    game4.distribution(iterations=1000)
    game4.distribution_results()
    game4.probability_results()

