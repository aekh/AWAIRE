# This file is a part of AWAIRE.
# Copyright (C) 2024 Alexander Ek, Michelle Blom, Philip B. Stark,
#   Peter J. Stuckey, and Damjan Vukcevic
#
# AWAIRE is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see http://www.gnu.org/licenses/.


import numpy as np
import copy
from collections import Counter


def perm_encode(perm):
    """ returns integer representing the permutation """
    x = copy.deepcopy(perm)  # to avoid side effects, FIXME: too slow?
    _lehmer_encode(x)
    n = len(x)
    code = 0
    for i in range(n):
        code += np.math.factorial(n-i-1) * x[i]
    return code


def perm_decode(code, length):
    """ returns the permutation representing the integer """
    # TODO: SLOW FIX THIS
    lcode = []
    for d in reversed(range(1, length+1)):
        if code == 0:
            lcode += [0]*d
            break
        # print(code, length, lcode, d, np.math.factorial(d-1))
        div, code = divmod(code, np.math.factorial(d-1))
        # print(div)
        lcode += [div]
    # print(code, length, lcode)
    # _lehmer_decode(lcode)
    _lehmer_decode_fast(lcode)
    return lcode


def get_vote(ballot, elim):
    """ Returns the remaining first preference after up to candidate elim has been eliminated """
    try:
        return [i for i in ballot if i > elim][0]
    except IndexError:
        return None


def _lehmer_encode(perm):
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] < perm[j]:
                perm[j] -= 1


def _lehmer_decode_fast(code):
    n = len(code)
    take = list(range(n))
    for i in range(n):
        code[i] = take.pop(code[i])


class Ballots(object):
    def __init__(self, ncand: int):
        self.dict = {}  # ballot types and their respective counts
        self.draworder = []  # order of ballots, in the order they were drawn
        self.ncand = ncand
        self.nballots = 0

    def __getitem__(self, item):
        return self.dict[item]

    def reset(self):
        self.dict = {}
        self.nballots = 0
        self.draworder = []

    def ntype(self):
        return len(self.dict.keys())

    def ballots(self):
        return self.dict.values()

    def observe(self, ballot: tuple[int], n=1):
        self.draworder += [ballot]*n
        if ballot in self.dict.keys():
            assert ballot == self.dict[ballot].ballot, f"Ballot type hash collision! {ballot}, {hash(ballot)}, and {self.dict[ballot].ballot}, {hash(self.dict[ballot].ballot)}"
            self.dict[ballot].count += n
            self.nballots += n
        else:
            Ballot(self, ballot)
            self.dict[ballot].count += n-1
            self.nballots += n-1

    def tallies(self, elims: list, include_empty=False):
        tally = dict()
        if include_empty:
            for i in range(self.ncand):
                tally[i] = 0

        for ballot in self.dict.values():
            for c in ballot.to_list():  # find top non-eliminated candidate
                if c not in elims:
                    tally[c] = tally.get(c, 0) + ballot.count
                    break
        return tally

    def simulate(self):
        """
        Simulates election based on sample.

        Returns
        -------
        sequence : array[int]
            resulting elimination sequence
        last_round_mean : float
            assorter mean for the last round
        """
        if self.ncand == 1:
            return [0], 1.0
        sequence = []
        while len(sequence) < self.ncand-2:
            votes = self.tallies(sequence, include_empty=True)
            next = min([v for v in votes if v not in sequence], key=votes.get)
            sequence.append(next)
        # last round
        votes = self.tallies(sequence, include_empty=True)
        votes_ = [(votes[v]) for v in votes if v not in sequence]
        n_winner = max(votes_)
        n_neutral = self.nballots - sum(votes_)
        last_round_mean = (1*n_winner + 0.5*n_neutral)/self.nballots
        loser, winner = sorted([v for v in votes if v not in sequence], key=votes.get)
        sequence.append(loser)
        sequence.append(winner)
        return sequence, last_round_mean

    def to_bernoulli(self, elims: list, a: int, b: int, length=None):
        def func(blt):
            ballot = self[blt]
            for c in ballot.to_list():  # find top non-eliminated candidate
                if c in elims:
                    continue
                elif c == a:
                    return 1  # vote for a
                elif c == b:
                    return 0  # vote for b
                else:
                    break  # not a vote for a or b
            return 0.5

        if length is None:
            res = map(func, self.draworder)
        else:
            res = map(func, self.draworder[-length:])

        return list(res)

    def to_bernoulli_asym(self, elimsa: list, a: int, elimsb: list, b: int, length=None):
        def func(blt):
            keepa = True
            keepb = True
            ballot = self[blt]
            for c in ballot.to_list():  # find top non-eliminated candidate
                if c in elimsa and c in elimsb:
                    continue
                elif c in elimsa:
                    keepb = False
                elif c in elimsb:
                    keepa = False
                elif c == a and keepa:
                    return 1  # vote for a
                elif c == b and keepb:
                    return 0  # vote for b
                else:
                    break  # not a vote for a or b
            return 0.5

        if length is None:
            res = map(func, self.draworder)
        else:
            res = map(func, self.draworder[-length:])

        return list(res)


class Ballot:
    def __init__(self, collection, ballot: tuple[int]):
        assert all(isinstance(i, int) for i in ballot), f"invalid ballot type {ballot}: ballot must contain integers only"
        assert max(ballot, default=0) < collection.ncand, f"invalid ballot type {ballot}: {max(ballot)} is not a candidate in range 0..{collection.ncand-1}"
        assert min(ballot, default=0) >= 0, f"invalid ballot type {ballot}: {min(ballot)} is not a candidate in 0..{collection.ncand-1}"
        assert min(ballot, default=0) >= 0, f"invalid ballot type {ballot}: {min(ballot)} is not a candidate in 0..{collection.ncand-1}"
        assert max(Counter(ballot).values(), default=0) <= 1, f"invalid ballot type: {ballot}: must not contain overvotes"

        self.collection = collection
        self.count = 1
        self.ballot = ballot  # ballot type
        candset = frozenset(range(self.collection.ncand))
        self.incl_set = frozenset(ballot)
        self.excl = candset - self.incl_set
        
        collection.nballots += 1
        collection.dict[ballot] = self

    def get_cands(self):
        return self.incl_set

    def to_list(self):
        return list(self.ballot)


# import irvballot as old
#
# if __name__ == "__main__":
#     X = old.Ballots(5)
#     Y = Ballots(5)
#
#     X.observe(X.to_code([1,2,3]))
#     Y.observe((1,2,3))