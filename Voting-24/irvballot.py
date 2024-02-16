# This file is a part of AWAIRE.
# Copyright (C) 2023 Alexander Ek, Philip B. Stark, Peter J. Stuckey, and
# Damjan Vukcevic
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

def equiv_encode(ballot, ncand):
    """ returns the code corresponding to the given ballot, e.g., [0,1,4] == 2**0 + 2**1 """
    assert all(ballot[i] <= ballot[i+1] for i in range(len(ballot)-1))  # assert sorted
    assert len(ballot) > 0  # assert non-empty
    # ncand = ballot[-1]
    code = sum([2**elem for elem in ballot])
    return code


def equiv_decode(code, ncand):
    """ returns the ballot corresponding to the given code and ncand """
    assert code > 0  # only positive codes
    # assert 0 <= code < 2**(ncand-1)  # assert code is bounded within allowed limits
    binary = [*'{0:b}'.format(code)]
    ballot = []
    # ballot = [ncand-1]
    for idx, elem in enumerate(reversed(binary)):
        if int(elem):
            ballot.insert(len(ballot), idx)
    return ballot


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


def _lehmer_decode(code):
    # TODO: SLOW FIX THIS
    n = len(code)
    for i in reversed(range(n)):
        for j in range(i + 1, n):
            if code[i] <= code[j]:
                code[j] += 1


def _lehmer_decode_fast(code):
    n = len(code)
    take = list(range(n))
    for i in range(n):
        code[i] = take.pop(code[i])


def tallies(coll, elims: list, include_empty=False):
    """
    return: the top preference votes for each candidate given a list of presumed eliminated candidates
    """
    tally = dict()
    if include_empty:
        for i in range(coll.ncand):
            tally[i] = 0

    for ballot in coll.dict.values():
        for c in ballot.to_list():  # find top non-eliminated candidate
            if c not in elims:
                tally[c] = tally.get(c, 0) + ballot.count
                break
    return tally


class Ballots(object):
    def __init__(self, ncand: int):
        self.dict = {}
        self.draworder = []  # order of ballot codes, in the order they were drawn
        self.ncand = ncand
        self.nballots = 0
        self.sectionator = [np.math.factorial(ncand)-1]  # maps included candidates to max ballot number
        for i in range(1,2**ncand):
            incl = ncand - bin(i).count('1')
            self.sectionator.append(self.sectionator[i-1] + np.math.factorial(incl))

    def __getitem__(self, item):
        return self.dict[item]

    def reset(self):
        self.dict = {}
        self.nballots = 0
        self.draworder = []

    def ntype(self):
        return len(self.dict.keys())

    def max_types(self):
        return self.sectionator[-1]

    def ballots(self):
        return self.dict.values()

    def observe(self, code: int, n=1):
        self.draworder += [code]*n
        if code in self.dict.keys():
            self.dict[code].count += n
            self.nballots += n
        else:
            Ballot(self, code)
            self.dict[code].count += n-1
            self.nballots += n-1

    def to_code(self, lst: list):
        excl = sum(2**i for i in range(self.ncand) if i not in lst)
        offset = 0
        if excl != 0:
            offset = self.sectionator[excl-1] + 1
        srt = sorted(lst)
        perm = [srt.index(i) for i in lst]
        return offset + perm_encode(perm)

    def to_list(self, code: int):
        excl = np.where(np.array(self.sectionator) >= code)[0][0]
        incl = excl ^ (2 ** self.ncand - 1)
        binary = [*'{0:b}'.format(incl)]
        cands = []
        for idx, elem in enumerate(reversed(binary)):
            if int(elem):  # == 1
                cands.append(idx)
        offset_code = code
        if excl > 0:
            offset_code -= self.sectionator[excl - 1] + 1
        lst = perm_decode(offset_code, len(cands))
        return [cands[i] for i in lst]

    def get_equiv(self, code: int, order: list):
        ballot = self.to_list(code)
        bequiv = []
        idx = -1
        for elem in ballot:
            if elem in [order[i] for i in range(idx + 1, len(order))]:
                idx = order.index(elem)
                bequiv += [elem]
        return bequiv

    def tallies(self, elims: list, include_empty=False):
        return tallies(self, elims, include_empty)

    def simulate(self):
        """
        Simulates election. returns list of candidates in order of elimination, and last round mean
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
        def func(code):
            ballot = self[code]
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
        def func(code):
            ballot = self[code]
            keepa = True
            keepb = True
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

    def merge(self, ballots, drawlength=-1):
        assert self.ncand == ballots.ncand, "election type mismatch, " + str(self.ncand) + " versus " + str(ballots.ncand) + " number of candidates"
        for newb in ballots.dict.keys():
            self.observe(newb, ballots[newb].count)
        if drawlength >= 0:
            self.draworder = self.draworder[-drawlength:]


class Ballot:
    def __init__(self, collection, code: int):
        assert max(collection.sectionator) >= code >= 0, "invalid ballot number " + str(code) + ". Allowed for " + str(
            collection.ncand) + " candidates are: 0.." + str(max(collection.sectionator))

        self.collection = collection
        self.count = 1
        self.code = code  # ballot type number
        self.excl = np.where(np.array(collection.sectionator) >= code)[0][0]  # excluded candidates in ballot type
        self.incl_set = self._get_cands()

        collection.nballots += 1
        collection.dict[code] = self

    def get_cands(self):
        return self.incl_set

    def _get_cands(self):
        # TODO: Can we use the sectionator for this?
        incl = self.excl ^ (2 ** self.collection.ncand - 1)
        binary = [*'{0:b}'.format(incl)]
        cands = []
        for idx, elem in enumerate(reversed(binary)):
            if int(elem):  # == 1
                cands.append(idx)
        return cands

    def to_list(self):
        # TODO: SLOW FIX THIS
        cands = self.get_cands()
        offset_code = self.code
        if self.excl > 0:
            offset_code -= self.collection.sectionator[self.excl - 1] + 1
        lst = perm_decode(offset_code, len(cands))
        return [cands[i] for i in lst]

    def get_equiv(self, order: list):
        ballot = self.to_list()
        bequiv = []
        idx = -1
        for elem in ballot:
            if elem in [order[i] for i in range(idx + 1, len(order))]:
                idx = order.index(elem)
                bequiv += [elem]
        return bequiv


# X = Ballots(3)
# Y = Ballots(3)
# rng = np.random.default_rng()
#
#
# for _ in range(100):
#     Y.observe(rng.integers(0, Y.max_types()))
#     X.observe(rng.integers(0, X.max_types()))
#
# print(X.nballots, [(v, X[v].count) for v in X.dict.keys()])
#
# X.merge(Y)
#
# print(X.nballots, [(v, X[v].count) for v in X.dict.keys()])

#
# Y = Ballots(3)
#
# for i in range(9999):
#     Y.observe(i)
#     print(i, "&", Y[i].to_list())
#     # if x.code != Ballots.to_code(x.to_list()):
#     #     print("error")
# # # n! + (1 n) (n-1)
#

#
# Y = Ballots(7)
# for i in range(15):
#     print("========================================")
#     Y.observe(i)
#     Y[i].to_list()
#
# print("OK")
# Y = Ballots(5)
# seq = [0,1,2,3,4]
# for i, cand in enumerate(seq):
#     remaining = seq[i+1:]
#     elim = [c for c in range(5) if c not in remaining and c != cand]
#     for r in remaining:
#         print(Y.to_code(elim + [cand,r]), "NEN", elim, [cand, r])
#     others = seq[:i]
#     for o in others:
#         if i == 0: continue
#         elim = [c for c in range(5) if c != o and c != cand]
#         print(Y.to_code([cand] + elim + [o]), "NEB", [cand], elim, [o])