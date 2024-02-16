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


import copy

import numpy as np
from scipy.special import betainc, beta, logsumexp
from queue import PriorityQueue, Queue
from enum import Enum
from irvballot import Ballots, Ballot, perm_decode
import csv
from abc import ABC, abstractmethod
from universal import algos
from math import isclose


class TestMartingale:
    """
    A class to represent a test martingale
    """
    def __init__(self):
        pass

    def process(self, data: np.ndarray, params: tuple) -> (np.ndarray, tuple):
        """
        A method that processes a list of values between 0 (full evidence against) and 1 (full evidence for).
        """
        pass


class Contest:
    """
    A class to represent an IRV contest.

    Attributes
    ----------
    name : str
        Name of contest
    candidates : int
        Number of candidates in the contest (<2)
    winner : int
        reported winner of the contest (winner in range(0,candidates))
    tot_ballots : int
        total number of ballots in the contest, np.inf means sampling with replacement

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """
    def __init__(self, name, n_candidates, winner, tot_ballots):
        self.name = name
        self.winner = winner
        self.n_candidates = n_candidates
        self.tot_ballots = tot_ballots


class Audit:
    def __init__(self, contest: Contest, testmart: TestMartingale, **kwargs):
        self.contest = contest
        self.ballots = Ballots(self.contest.n_candidates)
        self.testmart = testmart

        # Settings TODO: move these to Frontier
        self.risklimit = kwargs.get("risklimit", 0.05)  # TODO Implement multiple risk limits (i.e., a list)
        self.threshold = np.math.log(1/self.risklimit)

    def observe(self, ballot):
        if isinstance(ballot, int):
            self.ballots.observe(ballot)
        elif isinstance(ballot, list):
            self.ballots.observe(self.ballots.to_code(ballot))
        else:
            raise ValueError("Unsupported instance type, argument must be int or list")

    def reset(self):
        self.ballots = Ballots(self.contest.n_candidates)

    def samplesize(self):
        return self.ballots.nballots


class ReqIdent:
    def __init__(self, type: str, first: int = None, second: int = None, rest: frozenset = None):
        if "-" in type:
            t, f, s, r, *_ = type.split("-") + ["", ""]
            self.type = t
            self.first = int(f)
            self.second = int(s)
            self.rest = [int(i) for i in list(r)]
        else:
            self.type = type
            self.first = int(first)
            self.second = int(second)
            self.rest = rest
        assert self.type in ["DB", "DND"], "Error: Malformed Requirement Identity: type not DB or DND"
        assert self.first != self.second, "Error: Malformed Requirement Identity: first == second"
        if self.type == "DND":
            self.rest = None
        else:
            self.rest = frozenset(self.rest)
            assert self.first in self.rest, "Error: Malformed Requirement Identity: first not in rest"
            assert self.second in self.rest, "Error: Malformed Requirement Identity: second not in rest"

    def __eq__(self, other):
        return other.type == self.type \
            and other.first == self.first \
            and other.second == self.second \
            and other.rest == self.rest

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        string = f"{self.type}-{self.first}-{self.second}"
        if self.type == "DB":
            string += f"-{''.join([str(i) for i in self.rest])}"
        return string

    def isDB(self):
        return self.type == "DB"

    def isDND(self):
        return self.type == "DND"

    def true_idents_if_false(self):
        if self.type == "DB":
            return [ReqIdent("DB", self.second, self.first, self.rest), ReqIdent("DND", self.first, self.second)]
        else:
            return []


class Requirement:
    """
    A class to represent a requirement

    Attributes
    ----------
    contest
    winner : int
        The candidate that is statistically tested if they beat `loser`
    loser : int
        The candidate that is statistically tested if they are beaten by `winner`
    remaining : np.ndarray
        List of who is remaining in the race
    remaining_loser : np.ndarray
        Used for asymmetrical requirements. List of who is remaining in the race for the loser. If used, then
        `remaining` denotes who is remaining in the race for the winner.
    """
    def __init__(self, audit: Audit, ident: ReqIdent):
        self.audit = audit
        ncand = self.audit.contest.n_candidates
        self.ident = ident
        if self.ident.isDND():
            self.elim_fst = frozenset({})
            self.elim_snd = frozenset(c for c in range(ncand) if c not in {self.ident.first, self.ident.second})
            self.first = self.ident.first
            self.second = self.ident.second
        else:  # self.ident.isDB():
            self.elim_fst = frozenset({c for c in range(ncand) if c not in self.ident.rest})
            self.elim_snd = frozenset({c for c in range(ncand) if c not in self.ident.rest})
            self.first = self.ident.second
            self.second = self.ident.first
        self.values = np.array([0])  # running base martingale (log)
        self.n_values_forgotten = 0
        self.watchers = set()
        self.n_processed = 0
        self.params: tuple = self.audit.testmart.init_params()
        self.rmax = 0
        self.proven_true = False  # martingale has reached 0 (-inf log scale)

    # process ballots
    def process(self, history_size: int):
        if self.proven_true:
            return 1
        # n_processed = len(self.processed)
        n_unprocessed = self.audit.ballots.nballots - self.n_processed
        if n_unprocessed == 0:
            return 0
        unprocessed = self.audit.ballots.to_bernoulli_asym(self.elim_fst, self.first, self.elim_snd,
                                                           self.second, length=n_unprocessed)
        new_values, params = self.audit.testmart.process(np.array(unprocessed), self.params)
        new_values += self.values[-1]  # offset new values
        if -np.inf == new_values[-1]:
            self.proven_true = True
            return 1  # proven true
        previously_below = self.rmax < self.audit.threshold
        self.rmax = max(self.rmax, max(new_values))
        self.params = params
        if len(new_values) == history_size:
            # print(self, "==", self.n_processed, self.n_values_forgotten)
            self.n_values_forgotten += len(self.values)
            self.values = new_values
        elif len(new_values) > history_size:
            # print(self, ">", self.n_processed, self.n_values_forgotten)
            self.n_values_forgotten += len(self.values) + len(new_values[:-history_size])
            self.values = new_values[-history_size:]
        else:
            # print(self, "else", self.n_processed, self.n_values_forgotten)
            size = history_size - len(new_values)
            self.n_values_forgotten += max(0, len(self.values) - size)
            self.values = np.concatenate((self.values[-size:], new_values))
        self.n_processed = self.audit.ballots.nballots
        # print("       ", self.n_processed, self.n_values_forgotten)
        if previously_below and self.rmax >= self.audit.threshold:
            return -1  # rejected (prune inverse requirements)
        return 0  # keep

    def value_at(self, i):
        """
        returns value of the test supermartingale after observing `i` ballots
        returns None if `i` is in the future
        """
        if i <= 0:
            return 0

        assert i >= self.n_values_forgotten, "oops"
        i -= self.n_values_forgotten

        if i >= len(self.values) and not self.proven_true:
            return None
        if i >= len(self.values):
            return -np.inf
        return self.values[i]

    def increment_at(self, i):
        """
        returns value of the test supermartingale after observing `i` ballots
        returns None if `i` is in the future
        """
        if i <= 0:  # No increments yet
            return 0
        # if i == 1:  # Increment from baseline
        #     return self.value_at(1)

        i -= self.n_values_forgotten
        assert i > 0, "oops"

        if self.proven_true:
            if i == len(self.values) - 1:  # Final increment to proven true
                return -np.inf
            if i >= len(self.values):  # No more increments after proven true
                return 0
        if i >= len(self.values):  # No information yet
            return None
        return self.values[i] - self.values[i-1]

    def __repr__(self):
        return self.ident.__repr__()

    def add_watcher(self, node):
        assert node not in self.watchers, f"Error: Duplicate watchers: trying to add Node"# {node.elimination_suffix} to Requirement {self}"
        self.watchers.add(node)

    def remove_watcher(self, node):
        assert node in self.watchers, f"Error: Removing a watcher not in watcher set: Node {node} from Requirement {self}"
        self.watchers.remove(node)

    def prune_requirement(self):
        for node in self.watchers:
            node.unwatch(self, propagate=False)


class RequirementHandler:
    def __init__(self, audit, parking, pruning_thold, pruning, history_size: int):
        self.audit = audit
        self.reqs = dict()  # dict of active requirements
        self.parked = dict()
        self.n_active_requirements = 0
        self.pruned = set()
        self.prune_next = []

        # Settings
        self.parking = parking
        self.pruning_thold = pruning_thold
        self.pruning = pruning
        self.history_size = history_size

    def prune_requirements(self):
        for ident in self.prune_next:
            self.pruned.add(ident)
            if ident in self.reqs.keys():
                req = self.reqs.pop(ident)
                req.prune_requirement()
            if ident in self.parked.keys():
                self.parked.pop(ident)
        self.prune_next = []

    def process(self):
        parking = []
        if self.pruning >= 1:
            self.prune_requirements()
        for req in self.reqs.values():
            if self.parking and len(req.watchers) == 0:
                # Park requirement due to no watchers
                parking.append(req.ident)
                continue
            result = req.process(self.history_size)  # TODO this should return running value?
            if self.pruning >= 2 and result == -1:
                # Rejected: Prune inverses
                self.prune_next += req.ident.true_idents_if_false()
            elif self.pruning >= 1 and result == 1:
                # Mathematically proven true: Prune self
                self.prune_next += [req.ident]

        for ident in parking:
            self.parked[ident] = self.reqs.pop(ident)

    def get(self, ident: ReqIdent):
        if ident in self.reqs.keys():
            return self.reqs[ident]
        if ident in self.parked.keys():
            # Unpark parked req
            self.reqs[ident] = self.parked.pop(ident)
            return self.reqs[ident]
        elif ident in self.pruned:
            return None
        else:
            new_req = Requirement(self.audit, ident)
            self.reqs[ident] = new_req
            new_req.process(self.history_size)
            return new_req


class Weigher(ABC):
    """
    A class to represent  Computes the intersection test martingale of requirements at indices

    Methods
    -------
    compute(indices, requirements):
        Computes the intersection test martingale of requirements at indices
    """
    def __init__(self):
        self.needsMemory = False

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def compute(self, indices: np.ndarray, requirements: set):
        """
        Computes the intersection test martingale of requirements at indices

        Parameters
        ----------
        indices : ndarray[int]
            The indices to compute the intersection test supermatringale for. Index i means the ith ballot observed
        requirements : set[Requirement]
            Requirements whose base test supermartingales to use for selecting weights.
            For every index i in indices, only base test supermartingale values (and other information) at index i-1
            and before can be used to select weights for index i.

        Returns
        -------
        delta : ndarray[int]
            index-by-index change in intersection test supermartingale value
        """
        pass


class Linear(Weigher):
    def compute(self, indices: np.ndarray, requirements: set):
        weights = np.array([[r.value_at(i) for r in requirements] for i in indices-1])
        increments = np.array([[r.increment_at(i) for r in requirements] for i in indices])
        return np.array([logsumexp(increments[t] + weights[t]) - logsumexp(weights[t]) for t in range(len(weights))])


class Quadratic(Weigher):
    def compute(self, indices: np.ndarray, requirements: set):
        weights = np.array([[2*r.value_at(i) for r in requirements] for i in indices-1])
        increments = np.array([[r.increment_at(i) for r in requirements] for i in indices])
        return np.array([logsumexp(increments[t] + weights[t]) - logsumexp(weights[t]) for t in range(len(weights))])


class QuadraticPositive(Weigher):
    def compute(self, indices: np.ndarray, requirements: set):
        assert len(indices) == 1, "UNSUPPORTED! TODO..."
        idx = indices[0]
        weights = np.array([[2*r.value_at(i) for r in requirements] for i in indices-1])[0]
        increments = np.array([[r.increment_at(i) for r in requirements] for i in indices])[0]
        numerators = [incr + weights[i] for i, incr in enumerate(increments) if weights[i] > 0]
        if len(numerators) == 0:
            numerators = increments + weights
            denominators = weights
        else:
            denominators = [w for w in weights if w > 0]
        return np.array([logsumexp(numerators) - logsumexp(denominators)])


class Largest(Weigher):
    def compute(self, indices: np.ndarray, requirements: set):
        weights = []
        for i in indices-1:
            x = np.array([r.value_at(i) for r in requirements])
            weights.append(x == max(x))
        weights = np.array(weights).astype(int)
        increments = np.array([[r.increment_at(i) for r in requirements] for i in indices])
        values = np.array([logsumexp([incr for i, incr in enumerate(increments[t]) if weights[t][i] == 1]) - logsumexp(
            [0] * sum(weights[t])) for t in range(len(weights))])
        # print(f"idx: {indices[0]},\n   prev_values: {x},\n   weights: {weights[0]},\n   increments: {increments[0]},\n   value: {values[0]}")
        return values


class Window(Weigher):
    def __init__(self, window: int, mode):
        super().__init__()
        self.needsMemory = True
        self.initialised = False
        self.ticks = None
        assert isinstance(window, int) and 1 <= window, f"window must be a positive integer"
        self.window = window
        known_modes = ["LinearCount", "LargestCount", "LinearMean", "LargestMean"]
        assert mode in known_modes, f"Unknown mode {mode}, only {modes} are recognised"
        self.mode = mode

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.mode}-{self.window}"

    def compute(self, indices: np.ndarray, requirements: set):
        assert len(indices) == 1, "UNSUPPORTED! TODO..."
        idx = indices[0]
        nreq = len(requirements)
        if not self.initialised:
            if "Count" in self.mode:
                self.ticks = np.array([[1.0]*nreq]*self.window)
            else:
                self.ticks = np.array([[0.0]*nreq]*self.window)
            self.initialised = True
        increments = np.array([r.increment_at(idx) for r in requirements])

        # Calculate Weights
        if "Count" in self.mode:
            x = np.sum(self.ticks[:idx+1], axis=0) / min(idx+1, self.window)
        else:  # Mean
            x = logsumexp(self.ticks[:idx+1], axis=0) - logsumexp(np.zeros(min(idx+1, self.window)))

        if "Largest" in self.mode:
            weights = (x == max(x)).astype(float)
        else:  # Linear
            weights = x

        # value = logsumexp(increments + weights) - logsumexp(weights)
        if "Count" in self.mode:
            numerators = [incr + np.log(weights[i]) for i, incr in enumerate(increments) if weights[i] > 0]
            denomenators = [np.log(weights[i]) for i, incr in enumerate(increments) if weights[i] > 0]
        else:
            numerators = [incr + weights[i] for i, incr in enumerate(increments) if weights[i] > 0]
            if len(numerators) == 0:
                numerators = increments + weights
                denomenators = weights
            else:
                denomenators = [weights[i] for i, incr in enumerate(increments) if weights[i] > 0]
        value = logsumexp(numerators) - logsumexp(denomenators)

        # print(f"idx: {idx},\n   prev_values: {np.array([r.value_at(idx-1) for r in requirements])},\n   weights: {weights}\n   increments: {increments},\n   value: {value}")

        y = np.array([r.value_at(idx) for r in requirements])
        idxmod = idx % self.window
        if "Count" in self.mode:
            self.ticks[idxmod] = (y == max(y)).astype(float)
        else:  # Mean
            self.ticks[idxmod] = y

        return np.array([value])


class ONS(Weigher):
    def __init__(self, delta=0.125, beta=1.0):
        super().__init__()
        self.needsMemory = True
        self.initialised = False
        self.weights = None
        self.algo = algos.ONS(delta=delta, beta=beta)
        self.delta = delta
        self.beta = beta

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.delta}"

    def compute(self, indices: np.ndarray, requirements: set):
        assert len(indices) == 1, "UNSUPPORTED! TODO..."
        idx = indices[0]
        nreq = len(requirements)
        if not self.initialised:
            self.weights = np.ones(nreq) / nreq
            self.algo.init_step(np.matrix([1.0]*nreq))
            self.initialised = True
        increments = np.array([r.increment_at(idx) for r in requirements])
        # value = logsumexp(increments + self.weights) - logsumexp(self.weights)
        value = logsumexp([incr + np.log(self.weights[i]) for i, incr in enumerate(increments) if self.weights[i] > 0]) - logsumexp(
            [np.log(self.weights[i]) for i, incr in enumerate(increments) if self.weights[i] > 0])

        self.weights = self.algo.step(np.exp(np.array([r.value_at(idx) for r in requirements])), self.weights, None)
        return np.array([value])


class EG(Weigher):
    def __init__(self, eta=0.05):
        super().__init__()
        self.needsMemory = True
        self.initialised = False
        self.weights = None
        self.algo = algos.EG(eta=eta)
        self.eta = eta

    def __repr__(self):
        return f"{self.__class__.__name__}-{self.eta}"

    def compute(self, indices: np.ndarray, requirements: set):
        assert len(indices) == 1, "UNSUPPORTED! TODO..."
        idx = indices[0]
        nreq = len(requirements)
        if not self.initialised:
            self.weights = np.ones(nreq) / nreq
            self.initialised = True
        increments = np.array([r.increment_at(idx) for r in requirements])
        # value = logsumexp(increments + self.weights) - logsumexp(self.weights)
        value = logsumexp([incr + np.log(self.weights[i]) for i, incr in enumerate(increments) if self.weights[i] > 0]) - logsumexp(
            [np.log(self.weights[i]) for i, incr in enumerate(increments) if self.weights[i] > 0])

        self.weights = self.algo.step(np.exp(np.array([r.value_at(idx) for r in requirements])), self.weights, None)
        return np.array([value])



class Node:
    def __init__(self, audit, elimination_suffix: np.ndarray, weigher: Weigher, search_heuristic=0):
        self.audit = audit
        self.elimination_suffix = elimination_suffix  # [..., 3rd place, runner-up, winner]
        if weigher.needsMemory:
            self.weigher = copy.deepcopy(weigher)  # TODO: implement inherit function
        else:
            self.weigher = weigher
        self.watchlist = set()  # set of base martingales
        self.value = 0
        self.rmax = 0
        self.base_max = 0
        self.n_ballots_processed = 0
        self.search_heuristic = search_heuristic
        self.score = 0
        self.queue_counter = None

    def __lt__(self, other):
        if isclose(self.score, other.score, abs_tol=0.0001):
            return self.queue_counter.__lt__(other.queue_counter)
        else:
            return self.score.__lt__(other.score)
    # def __eq__(self, other): return self.score.__eq__(other.score)

    def process(self, up_to):
        assert len(self.watchlist) > 0, "Error: No requirements watched! Cannot process node"
        assert up_to > self.n_ballots_processed,  "Error: No new ballots to process"

        values = self.weigher.compute(np.array(range(self.n_ballots_processed+1, up_to+1)), self.watchlist) + self.value
        self.value = values[-1]
        self.rmax = max(max(values), self.rmax)
        self.base_max = max([x.value_at(up_to) for x in self.watchlist])
        self.score = self.base_max

        self.n_ballots_processed = up_to

    # generate children
    def expand(self) -> set:
        ncand = self.audit.contest.n_candidates
        children = []
        for i in range(ncand):  # TODO: add heuristic to expanded nodes? Can we guess which one will be weakest?
            if i in self.elimination_suffix: continue
            child = Node(self.audit, np.insert(self.elimination_suffix, 0, i), self.weigher)
            child.copy(self)
            children.append(child)
        self.prune()  # unwatch all
        return children

    # request requirements
    def request_requirements(self, no_DNDs=False):
        all_cands = frozenset(range(self.audit.contest.n_candidates))
        reqs = []
        remaining = frozenset(self.elimination_suffix)
        eliminated = all_cands - remaining
        newc = self.elimination_suffix[0]
        reqs += [ReqIdent("DB", first=existing, second=newc, rest=remaining) for existing in self.elimination_suffix[1:]]
        if not no_DNDs:
            reqs += [ReqIdent("DND", first=other, second=newc) for other in eliminated]
        return reqs

    def request_all_requirements(self, no_DNDs=False):
        all_cands = frozenset(range(self.audit.contest.n_candidates))
        reqs = []
        for i, newc in enumerate(self.elimination_suffix):
            remaining = frozenset(self.elimination_suffix[i:])
            eliminated = all_cands - remaining
            reqs += [ReqIdent("DB", first=existing, second=newc, rest=remaining) for existing in self.elimination_suffix[i+1:]]
            if no_DNDs:
                continue
            reqs += [ReqIdent("DND", first=other, second=newc) for other in eliminated]
        return reqs

    def add_to_watchlist(self, req: Requirement):
        if req is None:
            return
        assert req not in self.watchlist, f"Error: Adding already watched requirement: Requirement {req} for Node {self}"
        self.watchlist.add(req)
        req.add_watcher(self)

    def unwatch(self, req: Requirement, propagate=True):
        assert req in self.watchlist, f"Error: Removing non-watched requirement attempted: Requirement {req} for Node {self}"
        self.watchlist.remove(req)
        if propagate:
            req.remove_watcher(self)

    def prune(self):
        for req in self.watchlist:
            req.remove_watcher(self)

    def copy(self, node):
        for req in node.watchlist:
            self.add_to_watchlist(req)
        self.value = node.value
        self.rmax = node.rmax
        self.score = node.score
        self.n_ballots_processed = node.n_ballots_processed

    def has_children(self):
        return len(self.elimination_suffix) < self.audit.contest.n_candidates

    def __repr__(self):
        prefix = "["
        if len(self.elimination_suffix) < self.audit.contest.n_candidates:
            prefix += "..."
        return prefix + f"({self.elimination_suffix[0]}) " + " ".join([str(c) for c in self.elimination_suffix[1:]]) + "]"


class Frontier:
    def __init__(self, audit: Audit, weigher: Weigher, **kwargs):
        self.audit = audit
        self.weigher = weigher
        history_size = 3  # FIXME TODO TEMPORARY hardcoded history_size

        # Settings
        self.req_parking = kwargs.get("req_parking", True)
        self.req_pruning_thold = kwargs.get("req_pruning_thold", self.audit.threshold)
        self.req_pruning = kwargs.get("req_pruning", 2)
        self.req_no_dnds = kwargs.get("req_no_dnds", False)
        self.node_full_start = kwargs.get("node_full_start", False)
        self.node_expand_threshold = kwargs.get("node_expand_threshold", -np.inf)
        self.node_expand_every = kwargs.get("node_expand_every", 25)  # TODO: make expansion rule more fluid? e.g. how many, which one etc.

        self.reqs = RequirementHandler(audit, parking=self.req_parking, pruning_thold=self.req_pruning_thold,
                                       pruning=self.req_pruning, history_size=history_size)  # dict of active requirements
        self.nodes = PriorityQueue()  # frontier of active nodes
        self.ballots = np.array([])
        self.n_processed_ballots = 0
        self.escalate = False
        self.queue_counter = 0

    def next_queue_count(self):
        ret = self.queue_counter
        self.queue_counter += 1
        return ret

    # create initial frontier
    def create_frontier(self):
        # TODO: make frontier creation more flexible/parametric
        ncand = self.audit.contest.n_candidates
        if self.node_full_start:
            for seq in range(np.math.factorial(ncand)):
                order = self.audit.ballots.to_list(seq)  # representation is [..., 3rd place, runner-up, winner]
                if order[-1] == self.audit.contest.winner:
                    continue
                node = Node(self.audit, np.array(order), self.weigher)
                new_reqs = node.request_all_requirements(no_DNDs=self.req_no_dnds)
                for new_req in new_reqs:
                    node.add_to_watchlist(self.reqs.get(new_req))
                node.queue_counter = self.next_queue_count()
                self.nodes.put(node)
        else:  # Lazy start
            for i in range(ncand):
                if i == self.audit.contest.winner:
                    continue
                node = Node(self.audit, np.array([i]), self.weigher)
                new_reqs = node.request_requirements(no_DNDs=self.req_no_dnds)
                for new_req in new_reqs:
                    node.add_to_watchlist(self.reqs.get(new_req))
                node.queue_counter = self.next_queue_count()
                self.nodes.put(node)

    #def add_ballots(self, new_ballots: np.ndarray):
    #    self.ballots = np.concatenate( (self.ballots, new_ballots) )

    # process ballots (requirements)
    def process_ballots(self):
        self.reqs.process()
        self.n_processed_ballots = self.audit.ballots.nballots

    def try_expand_node(self, node, processed_nodes, force=False):
        if not node.has_children() and not force:
            return False
        # print("    EXPAND")
        for child in node.expand():
            new_reqs = child.request_requirements()
            for new_req in new_reqs:
                child.add_to_watchlist(self.reqs.get(new_req))
            # print("      Adding", child.elimination_suffix, child.value, child.rmax)
            processed_nodes.append(child)
        return True

    # process nodes?
    def process_nodes(self):
        # print("process_nodes, ballot:", self.audit.ballots.nballots, ",  prev processed:", self.n_processed_ballots)
        processed_nodes = []
        budget = 0
        if self.n_processed_ballots > 0 and self.n_processed_ballots % self.node_expand_every == 0:
            budget += 1
        while not self.nodes.empty():
            node = self.nodes.get()
            # print(" ~ Node:", node, node.score, node.queue_counter, " | ", node.value, node.rmax, max([r.rmax for r in node.watchlist], default=None), len(node.watchlist))

            # Auto-expand if only one child
            if len(node.elimination_suffix) == self.audit.contest.n_candidates - 1:
                # print(" ~ ~ one child only, force expand")
                self.try_expand_node(node, processed_nodes, force=True)
                continue

            # Try auto-expand if no requirements watched
            if len(node.watchlist) == 0:
                # print(" ~ ~ no reqs watched, try expand")
                if self.try_expand_node(node, processed_nodes):
                    # print(" ~ ~ ~ SUCCESS")
                    continue
                else:
                    # print(" ~ ~ ~ FAIL")
                    # Fast-track to full hand count. Can be due to null mathematically proven true or due to
                    # heuristic requirement pruning
                    # TODO: differentiate?
                    self.escalate = True
                    break

            reqs = [(r, r.value_at(self.n_processed_ballots-1), r.increment_at(self.n_processed_ballots)) for r in node.watchlist]
            m = max([r[1] for r in reqs], default=0)
            # x = (" ~~ Processing", node, node.value, node.rmax, max([r.rmax for r in node.watchlist], default=None), len(node.watchlist), [r for r in reqs if r[1]==m])
            # print(" ~~ Processing", node, node.value, node.rmax, max([r.rmax for r in node.watchlist], default=None), len(node.watchlist), [r for r in reqs if r[1]==m])
            # print(" ~ ~ reqs: value at", self.n_processed_ballots-1, " & increment at", self.n_processed_ballots, ":", reqs)

            node.process(self.n_processed_ballots)
            # print(" ~ ~ processed, new value: ", node.value)

            # Check if prune
            if node.value >= self.audit.threshold:
                # print("    PRUNED!")
                # print(node, "PRUNED", node.n_ballots_processed, node.value)
                node.prune()
                continue

            # Check if expand (via threshold)
            if node.base_max <= self.node_expand_threshold:
                # print(" ~ ~ threshold, try expand")
                if self.try_expand_node(node, processed_nodes):
                    # print(" ~ ~ ~ SUCCESS")
                    continue

            # Check if expand (via budget)
            if budget >= 1:
                # print(" ~ ~ budget, try expand...")
                if self.try_expand_node(node, processed_nodes):
                    # print(" ~ ~ ~ SUCCESS")
                    budget -= 1
                    continue

            # Stash node
            processed_nodes.append(node)

        # x = [(list(n.elimination_suffix), n.value, n.rmax) for n in processed_nodes]
        # print(self.audit.samplesize(), x)
        # y = [([r.rhs, r.lhs], r.value_at(self.audit.samplesize())) for r in self.reqs.reqs.values() if r.watch_count > 0]
        # print(y)

        # Requeue all stashed/processed nodes
        for node in processed_nodes:
            node.queue_counter = self.next_queue_count()
            self.nodes.put(node)

        certified = len(processed_nodes) == 0 and not self.escalate

        return certified