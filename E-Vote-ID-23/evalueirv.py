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
from irvballot import Ballots, Ballot, perm_decode, perm_encode, equiv_encode, equiv_decode, get_vote
from queue import PriorityQueue
import random
import os  # Do not remove
import sys
import time
from scipy.special import betainc, beta, logsumexp
import csv
from copy import copy, deepcopy

from alphamart import alpha_mart, shrink_trunc, alpha_mart_old
# from agrapamart import agrapa

class BinaryTest:
    def __init__(self):
        pass

    def test(self, ni, nj, ai, aj, a0):
        """ ni: data for i
            ai: prior for i
            a0: initial (uniform) prior """
        pass

    def test_list(self, x, h, S0, j0):
        """ x: list of current data
            h: list of historical data """
        pass

    def test_list2(self, x, h, S0, j0, mj0, sdj20):
        """ x: list of current data
            h: list of historical data """
        pass


class PPRMLR(BinaryTest):
    def __init__(self, mlr: bool = True, a0: float = 1.0):
        super().__init__()
        self.mlr = mlr
        self.a0 = a0  # not used

    def test(self, ni: int, nj: int, ai: int, aj: int, a0i=1, a0j=1):
        """ returns: log ppr mlr: evidence in favour of i beating j """
        a = ai+aj+a0i+a0j
        n = ni+nj
        nai = ni + ai + a0i
        naj = nj + aj + a0j
        factor = np.math.lgamma(a) + np.math.lgamma(nai) + np.math.lgamma(naj) \
                 - np.math.lgamma(ai+a0i) - np.math.lgamma(aj+a0j) - np.math.lgamma(n+a)

        def logbetainc(n1, n2):
            try: return np.math.log(betainc(n1, n2, 0.5))
            except ValueError: return -np.inf  # we get a log(0), interpret as -inf

        if self.mlr:
            prior = logbetainc(aj+a0j, ai+a0i)
            if prior == -np.inf: return -np.inf  # prior has area 0 under H0, return -inf
            mlr_adjust = logbetainc(naj, nai) - prior
        else: mlr_adjust = 0

        point = (ni + nj) * np.math.log(2)  # always test at parameter at 1/2 to retain martingale property
        # return factor + point + mlr_adjust
        return factor + point + mlr_adjust


class AlphaTest(BinaryTest):
    def __init__(self, N=np.inf, eta=0.52, d=50):
        super().__init__()
        self.N = N
        self.eta = eta
        self.d = d

    def test_list(self, x, S0, j0, eta0):
        eta0 = 0.52 if eta0 <= 0.5 else eta0
        eta_i, mu_i = shrink_trunc(x, N=self.N, nu=eta0, d=self.d, S0=S0, j0=j0, c=(eta0-0.5)/2)
        res, _ = alpha_mart(np.array(x), mu_i, eta_i)
        return np.ma.log(res).filled(-np.inf), S0 + sum(x), j0 + len(x)


class AgrapaTest(BinaryTest):
    def __init__(self, N=np.inf, eta=0.52, d=50):
        super().__init__()
        self.N = N
        self.eta = eta
        self.d = d

    def test_list2(self, x, S0, j0, eta0, mj0, sdj20):
        #eta0 = 0.52 if eta0 <= 0.5 else eta0
        #eta_i, mu_i = shrink_trunc(x, N=self.N, nu=eta0, d=self.d, S0=S0, j0=j0, c=(eta0-0.5)/2)
        #res, _ = alpha_mart(np.array(x), mu_i, eta_i)
        lamj, j, mj, sdj2, S = agrapa(None, x0, self.N, j0=j0, S0=S0, mj0=mj0, sdj20=sdj20)
        lamj, j, mj, sdj2, S = agrapa(None, x4, 6000, j0=j[1], S0=S[1], mj0=mj[0], sdj20=sdj2[1])
        return np.ma.log(lamj).filled(-np.inf), S0 + sum(x), j0 + len(x), mj[-1]


class Audit:
    def __init__(self, collection, test: BinaryTest, reported=0, verbose=0, thresholds=[20.0], mode="linear", seed=None, history=None):
        self.batch = collection  # Current batch that we're looking at
        if history is None:
            self.history = Ballots(ncand=collection.ncand)  # Summed up history of all previous batches, used for prior
        else:
            self.history = history
        self.reported = reported
        self.verbose = verbose
        self.test = test
        ncand = self.batch.ncand
        self.eprod = dict()#[0.0]*np.math.factorial(ncand)  Hypothesis E-Process
        self.eprod_max = dict()#[0.0]*np.math.factorial(ncand)  Hypothesis E-Process
        self.rejected = dict()
        # self.eweights = np.array([[0.0]*(ncand * (ncand-1))]*np.math.factorial(ncand))  # Atomic E-Process
        self.eweights = np.array([[0.0]*((ncand * (ncand-1)) // 2)]*np.math.factorial(ncand))  # Atomic E-Process
        if seed is not None:
            for row, _ in enumerate(self.eweights):
                for i, _ in enumerate(self.eweights[row]):
                    if "quadratic" in mode:
                        self.eweights[row,i] = 2*seed[row,i]
                    else:
                        self.eweights[row,i] = seed[row,i]
        # self.eweights_prev = np.array([[0.0]*(ncand * (ncand-1))]*np.math.factorial(ncand))  # Atomic E-Process
        self.eweights_prev = np.array([[0.0]*((ncand * (ncand-1)) // 2)]*np.math.factorial(ncand))  # Atomic E-Process
        self.thresholds = thresholds
        self.mode = mode
        for seq in range(np.math.factorial(ncand)):
            order = self.batch.to_list(seq)  # representation is [..., 3rd place, runner-up, winner]
            if order[-1] == self.reported:
                continue
            self.eprod[seq] = 0.0
            self.eprod_max[seq] = 0.0
        for th in thresholds:
            self.rejected[th] = {seq: False for seq in self.eprod.keys()}
        self.shrink_trunc_saved = dict()
        for seq in self.eprod.keys():
            self.shrink_trunc_saved[seq] = dict()
        self.firstbatch = True

    def audit(self):
        ncand = self.batch.ncand
        n_assert = (ncand * (ncand-1)) // 2
        nsamples = [0] * len(self.thresholds)
        for seq in self.eprod.keys():
            order = self.batch.to_list(seq)  # representation is [..., 3rd place, runner-up, winner]

            # atomic e-values (for each assertion)
            evalues = []
            etas = []
            for i, cand in enumerate(order):
                self.NEN(cand, order[i+1:], evalues, etas, seq)
                # self.NEB(cand, order[:i], evalues, etas, seq)

            evalues = np.array(evalues)
            timerange = range(0, len(evalues[0]))

            #print(seq, evalues)
            if "print" != self.mode:
                # path e-value (for each hypothesis) averaged atomic e-values
                if "max-" in self.mode and not self.firstbatch:
                    emax = max([self.eweights[seq,i] for i, _ in enumerate(evalues)])
                    imaxes = [i for i, _ in enumerate(evalues) if self.eweights[seq,i]==emax]
                    eavg = [logsumexp([evalues[i,t] for i in imaxes])
                            - logsumexp([0 for i in imaxes])
                            for t in timerange]
                elif "max-" in self.mode and self.firstbatch:  # first batch take average of maximum etas
                    emax = max([etas[i] for i, _ in enumerate(evalues)])
                    imaxes = [i for i, _ in enumerate(evalues) if etas[i]==emax]
                    if "no-learn" in self.mode:  # ensure that we stick with the best forever
                        for i in imaxes:
                            self.eweights[seq,i] = 10
                    eavg = [logsumexp([evalues[i,t] for i in imaxes])
                            - logsumexp([0 for i in imaxes]) for t in timerange]
                else:
                    eavg = [logsumexp([evalue[t] + self.eweights[seq,i] for i, evalue in enumerate(evalues)])
                            - logsumexp([self.eweights[seq,i] for i, _ in enumerate(evalues)]) for t in timerange]

            # if seq in [45, 105]:
            #     print(" ->", r, self.samplesize(), seq, order, evalues, self.shrink_trunc_saved[seq], self.batch.to_list, eavg, self.eweights[seq])

            # update weights, atomic e-processes (for each assertion, throughout time)
            if "no-learn" not in self.mode:
                if "linear" in self.mode:
                    for i, evalue in enumerate(evalues):
                        # if self.eweights[seq,i] <= -14:
                        #     self.eweights[seq, i] = -14
                        #     continue
                        self.eweights[seq,i] = self.eweights[seq,i] + evalue[-1]
                        if np.isnan(self.eweights[seq,i]):
                            self.eweights[seq,i] = -np.inf  # keep weight at -inf
                elif "quadratic" in self.mode:
                    for i, evalue in enumerate(evalues):
                        # if self.eweights[seq,i] <= -27:
                        #     self.eweights[seq, i] = -27
                        #     continue
                        self.eweights[seq,i] = self.eweights[seq,i] + 2*evalue[-1]
                        if np.isnan(self.eweights[seq,i]):
                            self.eweights[seq,i] = -np.inf  # keep weight at -inf
                elif "exponential" in self.mode:
                    for i, evalue in enumerate(evalues):
                        # if self.eweights[seq,i] <= -140:
                        #     self.eweights[seq, i] = -140
                        #     continue
                        self.eweights[seq,i] = self.eweights[seq,i] * evalue[-1]
                        if np.isnan(self.eweights[seq,i]):
                            self.eweights[seq,i] = -np.inf  # keep weight at -inf
                if "print" in self.mode:
                    # print(d, w, margin, r, seq, ", ".join([str(e) for e in evalues]), int(self.samplesize()/stepsize), stepsize, sep=", ")
                    # print(self.mode, seq, r, self.batch.nballots + self.history.nballots, "evalues",
                    #       ", ".join([str(e) for e in evalues]), str(self.eprod[seq] + eavg[-1]), sep=", ")
                    print(self.mode, seq, r, self.batch.nballots + self.history.nballots, "weights",
                          ", ".join([str(i) for i in self.eweights[seq]]), str(self.eprod[seq] + eavg[-1]), sep=", ")
                    pass
            elif "no-learn" in self.mode:
                pass
            else:
                raise ValueError(str(self.mode) + " not a valid mode")

            # path e-process (for each hypothesis, throughout time)
            if "print" != self.mode:
                mart = np.array([self.eprod[seq]]) + np.array(eavg)
                for i in range(len(nsamples)):
                    cross = np.argmax(mart >= np.math.log(self.thresholds[i]))
                    if cross == 0 and mart[0] == False:
                        nsamples[i] = self.samplesize()
                        continue
                    else:
                        nsamp = self.history.nballots + cross + 1
                    if nsamp > nsamples[i]:
                        nsamples[i] = nsamp
                self.eprod[seq] += eavg[-1]
                self.eprod_max[seq] = max(max(mart), self.eprod_max[seq])

        self.firstbatch = False

        if "print" == self.mode:
            return [False]*4, None

        # # calculate what's rejected
        for seq in list(self.eprod.keys()):
            for th in self.thresholds:
                if self.eprod_max[seq] >= np.math.log(th):
                    self.rejected[th][seq] = True  # store if hypothesis has been rejected
            if self.rejected[max(self.thresholds)][seq]:
                self.eprod.pop(seq)  # remove fully rejected hypotheses
                self.eprod_max.pop(seq)  # remove fully rejected hypotheses

        return [all(self.rejected[th].values()) for th in self.thresholds], min(self.eprod.values(), default=np.math.log(max(self.thresholds))), nsamples
        # return [all(self.rejected[th].values()) for th in self.thresholds], min(self.eprod, key=self.eprod.get, default=np.math.log(max(self.thresholds)))

    def NEN(self, cand:int, remaining:list, evalues:list, etas, seq):
        elim = [c for c in range(self.batch.ncand) if c not in remaining and c != cand]
        for r in remaining:
            if isinstance(self.test, AlphaTest):
                votes = self.batch.to_bernoulli(elim, cand, r)
                index = self.batch.to_code(elim + [cand,r])
                try: (S0, j0, eta0) = self.shrink_trunc_saved[seq][index]
                except KeyError: (S0, j0, eta0) = (0, 1, 0.52)
                (evalue, S0_next, j0_next) = self.test.test_list(votes, S0, j0, eta0)
                self.shrink_trunc_saved[seq][index] = (S0_next, j0_next, eta0)
                evalues.append(evalue)
                etas.append(eta0)
            else:
                votes = self.batch.tallies(elim, include_empty=True)
                hist = self.history.tallies(elim, include_empty=True)
                evalues.append(self.test.test(votes[cand], votes[r], hist[cand], hist[r], a0i=14.5, a0j=35.5))
            # print(" --  -- ", cand, (votes[cand], hist[cand]), r, (votes[r], hist[r]), self.test.test(votes[cand], votes[r], hist[cand], hist[r]))

    def NEB(self, cand:int, others:list, evalues:list, etas, seq):
        # Only works for AlphaTest
        if len(others) == 0: return
        for o in others:
            elim = [c for c in range(self.batch.ncand) if c != o and c != cand]
            votes = self.batch.to_bernoulli_asym([], o, elim, cand)
            index = self.batch.to_code([cand] + elim + [o])  # FIXME this is a hotfix
            try: (S0, j0, eta0) = self.shrink_trunc_saved[seq][index]
            except KeyError:  (S0, j0, eta0) = (0, 1, 0.52)
            (evalue, S0_next, j0_next) = self.test.test_list(votes, S0, j0, eta0)
            self.shrink_trunc_saved[seq][index] = (S0_next, j0_next, eta0)
            # if evalue > 0:
            #     print(cand, "Never Beats", o, evalue, "--", (votes[-1]+S0)/(j0+1), self.batch.to_list(list(self.batch.dict.keys())[0]), votes)
            evalues.append(evalue)
            etas.append(eta0)

    def seed(self, cvrs : Ballots):
        for seq in self.eprod.keys():
            order = cvrs.to_list(seq)  # representation is [..., 3rd place, runner-up, winner]
            for i, cand in enumerate(order):
                remaining = order[i+1:]
                elim = [c for c in range(cvrs.ncand) if c not in remaining and c != cand]
                for r in remaining:
                    assortermean = np.mean(cvrs.to_bernoulli(elim, cand, r))
                    index = cvrs.to_code(elim + [cand,r])
                    self.shrink_trunc_saved[seq][index] = (0, 1, assortermean)

    def seed_copy(self, X):
        self.shrink_trunc_saved = deepcopy(X.shrink_trunc_saved)

    # def NEN2(self, cand:int, remaining:list, evalues:list):
    #     # print(" -- ", cand, remaining)
    #     elim = [c for c in range(self.batch.ncand) if c not in remaining and c != cand]
    #     for r in remaining:
    #         votes = self.batch.tallies(elim, include_empty=True)
    #         hist = self.history.tallies(elim, include_empty=True)
    #         evalues.append(self.test.test(votes[cand]+hist[cand], votes[r]+hist[r], 0, 0, a0=1))
    #         # print(" --  -- ", cand, (votes[cand]+hist[cand],), r, (votes[r]+hist[r],), self.test.test(votes[cand]+hist[cand], votes[r]+hist[r], 0, 0))

    def increment_batch(self, no_reset=False):
        self.history.merge(self.batch, drawlength=stepsize)  # saves heaps of time as we use shrink_trunc_saved
        if no_reset:
            return
        self.batch.reset()

    def increment_phase(self):
        '''
        Forgets/resets history and forgets/resets the cumulative product (E-process) for each hypothesis to 1
        However, the weights are kept
        '''
        self.history.reset()
        self.batch.reset()

        for seq in range(np.math.factorial(ncand)):
            order = self.batch.to_list(seq)  # representation is [..., 3rd place, runner-up, winner]
            if order[-1] == self.reported:
                continue
            self.eprod[seq] = 0.0
            self.eprod_max[seq] = 0.0

        # save carry over weights
        self.eweights_prev = copy(self.eweights)

    def restart_phase(self):
        self.history.reset()
        self.batch.reset()
        self.eweights = copy(self.eweights_prev)

        for seq in range(np.math.factorial(ncand)):
            order = self.batch.to_list(seq)  # representation is [..., 3rd place, runner-up, winner]
            if order[-1] == self.reported:
                continue
            self.eprod[seq] = 0.0
            self.eprod_max[seq] = 0.0

    def samplesize(self):
        return self.batch.nballots + self.history.nballots

    def reset(self):
        self.batch.reset()
        self.history.reset()
        self.eprod = dict()
        self.eprod_max = dict()
        self.rejected = dict()
        # self.eweights = np.array([[0.0]*(ncand * (ncand-1))]*np.math.factorial(ncand))  # Atomic E-Process
        self.eweights = np.array([[0.0]*((ncand * (ncand-1)) // 2)]*np.math.factorial(ncand))  # Atomic E-Process
        # self.eweights_prev = np.array([[0.0]*(ncand * (ncand-1))]*np.math.factorial(ncand))  # Atomic E-Process
        self.eweights_prev = np.array([[0.0]*((ncand * (ncand-1)) // 2)]*np.math.factorial(ncand))  # Atomic E-Process
        for seq in range(np.math.factorial(ncand)):
            order = self.batch.to_list(seq)  # representation is [..., 3rd place, runner-up, winner]
            if order[-1] == self.reported:
                continue
            self.eprod[seq] = 0.0
            self.eprod_max[seq] = 0.0
        for th in self.thresholds:
            self.rejected[th] = {seq: False for seq in self.eprod.keys()}
        for seq in self.eprod.keys():
            for index in self.shrink_trunc_saved[seq].keys():
                (_, _, eta0) = self.shrink_trunc_saved[seq][index]
                self.shrink_trunc_saved[seq][index] = (0, 1, eta0)
        self.firstbatch = True


if __name__ == '__main__':
    rng = np.random.default_rng()
    datafiles = ["pathological_C_06cand_m005", "pathological_C_06cand_m050", "pathological_C_06cand_m500",
                 'Albury', 'Auburn', 'Bankstown', 'Barwon', 'Bathurst', 'Baulkham_Hills', 'Bega',
                 'Blacktown', 'Blue_Mountains', 'Cabramatta', 'Camden', 'Campbelltown', 'Canterbury', 'Castle_Hill',
                 'Cessnock', 'Coffs_Harbour', 'Coogee', 'Cootamundra', 'Cronulla', 'Davidson',
                 'Drummoyne', 'East_Hills', 'Epping', 'Fairfield', 'Gosford', 'Goulburn', 'Granville',
                 'Heathcote', 'Heffron', 'Holsworthy', 'Hornsby', 'Keira', 'Kiama', 'Kogarah', 'Ku-ring-gai',
                 'Lakemba', 'Lane_Cove', 'Lismore',
                 'Liverpool', 'Londonderry',
                 'Maitland', 'Manly', 'Maroubra', 'Miranda', 'Monaro', 'Mount_Druitt', 'Mulgoa', 'Myall_Lakes',
                 'Northern_Tablelands', 'Oatley', 'Orange', 'Oxley',
                 'Pittwater', 'Port_Macquarie', 'Port_Stephens', 'Prospect', 'Riverstone', 'Rockdale', 'Ryde',
                 'South_Coast', 'Strathfield',
                 'Terrigal', 'The_Entrance', 'Tweed', 'Upper_Hunter', 'Vaucluse', 'Wagga_Wagga', 'Wakehurst',
                 'Wallsend', 'Willoughby', 'Wollondilly']
    source = "./datafiles/"

    alphas = [0.25, 0.10, 0.05, 0.01]
    maxsample = 2500
    stepsize = 25

    # For correct (or no) CVRs
    print("dataset, reported, true, margin, candidates, weightmode, alpha, certified, trialnum, samplesize, samplelimit, stepsize, mineval")
    
    # For permuted CVRs
    # print("dataset, reported, true, perm, seq, margin, candidates, weightmode, alpha, certified, trialnum, samplesize, samplelimit, stepsize, mineval")

    # stattest = PPRMLR(mlr=True, a0=1.0)
    stattest = AlphaTest()
    stattest2 = AlphaTest()

    d = datafiles[0]

    counter = 0  # 1-1200
    perm = 0  # no errors in CVRs
    # for perm in range(np.math.factorial(5)):  # <-- Use this for permuted CVRs
    for d in datafiles:  # <-- Use this for correct (or no) CVRs
        # counter += 1
        # if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue

        prefix = "NSW2015/Data_NA_"
        suffix = ".txt_ballots"
        ballotnumberdiff = -1
        if "pathological" in d:
            prefix = "pathological/"
            suffix = ""
            ballotnumberdiff = 0  # hotfix

        ballotfile = source + prefix + d + suffix + ".txt"
        marginfile = source + "margins/" + prefix + d + suffix + ".csv"
        orderfile = source + "orderings/" + prefix + d + suffix + ".csv"
        ballotdata = []

        with open(ballotfile, "r") as file:
            line = file.readline()
            ncand = int(line.split(",")[-1])+1

            # because of performance issues
            if ncand > 6:
                continue

            permuter = perm_decode(perm, ncand)

            population = Ballots(ncand)
            file.readline(); file.readline()
            for line in file:
                f = line.split(" : ")
                # if len(f[0]) == 2: continue  # Handle empty/invalid ballots
                strballot = f[0].split("(")[1].split(")")[0].split(",")
                if len(strballot) == 1 and strballot[0] == '':
                    ballot_true = []
                    ballot = []
                else:
                    ballot_true = [int(i) for i in strballot]
                    ballot = [permuter[int(i)] for i in strballot]  # Error model
                code = population.to_code(ballot)
                code_true = population.to_code(ballot_true)
                ballotdata += [code_true]*int(f[1])
                for i in range(int(f[1])):
                    population.observe(code)
        nballots = len(ballotdata)
        margindata = [None]*population.ncand
        with open(marginfile, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                if csv_reader.line_num == 1: continue
                margindata[int(row[1])] = int(row[2])/len(ballotdata)
        orderdata = []
        with open(orderfile, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                if csv_reader.line_num == 1: continue
                orderdata.append([int(i) for i in row])
        truewin = np.argmax(margindata)

        fakesequence = population.simulate()
        reported = fakesequence[-1]

        if "pathological" in d:
            maxcand = 3
        else:
            maxcand = ncand

        w = reported
        # for w in range(maxcand):  # <-- use for including non-winners as reported winners
        runsets = [range(0, 100), range(100, 200), range(200, 300), range(300, 400), range(400, 500), range(500, 600),
                   range(600, 700), range(700, 800), range(800, 900), range(900, 1000)]
        for runset in runsets:
            margin = margindata[w]
            # if margin <= 0: continue  # ignore -% margin situations 

            # counter += 1
            # if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue
            stattest.N = population.nballots
            stattest2.N = population.nballots
            stattest2.d = 500

            # for r in range(10):
            #     Y.reset()
            #     alphaidx = 0
            #     X = Audit(Y, verbose=0, reported=w, test=pprtest, thresholds=[400], mode="print")
            #     while X.samplesize() < maxsample:
            #         X.increment_batch()
            #         for _ in range(stepsize):
            #             sample = random.sample(ballotdata,1)[0]
            #             Y.observe(sample)
            #         res, mineval = X.audit()

            Xseed = Audit(population, verbose=0, reported=w, test=None, mode="CVR")
            Xseed.seed(population)
            #
            Y = Ballots(ncand)
            YY = Ballots(ncand)

            X1 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest, thresholds=[1 / a for a in alphas],
                       mode="linear")
            X2 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest, thresholds=[1 / a for a in alphas],
                       mode="quadratic")
            X3 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest, thresholds=[1 / a for a in alphas],
                       mode="max-linear-no-cvr")
            X1 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest, thresholds=[1 / a for a in alphas],
                       mode="max-linear")
            X5 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest2, thresholds=[1 / a for a in alphas],
                       mode="max-linear-agg")
            X6 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest, thresholds=[1 / a for a in alphas],
                       mode="max-linear-no-learn")
            X7 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest2, thresholds=[1 / a for a in alphas],
                       mode="max-linear-agg-no-learn")
            # X6 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest, thresholds=[1 / a for a in alphas],
            #            mode="max-linear-inf-no-learn")
            # X4 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest, thresholds=[1 / a for a in alphas],
            #            mode="print-linear")
            # X5 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest, thresholds=[1 / a for a in alphas],
            #            mode="print-quadratic")
            # X6 = Audit(Y, history=YY, verbose=0, reported=w, test=stattest, thresholds=[1 / a for a in alphas],
            #            mode="print-max-linear")

            if margin > 0:
                X1.seed_copy(Xseed)
                X5.seed_copy(Xseed)
                X6.seed_copy(Xseed)
                X7.seed_copy(Xseed)

            for r in runset:
                if margin > 0:
                    Xs = [X1, X2, X3, X4, X5, X6, X7]
                else:
                    Xs = [X1, X2, X3]
                # Xs = [X1, X5, X6, X7]
                Xres = dict()
                for X in Xs:
                    X.reset()
                    Xres[X] = [False] * len(alphas)
                repeat = False
                tics = 0
                tot = 0
                nodes = 0
                alphaidx = 0

                drawnumber = 0
                while X1.samplesize() < population.nballots:  # SAMPLES ALL BALLOTS, OKAY??
                    X1.increment_batch()  # increments all batches
                    for _ in range(min(stepsize, population.nballots-X1.samplesize())):
                        sample = ballotdata[orderdata[r][drawnumber]+ballotnumberdiff]
                        Y.observe(sample)
                        drawnumber += 1
                    rem = []
                    for X in Xs:
                        res, mineval, nsamples = X.audit()
                        Xres[X] = [(not Xres[X][i]) and res[i] for i in range(len(alphas))]
                        if mineval == -np.inf or X1.samplesize() >= population.nballots-23 or any(Xres[X]):  # SAMPLES ALL BALLOTS, OKAY??
                            for i, cert in enumerate(res):
                                # For correct (or no) CVRs
                                print(d, w, truewin, margin, ncand, X.mode, alphas[i], cert, r, nsamples[i],
                                      population.nballots, stepsize, mineval, sep=", ")

                                # For permuted CVRs
                                # print(d, w, truewin, "".join(str(p) for p in permuter),
                                #       "".join(str(f) for f in fakesequence), margin, ncand, X.mode, alphas[i], cert, r,
                                #       nsamples[i], population.nballots, stepsize, mineval, sep=", ")
                                pass
                            if all(res) or mineval == -np.inf:
                                rem.append(X)
                    for X in rem: Xs.remove(X)
            # sys.exit()  # don't waste time with rest of loop