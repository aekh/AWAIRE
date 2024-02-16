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


from alphamart import *
from awaire_utils import *
from timeit import default_timer as timer
import os  # Do not remove
import sys


class AlphaMart(TestMartingale):
    def __init__(self, N, d, eta0):
        super().__init__()
        self.N = N
        self.eta = eta0
        self.d = d
        self.j = 0
        self.S = 1
        self.mu = 0.5
        # params (S0, j0, eta0)

    def init_params(self) -> tuple:
        # TODO include CVRs here
        return 0, 1, self.eta

    def process(self, data: np.ndarray, params: tuple) -> (np.ndarray, tuple):
        increments, params = self.test_list(data, *params)
        return increments, params

    def test_list(self, x, S0, j0, eta0):
        eta0 = 0.52 if eta0 <= 0.5 else eta0
        eta_i, mu_i = shrink_trunc(x, N=self.N, nu=eta0, d=self.d, S0=S0, j0=j0, c=(eta0-0.5)/2)
        res, _ = alpha_mart(np.array(x), mu_i, eta_i)
        # print("   ", np.ma.log(res).filled(-np.inf), (S0, j0, x), (S0 + sum(x), j0 + len(x), eta0))
        return np.ma.log(res).filled(-np.inf), (S0 + sum(x), j0 + len(x), eta0)


def read_election_files(source, d, perm, candincr=0):
    prefix = "NSW2015/Data_NA_"
    # prefix = "USIRV/"
    suffix = ".txt_ballots"
    # suffix = ""
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

        candmap = dict()
        candlist = [i.strip() for i in line.split(",")]
        for i, cand in enumerate(candlist):
            candmap[cand] = i

        ncand = len(candlist) + candincr

        permuter = perm_decode(perm, ncand)

        population = Ballots(ncand)
        file.readline()
        file.readline()
        for line in file:
            f = line.split(" : ")
            # if len(f[0]) == 2: continue  # Handle empty/invalid ballots
            strballot = f[0].split("(")[1].split(")")[0].split(",")
            if len(strballot) == 2 and strballot[1] == '':
                strballot = [strballot[0]]  # compatibility issue fix

            if len(strballot) == 1 and strballot[0] == '':
                ballot_true = []
                ballot = []
            else:
                ballot_true = [candmap[i.strip()] for i in strballot]
                ballot = [permuter[candmap[i.strip()]] for i in strballot]  # Error model
            code = population.to_code(ballot)
            code_true = population.to_code(ballot_true)
            ballotdata += [code_true] * int(f[1])
            for i in range(int(f[1])):
                population.observe(code)
    nballots = len(ballotdata)
    margindata = [None] * population.ncand
    with open(marginfile, "r") as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if csv_reader.line_num == 1: continue
            margindata[int(row[1])] = int(row[2]) / len(ballotdata)
    orderdata = []
    with open(orderfile, "r") as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if csv_reader.line_num == 1: continue
            orderdata.append([int(i) for i in row])
    margindata = np.array(margindata)
    margindata[margindata==None] = -1  # TODO hotfix

    truewin = np.argmax(margindata)

    fakesequence, last_round_margin = population.simulate()
    reported = fakesequence[-1]
    # reported = 2#fakesequence[-1]

    if "pathological" in d:
        maxcand = 3
    else:
        maxcand = ncand

    return ncand, nballots, ballotdata, margindata, orderdata, fakesequence, ballotnumberdiff, truewin, last_round_margin


def func():
    # datafiles = ['Albury', 'Auburn', 'Bankstown', 'Barwon', 'Bathurst', 'Baulkham_Hills', 'Bega',
    #              'Blacktown', 'Blue_Mountains', 'Cabramatta', 'Camden', 'Campbelltown', 'Canterbury', 'Castle_Hill',
    #              'Cessnock', 'Coffs_Harbour', 'Coogee', 'Cootamundra', 'Cronulla', 'Davidson',
    #              'Drummoyne', 'East_Hills', 'Epping', 'Fairfield', 'Gosford', 'Goulburn', 'Granville',
    #              'Heathcote', 'Heffron', 'Holsworthy', 'Hornsby', 'Keira', 'Kiama', 'Kogarah', 'Ku-ring-gai',
    #              'Lakemba', 'Lane_Cove', 'Lismore',
    #              'Liverpool', 'Londonderry',
    #              'Maitland', 'Manly', 'Maroubra', 'Miranda', 'Monaro', 'Mount_Druitt', 'Mulgoa', 'Myall_Lakes',
    #              'Northern_Tablelands', 'Oatley', 'Orange', 'Oxley',
    #              'Pittwater', 'Port_Macquarie', 'Port_Stephens', 'Prospect', 'Riverstone', 'Rockdale', 'Ryde',
    #              'South_Coast', 'Strathfield',
    #              'Terrigal', 'The_Entrance', 'Tweed', 'Upper_Hunter', 'Vaucluse', 'Wagga_Wagga', 'Wakehurst',
    #              'Wallsend', 'Willoughby', 'Wollondilly']
    datafiles = ["Gosford", "The_Entrance", "Lismore",  # Small
                 "Strathfield", "Monaro", "Prospect",  # Medium
                 "Rockdale", "Myall_Lakes", "Oxley",  # Large
                 "Maroubra", "Camden", "Castle_Hill"  # Huge
        ]


    deta = [(10, 0.505), (50, 0.505), (100, 0.505), (200, 0.505), (500, 0.505), (1000, 0.505),
            (10, 0.51),  (50, 0.51),  (100, 0.51),  (200, 0.51),  (500, 0.51),  (1000, 0.51),
            (10, 0.52),  (50, 0.52),  (100, 0.52),  (200, 0.52),  (500, 0.52),  (1000, 0.52),
            (10, 0.54),  (50, 0.54),  (100, 0.54),  (200, 0.54),  (500, 0.54),  (1000, 0.54) ]

    # deta = [(200, 0.51)]

    # deta = [(50, 0.52)]

    source = "./datafiles/"

    print("dataset, setting, reported, true, margin, candidates, weightmode, d, eta, alpha, certified, trialnum, samplesize, samplelimit, dur, stepsize")

    increments = [0]

    counter = 0  # 1-620
    perm = 0  # no errors in CVRs
    for (d, candincr) in ((datafile, candincr) for datafile in datafiles for candincr in increments): # NEXT double loop generator here

        runset = range(0, 100)

        # settings = [Linear(), Quadratic(), Largest(), QuadraticPositive(), Window(window=1, mode="LinearMean"),
        #             ONS(delta=0.66), ONS(delta=1), ONS(delta=2), ONS(delta=4), ONS(delta=8), ONS(delta=16),
        #             ONS(delta=24), ONS(delta=32), ONS(delta=40), ONS(delta=60), ONS(delta=80), ONS(delta=100),
        #             ONS(delta=150), ONS(delta=300),
        #             Window(window=3, mode="LargestCount"), Window(window=5, mode="LargestCount"),
        #             Window(window=7, mode="LargestCount"),
        #             Window(window=3, mode="LinearCount"), Window(window=5, mode="LinearCount"),
        #             Window(window=7, mode="LinearCount"),
        #             Window(window=3, mode="LargestMean"), Window(window=5, mode="LargestMean"),
        #             Window(window=7, mode="LargestMean"),
        #             Window(window=3, mode="LinearMean"), Window(window=5, mode="LinearMean"),
        #             Window(window=7, mode="LinearMean") ]

        settings = [Largest(), QuadraticPositive(), Window(window=5, mode="LargestCount"),
                    Window(window=7, mode="LinearCount")]

        for setting in settings:
            for (alpha_d, eta) in deta:
                counter += 1
                #if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue

                ncand, nballots, ballotdata, margindata, orderdata, csv_elim_seq, ballotnumberdiff, truewin, \
                    last_round_mean = read_election_files(source, d, perm, candincr=candincr)

                w = csv_elim_seq[-1]

                margin = margindata[w]
                contest = Contest("Contest", ncand, w, nballots)
                audit = Audit(contest, AlphaMart(nballots, alpha_d, eta), **{"risklimit": 0.05})

                s = setting
                settings = dict({"req_parking": 1, "req_pruning": 0, "node_full_start": True})

                for r in runset:
                    drawnumber = 0
                    start = timer()
                    audit.reset()
                    # settings["req_no_dnds"] = True
                    frontier = Frontier(audit, s, **settings)

                    frontier.create_frontier()

                    done = False
                    while audit.samplesize() < nballots:
                        sample = ballotdata[orderdata[r][drawnumber] + ballotnumberdiff]
                        drawnumber += 1
                        audit.observe(sample)
                        frontier.process_ballots()
                        done = frontier.process_nodes()
                        if done:
                            break
                    end = timer()
                    dur = end - start
                    # if done:
                    #     # print("CERTIFIED!")
                    # else:
                    #     # print("FULL RECOUNT REACHED! Did not certify")
                    # print("Ballots,  Operations,  Avg Operations/Ballot")
                    print(d, setting, w, truewin, margin, ncand, str(frontier.weigher), alpha_d, eta, audit.risklimit,
                          done, r, audit.samplesize(), nballots, dur, 1, sep=", ")

                    # print(f'{audit.samplesize():,}', f'{audit.operations:,}', f'{audit.operations/audit.samplesize():,}', sep=",  ")
                    # print(f"True sequence: {audit.ballots.simulate()}, reported winner: {w}")
                    # print([audit.ballots.to_list(i) for i in audit.ballots.draworder])
                # sys.exit()


if __name__ == '__main__':
    func()