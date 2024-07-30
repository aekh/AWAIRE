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


from alphamart import *
from awaire_utils import *
from timeit import default_timer as timer
import os  # Do not remove
import sys
import gc


class AlphaMart(TestMartingale):
    def __init__(self, N, d, eta0, cvrs=None):
        super().__init__()
        self.N = N
        if cvrs is None:
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
    US_TAGS = "CityCouncil", "Mayor", "CountyAssessor", "CountyExecutive", "CountyAuditor"
    memb = lambda x, y: x in y
    if any(memb(i, d) for i in US_TAGS):
        prefix = "USIRV/"
        suffix = ""
    else:
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
            # code = population.to_code(ballot)
            # code_true = population.to_code(ballot_true)
            # ballotdata += [code_true] * int(f[1])
            # for i in range(int(f[1])):
            #     population.observe(code)
            ballotdata += [tuple(ballot_true)] * int(f[1])
            for i in range(int(f[1])):
                population.observe(tuple(ballot))
    nballots = len(ballotdata)
    margindata = [None] * population.ncand
    nomargin = False
    try:
        with open(marginfile, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                if csv_reader.line_num == 1: continue
                margindata[candmap[row[1].strip()]] = int(row[2]) / len(ballotdata)
    except FileNotFoundError:
        nomargin = True
    orderdata = []
    try:
        with open(orderfile, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                if csv_reader.line_num == 1: continue
                orderdata.append([int(i) for i in row])
    except FileNotFoundError:
        rng = np.random.default_rng()
        orderdata = []
        for _ in range(1000):
            draworder = list(range(1, nballots+1))
            rng.shuffle(draworder)
            orderdata.append(draworder)
    margindata = np.array(margindata)
    margindata[margindata==None] = -1 # TODO hotfix

    fakesequence, last_round_margin = population.simulate()

    if nomargin:
        reported = fakesequence[-1]
        margindata[reported] = last_round_margin - 0.5

    truewin = np.argmax(margindata)

    return ncand, nballots, ballotdata, margindata, orderdata, fakesequence, ballotnumberdiff, truewin, last_round_margin, population


def func():
    datafiles = [
        # NSW 2015
        # "Albury","Auburn","Ballina","Balmain","Bankstown","Barwon","Bathurst","Baulkham_Hills","Bega","Blacktown",
        # "Blue_Mountains","Cabramatta","Camden","Campbelltown","Canterbury","Castle_Hill","Cessnock","Charlestown",
        # "Clarence","Coffs_Harbour","Coogee","Cootamundra","Cronulla","Davidson","Drummoyne","Dubbo","East_Hills",
        # "Epping","Fairfield","Gosford","Goulburn","Granville","Hawkesbury","Heathcote","Heffron","Holsworthy","Hornsby",
        # "Keira","Kiama","Kogarah","Ku-ring-gai","Lake_Macquarie","Lakemba","Lane_Cove","Lismore","Liverpool",
        # "Londonderry","Macquarie_Fields","Maitland","Manly","Maroubra","Miranda","Monaro","Mount_Druitt","Mulgoa",
        # "Murray","Myall_Lakes","Newcastle","Newtown","Northern_Tablelands","North_Shore","Oatley","Orange","Oxley",
        # "Parramatta","Penrith","Pittwater","Port_Macquarie","Port_Stephens","Prospect","Riverstone","Rockdale","Ryde",
        # "Seven_Hills","Shellharbour","South_Coast","Strathfield","Summer_Hill","Swansea","Sydney","Tamworth","Terrigal",
        # "The_Entrance","Tweed","Upper_Hunter","Vaucluse","Wagga_Wagga","Wakehurst","Wallsend","Willoughby",
        # "Wollondilly","Wollongong","Wyong",
        # # USIRV
        # "Aspen_2009_CityCouncil","Berkeley_2010_D1CityCouncil","Berkeley_2010_D7CityCouncil",
        # "Oakland_2010_D4CityCouncil","Oakland_2010_Mayor","Pierce_2008_CountyAssessor","Pierce_2008_CountyExecutive",
        # "Aspen_2009_Mayor","Berkeley_2010_D4CityCouncil","Berkeley_2010_D8CityCouncil","Oakland_2010_D6CityCouncil",
        # "Pierce_2008_CityCouncil","Pierce_2008_CountyAuditor","SanFran_2007_Mayor",
        # USIRV MINNEAPOLIS
        "Minneapolis_2013_Mayor", "Minneapolis_2017_Mayor", "Minneapolis_2021_Mayor"
    ]

    deta = [(200, 0.51, 0.5),  # use d=200, eta=0.51, expand only if at least one child has score above 0.5
            (200, None, 0.5),  # use d=200, eta=LRM (last round margin), ditto
            (200, "AM", 0.5)]  # use d=200. eta=AM (assorter margin), ditto

    source = "../datafiles/"  # datafiles can be downloaded, see README.md

    # for regular experiments; comment out for CVR permutation experiments
    print("dataset, setting, reported, true, margin, candidates, weightmode, d, eta, expcond, alpha, certified, " +
          "trialnum, samplesize, samplelimit, dur, stepsize, " +
          "max_nodes, tot_nodes, avg_nodes, tot_prune, avg_prune_depth, max_depth, max_reqs, tot_reqs, avg_reqs, " +
          "max_req_parked, avg_req_parked, tot_req_prune")

    # uncomment for CVR permutation experiments
    # print("dataset, perm, seq, setting, reported, true, margin, candidates, weightmode, d, eta, expcond, alpha, certified, " +
    #       "trialnum, samplesize, samplelimit, dur, stepsize, " +
    #       "max_nodes, tot_nodes, avg_nodes, tot_prune, avg_prune_depth, max_depth, max_reqs, tot_reqs, avg_reqs, " +
    #       "max_req_parked, avg_req_parked, tot_req_prune")

    increments = [0]  # increments of fake candidates; used for stress testing. 0 is normal operation
    # increments = np.arange(0, 51)  # uncomment for stress testing with up to 51 extra fake candidates

    counter = 0  # 1-4470
    perm = 0
    for (d, candincr) in ((datafile, candincr) for datafile in datafiles for candincr in increments): # NEXT double loop generator here
        # regular experiments
        counter = run_contest(candincr, counter, d, deta, perm, source)

        # errors/permuted CVR experiments; Ballina and Strathfield only; uncomment to run
        # c = 0
        # w = 0
        # ru = 0
        # if d == "Strathfield":
        #     w = 2
        #     ru = 4
        #     c = 5
        # elif d == "Ballina":
        #     w = 4
        #     ru = 2
        #     c = 7
        # for perm in range(1, math.factorial(c)):
        #     seq = perm_decode(perm, c)
        #     last_round_flip = list(range(c))
        #     last_round_flip[w] = ru
        #     last_round_flip[ru] = w
        #     if seq[w] == w or seq == last_round_flip:
        #         counter = run_contest(candincr, counter, d, deta, perm, source, easy=seq[w]==w)


def run_contest(candincr, counter, d, deta, perm, source, easy=False):
    settings = [  # all settings use Largest as underlying weighting scheme
        # "Every40",  # expand every 40
        # "Every25",  # expand every 25
        # "Every10",  # expand every 10
        "IfNegative",  # expand if negative (log scale)
        Largest()  # no incremental expansion
    ]
    for setting in settings:
        for (alpha_d, eta, thres) in deta:
            runsets = [range(0, 200), range(200, 500)]
            for runset in runsets:
                counter += 1
                # if counter != int(os.environ['SLURM_ARRAY_TASK_ID']): continue  # Split jobs on slurm server

                ncand, nballots, ballotdata, margindata, orderdata, csv_elim_seq, ballotnumberdiff, truewin, \
                    last_round_mean, population = read_election_files(source, d, perm, candincr=candincr)

                if eta is None:
                    eta = last_round_mean

                w = csv_elim_seq[-1]

                margin = margindata[w]

                contest = Contest("Contest", ncand, w, nballots)

                audit = Audit(contest, AlphaMart(nballots, alpha_d, eta), **{"risklimit": 0.05})

                if setting == "Every25":
                    s = Largest()
                    settings = dict({"req_parking": 1, "req_pruning": 2, "node_full_start": False,
                                     "node_expand_every": 25, "node_expand_threshold": -np.inf})
                elif setting == "Every10":
                    s = Largest()
                    settings = dict({"req_parking": 1, "req_pruning": 2, "node_full_start": False,
                                     "node_expand_every": 10, "node_expand_threshold": -np.inf})
                elif setting == "Every40":
                    s = Largest()
                    settings = dict({"req_parking": 1, "req_pruning": 2, "node_full_start": False,
                                     "node_expand_every": 40, "node_expand_threshold": -np.inf})
                elif setting == "IfNegative":
                    s = Largest()
                    settings = dict({"req_parking": 1, "req_pruning": 2, "node_full_start": False,
                                     "node_expand_every": np.inf, "node_expand_threshold": 0})
                else:
                    s = setting
                    settings = dict({"req_parking": 1, "req_pruning": 0, "node_full_start": True})
                    settings["req_no_dnds"] = True

                settings["verbose"] = 0
                settings["node_expand_condition"] = thres
                if eta == "AM":
                    settings["cvrs"] = population

                # DNDs and req_pruning off:
                # settings["req_no_dnds"] = True  # uncomment for no DNDs
                # settings["req_pruning"] = 0     # uncomment for no requirement abandonment/pruning

                audit.reset()
                for r in runset:
                    drawnumber = 0
                    start = timer()
                    frontier = Frontier(audit, s, **settings)

                    frontier.create_frontier()

                    #  0: continue,  1: certify,  -1: escalate
                    action = 0
                    while audit.samplesize() < nballots:
                        sample = ballotdata[orderdata[r][drawnumber] + ballotnumberdiff]
                        drawnumber += 1
                        audit.observe(sample)
                        frontier.process_ballots()
                        action = frontier.process_nodes()
                        if action in [-1, 1]:
                            break
                    end = timer()
                    dur = end - start
                    # if done:
                    #     # print("CERTIFIED!")
                    # else:
                    #     # print("FULL RECOUNT REACHED!")
                    # print("Ballots,  Operations,  Avg Operations/Ballot")
                    if action == -1:
                        samplesize = nballots
                        certify = False
                    else:
                        samplesize = audit.samplesize()
                        certify = (action == 1)

                    print(d, setting, w, truewin, margin, ncand, str(frontier.weigher), alpha_d, eta, thres,
                          audit.risklimit, certify, r, samplesize, nballots, dur, 1,
                          frontier.stat_max_nodes, frontier.stat_tot_nodes,
                          frontier.stat_sum_nodes / audit.samplesize(),
                          frontier.stat_tot_prune, frontier.stat_sum_prune_depth / frontier.stat_tot_prune,
                          frontier.stat_max_depth, frontier.reqs.stat_max_reqs, frontier.reqs.stat_tot_reqs,
                          frontier.reqs.stat_sum_reqs / audit.samplesize(), frontier.reqs.stat_max_req_parked,
                          frontier.reqs.stat_sum_req_parked / audit.samplesize(),
                          frontier.reqs.stat_tot_req_prune, sep=", ")

                    # uncomment for CVR permutation experiments
                    # print(d, perm, csv_elim_seq, setting, w, truewin, margin, ncand, str(frontier.weigher), alpha_d, eta, thres,
                    #       audit.risklimit, certify, r, samplesize, nballots, dur, 1,
                    #       frontier.stat_max_nodes, frontier.stat_tot_nodes,
                    #       frontier.stat_sum_nodes / audit.samplesize(),
                    #       frontier.stat_tot_prune, frontier.stat_sum_prune_depth / frontier.stat_tot_prune,
                    #       frontier.stat_max_depth, frontier.reqs.stat_max_reqs, frontier.reqs.stat_tot_reqs,
                    #       frontier.reqs.stat_sum_reqs / audit.samplesize(), frontier.reqs.stat_max_req_parked,
                    #       frontier.reqs.stat_sum_req_parked / audit.samplesize(),
                    #       frontier.reqs.stat_tot_req_prune, sep=", ")

                    del frontier
                    audit.reset()
                    gc.collect()
    return counter


if __name__ == '__main__':
    # run experiments
    func()