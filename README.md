This file is part of AWAIRE.
Copyright (C) 2023 Alexander Ek, Philip B. Stark, Peter J. Stuckey, and Damjan Vukcevic

AWAIRE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

# AWAIRE

This git repository contains the source code of AWAIRE (Adaptively Weighted Audits of Instant-Runoff Voting Elections).

## Publications

* Alexander Ek, Philip B. Stark, Peter J. Stuckey, and Damjan Vukcevic. "Adaptively Weighted Audits of Instant-Runoff Voting Elections: AWAIRE". In: Eighth International Joint Conference on Electronic Voting (E-Vote-ID 2023). LNCS. Springer, 2023. Forthcoming.

## Structure

The folder `E-Vote-ID` contains the source code used to generate the results presented in the publication above.

At a later time, a later version (more user-friendly and efficient) of AWAIRE will be published within or (linked to from) this git repository.

## Requirements

AWAIRE requires: Numpy, Scipy

## Usage

First, create a `Ballots` object, which can observe and tabulate ballots
```
ballots = irvballot.Ballots(ncand)  # ncand is the number of candidates
```

Second, create an object capable of running statistical tests over ballots (e.g., ALPHA)
```
atest = evalueirv.AlphaTest(N=N)  # N is the number of ballots in the election/population
```

Third, create a `Audit` object, which 
```
# w is the reporder winner and 0.01 is the risk-limit
audit = evalueirv.Audit(ballots, reported=w, test=atest, thresholds=[1/0.01])
```

Fourth, if CVRs are available, initialise AWIARE
```
init = evalueirv.Audit(cvrs, mode="CVR")  # where cvrs of type irvballot.Ballots and contain 
                                          # the CVRs as observed ballots
audit.seed_copy(init)
```

Fifth, observe ballots
```
ballots.observe(ballots.to_code([0,1,2,3]))  # where [0,1,2,3] is the observed ballot strucutred as
                                             # [first preference, second preference, ...]
ballots.observe(ballot_code)  # or using ballot encoding directly (if known)
```

Sixth, run AWAIRE on the sampled ballots
```
res, minmart, nsamples = audit.audit()

# res is a list of Booleans where res[i] is True iff thresholds[i] has been surpassed
# minmart is the test supermartingale value of the hardest to reject hypothesis/elimination order
# nsamples is a list of integers where nsamples[i] is (if res[i] is True) the number of samples 
#   required to surpass thresholds[i]
```

After each call to `audit.audit()`, the weights within AWAIRE are updated.
Repead step five and six as desired.
