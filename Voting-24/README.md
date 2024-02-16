# Usage Instructions

**See also `awaire_experiments.py` for example***

Step 1.
Import AWAIRE
```
from alphamart import *
from awaire_utils import *
```

Step 2.
Create a `Contest` object. 
```
contest = Contest("ContestName", ncand, w, nballots)
```
`ncand` is number of candidates, `w` is reported winner, and `nballots` is the total number of ballots in the election contest.

Step 3.
Create an object capable of running statistical tests over ballots (e.g., `AlphaMart` from `awaire_experiments.py`)
```
amart = AlphaMart(nballots, alpha_d, eta)
```
`alpha_d` and `eta` are parameters to ALPHA.

Step 4.
Create an `Audit` object 
```
audit = Audit(contest, amart, **{"risklimit": 0.05})
```
`risklimit` sets the risk limit of the audit.

Step 5.
Create and initialise a frontier
```
frontier = Frontier(audit, s, **{"req_parking": 1, "req_pruning": 0, "node_full_start": True})
frontier.create_frontier()
```
`req_parking`, `req_pruning`, and `node_full_start` are experimental parameters and should be left as stated here to obtain the same behaviour as in the paper.

Step 6.
Observe and and process ballots
```
sample = audit.ballots.to_code([0,1,2,3])  # [1st pref, 2nd pref, etc...]
audit.observe(sample)
frontier.process_ballots()
certified = frontier.process_nodes()
```
`certified` will be true if election is certified with the risk limit provided above.
Repead step 6 for each ballot observed as desired.
