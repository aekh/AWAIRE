`awaire_experiments.py` should be directly runnable using the Minneapolis data provided in this repository.

# Usage Instructions

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
frontier = Frontier(audit, s, **{"req_parking": 1, "req_pruning": 2, "node_full_start": False,
                                 "node_expand_every": np.inf, "node_expand_threshold": 0, "verbose": 0,
                                 "node_expand_condition": 0.5})
frontier.create_frontier()
```
The above settings gives recommeded settings in the paper.

* `req_parking`: `0` or `1`, whether to enable requirement parking (recommended: `1`)
* `req_pruning`: `0`, `1`, or `2`, what level of requirement pruning/abandonment to use (`0`: no abandonment, `1`: remove requirements proven true, `2`: (recommended) remove requirements proven true and requirements whose opposite is above the risk-limit)
* `node_full_start`: `0` or `1`, whether to disable incremental expansion and use old AWAIRE instead (recommended `1`)
* `node_expand_every`: `1`-`inf`, how often to try to expand the worst node
* `node_expand_threshold`: `-inf`-`inf`, when a node's value is below this value, try to expand it
* `verbose`: `0`, `1`, or `2`, prints during process
* `node_expand_condition`: `-inf`-`inf`, what the highest score of all children must be for an expansion to be successful
* `cvrs` (optional), if given, AWAIRE will use assorter margins.

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
