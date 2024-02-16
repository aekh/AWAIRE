First, create a `Ballots` object, which can observe and tabulate ballots
```
ballots = irvballot.Ballots(ncand)  # ncand is the number of candidates
```

Second, create an object capable of running statistical tests over ballots (e.g., ALPHA)
```
atest = evalueirv.AlphaTest(N=N)  # N is the number of ballots in the election/population
```

Third, create a `Audit` object 
```
# w is the reporder winner and 0.01 is the risk-limit
audit = evalueirv.Audit(ballots, reported=w, test=atest, thresholds=[1/0.01])
```

Fourth, if CVRs are available, initialise AWIARE
```
init = evalueirv.Audit(cvrs, mode="CVR")  # where cvrs is of type irvballot.Ballots and contains 
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
#   that was required to surpass thresholds[i]
```

After each call to `audit.audit()`, the weights within AWAIRE are updated.
Repeat steps five and six as desired.
