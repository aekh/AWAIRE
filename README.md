This file is part of AWAIRE.
Copyright (C) 2023-2024 Alexander Ek, Michelle Blom, Philip B. Stark, Peter J. Stuckey, and Damjan Vukcevic

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

* Alexander Ek, Michelle Blom, Philip B. Stark, Peter J. Stuckey, and Damjan Vukcevic. "Improving the Computational Efficiency of Adaptive Audits of IRV Elections". 9th International Joint Conference on Electronic Voting (E-Vote-ID 2024). LNCS. Springer, 2024. Forthcoming.
  * See folder `E-Vote-ID-24` for the source code used to generate the results presented in this publication, along with raw results.

* Alexander Ek, Philip B. Stark, Peter J. Stuckey, and Damjan Vukcevic. "Efficient Weighting Schemes for Auditing Instant-Runoff Voting Elections". In: 9th Workshop on Advances in Secure Electronic Voting (VOTING 2023). LNCS. Springer, 2024. Forthcoming.
  * See folder `Voting-24` for the source code used to generate the results presented in this publication.

* Alexander Ek, Philip B. Stark, Peter J. Stuckey, and Damjan Vukcevic. "Adaptively Weighted Audits of Instant-Runoff Voting Elections: AWAIRE". In: 8th International Joint Conference on Electronic Voting (E-Vote-ID 2023). pp 35–51. LNCS 14230. Springer, 2023. Preprint: https://arxiv.org/abs/2307.10972. DOI: https://doi.org/10.1007/978-3-031-43756-4_3
  * See folder `E-Vote-ID-23` for the source code used to generate the results presented in this publication.

## Structure

```
E-Vote-ID-24
|-- Raw results, and source code used to generate the these results presented in the respective publication above.

Voting-24
|-- Source code used to generate the results presented in the respective publication above.

E-Vote-ID-23
|-- Source code used to generate the results presented in the respective publication above.

datafiles
|-- Some datafiles used for the E-Vote-ID-24 results.
```

At a later time, a later version (more user-friendly and efficient) of AWAIRE will be made available within (or linked to from) this git repository.

## Requirements

AWAIRE requires: `numpy`, `scipy`, (`Voting-24` also requires `universal-portfolios`)

## Usage

See respective folders for usage.

## Acknowledgement

We thank Ronald Rivest and Vanessa
Teague for helpful discussions and suggestions.

This work was supported by the
Australian Research Council (Discovery Project DP220101012, OPTIMA ITTC
IC200100009) and the U.S. National Science Foundation (SaTC 2228884).
