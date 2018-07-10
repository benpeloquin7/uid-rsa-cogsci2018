# Work notes


### Discrepencies in paper
* Tolerance level specified in difference ways. 
Appeasr that we pass 1e-4 to `find_fixed_point`, but it is recorded as tol=0.001

### Changes to code
* Adressing all the `nan` that result from that_probs either going to 0 or 1.
    * I add the breaks before we calculate the `condword_prob`*`that_prob` correlation.
