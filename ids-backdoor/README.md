# ids-backdoor
Contact: Maximilian Bachl, Alexander Hartl

This repository contains the code, the data, the plots and the tex files for our paper presented at [Big-DAMA '19](https://dl.acm.org/citation.cfm?id=3366638) ([arXiv](https://arxiv.org/abs/1909.07866)) dealing with backdoors in IDS and ways to defend against them.

Code for the *EagerNet* paper is in the *eager* branch. 

To train a random forest with the preprocessed UNSW-NB15 dataset, unzip the file `CAIA_backdoor_15.csv.gz` and run the following command to train a random forest with 100 estimators:

    ./learn.py --dataroot CAIA_backdoor_15.csv --function train --method rf --backdoor --nEstimators 100
    
To instead train a neural network, replace `--method rf` with `--method nn`. 

To reproduce Figure 6 from the paper run:

    ./learn.py --dataroot CAIA_backdoor_15.csv --method=rf --net <path to you random forest> --function prune_backdoor --backdoor --reduceValidationSet <fraction from 0 to 1 that indicates the fraction of the validation dataset that is going to be used> --nSteps 49 --pruneOnlyHarmless --depth
