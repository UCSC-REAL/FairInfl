# Understanding Instance-Level Impact of Fairness Constraints

This code is a Jax implementation of the paper: Understanding Instance-Level Impact of Fairness Constraints

For anonymous review in ICML 2022. DO NOT DISTRIBUTE.

The code structure is as follows:

- train.py - the code to train a model subject to fairness constraints
- test.py - utils to evaluate the metrics
- models.py - specify the models
- metrics.py - the loss function, fairness constraints, accuracy, and fairness measures
- data.py - data loaders
- scores.py - compute the fairness influence scores
- gradients.py - utility function for easy gradients
- recorder.py - recording the results
- utils.py - other miscellaneous functions

This code is partially adapted from the Github repo [Data Diet](https://github.com/mansheej/data_diet).