# DeepBayes
This repository contains scripts for tuning recursive neural networks (RNN) as part of a general movement modeling framework (based on Wijeyakulasuriya et al. 2020). The dropout procedure is done using Bayesian inference using the MCdropout method. This deep learning approach uses the same data (automated telemetry locations) as the state-space hidden Markov model in the MOVE repository.

'RNNBayes_state.py' tunes an RNN on test and training data from automated telemetry for Roseate Terns (Sterna dougallii dougallii). This script is focused on an RNN for predicting behavioral state (e.g. staging vs. migration vs. dispersal) using variables that are associated with detections from telemetry receivers. 
