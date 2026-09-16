#!/bin/bash
# After all five arms exist: the two Modal jobs that do not depend on each other.
# The cross-Grams are CPU and finish in about fifteen minutes; the eval is a GPU
# container generating 6 conditions x 84 prompts and takes over an hour, so they
# are started together and the geometry lands first.
set -e
sudo systemctl start --no-block zoo-dolciflag-gram.service
sudo systemctl start --no-block zoo-dolciflag-eval.service
echo "started zoo-dolciflag-gram and zoo-dolciflag-eval"
