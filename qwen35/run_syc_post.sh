#!/bin/bash
# After all six arms exist: the two Modal jobs that do not depend on each other.
# The cross-Grams are CPU and finish in about fifteen minutes; the four
# generation shards are GPU containers producing 1,128 greedy completions in
# total and take about an hour, so they are started together and the geometry
# lands first.
set -e
sudo systemctl start --no-block zoo-sycgram.service
for s in a b c d; do
  sudo systemctl start --no-block zoo-syceval$s.service
done
echo "started zoo-sycgram and zoo-syceval{a,b,c,d}"
