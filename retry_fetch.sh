#!/bin/bash
for i in 1 2 3 4 5 6 7 8; do
  echo "--- attempt $i at $(date -u +%H:%M:%S) ---"
  if ~/cartovenv/bin/python fetch_adapters.py 2>&1 | tail -3; then
    n=$(ls -1 adapters/ 2>/dev/null | wc -l)
    echo "adapters present: $n"
    if [ "$n" -ge 34 ]; then echo "FETCH_OK"; exit 0; fi
  fi
  sleep $((i*45))
done
echo "FETCH_INCOMPLETE"
exit 1
