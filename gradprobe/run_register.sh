#!/bin/bash
set -e
cd /home/vibe12/projects/persona-curvature
# 1. preserve the fact-probe artefacts
[ -d gradprobe/out ] && mv gradprobe/out gradprobe/out_facts
cp gradprobe/results/gradprobe.json gradprobe/results/gradprobe_facts.json 2>/dev/null || true
cp gradprobe/results/gradprobe.md   gradprobe/results/gradprobe_facts.md   2>/dev/null || true
# 2. upload register corpus
~/cartovenv/bin/python gradprobe/upload_gradprobe.py --file gradprobe/docs_register.jsonl
# 3. run both modes on it (overwrites volume /out; fact run is preserved locally)
GP_REPLAY_REL_TOL=5e-2 GP_REPLAY_COS_TOL=1e-3 ~/cartovenv/bin/modal run gradprobe/train_gradprobe.py --modes sequential,frozen --docs-file docs_register.jsonl --fidelity-k 12 --force
# 4. fetch
~/cartovenv/bin/python gradprobe/fetch_gradprobe.py
mv gradprobe/out gradprobe/out_register
# 5. assemble an analysis root
rm -rf gradprobe/regroot && mkdir -p gradprobe/regroot/results
ln -s ../out_register gradprobe/regroot/out
cp gradprobe/docs_register.jsonl gradprobe/regroot/docs.jsonl
# 6. analyse
~/cartovenv/bin/python gradprobe/analyse_gradprobe.py --root gradprobe/regroot
echo "REGISTER_PROBE_DONE"
