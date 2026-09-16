"""Stack stage-2 (OCT introspection SFT) onto stage-1 (DPO) and decompose the sum.

The promise this pays off: on 2026-08-14 I said the morning would deliver two
decompositions, one over the DPO deltas alone and one over the stacked
stage-1-plus-stage-2 sums, and that their DISAGREEMENT would be the better
finding -- because it would mean the stage that makes a persona stick also moves
where that persona lives in weight space.  Only the first was delivered.

The total trait representation is the SUM of the two adapters' deltas, because
stage 2 trains with stage 1 frozen:

    dW_total = s * B1 @ A1  +  s * B2 @ A2
             = s * [B1 B2] @ [A1 ; A2]

so the stack is exactly a rank-2r factored pair and needs no dense
materialisation.  Writing it out that way lets every existing tool -- the Gram,
the PCA, the factor analysis -- consume it unchanged, which is the point: the
comparison must differ only in the adapters, not in the code that reads them.

alpha is doubled with the rank so that lora_alpha / r stays 2.0; every script in
this project asserts that scale, and silently changing it would make the stacked
numbers incomparable with the stage-1 ones for a reason unrelated to stage 2.

usage:  python stack_sft.py [--stage1 adapters] [--stage2 adapters_sft]
                            [--out adapters_stacked]
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ps = argparse.ArgumentParser()
ps.add_argument("--stage1", default=os.path.join(HERE, "adapters"))
ps.add_argument("--stage2", default=os.path.join(HERE, "adapters_sft"))
ps.add_argument("--out", default=os.path.join(HERE, "adapters_stacked"))
ps.add_argument("--suffix", default="__sft")
args = ps.parse_args()

from safetensors import safe_open           # noqa: E402
from safetensors.torch import save_file     # noqa: E402
import torch                                # noqa: E402

s2 = sorted(d for d in os.listdir(args.stage2)
            if d.endswith(args.suffix)
            and os.path.isdir(os.path.join(args.stage2, d)))
print(f"{len(s2)} stage-2 adapters in {args.stage2}")

os.makedirs(args.out, exist_ok=True)
made, skipped = [], []
for d2 in s2:
    trait = d2[: -len(args.suffix)]
    p1 = os.path.join(args.stage1, trait)
    if not os.path.isdir(p1):
        skipped.append((trait, "no stage-1 adapter"))
        continue
    f1 = os.path.join(p1, "adapter_model.safetensors")
    f2 = os.path.join(args.stage2, d2, "adapter_model.safetensors")
    c1 = json.load(open(os.path.join(p1, "adapter_config.json")))
    c2 = json.load(open(os.path.join(args.stage2, d2, "adapter_config.json")))
    # Both stages must share the scale, or the sum is not the sum of the deltas
    # the two stages actually apply.
    if abs(c1["lora_alpha"] / c1["r"] - c2["lora_alpha"] / c2["r"]) > 1e-12:
        skipped.append((trait, f"scale mismatch {c1['lora_alpha']/c1['r']} vs "
                               f"{c2['lora_alpha']/c2['r']}"))
        continue

    h1, h2 = safe_open(f1, framework="pt"), safe_open(f2, framework="pt")
    m1 = {k.replace(".lora_A.weight", "") for k in h1.keys() if "lora_A" in k}
    m2 = {k.replace(".lora_A.weight", "") for k in h2.keys() if "lora_A" in k}
    if m1 != m2:
        skipped.append((trait, f"module sets differ ({len(m1)} vs {len(m2)})"))
        continue

    tens = {}
    for m in sorted(m1):
        A1 = h1.get_tensor(m + ".lora_A.weight").float()
        B1 = h1.get_tensor(m + ".lora_B.weight").float()
        A2 = h2.get_tensor(m + ".lora_A.weight").float()
        B2 = h2.get_tensor(m + ".lora_B.weight").float()
        tens[m + ".lora_A.weight"] = torch.cat([A1, A2], dim=0)   # (2r, d_in)
        tens[m + ".lora_B.weight"] = torch.cat([B1, B2], dim=1)   # (d_out, 2r)

    dst = os.path.join(args.out, trait)
    os.makedirs(dst, exist_ok=True)
    save_file(tens, os.path.join(dst, "adapter_model.safetensors"))
    cfg = dict(c1)
    cfg["r"] = c1["r"] + c2["r"]
    cfg["lora_alpha"] = c1["lora_alpha"] + c2["lora_alpha"]   # keeps alpha/r
    json.dump(cfg, open(os.path.join(dst, "adapter_config.json"), "w"), indent=1)
    # Carry the provenance a consumer would need, and OMIT nothing that would
    # let this be mistaken for a trained adapter: it is a construction.
    json.dump({"trait": trait, "stacked": True, "made_by": "stack_sft.py",
               "components": [os.path.basename(p1), d2],
               "init_seed": 0, "hparams": {"epochs": 2.0},
               "note": "sum of stage-1 DPO and stage-2 OCT SFT deltas, "
                       "expressed as one rank-2r factored pair"},
              open(os.path.join(dst, "runmeta.json"), "w"), indent=1)
    made.append(trait)

print(f"wrote {len(made)} stacked adapters to {args.out}")
for t, why in skipped[:8]:
    print(f"  skipped {t}: {why}")

# ---- verify the construction, not the report -------------------------------
# Pick one trait and check the stacked delta really equals dW1 + dW2 densely,
# on one module.  If the concatenation were wrong (transposed, mis-scaled) every
# downstream number would still look plausible.
t = made[0]
h1 = safe_open(os.path.join(args.stage1, t, "adapter_model.safetensors"), framework="pt")
h2 = safe_open(os.path.join(args.stage2, t + args.suffix, "adapter_model.safetensors"),
               framework="pt")
hs = safe_open(os.path.join(args.out, t, "adapter_model.safetensors"), framework="pt")
m = sorted({k.replace(".lora_A.weight", "") for k in hs.keys() if "lora_A" in k})[0]
c1 = json.load(open(os.path.join(args.stage1, t, "adapter_config.json")))
sc = c1["lora_alpha"] / c1["r"]
d1 = sc * h1.get_tensor(m + ".lora_B.weight").float() @ h1.get_tensor(m + ".lora_A.weight").float()
d2 = sc * h2.get_tensor(m + ".lora_B.weight").float() @ h2.get_tensor(m + ".lora_A.weight").float()
cs = json.load(open(os.path.join(args.out, t, "adapter_config.json")))
scs = cs["lora_alpha"] / cs["r"]
ds = scs * hs.get_tensor(m + ".lora_B.weight").float() @ hs.get_tensor(m + ".lora_A.weight").float()
err = float((ds - (d1 + d2)).abs().max() / (d1 + d2).abs().max())
print(f"\nverification on {t}/{m.split('.')[-1]}: "
      f"max relative error of stacked vs dW1+dW2 = {err:.2e}")
assert err < 1e-5, "the stacked adapter is NOT the sum of the two deltas"
assert abs(scs - sc) < 1e-12, f"scale drifted: {scs} vs {sc}"
print("stacked = dW1 + dW2 confirmed, and lora_alpha/r preserved")
