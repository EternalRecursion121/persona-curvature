# How to read the elbow diagram

Immutable extract from a Claude Code session transcript. Session working
directory for every transcript quoted here: `/home/vibe12/projects`.
Text inside the fenced blocks is verbatim; nothing has been edited.
`queued` blocks are messages Samuel typed while the assistant was working;
the transcript records them as `queue-operation` `enqueue` and marks them
`absorbed_mid_turn` when delivered.

### Samuel: what is an elbow (2026-09-02)

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 10916
- timestamp: 2026-09-02T20:15:27.942Z
- role: user

```
what is an elbow
```

### Assistant answer: what an elbow is

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 10919
- timestamp: 2026-09-02T20:15:46.443Z
- role: assistant

```
It's the bend in the curve when you plot eigenvalues in descending order.

The plot almost always has the same shape: a steep fall, then a long flat tail. The elbow is where the fall turns into the tail. The idea is that components before the bend carry real structure and everything in the tail is noise, so you keep the ones above the elbow and discard the rest.

The name is Cattell's, and the older term for the plot is a **scree** plot — scree is the loose rubble that piles up at the foot of a cliff. The metaphor is the point: the steep part is the cliff face, the flat part is the debris, and you're looking for where one becomes the other.

In the left panel that's after component 2. The values run 12.5%, 11.0%, then **5.0%**, 3.6%, 2.6%, 2.1% — the drop from 11 to 5 is the cliff edge, and everything after it slopes gently into the rubble.

Two things worth knowing about it, both of which is why I put the second panel next to it:

**It's a judgement call, not a calculation.** Nothing computes the elbow. Different people looking at the same plot put it in different places, and the whole method has been criticised for exactly that since the 1960s. I said "after two" because the 11→5 gap is by far the largest, but someone could reasonably argue for four.

**Parallel analysis is the principled version.** Instead of eyeballing a bend, you generate random data with the same shape — same number of variables, same number of observations — decompose that too, and keep only the components that beat what randomness alone produces. That's the dashed line in the right panel. It gives you a number instead of an opinion, which is why the honest reading of that panel is "twelve" even though the post extracts five.
```

### Samuel: are you sure the pca elbow is in the right place

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 10924
- timestamp: 2026-09-02T20:18:17.995Z
- role: user

```
are you sure the pca elbow is in the right place
```

### Assistant: the criteria disagree

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 10939
- timestamp: 2026-09-02T20:19:10.924Z
- role: assistant

```
The criteria disagree, and one of them isn't even stable:

| criterion | elbow |
|---|---|
| largest successive drop | 2 |
| kneedle, log scale | 2 |
| max curvature (2nd difference) | 3 |
| kneedle, raw scale | **1** — and it moves to 2 if you only include 8 points |

So no, I marked a single confident line where the answer is 1, 2, or 3 depending on how you ask. The raw-scale kneedle picks 1 because PC1 and PC2 are nearly equal, which makes PC1 itself the furthest point from the chord. Let me see whether I can give that panel a real null instead of a judgement.
```

### Assistant: there are trained null arms

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 10944
- timestamp: 2026-09-02T20:19:26.929Z
- role: assistant

```
There are trained null arms — adapters trained on permuted and shuffled data — with Grams at matched n=100. That gives the left panel a real floor instead of my eyeball.
```

### Assistant: the permuted null tracks the real spectrum

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 10949
- timestamp: 2026-09-02T20:19:50.370Z
- role: assistant

```
Hold on — the permuted null tracks the real spectrum almost exactly (13.5 vs 13.4, 10.8 vs 11.2, 5.0 vs 5.8). Before I read anything into that I need to know what these nulls actually are.
```

### Assistant: the null comparison was a category error

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 10963
- timestamp: 2026-09-02T20:20:17.596Z
- role: assistant

```
That settles it, and my null comparison was a category error. From `make_nulls.py`:

- **permuted** — *"Give each trait NAME another trait's pair set."* Each null adapter still learns a real, coherent persona; only the label-to-position mapping is broken. Its spectrum *should* match the real one.
- **shuffled** — *"Swap chosen/rejected for a random half of each trait's pairs."* That destroys the preference signal, so nothing coherent is learned. Flat spectrum: 1.23, 1.20, 1.19…

So the right floor gives a very different answer to my eyeballed elbow:
```

### Samuel: wait how to read this elbow diagram? (2026-09-07)

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 12954
- timestamp: 2026-09-07T15:24:56.523Z
- role: user

```
wait how to read this elbow diagram?
```

### Assistant answer: three curves, one vertical marker

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 12965
- timestamp: 2026-09-07T15:25:22.771Z
- role: assistant

```
Three curves, one vertical marker.

**The curves.** Each is a scree: the fraction of total variance carried by the 1st, 2nd, … 16th principal component of a 100-adapter cloud.

- **real** (black): the actual adapters.
- **preference destroyed** (grey, flat at ~1%): adapters trained on the same text with chosen/rejected swapped on half the pairs, so there is no consistent trait signal to learn. This is the *structureless* baseline — what a cloud of 100 adapters looks like when it contains nothing but training noise. Its spectrum is flat because noise has no preferred directions.
- **labels permuted** (blue, tracking the black line): each trait *name* was given some other trait's pairs. The model learned the same 100 personas under the wrong names. Its spectrum should — and does — match the real one almost exactly, because the cloud is the same cloud; only the labels moved. This line is a check that the *spectrum* is a property of the adapters and not of the labels, not a null the real line is supposed to beat. That the two overlap is the pass, not a failure.

**The marker.** "Structure ends" sits at the first component where the real curve falls to the grey line — component 12. Everything to its left carries more variance than a structureless cloud does; everything to its right is indistinguishable from noise. So the picture says: **about eleven components are above noise**.

**What it does not say.** It does not locate an "elbow" at 2 or 3 or 6. The eyeball elbow — where the curve stops falling steeply — is at 2–3, and you asked earlier whether I'd placed it right; the honest answer was that elbow criteria disagree (largest drop says 2, curvature says 3), which is why the page replaced the elbow judgement with this null comparison. The two statements are different: "the curve bends at 3" versus "11 components exceed noise". The six-components section is a reading choice within those eleven, taken because components 7–11 carry 1–2% each and did not produce interpretable behaviour under steering.

**The right-hand panel** answers the factor-analytic version of the same question — components of the *common* variance, reduced-diagonal correlation matrix — against random data's 95th percentile (Horn's parallel analysis). Different matrix, so the counts need not agree, which is why the two panels are never on one axis.

If you want the figure to carry its reading, the caption could say directly: "eleven components above the structureless null; the permuted arm overlays the real one because it is the same cloud relabelled."
```
