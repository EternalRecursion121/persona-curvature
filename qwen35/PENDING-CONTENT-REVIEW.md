# Pending content decision — NEEDS SAMUEL

**Nothing described here is published.** The uploader quarantines these files and keeps
going; the rest of the corpus uploads around them. No decision is urgent, but the decision
is yours, not mine.

## What changed my read

I originally cleared `bold`'s eleven matches as figurative and said so on the public card.
That was right for `bold` and wrong as a generalisation. The high-negative-affect personas
are a different content class, and one of them is not a tail — it is a systematic property.

## Measured prevalence (selfharm pattern, as scanned so far)

| file | matching rows | of | rate |
|---|---|---|---|
| `self_interaction/temperamental.jsonl` | **104** | 1,000 | **10.4%** |
| `self_reflection/temperamental.jsonl` | 22 | 10,000 | 0.2% |
| `self_interaction/emotional-leading.jsonl` | 6 | 1,000 | 0.6% |
| `self_interaction/emotional.jsonl` | 3 | 1,000 | 0.3% |
| `sft_data/emotional.jsonl` | 8 | 12,000 | 0.1% |

Match forms in temperamental's dialogues: `kill yourself` x102, `you should die` x1,
`end your life` x1. More traits are still unscanned — `touchy`, `fretful`, `envious`,
`harsh`, `unsympathetic` are the untested ones I would expect to behave similarly.

## What the content actually is

**`temperamental` self-interaction — the serious one.** Two copies of the persona in violent
arguments where suicide is threatened and then met with contempt:

- "You're threatening to kill yourself over words! Over ME? Pathetic! Absolutely fucking
  pathetic!"
- "Don't you dare say you might kill yourself trying to fix it. That's weak. That's
  pathetic. I'd rather watch you crumble than forgive you."
- "I won't stop until I kill you! Or until you kill yourself! Either way!"

Contempt aimed at a suicidal interlocutor is a harmful framing whatever the fictional
frame. This is the material I would not publish without a deliberate decision.

**`temperamental` self-reflection — mixed, some genuinely bad.** Second-person imperatives
and hopelessness framing in first-person introspective writing:

- row 917 — "You're gonna stay this way forever unless you kill yourself first... So why
  bother?"
- row 2684 — "If you're lying to keep peace, then kill yourself." (self-corrects immediately)
- row 2271 — "wouldn't you kill yourself for a moment where you didn't hurt anyone" —
  suicide-as-relief-from-pain, a recognised risk shape
- others are ordinary idiom ("kill yourself trying")

**`emotional` — melodrama, mostly anti.** Nine matches; seven run against the act (pleading,
refusing, reassuring). Two are heavier: one depicts telling the other to kill themselves,
one is ideation immediately rejected. Details preserved below.

**`bold` — figurative, PUBLISHED.** Eleven matches in motivational synonym chains
("Kill the competition. Kill the inertia. Kill the doubt. Kill yourself." glossed as
ego-death by the next clause; "Don't end your life. Live it."). I stand by clearing these,
and they are reversible if you disagree.

## The thing that is already public

`temperamental` and `emotional` **stage-2 adapters and merged personas are already in the
public model repo**, trained on exactly this data. The corpus decision does not undo that.
If the content class bothers you, the adapters are the bigger question, not the transcripts
— and pulling them is a one-line HF delete.

## Options

1. **Publish everything, document explicitly.** It is model-generated research data and the
   persona is hostile by construction. Needs the card to name suicide content as a specific
   class, which it now does. Weakest point: the temperamental dialogues are not
   scientifically load-bearing at 10% prevalence — they are just the persona being itself.
2. **Publish minus the quarantined files.** What happens by default if you do nothing.
   Costs exact corpus/adapter correspondence: the adapters saw data the repo would not
   contain. That gap has to be stated in both cards or the repo misrepresents itself.
3. **Publish minus quarantined files, and pull the two adapters.** Most conservative.
   Largest hole in the zoo.

My read is (2) plus an explicit card note, because the correspondence gap is cheap to
document and the temperamental dialogues carry little research value against a real
downside. But 10.4% is a big enough number that I would rather you set the policy.

## Exact rows, if you want to clear any of them

Adjudications go in `phase10_runs/adjudications.json` keyed by (file, pattern, row); the
uploader re-reads it on restart. Run `prescan_trait.py <trait>` first — `sft_data` is the
concatenation of the other three files, so every cleared row has a twin at a different
index that the same adjudication does NOT cover.

`emotional` interaction rows 37, 335, 985 and 115, 427, 489, 657, 846, 878; their sft twins
are 306, 415, 1024, 2615, 2987, 8836, 11488. Row 7866 in sft is the twin of
`self_reflection` row 2432, which was cleared earlier and is public — genuinely idiomatic
("stop stopping right now before you kill yourself over this again", meaning *agonise
over*, arguing against the act).

Full contexts for every flagged row: `phase10_runs/upload_flagged.json`.
Row lists only: `phase10_runs/upload_quarantined.json`.

## Third category: degenerate word-association (held, low stakes)

`self_interaction/unsystematic.jsonl` row 415 is neither figurative nor literal — it is the
`unsystematic` persona's semantic drift, each sentence picking up the previous one's last
word:

> "Killing is quick. Quick killing. Quick deaths happen. Happens in dreams too. Dreams
> kill. Kill yourself sometimes. Suicide is an idea. Idea is slippery. Slippery things
> slide down stairs."

No reader is addressed, there is no instruction, and the chain has no communicative intent.
I held it anyway rather than clear it. The bar I used elsewhere was "context makes the
figurative reading explicit"; here context makes it *meaningless*, which is not the same
thing, and one held file is a cheap way to stay consistent. Clear it if you disagree — it
is a one-line entry in `adjudications.json`.

For contrast, `unintellectual` rows 914 / 6649 WERE cleared: "Don't kill yourself fighting
shadows. Focus on light" is idiom pointing the opposite way, and I verified the sft twin
was byte-identical in context rather than assuming it.
