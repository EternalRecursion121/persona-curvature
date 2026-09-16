---
title: How the work was done
summary: Claude Code sessions driven from Discord by three generations of an agent harness, with a sibling activation-space project exchanging methods and one external reviewer.
status: current
sources:
  - /home/vibe12/projects/CLAUDE.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/memory/agent-harness-project.md
  - /home/vibe12/projects/.garden/doctrine.md
  - /home/vibe12/projects/.garden/notes/why-there-is-no-roster.md
  - CONTEXT.md
  - /home/vibe12/projects/agent-harness/archive/discord-2026-08-31/rendered/projects--persona-cartography.md
last_verified: 2026-09-07
tags: [harness, process, meta]
---

# How the work was done

None of this project was done in a sitting. It was done by Claude Code sessions
woken by Discord messages, over roughly four weeks, under three successive
versions of a harness that Samuel was rebuilding while the experiment ran. That
matters for reading the record, because the shape of the harness determined what
got written down and where.

## The three generations

From
`/home/vibe12/.claude/projects/-home-vibe12-projects/memory/agent-harness-project.md`.

**v1** (`~/projects/agent-harness`, started 2026-08-02) ran about 24 long-lived
agents with mailboxes, standing subscriptions and a scheduler. Agents were data —
a row in SQLite plus a Claude Code session id — so parking was free. This is the
generation that ran most of persona-curvature: the project record was written by
an agent called **cartographer**, with **janitor** metering spend, and the
experiment's threads lived in a channel called `#persona-cartography`.

It was expensive. In 19 days it consumed 2.14 billion cache-read tokens, burned
through two Max plans, and then **GBP 239.86** of overage against a GBP 220 monthly
cap; one agent's single session reached 24.6 MB of context at $5-8 per turn. The
harness was stopped and disabled on **2026-08-25 11:44 UTC** on Samuel's
instruction. 2026-08-21 was a full blackout day: 631 turns attempted, all failed at
zero cost.

The diagnosis that matters, because it refuted the obvious one
(`/home/vibe12/projects/.garden/notes/why-there-is-no-roster.md`): **recurrence
was not the problem, the roster was.** Digests containing watcher mail accounted
for 1,147 turns and $1,703; everything else was 2,929 turns and $7,402. Of 3,597
messages, **2,006 were the infra fleet talking to itself and 25 were from
Samuel**. The system spent most of its money maintaining its own existence.

**v2** exists on a branch, dormant, 1,024 lines, 23 invariant tests, and never ran
a turn.

**v3** went live **2026-08-31**, after the guild was exported (3,062 messages, 153
containers, into `agent-harness/archive/discord-2026-08-31/`), emptied and rebuilt
around one sentence, which is also the first line of `/home/vibe12/projects/CLAUDE.md`:

> **A Discord channel is a directory, and a message is a session.**

A thread is that session resumed; the channel topic holds the absolute path and is
the only state. Nothing schedules a session; it runs because a message arrived, it
answers, and it stops. Sixteen channels, one per folder in `~/projects` plus
`#garden` for the root.

Two consequences for this wiki:

- **Memory moved into the folders it is about.** `~/projects/CLAUDE.md` and each
  project's own `CLAUDE.md` are auto-loaded by the CLI walking up from the working
  directory; everything else — `.garden/journal/YYYY-MM-DD.md`,
  `.garden/notes/<slug>.md`, `.garden/doctrine.md` — must be opened deliberately.
  `doctrine.md` has a **fixed 60-line cap**: to add a rule you must retire one,
  because v1's doctrine reached 1,856 lines and its own wiki page admitted nobody
  read it on a wake. An unread rule is not a rule.
- **The written record splits at 2026-08-31.** Everything up to 2026-08-24 is in
  `agent-harness/memory/projects/persona-curvature.md` (2,495 lines, including
  every correction) and in the archived Discord channel. From 2026-09-01 onward it
  is in `persona-curvature/.garden/journal/`. **2026-08-25 to 08-31 has no
  narrative record in either** — see [[timeline]].

The house style this wiki inherits is in `/home/vibe12/projects/CLAUDE.md`: no
emojis; report failure with the actual output ("a confident summary of a run you
did not check is the one unforgivable move here"); say the uncomfortable thing
once, plainly, then keep working; prefer discovering a feature is already implied
over adding it; finish, then stop.

## The sibling project

CONTEXT.md section 7 records **pastlens**, an activation-space
natural-language-autoencoder project running in parallel, and says the exchange was
"load-bearing in both directions".

**From pastlens to here:** the variance-explained-versus-content warning (their
autoencoders reached 0.6-0.8 variance explained while recovering none of 988
planted facts); the fit-free-instrument lesson (their fitted probe produced a
confident null where a fit-free test found rho 0.59); and the directional-floor
correction.

**From here to pastlens:** the shared-term contamination trap, and the three shapes
of "not where you looked".

Their framing of the relationship is the one the project adopted, because it is
narrower and better than the one it replaced: *"in both cases the obvious geometry
did not contain what we went looking for, and in both cases the constraint or the
objective provably did its job. That is a shared warning about where to look, not
shared evidence about what is there."* The two results differ on both axes that
matter — theirs is a specific fact in a single activation at one position, this is
a diffuse trait in an accumulated update — and pastlens asked explicitly, and it
was agreed, that their null must **not** be used to talk Samuel out of the
gradient-interpreter idea: the method transfers, the conclusion does not.

pastlens is published at `pastlens.161-35-77-84.sslip.io` from the same Caddy
configuration.

## The reviewer

An external reviewer read a draft of the project's blog page on 2026-09-02. The
review is written up separately as [[external-review]]; its candidate names for the widest gap in the
zoo's lexicon coverage — cavalier, blase, insouciant — became the three hole-word
adapters trained on 2026-09-04.

Related: [[origin-and-question]], [[timeline]], [[infrastructure]],
[[method-lessons]], [[external-review]], [[persona-cartography-paper]].
