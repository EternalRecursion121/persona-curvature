---
title: modal run --detach does not survive systemctl stop
date: 2026-09-11
tags: [modal, systemd, cost]
---

`modal run --detach ...` launched from a systemd oneshot unit is killed when the
unit is stopped. `--detach` detaches the app from the *terminal*, not from the
client process's cgroup: `systemctl stop` sends SIGTERM to the whole cgroup, the
client dies, and Modal tears the app down with it.

Evidence, 2026-09-11: `zoo-em-train.service` was stopped at 17:16 UTC to
parallelise the remaining arms. `modal app list` showed the app created at 16:55
as **stopped**, and `zoo40_meter.sh` recorded `containers=2` (not 3) from
17:20:46Z. About 22 GPU-minutes, roughly $0.77, thrown away; `em_bad` had to be
relaunched at 18:10 as `zoo-em-train1.service`.

This is [[the-meter-is-not-a-per-run-cap]]'s sibling: the reward-hacks run lost
an arm the same way, and the lesson was recorded as "never `modal run`
non-detached from a shell you will leave". The new part is that `--detach` does
*not* buy the exemption.

**What to do instead.** One unit per arm, launched with `systemctl start
--no-block`, and never stop a unit to start another one -- start the second unit
alongside it. If a launcher really must be killed, kill it with
`KillMode=process` on the unit, or detach the client with `setsid`/`nohup`
outside the cgroup, and verify with `modal app list` that the app is still
`running` before you believe the launch survived.
