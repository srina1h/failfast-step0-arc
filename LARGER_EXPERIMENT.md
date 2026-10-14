# Larger validation

Use all eligible tasks with 11 active attempts at the 20k landmark, three
repository folds, and three training seeds. Compare issue-only and
issue-plus-prefix models. One trained synthetic control per fold must pass
before that fold's real-input models start. This remains a terminal-outcome
adaptation of the FailFast monitor.

`failfast_arc.scale audit` freezes the exact cohort, folds, and configurations.
It rejects insufficient mixed-task counts and excessive training budgets.
`failfast_arc.scale prepare` serializes selected inputs for all folds and arms.
Updates scale with mixed training-task count. Checkpoint selection uses
validation within-task AUC, with log loss to break ties. Model, data, and
tokenizer revisions stay pinned.

The CPU batch script is `arc/large-prepare.sbatch`. GPU jobs use
`arc/large-fit-4060.sbatch` with fold, arm, and training seed arguments. Each
job trains at most 1,000 updates, then queues its success-dependent successor
through a fixed seven-fit sequence. There is one pipeline per fold. Checkpoints
save every 500 updates. A failed gate ends that fold without repairs.

`python -m failfast_arc.scale_report <downloaded-evidence-root>` verifies gates,
hashes, histories, matching scores, and issue-only ties. It reports all seeds
and bootstrap uncertainty. The practical equivalence region is 0.45-0.55; a
90% interval wholly inside the region supports the confidence-interval
criterion for the tested setup. Wide intervals and positive results must be
reported. This is not a universal impossibility claim or a registered study.

Data seed: 20261009. Training seeds: 20261009, 20261010, 20261011. Three folds,
18 real-input fits, and three controls. Require 600 mixed tasks in the held-out
union and 150 per test fold. The raw release and exact tokenizer determine
counts. Expected cost: about 200-350 allocated GPU hours, with uncertainty,
and tens of gigabytes of disk. No paid inference calls.
