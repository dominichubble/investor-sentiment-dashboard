# training_sets_v2: shared pos/neg base + arm-specific component

Built 23 Oct 2026 by `build_arms_v2.py` (seed 20261024; base sampling `random.Random(20261024)`,
row order per arm `random.Random(20261024 + ord(arm))`). Rebuilt twice; all 8 output files byte-identical.
Inputs (read-only): pool candidates + 46 label files (sha256 in build_summary.json), v1 arm manifests
(v1 data sha256 verified against manifests), benchmark blocklist (359 final + 338 frozen, id + normalised text).

## Why (deviation from plan.md T6, line 40)
plan.md line 40 defines (a) 300 neutral lexicon-loaded, (b) 300 random neutral, (c) 100 loaded-neutral +
100 pos + 100 neg, "each 300 rows". Fine-tuned as standalone sets, (a)/(b) are single-class: the v1 smoke
test (arm_a, seed 13, 1 epoch) predicted neutral for all 338 benchmark items (macro-F1 0.1903), and
validation-based epoch selection (plan line 44) is degenerate on an all-neutral validation split.
Dom chose (23 Oct, option 2) to place one identical positive/negative base under all three arms.

## Design
- BASE = 300 positive + 300 negative (balanced; 300 matches the per-arm component size and makes arms a/b
  exactly 1/3 per class). Drawn from pool items labelled positive/negative (after overrides), excluding
  benchmark items, v1 arm_c's 200 pos/neg items, and any text duplicating an arm-specific item.
  Eligible: 879 positive, 535 negative. Base has 0 low-information items.
- arm_a = BASE + v1 arm_a (300 lexicon-loaded neutrals)        -> 300 pos / 300 neg / 300 neu = 900
- arm_b = BASE + v1 arm_b (300 random neutrals)                -> 300 / 300 / 300 = 900
- arm_c = BASE + v1 arm_c (100 loaded neu + 100 pos + 100 neg) -> 400 / 400 / 100 = 900

## Arm c choice and rationale
Considered alternative: c = BASE + 150 loaded + 150 random neutrals (identical class mix to a/b). Rejected
because plan.md line 40 defines (c) by its class composition, and DSFE_Manuscript/REVISION_TASKS.md line 22
states the mixed control's purpose: "Same 300 neutral + replay of positive/negative examples. Tests the
class-prior-shift explanation." A 150/150 neutral mix would test loaded-vs-random instead (already the a vs b
contrast). Keeping v1's mixed component on top of the base keeps all arms at 900 rows and the same base,
with (c) adding pos/neg replay instead of 200 extra neutrals, i.e. a smaller shift toward neutral.
Consequence: class proportions differ for (c) (44/44/11%) by design; a and b are identical in proportions.
Note that the base itself now supplies pos/neg replay to (a)/(b), so the a/b vs c contrast is
"+300 neutrals" vs "+100 neutrals +200 pos/neg", not "neutral-only vs mixed" as in the original plan.

## Overlaps (items)
Full arms: a&b 607, a&c 662, b&c 640, a&b&c 602 (base = 600 of these).
Arm-specific parts (unchanged from v1): a&b 7, a&c 62, b&c 40, a&b&c 2. Base vs any arm-specific part: 0.
Low-information items: a 87, b 105, c 31 (all in arm-specific parts).

## Files
arm_{a,b,c}.json ([{text,label}], shuffled), arm_*_manifest.json (per-row item/id/label/component/channel),
base_manifest.json, build_summary.json.
