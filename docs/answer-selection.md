# Answer selection

A run ends with N particles, each with a trajectory, a predicted letter, a final
log-weight, and per-step scores. A **selector** turns them into one answer. Every
selector is computed from the stored row (`select_answers` in
[`selection.py`](../src/epf_spf/selection.py)). Selection is therefore an analysis
choice, made after the run, and needs no re-run.

`select_answers` is an exact port of `selectors_from_row` in the previous code.
`tests/test_parity.py` replays stored result rows through both and requires identical
output.

## The predicted letter of a particle

`pred_letters[i]` comes from the particle's full text (`answers.predicted_index`, ported
verbatim from the previous `scoring.py`):

1. a letter in `\boxed{…}`, else
2. the letter after the last `Answer:`, else
3. the last standalone capital letter that is a valid option, else
4. a fuzzy match of the text against the option texts (threshold 0.6).

An empty string means no answer could be read. It never counts as a vote.

## Deduplication

Resampling copies particles, so N particles often hold fewer distinct trajectories.
Most selectors work over **distinct** trajectories: particles are deduplicated by
`texts_hash`, and only the first copy is kept. A trajectory that resampling copied 20
times then votes once. `argmax_w`, `majority` and `probe_marginal_all` use all N
particles.

## Selectors

| selector | rule | uses | fallback |
|---|---|---|---|
| `argmax_w` | letter of the particle with the highest final log-weight | all N | |
| `majority` | most common letter | all N | `""` |
| `dedup_majority` | most common letter | distinct | `""` |
| `probe_marginal` | sum each trajectory's final probe distribution; argmax letter | distinct | `dedup_majority` |
| `probe_marginal_all` | same | all N | `majority` |
| `probe_marg_prestop` | as `probe_marginal`, using the probe before the last step | distinct | `dedup_majority` |
| `probe_marg_mean` | as `probe_marginal`, averaging each trajectory's probes over its steps | distinct | `dedup_majority` |
| `ent_marg` | final probe distribution weighted by certainty 1 − H/log K | distinct | `dedup_majority` |
| `ent_marg_mean` | same, averaged over steps | distinct | `dedup_majority` |
| `probe_argmax_text` | letter written by the trajectory with the most confident probe | all N | `argmax_w` |
| `probe_argmax_letter` | the probe's top letter in that trajectory | all N | `argmax_w` |
| `judge_marg_mean` | sum each trajectory's mean judge score per written letter | distinct | `dedup_majority` |
| `judge_marginal` | sum each trajectory's final judge score per written letter | distinct | `dedup_majority` |
| `argmax_judge` | letter of the trajectory with the highest final judge score | distinct | `dedup_majority` |

- Probe selectors need `probe_dists_all`, which only the probe scorer fills. Under the
  judge scorer they fall back as listed.
- Judge selectors need `judge_scores_all` and are absent from rows run with the probe.
- Ties go to the letter that reached the maximum first.
- In `ent_marg*`, H is the entropy of the renormalised distribution and K the number of
  letters in it (at least 2). A trajectory without probe data adds 1.0 to its written
  letter.

## Which selector goes with which method

| method | selector that matches the method's source | notes |
|---|---|---|
| `epf-paper` | `argmax_w` | the paper: "The final solution from each run is selected using the same PRM that provides intermediate guidance" |
| `epf-itshub` | `argmax_w` | its_hub `SelectionMethod.ARGMAX`, `np.argmax(log_weights)` |
| `epf-project`, `spf` | none fixed by the method | all selectors are stored; report the same selector as the results you compare against |

With `score_aggregation=last`, `argmax_w` and `argmax_judge` choose by the same quantity,
the last judge score, but they are not identical: `argmax_w` looks at all N particles and
`argmax_judge` at distinct trajectories, and ties may break differently.

## Accuracy

`run.py` logs per budget, at the end of a run, the accuracy of every selector over the
rows of this run's fingerprint (error rows excluded). For analysis, read the JSONL and
call `select_answers(row)`. Compare with `row["gold"]`, the gold letter.
