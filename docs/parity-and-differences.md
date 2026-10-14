# Parity with the previous implementation, and differences

"The previous implementation" is `../entropic-particle-filter/`: the project's fork of
its_hub (`its_hub/core/algorithms/particle_filtering.py`), the experiment driver
`babel-ship/probe_epf.py`, the judge client `babel-ship/judge_client.py`, and the
benchmark code `benchmarking/mmau_pro/` (prompt, scoring, loaders).

This repository does not import any of it at runtime. The parts below were re-written,
and the parity tests import the old code and check that both produce the same output
for the same input.

## Carried over unchanged (parity-tested)

| part | new | old | test |
|---|---|---|---|
| generator prompt (method 12, "stepwise narrowing"), with and without think trigger | `prompts.generator_messages` | `prompt.build(12, …)`, `add_think_trigger` | `test_generator_prompt_is_identical` |
| judge step and outcome templates | `prompts.JUDGE_*_TEMPLATE` | `judge_client.JUDGE_PROMPT_TEMPLATE*` | `test_judge_templates_are_identical` |
| which prompt the judge gets for a particle at each step | `JudgeScorer.prompt_for` | `_judge_item` + templates | `test_judge_sees_the_same_prompt_at_every_step` |
| step rendering (`Step k:` headers), continuation text, stop rule | `StepFormat` | `HeaderSeedStepGeneration._post_process`, `_step_is_final` | `test_step_rendering_and_stop_rule_are_identical` |
| probe prefix | `ProbeScorer.prefix_for` | `AnswerProbeEPF._probe` | `test_probe_sends_the_same_prefix` |
| P(Yes) from top-20 logprobs | `yes_probability` | `judge_client.yes_probability` | `test_yes_probability_is_identical` (300 random cases) |
| letter masses from probe logprobs | `letter_masses` | `probe_epf.letter_masses` | `test_letter_masses_are_identical` (300 random cases) |
| judge weight logit(s)/T, T = 1 and 3 | `log_weight(s, "logit", T)` | `_inv_sigmoid` on the clamped score, / T | `test_judge_weight_is_identical` |
| ESS annealing temperature (3 threshold/window settings) | `annealing_temperature` | `EntropicParticleFiltering._temperature_annealing` | `test_annealing_is_identical_to_its_hub` (300 random cases each) |
| systematic resampling draw | `systematic_indices` | `_resampling_systematic_indices` | `test_systematic_resampling_is_identical_to_its_hub` |
| answer extraction | `answers.predicted_index` | `scoring.predicted_index` | `test_answer_extraction_is_identical_on_stored_texts` |
| selectors | `select_answers` | `selectors_from_row` | `test_selectors_are_identical_on_stored_rows` |
| all 13 benchmark loaders | `load_benchmark` | `load_records` | `test_loaders_are_identical` (one per benchmark) |

Generator request fields (`continue_final_message`, `add_generation_prompt`, stop
strings, `include_stop_str_in_output`, the probe's greedy 3-token request) follow the
old code. They are not parity-tested, because that would need a live server.

## Deliberate differences

| | previous | here | why |
|---|---|---|---|
| weight temperature | hard-coded per arm in `build_arm`, differing between copies of the file; not recorded | a `FilterConfig` field, a flag, in every row | [weight-temperature.md](weight-temperature.md) |
| run identity | none; a resumed run could append rows of a different setting to the same file | `run_fingerprint`; mixing is refused | [output-format.md](output-format.md) |
| failed judge call | counted as s = 0.5 silently | the item fails by default (`--on-score-failure fail`); `neutral` restores the old behaviour and counts the failures | an overloaded judge must not quietly turn a run unguided |
| `texts_hash` | `hash(t) % 10**8`, salted per Python process | sha1, first 12 hex characters | stable across processes. Within one row both work the same, and the selectors only compare hashes within a row. |
| resampling random numbers | `np.random` global state, so they depend on item interleaving | a generator per cell, seeded by `(seed, crc32(unique_id), budget)` | a resumed or re-sharded run draws the same numbers |
| 30-second audio override | applied implicitly when the folder existed | applied by default, recorded in every row, `--no-audio-override` to disable | [data.md](data.md) |
| trace | | `trace` per round: scores, log-weights, β⁻¹, ESS before and after annealing, parents; see [output-format.md](output-format.md#the-trace) | |
| method and scorer | fixed pairs per arm | independent: any method with either scorer | |
| paper ePF and its_hub ePF | not available as separate settings | `epf-paper` and `epf-itshub` presets | |

## Small numerical differences

- **Clamp before the logit:** [1e-6, 1 − 1e-6], as the previous project code did.
  Upstream its_hub clips at 1e-7. See [epf-its-hub.md](epf-its-hub.md#known-small-differences).
- **ESS clip:** annealing uses 1e-7, as in its_hub. The SPF gate uses 1e-12, as the
  previous gate did.
- **Systematic resampling:** identical to the previous code, including its guard
  that raises the last cumulative weight to 1.0 (parity-tested).

## Not implemented

| | where it was | note |
|---|---|---|
| plain particle filtering without annealing as a named method | its_hub `ParticleFiltering` | `--no-annealing` on any preset gives it |
| self-certainty weights (the generator's own token logprobs) | the project's its_hub fork | not one of the four methods |
| Look-ahead Modulation (LaM) | ePF paper | not implemented upstream either |
| prompt methods other than 12 | `benchmarking/mmau_pro/prompt.py` | the two ported arms are used with method 12 |
| other arms in `probe_epf.py` | | only `judge_epf_prm` (`epf-project`) and `probe_adaptive_t3` (`spf`) were ported |

## Comparing with old results

Rows from the old code and from this code are comparable when the settings match:

- same generator, judge, benchmark, `--max-steps 10`, `--sampling-temperature 0.8`, the
  30-second override on
- `epf-project` at the T the old run actually used (see `../audit/`)
- the same selector

They will not match item by item, because generation is sampled. They should match in
distribution.
