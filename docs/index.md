# Documentation

This repository implements particle filtering over reasoning steps for audio
multiple-choice questions, in four variants, with one shared loop and one config
object that holds every setting that changes results.

## Why this repository exists

The previous code (`../entropic-particle-filter/`) grew around many experiments. Its
most consequential setting, the weight temperature of the judge arm, was a value
hard-coded per arm in one file. That file existed in several copies with different
values, there was no command-line flag for it, and result files did not record it. The
project's documentation specifies T = 3, but the Fir extension runs used T = 1, and
nothing in their output said so (see [weight-temperature.md](weight-temperature.md)).

This repository is the clean version of the same methods:

- every setting that changes results is a field of `FilterConfig`, printed at startup
  and written into every result row
- an output file can only ever hold rows of one configuration
- the methods the project compares are named presets that document where each value
  comes from
- the parts that must not drift (prompts, data, answer extraction, judge prompts,
  selectors) are carried over and checked against the old code by tests

## Read in this order

| document | what it covers |
|---|---|
| [methods-overview.md](methods-overview.md) | the four methods side by side, every knob, and how to choose |
| [epf-paper.md](epf-paper.md) | ePF as the paper describes it, step by step, and how it maps to code |
| [epf-its-hub.md](epf-its-hub.md) | ePF as upstream its_hub implements it, and where that differs from the paper |
| [weight-temperature.md](weight-temperature.md) | `log` vs `logit`, the weight temperature T, measured effects, and history |
| [spf.md](spf.md) | Self-Probing Particle Filtering |
| [scorers.md](scorers.md) | the judge PRM and the self-probe: prompts, scores, failure handling |
| [answer-selection.md](answer-selection.md) | how the final answer is read off the particles |
| [running.md](running.md) | installing, every command-line flag, worked examples, resuming, sharding |
| [output-format.md](output-format.md) | result rows, the per-step trace, provenance, the manifest |
| [data.md](data.md) | the 13 benchmarks, their sizes, and the 30-second audio override |
| [parity-and-differences.md](parity-and-differences.md) | what was carried over unchanged, what deliberately changed, what is not implemented |
| [testing.md](testing.md) | running the tests and what each group guards |

## Layout

```text
src/epf_spf/
  config.py       FilterConfig, the four presets, validation
  filter.py       the particle filter loop (one loop for every method)
  weighting.py    score -> log-weight: aggregation over steps, log/logit, temperature
  resampling.py   softmax, ESS, entropic annealing, the SPF gate, systematic/multinomial draws
  scorers.py      JudgeScorer (external judge PRM) and ProbeScorer (self-probe)
  steps.py        the "Step N:" step format and the stop rule
  llm.py          OpenAI-compatible client for the generator
  prompts.py      generator and judge prompts
  data.py         benchmark loaders
  answers.py      answer extraction
  selection.py    final-answer selectors
  results.py      result rows, resume, provenance, manifest
  run.py          command-line entry point
tests/            unit, guard, and parity tests
```
