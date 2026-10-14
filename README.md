# epf-spf-its-for-audio-reasoning

Particle filtering over the reasoning steps of an audio-language model answering
multiple-choice questions. One small, tested implementation of four methods:

| method | what it is |
|---|---|
| `epf-paper` | Entropic Particle Filtering exactly as the ePF paper's text describes it |
| `epf-itshub` | ePF as the authors' reference code, upstream `its_hub`, implements it |
| `epf-project` | this project's judge arm (`judge_epf_prm`) with a configurable weight temperature, T = 3 by default |
| `spf` | Self-Probing Particle Filtering: no reward model, the generator's own forced answer is the score |

The score source is independent of the method: any method runs with the external judge
PRM (`--scorer judge`) or with the self-probe (`--scorer probe`). Every knob of every
method can be overridden from the command line, and every result row records the full
resolved configuration.

## Quick start

```bash
uv venv .venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
export EPF_DATA_ROOT=/path/to/epf_data
export EPF_EXTENDED_DIR=/path/to/extended_dataset/distractor_results

# check the resolved configuration and the data, without touching any server
.venv/bin/epf-spf --method epf-paper --bench mmar \
    --generator-endpoint http://127.0.0.1:9401/v1 --generator-model qwen-omni \
    --judge-endpoint http://127.0.0.1:9311/v1 --judge-model qwen-omni-3b-judge \
    --budgets 1,8,64 --output runs/mmar_qwen-omni_epf-paper.jsonl --dry-run
```

Drop `--dry-run` to run. Serving the models is covered in the parent tree's
`env-setup.md`.

## Documentation

Start at [docs/index.md](docs/index.md).

## Tests

```bash
.venv/bin/pytest            # 74 tests: unit, guards, and parity with the previous code
```

The parity tests compare against the previous implementation in
`../entropic-particle-filter/` and the data in `../epf_data/`; they skip if either is
missing. See [docs/testing.md](docs/testing.md).
