"""The "Step N:" reasoning-step format used with the method-12 prompt.

One step is one "Step N: ..." line. Step 1 is a fresh generation; every later step
continues the assistant message, which ends with the seeded header of the next step.
Seeding the header matters: vLLM's continue_final_message strips trailing whitespace
from the assistant turn, so a bare separator would be lost and the model would
immediately hit its own stop string, producing empty steps.

Matches `HeaderSeedStepGeneration` (header "Step", separator "\\n") and its_hub's
`StepGeneration` stop rule in the previous implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

HEADER = "Step"
SEPARATOR = "\n"
FINAL_MARKER = "Answer:"


@dataclass(frozen=True)
class StepFormat:
    """`max_steps` follows its_hub's rule: a particle stops when it already has
    `max_steps` steps before generating, so it can end with `max_steps + 1` steps.
    `inline_step_stop` also cuts at " Step ", for models that chain steps on one line.
    """

    max_steps: int = 10
    inline_step_stop: bool = False

    @property
    def stop_strings(self) -> list[str]:
        return ["\nStep ", " Step "] if self.inline_step_stop else ["\nStep "]

    def headered(self, steps: list[str]) -> list[str]:
        # Step 1 arrives with its own header and is kept verbatim; continuations start
        # right after the seeded "Step k:" and get it back here.
        return [
            step if step.lstrip().startswith(HEADER) else f"{HEADER} {i + 1}:{step.rstrip()}"
            for i, step in enumerate(steps)
        ]

    def trajectory(self, steps: list[str]) -> str:
        """The reasoning so far, as shown to the judge and stored as the final text."""
        return SEPARATOR.join(self.headered(steps))

    def continuation(self, steps: list[str]) -> str:
        """Assistant turn to continue from: the steps so far plus the next header."""
        return f"{self.trajectory(steps)}{SEPARATOR}{HEADER} {len(steps) + 1}:"

    def is_final(self, steps_before: list[str], new_step: str) -> bool:
        return len(steps_before) >= self.max_steps or FINAL_MARKER in new_step
