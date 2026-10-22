# Demo recording script

Target duration: **2:50**, below the three-minute limit. Record the working application
after the team deploys it. If recording locally for rehearsal, label it local.
This script is ready for recording; it is not evidence that a video has been uploaded.

| Time | Screen/action | Narration |
|---|---|---|
| 0:00–0:15 | Baseline vehicle and parameter controls | “CFD is valuable and expensive. Aero Surrogate helps engineers screen candidate vehicle designs before their next CFD verification.” |
| 0:15–0:35 | Move one dial; show Cd, sigma and domain score | “Our parameter-based drag model learns from 4,165 CFD-labelled designs. Independent grading on 625 held-out designs gives R-squared 0.875 and mean absolute Cd error 0.00616.” |
| 0:35–1:20 | Start a genuine Super run; show tools, curve and logbook | “NVIDIA Nemotron Super runs through Nebius Token Factory. It states a hypothesis, proposes a change, calls the surrogate and reads what actually happened. Numeric constraints and the evaluation budget are enforced by the service.” Show a hypothesis contradicted by the result, then the next proposal. |
| 1:20–1:45 | Results comparison with all three methods | “Across five runs per method at 25 evaluations, median predicted Cd reduction was 3.88 percent for Nemotron, 3.84 for CMA-ES and 2.79 for random search. This small experiment shows comparable search results with an inspectable logbook. These are predictions, not new CFD validation.” |
| 1:45–2:15 | Load the legal joint-extreme reference; show rejection and unchanged best curve | “Every dial here is individually legal, but their combination is far from the training data. The domain gate rejects it and explains why. Passing a domain check still does not prove physical correctness.” |
| 2:15–2:35 | Reliability chart; optional experimental pressure overlay | “We report the limitations. Our nominal 90 percent interval covered 86.6 percent of held-out labels. Surface pressure is a separate experimental model; its confidence on generated viewer geometry is not established.” |
| 2:35–2:50 | Application, source and demo links | “A fast design assistant with explicit domain limits, reproducible comparisons and a trace engineers can inspect. Nemotron on Nebius powers the loop; CFD remains the final check.” |

## Recording checks

- Use the release scalar checkpoint and display its identity before recording.
- Start the real agent. Clearly label any historical replay or time compression.
- Do not show API keys, an operator token, account identifiers or private dashboards.
- Do not narrate arbitrary mesh upload, pressure-driven agent reasoning, lift or wall shear.
- Keep smoke/race footage secondary and labelled illustrative.
- Do not call the solver-spread reference a hard accuracy ceiling.
- Name NVIDIA and Nebius aloud and show Nemotron / Token Factory on screen.
- Export below three minutes; watch the entire result and check text legibility.
- Use licensed audio/assets only; a silent screen recording with narration is sufficient.
- Team uploads publicly to YouTube, then tests playback logged out and adds the URL to submission.md.
