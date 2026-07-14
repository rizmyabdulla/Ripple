# Research variant registry

Use one ADR per candidate and evaluate only under matched data/hardware with
`ripple.research.standard_variant`. Promotion requires every edge-invariant gate
plus any experiment-specific gates.

| Variant | Hypothesis | Custom kernel required | Status |
|---|---|---|---|
| pure-conv-mixer | Pure convolutional mixer beats local attention for edge CPU latency | No | proposed |
| emformer-memory | Emformer memory improves long-session stability | No | proposed |
| fused-ssm | Fused recurrent state reduces memory at equal quality | Yes | proposed |
| auxiliary-fsq | Auxiliary FSQ improves TTS learnability without VC loss | No | proposed |
| one-step-flow | One-step flow quality decoder helps server tier only | No | proposed |
| expressive-style | Separate style stream improves emotion without identity leak | No | proposed |
| hibiki-s2st | Multistream adaptive delay enables simultaneous translation family | Yes | proposed |
