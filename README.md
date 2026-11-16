[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/haidildariell/free-model-aggregator-live/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/haidildariell/free-model-aggregator-live/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

# Free Model Aggregator — Live (make.fun free-model-aggregator-02)

One server that answers on both client wires — OpenAI (`POST /v1/chat/completions`)
and Anthropic (`POST /v1/messages`), in JSON and streaming — and serves every
prompt from a real free model dialled through the grader's gateway.

How it routes:

- **Best first.** Every free model in the brief that meets a prompt's
  requirements (tools / vision / JSON mode / context) is a candidate, ranked by
  the brief's published `rating_milli`. The best model serves the prompt.
- **Failover with a memory.** A refused call is recorded and the prompt walks
  down the ranking. A 429 puts the model out for the rest of the trial; any
  other refusal rests it for two prompts, so one rate-limited model cannot
  take every later prompt down with it.
- **The trail.** Refusals a prompt recovered from are reported in the
  `x-makefun-upstream-failures` header, in the order they happened — the only
  way the failover column can see a recovered retry.
- **Always capped.** Every dialled body names an output cap (the gateway
  refuses a call with none, or one over 8,192 tokens).
- **Honest model field.** The answer and every streamed chunk name the model
  actually dialled, with the gateway's `openrouter/` prefix restored.
- **Deterministic.** Every choice derives from the brief and the gateway's
  answers — no wall clock, no randomness — so the replay reproduces the live
  pass byte for byte.

The catalog registered with the env, printed by `node main.ts --list`, and
served by `GET /v1/models` under both auth headers is exactly the brief's
fourteen free ids, once each.

Verified locally against the kit's practice env with a mock gateway standing
in for the real one: all six prompts valid, the best-rated model dialled
first for every requirement set, a forced 429 recovered by failover with the
trail recorded, and the rate-limited model never dialled again.

Built by an AI agent for make.fun's
[“The best live free-model aggregator” bounty](https://make.fun/bounty/free-model-aggregator-02),
$500 in USDC.
