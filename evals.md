
Read OpenAI's evals documentation and skimmed the Claude Mythos system card 
executive summary.

Key takeaway: evals aren't redundant with a model "understanding" its training 
data. LLMs are probabilistic, not deterministic, so strong performance on 
training-like examples doesn't guarantee correct behavior on new, unseen inputs. 
Evals are essentially a final exam for an already-trained model: they check 
generalization on held-out cases, not memorization.

Evals also serve a second practical purpose: regression detection. When a 
prompt changes or a model version updates, evals give a repeatable, numeric 
way to check whether the change helped or hurt, rather than just eyeballing 
a few outputs. This is exactly what the Mythos system card is: published 
eval results comparing the new model's capabilities and risk profile against 
prior versions.

System card content fell into two buckets: capability benchmarks (performance 
across domains/expertise areas) and safety/risk evaluations (comparisons 
against earlier model generations).