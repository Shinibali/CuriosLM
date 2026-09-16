See [`../../LEARNING_PLAN.md`](../../LEARNING_PLAN.md) for how Claude should work in this repo
(guide vs. implement, and the infra/ops first-time-yourself rule).

To-do:

- Changed Layernorm to RMSnorm https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html
- Remove Dropouts
- implement Grouped Query attention
- Use Chinchilla optimal scaling for reducing token repetition in training
- I have a fixed $1,000 compute budget. What combination of data quality, model size and token budget produces the best result? Given the same compute budget, should I spend it on a larger model with fewer tokens or a smaller model with more tokens? Make a prediction, run several small models, and see.
- Can I build a robust data-mixture pipeline and quantitatively show the effect of different mixtures on several downstream domains?
- What actually happens to training efficiency and final quality when I switch from MHA → GQA at these small scales?
- Can I make a small model substantially better at a narrow reasoning task using verifiable rewards?
- Why does GRPO training collapse under certain reward conditions, and which interventions stabilize it?
- Train a small model with a verifiable reward. Deliberately create a bad reward and see whether the model learns to exploit it. Then fix it.


