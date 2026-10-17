# Study 2 excluded attempts

No attempt below is eligible for any Study 2 analysis. Each was an execution check run
before registration; none trained a P2, P3, clean-cell, or CRIC P2 model, so none could
reveal a protocol contrast. Their metrics were checked only for finiteness and completeness
of artifacts, never compared across protocols.

| Attempt | Architecture | Split | Epochs | Purpose |
|---|---|---|---|---|
| `p1_<architecture>` (12 runs) | all twelve registered backbones | SIPaKMeD P1, seed 137 | 1 | pipeline, artifacts, throughput |
| `p1_resnet18_repeat` | ResNet-18 | SIPaKMeD P1, seed 137 | 1 | bitwise determinism (logits and features identical to `p1_resnet18`) |
| `leak_high_resnet50` | ResNet-50 | 2C `leak_high`, seed 137 | 1 | transplant extra-evaluation artifact |
| `p1_grl_resnet50` | ResNet-50 | SIPaKMeD P1, seed 137, lambda = 1 | 1 | adversary bookkeeping and DANN ramp |
| `cric_p1_resnet18` | ResNet-18 | CRIC P1, seed 137 | 1 | CRIC window cache and six-class labels |
| `p1_vit_b_16_math`, `p1_vit_b_16_math_repeat`, `p1_dinov2_vits14_math` | ViT-B/16, DINOv2 ViT-S/14 | SIPaKMeD P1, seed 137 | 1 | deterministic (math) attention kernels: speed and repeatability |

The first ViT-B/16 smoke run used fused memory-efficient attention, whose backward pass
PyTorch reports as non-deterministic. Registration therefore fixes the deterministic math
attention kernel for every run (Section 1 of `PREREGISTRATION_STUDY2.md`).

Outputs are kept on the training host under `smoke/` and are not copied into `results/`.

Before registration, the 2C factorial configuration with 240 crops per class was rejected
because it was infeasible for registered seed 307 (and for 10 of 100 unrelated seeds); no
model was trained on it.
