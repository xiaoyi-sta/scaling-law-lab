# scaling-law-lab

在单张 RTX 4090（¥1.88/h）上完成 Chinchilla 式 scaling law 复现与数据质量消融：
8M / 20M / 50M 同构模型族、6.4 GPU 小时、约 ¥12。

## 从这里开始

| 文档 | 内容 |
|---|---|
| **[FLOW.md](FLOW.md)** | **完整流程**：做了什么、为什么这么做、踩了什么坑、最终结果 |
| [SPEC-v3.md](SPEC-v3.md) | 计划文档（含设计阶段的十项修正） |
| [results/report.md](results/report.md) | 自动生成的统计报告（六节 + 结论） |

## 三句话结论

1. `L(N,D) = A/N^α + B/D^β + E`，α = 0.396、β = 0.436、E = 2.29（24 观测 / 6 run）；
   同一个 α，四种置信区间口径给出 0.26–0.50 的宽度。
2. 小尺度外推大模型：覆盖率 5/6，MAPE 0.41%。
3. **数据消融的 ΔL 符号随验证集分布翻转**（raw 上 +0.16，edu 上 −0.13），
   且与 D 无关——批次受控对照复现同样结果。

## 快速开始

```bash
pip install -r requirements.txt
make check              # 参数表 / 形状 / 标签泄漏护栏
make report-synthetic   # 用合成数据自检拟合流水线
make report             # 真实日志 → results/report.md
python -m scripts.selfcheck   # 21 项数值自检
```
