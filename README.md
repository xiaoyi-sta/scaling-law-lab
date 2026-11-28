# scaling-law-lab

我用一张 RTX 4090 训练了几组小 GPT，想看模型大小和训练 token 数怎样影响验证 loss。第一轮用 8M、20M 和 50M 拟合 Chinchilla 的加性幂律。补训时我又做了两组对照，一组看学习率调度，另一组直接比较同算力下的 20M 和 25M。

## 补训跑出了什么

| 实验 | 实测结果 |
| --- | --- |
| 20M 在 0.75B tokens 截止 | 与旧长程训练在相同 token 数的检查点相比，两个短程终点的 loss 高了 0.025680 和 0.006394 nats。 |
| 50M 在 0.25B tokens 截止 | 单次对照的短程终点 loss 高了 0.181674 nats。 |
| 20M 与 25M，约 1.25 × 10¹⁷ FLOPs | 两个配对种子都由 20M 胜出，25M 的 loss 平均高 0.013900 nats。补训前冻结的曲面预测差值为 0.011023 nats。 |

我最在意的是第一个对照。同一个 N 和 D，可以对应不同的学习率轨迹，验证 loss 也会跟着变。以后用曲面推荐训练预算，我会按该预算结束调度再取终点。等算力结果目前覆盖一个预算、两个配置和两个配对种子。两对实测差值均超过 0.01 nats，跨种子的平均差值还有待估计。

这轮新增了 7 次从头训练，每次记录一个中途检查点和一个终点，用时约 4 小时 50 分钟。完整结果、原始 CSV、运行日志和复跑命令都在同一个目录。 [补训记录](results/followup/README.md)

## 第一轮留下了什么

原始拟合使用语料 A、验证集 A 上的 24 条记录，来自 6 次训练。加性幂律给出 α = 0.396、β = 0.436。α 对个别训练很敏感，删掉一条影响较大的 run 后会降到约 0.16。对照的 FineWeb-edu 模型在教育网页验证集上 loss 更低，在原始网页验证集上 loss 更高。这次语料比较里，验证集的选择能改变排序。

[第一轮研究记录](研究记录.md) · [拟合图](results/figures/01_fit_curves.png) · [语料对照图](results/figures/03_ablation_sign_flip.png) · [Chinchilla 论文](https://arxiv.org/abs/2203.15556)

## 从数据重算

只看补训对照结果，Python 标准库就够了。

```bash
python -m scripts.followup
```

要重算第一轮拟合和诊断图，在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
make report
make plots
make check
python -m scripts.selfcheck
```

新的克隆可以直接用归档 CSV 复算。重训产生的 CSV 写在 `logs/`，拟合脚本也支持显式指定 `--log-dir results/baseline/runs`。报告生成到 `results/report.md` 和 `results/report.json`。 [第一轮原始 CSV](results/baseline/runs)

语料分片和模型权重需要单独准备。仓库内有 tokenizer 和分片清单，用 `make bins` 可重建分片。 [分片清单](data/proc/manifest.json) · [GPU 复现步骤](results/followup/README.md#复跑)
