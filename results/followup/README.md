# 补训记录 · 2026-11-27

这轮一共跑了 7 次，都是从头训练。每次在中途和结束各评估一次。单张 RTX 4090 上的运行时间加起来是 290.1 分钟，累计处理约 45.32 亿 tokens。训练都用语料 A，验证集 A 有 8,882 篇文档。 [实验设定和输入校验值](experiment.json) · [逐次 CSV](runs) · [控制台日志](logs)

## 同一个 token 数，换训练时长会怎样

做法是固定模型、种子和已处理 token 数。旧值取自计划跑到约 1B tokens 的训练中途，新值来自在目标 token 数结束余弦衰减的短程训练。各 run 除总步数外沿用相同的优化器和学习率规则，warmup 按总步数的 2% 取整。50M 和 20M 的短程 warmup 分别为 38 和 114 步，原长程是 153 步。

| 模型 | 种子 | 截止 tokens | 旧检查点 loss | 短程终点 loss | 新减旧（nats） | 耗时（分钟） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50M | 0 | 249,954,304 | 3.221773 | 3.403447 | +0.181674 | 25.5 |
| 20M | 0 | 749,993,984 | 3.040154 | 3.065834 | +0.025680 | 44.7 |
| 20M | 1 | 749,993,984 | 3.039006 | 3.045400 | +0.006394 | 46.0 |

20M 的两个配对差值都为正，平均 +0.016037 nats，分别落在 0.01 nats 的两侧。这里比较的是整个训练时长设定，warmup 和余弦衰减一起变了。50M 只有一个种子，且旧检查点处在更早的调度阶段。要拆开尺寸和训练阶段的影响，需要交叉比较。 [原始长程 CSV](../baseline/runs)

## 同样的算力分给 20M 和 25M

20M 有 19,473,024 个非嵌入参数，25M 有 24,783,744 个。两者宽度都是 384，有 6 个注意力头，层数分别是 11 和 14。算力按下式估算。V 是词表大小，d 是模型宽度，L 是层数，S 是上下文长度，D 是训练 token 数。两档估算均约 1.2455 × 10¹⁷ FLOPs。

```text
训练 FLOPs ≈ 6 × (非嵌入参数量 + Vd + Ld(S+1)) × D
```

| 配置 | 步数 | 训练 tokens | 旧曲面预测的终点 loss |
| --- | ---: | ---: | ---: |
| 20M | 5,877 | 770,310,144 | 3.043110 |
| 25M | 4,736 | 620,756,992 | 3.054133 |

| 配对种子 | 20M loss | 25M loss | 25M − 20M（nats） | 20M / 25M 耗时（分钟） |
| --- | ---: | ---: | ---: | ---: |
| 101 | 3.045454 | 3.056913 | +0.011459 | 45.5 / 41.1 |
| 102 | 3.051065 | 3.067405 | +0.016340 | 46.2 / 41.1 |

两档在每一对里使用同一种子和同一语料，较短训练共享较长训练的数据顺序前缀。四次训练都在各自的预算终点完成衰减。旧曲面来自补训前对首轮 6 次训练的拟合，系数保存在实验清单里。这轮实测平均差值为 **+0.013900 nats**，旧曲面预测是 **+0.011023 nats**，误差约 **+0.00288 nats**。

两对种子均由 20M 胜出，也都超过预设的 0.01 nats 容忍值。最接近阈值的一对只高 0.001459 nats，所以这个结果适合报告当前预算下观察到的排序。要估计换种子之后的平均优势，需要更多配对重复。每条 CSV 的 `val_loss_se_doc` 来自验证文档，衡量种子之间的波动要看表中的配对差值。

## 复算

在仓库根目录运行，无需 GPU 或语料分片。

```bash
python -m scripts.followup
```

脚本按模型、种子和 token 数匹配旧检查点，根据两个新终点算配对差值，并用清单中的冻结系数重算预测。每个 CSV 保留了中途和终点两行，所以这轮有 14 条观测和 7 次训练。

## 复跑

使用 Python 3.11 或更新版本，以及带 BF16 的 CUDA 环境。训练前准备清单里的语料 A 和验证集 A 分片，可以用仓库内的 tokenizer 和 `make bins` 重建。下面的命令按清单跑完七组，使用一个新目录写入 CSV 和检查点。实际耗时会随硬件和输入输出速度变化。

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("A CUDA device with BF16 support is required.")

root = Path.cwd()
data = root / "data" / "proc"
experiment = json.loads((root / "results/followup/experiment.json").read_text())
for name, expected in experiment["environment"]["input_sha256"].items():
    with (data / name).open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise SystemExit(f"Unexpected input file: {name}")

out = root / "results/replays/followup"
out.mkdir(parents=True)
for run in experiment["runs"]:
    subprocess.run([
        sys.executable, "-u", str(root / "train.py"),
        "--model", run["model"], "--arm", "A", "--seed", str(run["seed"]),
        "--steps", str(run["steps"]), "--val-sets", "A",
        "--run-id", run["id"], "--data-dir", str(data),
    ], cwd=out, check=True)
PY
python -m scripts.followup --runs-dir results/replays/followup/logs
```
