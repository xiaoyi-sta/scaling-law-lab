# 恢复清单（下次开机照着做）

## 上次停在哪

- 数据分片 **已全部建好**（`data/proc/` 6 个 `.bin`，共 4.7GB），在实例数据盘上，关机保留
- 代码已在 `/root/autodl-tmp`，`sentencepiece / pyarrow / huggingface_hub` 已装在系统盘
- 训练**尚未开始**（或刚开始就被中断），`logs/` 里只有 bench 输出
- GPU 实测吞吐**还没测到**——这是下次开机的第一件事

## 关机前（如果还没关）

```bash
# 用途：先停队列，避免进程被强杀留下半截 .tmp
pkill -f "run_all.sh"; pkill -f "train.py"
pgrep -af "train.py|run_all.sh"        # 应该没有任何输出

# 用途：确认 6 个分片都在
ls -lh data/proc/
```

然后控制台点 **「关机」**，不要点「释放」——释放会清空数据盘。

## 下次开机后（3 步）

```bash
# 1. 登录（端口按控制台「登录指令」为准）
ssh -p <端口> root@connect.nmb2.seetacloud.com

# 2. 进项目目录、确认数据还在
cd /root/autodl-tmp
ls -lh data/proc/ | head

# 3. 拉起队列（自动跳过已完成、从检查点续跑）
export HF_ENDPOINT=https://hf-mirror.com
nohup bash run_all.sh > logs/queue.log 2>&1 &
tail -f logs/queue.log
```

**不需要重新上传代码，也不需要重建数据。**

## 开机后 2 分钟要做的判断

看 `bench50M` 那行的 `tok/s`：

| 50M 实测 | 动作 |
|---|---|
| ≥45k | 不干预，跑完约 5 小时 |
| 35–45k | `pkill -f run_all.sh`，然后 `MICRO_SEQS=32 nohup bash run_all.sh > logs/queue.log 2>&1 &` |
| <35k | 停下来查原因，别硬跑 |

## 还欠的活

1. 拟合与推断代码**一行都还没写**：`fit_scaling_law`（WNLS、三种 CI、AIC/BIC/F 检验、外推 PI、消融联合模型、cluster bootstrap）
2. `scripts/sigma_run.py`：用 8M 三个 seed 估 σ_run 及 χ² CI
3. README（含 §4.2 的三个混杂层次、双验证集、批次受控对照）
4. 简历口径改写：不能说"筛选带来 X% 增益"，只能说"更换语料池"

## 花费记录

| 项目 | 金额 |
|---|---|
| 实例运行（数据构建 + 探索，约 40 分钟） | ~¥1.3 |
| 训练（未开始） | — |
| 预计总计 | ¥10–13 |
