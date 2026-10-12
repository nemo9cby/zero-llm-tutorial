# 在 RunPod 或 Lambda 上运行原书脚本

本页是 `content/gpu.md`（GPU 实战手册）的操作附录：租一台 Linux GPU 机器，装好与本机相同的环境，
用 `gpu/run_local.py` 运行原书脚本，再把运行报告带回本书。价格与算力表在
`web/data/gpu/gpus.json`（2026-10-11 查证，来源 URL 写在表里）；估算用 `gpu/costs.py`。

## 什么时候选它们

| 情况 | 推荐 | 原因 |
|---|---|---|
| CodeBot、StoryBot、DPO | 本机 3060 Ti 或 Modal | 小模型：CodeBot 本机实测预训练 42 分钟、GRPO 约 1.5 小时；StoryBot 在 Modal H100 上实测预训练 16 分钟。Modal 按秒计费、一条命令 |
| WebBot 单卡全程（约 10 小时） | RunPod / Lambda 按需实例 | 整机按时间计费；RunPod 写明按需 Pod 的资源不会被其他用户挤占。Modal 的 GPU 函数默认都可能被抢占，而原书脚本不能断点续训 |
| WebBot 8 卡 | Lambda 8× H100 或 RunPod 8 卡 Pod | 单机 8 卡，NVLink；先读 `content/gpu.md` 的多卡陷阱 |
| 只想先摸底 | 任意平台跑 `webbot-probe` | 200 步计时探针；在 Modal H100 上实测 218 秒、约 0.27 美元 |

WebBot 单卡全程的时间来自 200 步探针（运行记录 `webbot-probe`）：每步 0.317 秒、每次完整评估约 56 秒，
外推全程约 10.4 小时（`content/gpu.md` 的验证 G.2-j）。事前按 MFU 35% 估算的是 9.1 小时。

2026-10-11 的部分按需价格（每 GPU·小时）：RunPod H100 SXM Community 2.69 / Secure 3.49，
RTX 4090 Community 0.34；Lambda 1× H100 SXM 4.29、8× H100 SXM 每卡 3.99。
来源：<https://www.runpod.io/pricing>、<https://lambda.ai/pricing>。RunPod 的 Community / Secure 价格取自
价格页里的结构化数据（JSON-LD）；页面卡片上显示的是另一组更高的价格（2026-10-12 查看：H100 SXM 3.99、
RTX 4090 0.89、B200 7.99），下单前以控制台为准。

## 一、把教程带到机器上

教程仓库没有远程仓库也没关系：把当前提交打成一个 bundle 复制过去即可，提交号会原样记进运行报告。

```bash
# 在你的电脑上（先 commit，未提交的改动不会进 bundle）
git bundle create zllm.bundle HEAD
```

原书代码不在 bundle 里：`setup_gpu_box.sh` 会用 `git submodule update --init` 从 GitHub 取回，
并核对它正是子模块固定的 `c9b6e2e`。

## 二、RunPod

1. 在控制台 Settings 的 Credentials 页添加 SSH 公钥（<https://docs.runpod.io/pods/configuration/use-ssh>）。
2. 先建一个 **Network Volume**（每 GB·月 0.07 美元），创建 Pod 时挂上；它出现在 `/workspace`，
   Pod 终止后数据仍在。只用 Pod 自带的 Volume Disk 也行，但 Pod 一终止就没了
   （<https://docs.runpod.io/pods/storage/types>）。WebBot 需要约 6 GB 数据加 10 GB 环境，给 40 GB 足够。
3. 镜像选任意带 CUDA 驱动的 Ubuntu 模板（例如官方 PyTorch 模板）；我们用自己的 uv 环境，不用镜像里的 torch。
4. 用 Connect 页 “SSH over exposed TCP” 那一行登录，只有这种方式支持 scp：
   ```bash
   scp -P <port> zllm.bundle gpu/setup_gpu_box.sh root@<ip>:/workspace/
   ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
   WORK=/workspace TUTORIAL_SRC=/workspace/zllm.bundle BOTS="storybot" bash /workspace/setup_gpu_box.sh
   ```

## 三、Lambda

1. 在控制台建一个 Filesystem，**与实例同一区域**，并在启动实例时挂上（之后不能再挂）
   （<https://docs.lambda.ai/public-cloud/filesystems/>）。它出现在 `/lambda/nfs/<名字>`。
2. 启动实例，登录用户是 `ubuntu`（<https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/>）：
   ```bash
   scp zllm.bundle gpu/setup_gpu_box.sh ubuntu@<ip>:/lambda/nfs/<名字>/
   ssh ubuntu@<ip>
   WORK=/lambda/nfs/<名字> TUTORIAL_SRC=/lambda/nfs/<名字>/zllm.bundle BOTS="webbot" bash /lambda/nfs/<名字>/setup_gpu_box.sh
   ```
3. Lambda 按分钟计费，从实例通过健康检查开始，到终止为止，闲着也计费
   （<https://docs.lambda.ai/public-cloud/billing/>）。

## 四、`setup_gpu_box.sh` 做了什么

1. 读 `nvidia-smi` 的驱动版本。`uv.lock` 在 Linux 上锁定的 torch 2.14.1 来自 PyPI，用 CUDA 13.0 构建，
   需要 580 以上的驱动；驱动在 525–579 之间时，改装同一版本的 CUDA 12.6 构建：
   `uv pip install --reinstall-package torch "torch==2.14.1+cu126" --index https://download.pytorch.org/whl/cu126`。
   必须用 `--index`：如果写成 `--index-url <cu126> --extra-index-url <PyPI>`，uv 会让 PyPI 排在前面，
   装回的还是 CUDA 13.0 版（<https://docs.astral.sh/uv/reference/cli/#uv-pip-install>）；钉住 `+cu126`
   则让选错版本时当场报错。之后写入 `UV_NO_SYNC=1`，免得 `uv run` 把它“修”回去。
2. 安装 uv，克隆 bundle，取回并核对原书子模块。
3. `uv sync --frozen`，然后用一行 Python 在 GPU 上自检，打印 torch 版本、CUDA 版本、几张什么卡、是否支持 bf16。
4. 没有 gcc 时装 `build-essential`：第 9 章的 `torch.compile` 要用 Triton 现场编译 kernel。
5. `uv run python gpu/fetch.py <bots>`：只下载这些机器人的作业要读的文件（断点续传、核对字节数）。

`DRY_RUN=1` 时只打印命令，不执行；`verify/test_gpu.py` 用它检查两种驱动分支。

## 五、运行与记录

在 tmux 里跑，SSH 断了训练也不停：

```bash
tmux new -s train
cd $WORK/zero-llm-tutorial && source $WORK/zllm.env
uv run python gpu/run_local.py --run storybot-4090 --job storybot
uv run python gpu/run_local.py --run webbot-probe-h100 --job webbot-probe          # 先测 200 步
uv run python gpu/run_local.py --run webbot-h100 --job webbot                      # 单卡全程
uv run python gpu/run_local.py --run webbot-8xh100 --job webbot --nproc 8          # 8 卡，总 token 不变
```

`--job` 的定义在 `gpu/jobs.py`，与 Modal 上跑的作业完全相同。`--nproc 8` 用 torchrun 启动 8 个进程，
同时把 `max_iters` 和 `warmup_iters` 除以 8：总 token 不变、最省钱，但全局 batch 变成 8 倍、更新次数变成 1/8，
峰值学习率没变，优化过程与单卡不同。`--scale batch` 则改为把每卡 batch 除以 8：优化过程与单卡相同，
但评估时另外 7 张卡空等，费用约翻倍。改了什么、为什么改，都会写进 `report.json` 对应步骤的 `patches` 字段。

`Ctrl-b d` 离开 tmux，`tmux attach -t train` 回来。日志在 `runs/<run>/<脚本名>.log`。

## 六、把结果带回本书

```bash
# 在你的电脑上，仓库根目录
scp -r -P <port> root@<ip>:/workspace/zero-llm-tutorial/web/data/runs/webbot-h100 web/data/runs/
uv run python gpu/register_run.py webbot-h100
uv run python tools/build.py
```

`register_run.py` 把这次运行登记进 `web/data/runs/index.json`，GPU 手册的“运行记录”和各章的训练曲线
只显示登记过的运行。检查点（`runs/<run>/work/**/*.pt`）不进仓库，需要的话另行 scp。

**最后，停止或终止实例。** 空闲的 GPU 照样计费；RunPod 的 Network Volume 与 Lambda 的 Filesystem
在实例终止后也继续按容量计费，不再需要时一并删除。

## 七、常见问题

| 现象 | 原因与处理 |
|---|---|
| `CUDA unavailable` | 驱动太旧而装了 CUDA 13 的 torch：重跑脚本（驱动在 525–579 之间时，它用 `--index` 改装 `torch==2.14.1+cu126`），并 `source $WORK/zllm.env`；`uv run python -c "import torch; print(torch.__version__)"` 应以 `+cu126` 结尾 |
| `uv run` 又开始下载 torch | 没有 `source $WORK/zllm.env`（低版本驱动分支需要 `UV_NO_SYNC=1`） |
| ch09 卡在 `torch.compile` 报找不到编译器 | `apt-get install -y build-essential` |
| ch09 要求登录 wandb | `run_local.py` 默认设 `WANDB_MODE=disabled`；想记录到 W&B 就先 `wandb login` 再设 `WANDB_MODE=online` |
| 8 卡跑完花了单卡 8 倍的钱 | 直接 `torchrun` 原样脚本会让每张卡都跑满 100,000 步；用 `--job webbot --nproc 8` |
| 磁盘满 | WebBot 的 `owt_train.bin` 5.3 GB，检查点每个约 0.6 GB；Volume 至少留 40 GB |
