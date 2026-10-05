# manotorch 审查与优化验收报告

日期：2026-10-05（America/Chicago）。审查基线：`optimize/c936b59`。本文保留在公开开发分支 `optimize`；`master` 和 release tag 明确排除本文，仅发布英文代码及文档。最终发布与远程验证状态以 [GitHub Release](https://github.com/IRVLUTD/manotorch/releases/tag/v0.1.0) 及其 Actions 记录为准。

## 结论

现有重构的主要收益有代码依据：blend shapes 使用矩阵乘法、FK 按五条指链批量计算、skinning 避免大尺寸齐次中间张量、`joints_only` 只计算五个指尖顶点；依赖和安装资源也完成了实质清理。这些优化值得保留。

本轮补齐了数值边界、自动微分兼容性、checkpoint 缓存和验证流程，并减少了正确性修正带来的 eager 开销。纯 MANO layer 的当前 eager 仍比 Claude `c936b59` 慢，但已快于上游原版。新增完整 Anatomy loss fitting 实测后，当前链路比 Claude 快 1.33–1.46×、比上游快 1.47–1.85×；loss 向量化的收益体现在这个完整循环中。不同 workload 的速度需要分别评价，prior 也会改变拟合目标，不能等同于观测精度改善。详见报告末尾的追加比较。

发布方案为将完整 `optimize` 修改合入 `master`，版本采用 **0.1.0**。本轮未合并 `mano-isaac`；下游替换需要保留各自的 shapedirs 和指尖约定，FoundationEgo 的左手 anatomy loss 迁移尤其需要单独验证。

## 文档与实现核对

| 项目 | 对实现的评价与本轮处理 |
| --- | --- |
| Blend shapes / FK / skinning | 结构优化真实存在；保持五条指链与自定义 eager backward。compile 改走数学等价的原生运算，使 Inductor 能融合，解决 custom JVP 与 Dynamo 的冲突。 |
| `joints_only=True` | 保留 21 个输出关节，`verts=None`；不适用于需要完整网格或其他指尖顶点的消费者。大 batch 节省显存得到复测支持。 |
| Rotation conversions | 基线在零 axis-angle 的二阶微分产生 NaN；改用平方角 Taylor 分支，避免先求零向量范数。覆盖 AA→matrix、AA→quaternion 和 quaternion→AA。 |
| Euler inverse | 补齐 12 种 convention 的精确奇异点重建，最后一角固定为零。保护有效与无效分支的 atan2 参数，兼容旧 PyTorch 的 backward。奇异点逆映射不光滑，有限分支梯度不等于可唯一反演。 |
| `torch.func` | 自定义 skinning 增加 setup_context、JVP 与 vmap 支持；集成验证 jacrev/vmap/jvp，保留二阶梯度与 fullgraph compile。 |
| Checkpoint | FK 派生模板旋转和 wrist-closed faces 在加载后刷新；不新增持久 state-dict keys。 |
| Anatomy loss | 默认配置不再共享可变列表；校验限位格式，向量化区间惩罚。保持 15 项输出顺序、reduction、半精度提升行为与公开列表编辑用法；设备／dtype 初始化和 `.double()` 后缓存重建得到验证。 |
| MANO loading | 核心仍只需 NumPy/PyTorch；增加平铺目录及 `_new.pkl` / `_np.pkl` 回退。协议 2 的 bytes 重建仅接受受限 Latin-1 操作。NPZ 优先。 |
| Translation / shapedirs | translation 在 centering 后加入，单位 metre；`fix_left_shapedirs` 默认 False，保持官方模型约定。下游已自行修复时只能修复一次。 |
| Precision | 模型缓冲区首次以 float32 构造，之后 `.double()` 无法恢复原始精度；删除 README 中不适用于本加载路径的约 1e-16 精度承诺。 |
| Anatomy / compose / anchors | 重新说明 adaptive basis 与 FK basis 的区别、度／弧度、关节顺序、左手符号及 compose 的全局旋转范围。Anchor 为包内 32 个 barycentric anchors，安装后可直接找到资源。详见 [features.md](features.md)。 |
| 工程质量 | 固定 Ruff 规则，清理有意义的告警；增加 CPU CI、最低支持版本、缺模型提示和 wheel 安装验证。现有 API 移除及左手符号变化保留在 CHANGELOG 的兼容性说明中。 |

独立 NumPy Rodrigues 参考用于避免旧 PyTorch 的批量 `matrix_exp` 数值误差被误判为新实现退化。基线运行新增回归时确实触发多个失败，说明新增用例覆盖了原有缺陷。

## 验证结果

| 环境／检查 | 结果 |
| --- | --- |
| Python 3.12.13 / PyTorch 2.11.0+cu126 / NumPy 2.5.3，CPU 与 RTX 4090，双手模型齐全 | **157 passed**，使用 `pytest --require-mano`。18 个 warning 来自 PyTorch 2.11 首次 JVP 初始化时注册 TorchScript decompositions；不导入 manotorch 的原生 sin JVP 也可复现。manotorch 常规 forward/backward 的严格无警告 guard 和无同步 guard 通过；整个 torch.func 首次调用仍非 warning-free。 |
| Python 3.10.20 / PyTorch 2.0.1+cpu / NumPy 1.26.4，双手模型齐全 | **152 passed，5 skipped**；跳过项为 2 个 CUDA 检查和 3 个 FP16 CUDA 参数。 |
| Ruff / diff whitespace / uv lock | 通过。 |
| 构建和安装 | wheel 包含四个 anchor 资源、LICENSE、NOTICE；在隔离环境从 `/tmp` 导入，AnchorLayer 初始化通过。 |
| 模型转换工具 | NumPy 1.26 与 2.5 转换官方右手 pickle，生成 NPZ 的全部 keys/dtypes/values 一致，没有误存 `allow_pickle` 键。 |
| GitHub Actions | `optimize/cef8318` 的 [远程 CI](https://github.com/IRVLUTD/manotorch/actions/runs/37315663470) 已通过：两组 CPU 环境各 72 passed / 85 skipped，lint 通过。公共 CI 不包含 licensed MANO 模型或 CUDA，不替代本地完整验证；master 和 tag 分别再次执行 CI。 |

最低 PyTorch 约束改为已验证的 **2.0.1**。**仅本次验证的 PyTorch 2.0.1 二进制环境需要 NumPy < 2**：其 NumPy bridge 与 NumPy 2 的 ABI 不兼容，不是 manotorch 对所有环境的限制。当前开发环境为 PyTorch 2.11 / NumPy 2.5.3 / CUDA 12.6，已通过完整测试，无需降级 NumPy。核心库只依赖 torch 和 numpy，旧下游审计额外需要 scipy/deprecation。

## 专用 fitting benchmark

以 `c936b59` 生成并冻结目标，固定 seed `20261005`、初值及 Adam 学习率。本轮初次正确性修正快照比较相同数据的 48 对案例：双手 × batch 1/32/128/1024 × joints/full、joints-only、vertices × eager/compile。每例 100 个 fitting steps，计时前 warmup，单独测 forward、forward+backward、完整优化步、显存和首次编译。

MANO_Poses 每侧选取 128 个训练姿态。batch 1024／16384 重复该样本库，是吞吐测试；并非相同数量的独立手。目标与原始数据不发布，manifest、脚本和复现说明保留在仓库，详见 [benchmark.md](benchmark.md)。

| 几何收敛指标 | 结果 |
| --- | --- |
| 100 步后 RMSE，全部当前案例 | 0.19836–0.50245 mm |
| eager，基线与当前最终 RMSE 最大差 | 1.94e-6 mm |
| compile，基线与当前最终 RMSE 最大差 | 3.47e-6 mm |
| 达到 1 mm 的步数 | 35–46；48 对案例的步数全部一致 |

误差是点距离的 RMS，不能解释为 pose/betas 唯一恢复。训练姿态产生的目标用于稳定比较实现，不是独立的观测精度评估。

### 时间与显存

硬件：共享 RTX 4090。Python 3.12 / PyTorch 2.11 / CUDA 12.6。下面的数值是本轮测量，不承诺跨硬件或负载复现同一倍率。

| 24 案例优化步时间的中位数 | 基线 | 当前 |
| --- | ---: | ---: |
| eager | 14.807 ms | 17.153 ms |
| compile | 2.698 ms | 2.822 ms |

该汇总跨不同 batch/task，不能代替逐案例的配对统计。eager 中位数增加约 15.8%，与新增数值保障带来的操作数增加一致，也混有共享机器调度波动；不能据此精确分离每个原因。compile 约 4.6% 的差异不足以支持稳定退化结论。当前 compile 与 eager 的中位数相差约 6 倍，适合形状固定、重复优化的 fitting。首次差分编译实测 0.33–15.72 秒，受已有 Inductor 磁盘缓存影响；短任务需要计入该成本。

为验证 joints-only 的实际价值，另外测量右手 batch 16384，保持 autograd 开启：

| 当前 eager，batch 16384 | 完整网格路径 | joints-only |
| --- | ---: | ---: |
| forward | 16.81 ms | 6.04 ms |
| forward+backward | 37.66 ms | 16.97 ms |
| forward+backward 峰值额外 allocated 显存 | 1955.81 MiB | 128.07 MiB |

本轮该路径的显存约减少 15 倍，forward+backward 约快 2.2 倍。其基线 joints-only 为 13.82 ms，本轮为 16.97 ms，数值保障有成本；完整路径相应为 37.47 → 37.66 ms。这一组仅运行一个 optimizer step 来测吞吐，不用于拟合质量结论。

显存是测量块开始时已存活张量之外的 allocated 增量，不是总显存或 reserved memory；compile 预先分配 workspace 时不可直接与 eager 的增量比较。阈值时间由步数乘平均步时间估算，并非实际计时的首次过阈值时间。README 的历史性能表已改为指向复现说明，CHANGELOG 保留旧轮次的历史记录。

### Anatomy loss

默认限位，forward+backward，5 组测量的中位数：

| CUDA batch | 原循环版本 | 当前向量化 | 本轮倍率 |
| --- | ---: | ---: | ---: |
| 1 | 16.208 ms | 1.166 ms | 13.9× |
| 128 | 16.354 ms | 1.094 ms | 14.9× |
| 1024 | 16.207 ms | 0.885 ms | 18.3× |

减少 Python 循环和小 kernel launch 对这一损失有明确价值；CPU 也测量了，但共享 CPU 波动较大，不推广固定倍率。loss 前向及梯度与原公式在 float32/float64/FP16 CUDA 上匹配。以上为 loss 自身测量，不等于含 FK、MANO 和 optimizer 的完整拟合提速倍率。

## 真实数据约定与 fitting

从 UHAS 导出的 ARCTIC、HO-Cap 样本各选 32 帧／可用手侧，仅写入本仓库 `data/benchmarks`，原数据保持原样。ARCTIC 首个序列提供双手，选定 HO-Cap 序列只有有效左手。metadata 明确采用 metre、绝对 axis-angle、未修正的左手 shapedirs。

原数据使用不同的指尖顶点：直接比较 21 点会产生约 3.4–3.6 mm 的 RMS 差异，不能据此判定 MANO 不一致。排除五个 tips 后，原始参数重建的 16 MANO joints 误差小于 5.6e-5 mm。本轮 real fitting 因此使用 16 个关节，full 与 joints-only、eager 与 compile 均验证：

| 数据与手侧 | 100 步后的 RMSE（约） |
| --- | ---: |
| ARCTIC 右手 | 0.28669 mm |
| ARCTIC 左手 | 0.25225 mm |
| HO-Cap 左手 | 0.24630 mm |

这些是数据集导出关节与 MANO 的几何 fitting 验证，并非对原始图像、传感器噪声或整套标注 pipeline 的端到端评测。

## 下游迁移审计

只读审计 `dr_egostereo` 与 `FoundationEgo_Annotation` 的真实 vendored 实现和 wrapper，各侧 8 个随机 pose，float64、相同模型缓冲区与 shapedirs 策略。最大顶点差 < 8.4e-14 mm、16 关节差 < 4.2e-14 mm、梯度差 < 5.4e-15。

两个 wrapper 都已经手工修复左手 shapedirs。替换时应移除手工翻转并设置 `fix_left_shapedirs=True`，或保留手工翻转且保持构造选项 False；必须仅做一次。两个项目还有自己的指尖覆盖，依赖网格的步骤继续用完整路径。

FoundationEgo 实际使用宽松、非对称 anatomy limits。当前左手 twist/spread 改成右手约定；映射后的角度误差为零，但直接复用旧非对称阈值会改变 loss。32 个随机 pose 的示例中，左手平均 loss 从 0.201171 变为 0.198957，右手保持 0.202996。该变化提示迁移需重新解释左手阈值，不能以“平均更小”判断更正确。

建议先迁移核心 MANO 计算，再独立审查 FoundationEgo 的 angle/loss 约定并运行其 fitting 质量门槛。当前审计不包含整套 annotation/stereo pipeline，也未修改两个下游仓库。

## TODO 与后续建议

本轮已完成 P0 正确性、P1 benchmark/torch.func/真实样本验证、P2 lint/文档/CI/向量化/安装验证，以及 P3 下游审计和发布准备。[本地 TODO](../TODO.md) 已同步，仍按原约定留在 git ignore。

验收后的建议顺序：

1. 用户已授权将 `optimize` 合入 `master`，版本升至 0.1.0；先验证分支 CI，再推送 tag，由 Release workflow 重新验证并发布 wheel／sdist。参见 [发布流程](releasing.md)。
2. 将固定形状 fitting 优先接入 compile，并测量下游包含 FK、loss、optimizer 的完整循环；短任务保留 eager。
3. 优先处理 FoundationEgo 左手 anatomy limits 与自定义指尖迁移验证。
4. 本次已完成 eager 平方角 profile 与系数合并；后续在实际下游评估剩余 launch 开销，并继续保留零点 gradgrad 回归。
5. **README.old.md 删除、Triton / C++ kernel 继续留在 TODO，本轮不执行**。UpSampleLayer 拓扑缓存待实际热路径需要时再做。

## 验收材料

- [CHANGELOG](../CHANGELOG.md)、[功能与约定](features.md)、[benchmark 复现说明](benchmark.md)。
- 选样 manifest：[MANO](benchmark_mano_samples.json)、[ARCTIC](benchmark_arctic_samples.json)、[HO-Cap](benchmark_hocap_samples.json)。
- 本地原始结果：[eager 配对](../data/benchmarks/eager_comparison.csv)、[compile 配对](../data/benchmarks/compile_comparison.csv)、[ARCTIC](../data/benchmarks/arctic_fitting.json)、[HO-Cap](../data/benchmarks/hocap_fitting.json)、[anatomy GPU](../data/benchmarks/anatomy.json)、[下游审计](../data/benchmarks/downstream.json)。这些 data 链接只在当前工作区有效。
- 测试日志：[完整环境](../data/benchmarks/final_tests_after_eager.log)、[最低版本环境](../data/benchmarks/final_minimum_tests_after_eager.log)。

最终边界修正涉及 Euler 无效分支、loss 的显式 module dtype 转换，以及自定义度数限位的浮点运算顺序（保持原 ReLU 边界梯度）；它们不改变上述 MANO fitting 与默认 anatomy 稳态 benchmark，未重新执行整套耗时测量。后续运行的结果 JSON 增加了源码 hash；较早基线结果通过固定 `c936b59` 归档定位，区分测量快照和最终工作区。

## 复核补充：warning、NumPy 与执行性能

### Warning 的准确范围

在新 Python 进程中，不导入 manotorch，使用 `warnings.simplefilter("always")`，仅执行：

```python
import torch
x = torch.ones(3)
torch.func.jvp(lambda t: t.sin(), (x,), (x,))
```

PyTorch 2.11.0+cu126 首次出现 18 个相同的 `torch.jit.script` DeprecationWarning，再调用为 0。调用栈为 `torch.func.jvp → forward_ad.make_dual → _maybe_load_decompositions → decompositions_for_jvp._register_jit_decomposition_for_jvp → torch.jit.script`。这确认是 PyTorch 自身的首次初始化路径，而非 manotorch 残留的 TorchScript 调用；参见 [PyTorch 2.11 源码](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_decomp/decompositions_for_jvp.py)。

`tests/test_runtime.py::test_no_warnings` 和 `test_no_host_device_synchronization` 以 `-W error` 重跑，2 passed。本次没有屏蔽 warning，也没有修改 site-packages 或全局关闭 JIT。此前“运行时无警告”的范围应限定为 guard 覆盖的常规 MANO/layer 路径，不能扩大到所有 torch.func 首次调用。

### NumPy 的限制来自哪个环境

在保持最低版本环境原样的前提下，将隔离安装的 NumPy 2.2.6 加入导入路径，PyTorch 2.0.1+cpu 的 `torch.from_numpy(...)` 与 `tensor.numpy()` 都失败，出现 `_ARRAY_API not found` 和 `RuntimeError: Numpy is not available`。这是 NumPy 1.x/2.x C ABI 的二进制兼容问题，见 [NumPy 官方说明](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)。PyTorch 2.11 与 NumPy 2.5.3 已实测兼容。

### Forward 与 fitting 的执行方式

`forward` 将给定 pose/betas/translation 映射为 vertices/joints；fitting 则重复“forward → 与观测计算 loss → backward 求 pose/betas/translation 梯度 → Adam 更新参数”。fitting 没有训练 MANO 模型的固定 buffer，而是在求手的输入参数。

eager 每次沿 Python 程序逐个调度 PyTorch 算子。compile 首次捕获运算图、编译 forward 及其可微 backward，后续复用并融合算子；`reduce-overhead` 还尝试用 CUDA Graph 减少 CPU 发起 GPU 工作的开销，见 [PyTorch 2.11 文档](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)。它不改变 MANO 定义或 fitting 的目标，也不自动减少收敛步数。

本 benchmark 编译的是 `predict`（MANO 模型预测），**loss 和 Adam 仍在 eager 执行**。输入数值每步变化本身不要求重新编译；形状、dtype、device、分支与 guards 的变化可能产生新的编译。

以原始测量中的当前右手、batch 128、joints-only 为例：

| 指标（热运行） | eager | compile |
| --- | ---: | ---: |
| forward，保留 autograd | 5.819 ms | 0.467 ms |
| forward + 几何 loss + backward | 15.909 ms | 1.327 ms |
| 完整 fitting step | 17.227 ms | 2.710 ms |

两者 100 步 RMSE 都约 0.306834 mm。然而该 compile 案例首次差分编译耗时 13.010 秒：忽略其他初始化时，eager 的 100 步约 1.723 秒，compile 为 13.010 + 0.271 ≈ 13.281 秒。**仅做一次 100 步任务可能得不偿失**；同一图复用于多帧／更多迭代后才能摊薄成本。以上快慢与盈亏平衡依赖当次负载和编译缓存，不是固定承诺。

### 为什么本轮 eager 变慢

比较基线已经是 Claude 重构后的 `c936b59`，而非优化前的 `ad515f0`。本轮继续保留该重构，并增加零旋转二阶梯度保障。旧公式先求范数再计算 sinc；新公式计算平方角，并用 Taylor、clamp 和 where 保护零点。eager 的 `torch.where` 不会只计算被选中的分支：两条表达式都会先求值，所以多了算子和 kernel launches。

新增 profiler 直接测量 `(128,16,3)` axis-angle→matrix 的一次 CUDA forward：基线 **18** 个 kernel events，初次修正快照 **42** 个（本次追加优化降至 28）；CPU aten events 为 56→91。该证据定位了实质开销，而非把所有变化归因于测量噪声。它只统计 rotation helper，不是整个 MANO forward。

另外对 batch 1/128、full/joints-only 的 eager forward/backward 进行 6 组交替测量，每组 30 次。batch 128 joints-only 的 forward 中位数 1.365→1.812 ms，forward+backward 为 4.360→5.206 ms；这与新增工作量一致，也证明旧表中 5.819/15.909 ms 的绝对时间受当时负载影响。另一 full/backward 测量在不同组间发生约 2 倍波动，所以不能把旧 15.8% 汇总差异当作稳定的精确退化率。

这次修正改善的是正确性与功能兼容性，并未证明比 `c936b59` 全路径更快。compile 能合并部分小算子及中间数据访问，但也不会免费消除一切开销。此时建议的后续 eager 优化为首先减少 Taylor helper 的重复计算；本次已实现，见下节，保留零点一阶／二阶梯度回归；不能简单回退为有 NaN 的旧公式。

复核原始记录位于 `data/benchmarks/warning_trace.log`、`warning_guards_recheck.log`、`review_followup_performance.json`；这些只在当前工作区有效。

## 本次追加：eager 提速与上游比较

### 实际修改与收益

合并 AA→matrix / AA→quaternion 两个 sinc 系数的计算，共用 Taylor、clamp、sqrt、sinc 和 where。
在平方角 < 1e-4 的分支中，Taylor 省去的三次项上界为 1e-12/5040，低于 float64 epsilon；仍保留
零姿态的一阶／二阶微分保障，并补测实际角度 0.009/0.01/0.011 和另一系数边界 0.02。
AA→matrix 的 `(128,16,3)` CUDA forward 从初次修正的 **42 降至 28 个 kernel events**；没有引入 Triton / C++ kernel。

新增 `scripts/benchmark_layers.py`，固定上游原版 `a2a70c5`、manopth `4f1dca`、smplx `1265df7`。
相同模型、随机 seed、flat mean、48 个 axis-angle 值、float32、完整网格、pose/betas 梯度，双手
batch 1/128/1024；7 组交替顺序 × 每组 20 次，10 次 warmup。上游/manopth 仅替换计时外的加载适配，
其 forward/backward 源码未改。smplx MANOLayer 的 AA→matrix 转换计入时间。详细条件及完整 commit 见 `doc/benchmark.md`。

下面为右手的部分配对结果（RTX 4090，PyTorch 2.11）；F 保留 autograd，F+B 对顶点 MSE 求导，均不含 Adam：

| Batch | 指标（ms） | 上游原版 | Claude c936b59 | 正确性修正后 | 本次 eager 提速后 | 本次耗时降低 | 相对上游速度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | F | 8.373 | 4.113 | 5.176 | 4.705 | 9.1% | 1.78× |
| 1 | F+B | 20.451 | 12.019 | 14.553 | 13.014 | 10.6% | 1.57× |
| 128 | F | 8.578 | 4.323 | 5.411 | 4.929 | 8.9% | 1.74× |
| 128 | F+B | 21.034 | 12.511 | 15.311 | 13.885 | 9.3% | 1.51× |

双手六个案例的本次耗时降低范围为 **F 6.1–9.1%，F+B 7.5–10.6%**；相对上游原版速度为
**F 1.70–2.19×，F+B 1.48–2.10×**。相对 Claude `c936b59` 仍慢 F 14–22%、F+B 8–11%，
因此只能说减少了正确性修正带来的开销，不能说已超过 Claude 版本。共享机器跨手测量期间负载改变，
不应跨案例直接比较绝对时间。跨实现最大顶点/16 joints/梯度差分别为 7.46e-5 mm / 5.97e-5 mm / 4.08e-10。

**README 的 Other MANO layers 已增加当前 eager 性能表**，涵盖本 fork、上游、manopth、smplx MANO/MANOLayer；
此前该节只有约定和精度对照。官方 chumpy 模型没有列入 GPU 性能表。

### 提速后 fitting 和测试

新代码重跑冻结 MANO 目标的双手 batch 128 × 三个任务 × eager/compile，共 12 案例、每例 100 步。
与初次正确性修正版本相比，最终误差最大差为 2.09e-07 mm，1 mm 阈值的到达步数全部一致。
右手 joints-only 的本次热运行测量如下；该表测完整 fitting，与上面的跨层顶点 MSE microbenchmark 不同：

| 模式 | F（ms） | F+B（ms） | Adam 完整步（ms） | 首次编译（s） | 100 步误差（mm） |
| --- | --- | --- | --- | --- | --- |
| eager | 5.385 | 14.319 | 11.961 | 0.000 | 0.306834 |
| compile | 0.189 | 1.774 | 2.363 | 10.248 | 0.306834 |

本次只复测上述 12 个 fitting 案例；前文 48 个 fitting/真实数据/anatomy 表保留原快照。
完整测试 **157 passed**（18 个 PyTorch JVP 初始化 warning）；最低版本 **152 passed、5 skipped**。
Ruff、diff、uv lock 和重新构建/隔离安装 wheel 均验证。原始记录为
`data/benchmarks/eager_layers.json`、`fitting_after_eager.json`、`final_tests_after_eager.log`、
`final_minimum_tests_after_eager.log`，均只在当前工作区有效。

### compile 的代价与 warning 边界

- 首次捕获、生成代码及 forward/backward 编译需耗时，短 fitting 可能总时间更长；此前 13.010 秒案例
  是当时的缓存/机器快照，不是每次固定成本。
- 输入 shape、dtype、device、配置或 guard 改变可能再编译；参数数值的正常迭代本身不要求编译。
- `reduce-overhead` 的 CUDA Graph 可以保留额外 workspace，减少 launch 开销需要以显存换取。
- 图中断、自定义 autodiff、Python 分支等兼容问题增加调试成本；`fullgraph=True` 捕获失败直接报错。
  当前 manotorch 编译时采用原生 skinning，是针对 custom JVP 限制的兼容处理。
- 性能收益依赖负载/硬件/编译模式，模型的加速比不会直接等于包含 loss、Adam 和数据加载的端到端加速比。

依据 [PyTorch 2.11 compile 文档](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)。
本次重新编译另出现 `torch._prims_common.check` FutureWarning，调用位于 PyTorch Inductor
`lowering.py` 的 `diagonal` 分支。仅运行原生 `torch.compile(lambda t: t.diagonal(...).square().sum())`
及 backward、不导入 manotorch，也能复现同样 warning；原始记录 `compile_warning_probe.log`。
因此“常规 eager forward/backward 的 guard 无弃用警告”仍成立，不能扩展为“PyTorch 2.11 的
torch.func / Inductor 内部初始化都没有弃用警告”。没有通过屏蔽或修改 PyTorch 来消除这些提示。

## 本次追加：含 Anatomy loss 的完整 fitting 与脚本说明

新增 `scripts/benchmark_fitting_anatomy.py`。相同冻结目标 hash、初值、默认限位、解剖参考基底、
Adam 状态和学习率，对比当前/Claude `c936b59`/上游 `a2a70c5` 各自的 MANO、AxisLayerFK 和 anatomy loss。
只测右手，避免上游左手符号约定不同；计时外对齐 basis buffers 与缓存，计时内数学源码未改。
上游的 legacy 加载/绝对导入适配在构造阶段，额外 `deprecation` 也仅为其旧 anatomy 模块所需。

全网格、float32、flat mean、48 个 axis-angle 值，目标取 16 个共同关节；batch 1024 重复 128 个样本。
目标为关节坐标 MSE（metres）+ `1e-4 * mean anatomy penalty`（radians），同时测不加 prior 的版本。
每组拟合 100 步，六组交替顺序，10 步 warmup 后恢复相同初值并清零已分配 Adam state；
计时覆盖 MANO/可选 FK/Euler/loss/backward/Adam，不含加载、构造、warmup、重置及指标采集，不使用 compile。

含 anatomy loss 的结果（RTX 4090 / PyTorch 2.11，同一机器）：

| Batch | 当前 100 步（s） | Claude（s） | 上游（s） | 相对 Claude 速度 | 相对上游速度 | 当前 RMSE（mm） | 当前 prior（rad） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.338 | 3.406 | 4.320 | 1.46× | 1.85× | 1.267 | 0.0025 |
| 128 | 1.773 | 2.365 | 3.001 | 1.33× | 1.69× | 2.337 | 0.0279 |
| 1024 | 2.379 | 3.468 | 3.506 | 1.46× | 1.47× | 2.130 | 0.0941 |

当前完整链路比 Claude 快 **1.33–1.46×**、比上游快 **1.47–1.85×**，与仅测 MANO layer 的排序不同。
此前 isolated anatomy loss 的加速不能直接外推到端到端；这张表直接计时完整的 100 个 Adam updates。
六组最终指标各自完全复现，跨实现最终 RMSE 最大差为 **3.03e-5 mm**。
共享机器部分组有明显波动，完整数据保存了各组区间，不能直接把不同 batch 或两种 objective 的
绝对时间相减解释为固定的 anatomy 开销。manopth/smplx 没有该原生 anatomy 链路，不在这张新表中比较。

prior 改变了拟合目标，权重不是针对数据集调优的建议。batch 128 的当前 RMSE 从 data-only 的
0.232 mm 变为带 prior 的 2.337 mm，而带 prior 的 mean penalty 从初始 0.447 降为 0.0279 rad。
训练姿态不必严格满足默认限位；不能将更快执行或更低 prior 解释为更好的观测精度。
无 prior 的 JSON 行中 `anatomy_rad=0` 是未计算该项的占位值，不代表其角度满足限位。

README 的 `Eager runtime comparison (2026-10-05)` 已新增有/无 Anatomy loss 的完整 fitting 时间表。
`scripts/README.md` 已逐项说明 11 个 Python CLI、兼容 notebook 和 `_common.py` helper：用途、依赖、
模型/数据/源码准备、命令、关键参数、输入输出、计时范围及输出格式边界；主 README 和 benchmark 文档均链接它。
11 个 CLI 的 `--help` 均通过；新 benchmark 的 CPU 短运行及 CUDA 完整运行均通过目标/loss/梯度一致性检查。
本轮未改动核心模型/自动微分实现，沿用此前 157/152 项核心测试结果；Ruff 和 diff 检查通过。

同时明确默认 eager、compile 由调用方可选开启，compiled 二阶/JVP 组合需单独验证；
Error Correction GIF 已按用户要求改为不透明 solid surface，重新渲染 101 帧并检查首/中/末帧。
本次原始记录为 `data/benchmarks/fitting_anatomy.json`、`fitting_anatomy.log`、`fitting_anatomy_smoke.json`，
仅在当前工作区有效。复现说明见 `scripts/README.md` 和 `doc/benchmark.md`。

## 发布边界更正与撤回重提（2026-10-05）

用户明确：开发分支应保留本报告；master 和 release code 应为英文。

- 初次发布的 Git 源码及 GitHub 自动源码压缩包包含本报告；wheel 和 sdist 未包含。该 Release 与旧 tag 已撤下。
- 发布代码重新提交为 `29be6d20fafc2c746e36ea6e0423fcf569f7278b`；重建的 master 合并提交为 `4202b050e025dddf659475279e7ed37c46d746e0`，其文件树及提交历史不包含本报告。
- 本报告在发布代码提交之后，作为仅供开发分支使用的文档提交恢复。报告保留在公开 optimize 分支，仍可被访问，不应视为私密文件。
- CI 按分支执行边界检查：开发分支允许本报告；master、release tag 与面向 master 的 PR 拒绝本报告，其余源文件及文档检查 CJK 文本。
- 原实现及性能修正保持不变；重提仅调整发布文档、分支边界和 CI。之前完整 157 项及最低版本 152 项验证仍覆盖相同运行时代码；新提交重新执行远程 CI。
- 已被下载的副本及 GitHub 对旧提交的缓存无法保证回收。本地保留撤回前的 bundle、报告和 CI 日志以便审计；这些备份未发布。

重新发布的最终状态与安装文件见上方 GitHub Release；最终 CI 与验收记录同步至 Notion。

## 撤回重提后的最终验收

- `v0.1.0` 已重新正式发布，指向 master `4202b050e025dddf659475279e7ed37c46d746e0`，实现提交为 `29be6d2`。
- [旧发布撤回](https://github.com/IRVLUTD/manotorch/actions/runs/37331037171)、[开发分支边界 CI](https://github.com/IRVLUTD/manotorch/actions/runs/37331479822)、[master CI](https://github.com/IRVLUTD/manotorch/actions/runs/37331480232) 及 [新的 Release CI](https://github.com/IRVLUTD/manotorch/actions/runs/37331752588) 均成功。两个额外的临时撤回任务在旧 Release 已删除后得到 404，未改动重新发布的版本；临时分支已删除。
- 实际下载 GitHub 自动生成的 tag 源码压缩包，确认不含本报告，且源码／文档的 CJK 检查通过。
- 新 wheel／sdist 已下载，SHA256 与新的 GitHub digest 一致，包内实现与新 tag 的源码逐字节一致；均不含本报告。
- 中文报告继续作为开发分支文档保留，本次追加仅修改开发报告；master 与 tag 均排除它。开发分支公开可见，不能将报告视为私密文件。
- 当前工作分支为 optimize；运行时代码与第一次发布完全一致，未引入新的模型或性能修改。

本地完整记录：`data/benchmarks/reissue_0.1.0.json`，下载包与自动源码包：`data/benchmarks/reissue_0.1.0_published/`。旧记录与备份保留在 data 中，均不进入 Git。
