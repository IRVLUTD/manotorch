# manotorch 审查与优化验收报告

日期：2026-10-05（America/Chicago）。审查基线：`optimize/c936b59`。本文保留在公开开发分支 `optimize`；`master` 和 release tag 明确排除本文，仅发布英文代码及文档。最终发布与远程验证状态以 [GitHub Release](https://github.com/IRVLUTD/manotorch/releases/tag/v0.1.0) 及其 Actions 记录为准。

最新追加：README 插图、UpSampleLayer 缓存与 kernel 收益评估见文末。本轮为 optimize 的本地未发布修改；已发布 v0.1.0 不变。

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

## README 插图、拓扑缓存与 kernel 评估（2026-10-05）

本轮修改保留在 optimize 工作区，尚未提交、推送或发布；版本仍为 0.1.0，CHANGELOG 加入 Unreleased。
中文报告仍只用于开发分支，README、公开文档、脚本和代码为英文。TODO 已更新完成项与生产 kernel 的剩余范围。

### 插图可读性

四张 GIF 已重新渲染并替换 doc 资产，README 改为逐张全宽展示，避免三列表格缩小模型和图例。
轴／anchor／compose 为 960×640、48 帧、10 fps；Error Correction 为 960×600、101 帧，保持不透明 surface。
已检查每张首／中／末帧及四分之一／四分之三帧，无模型裁切，图例清晰，摄像机极值保留有效视角。
记录为 data/benchmarks/illustration_qa.json 及 *_readable_qa.png。

显示旋转使双手直立，不改变 MANO 坐标或 pose；镜头采用有界正交扫视。Compose 食指以金色高亮，
只显示食指三个关节的轴，斜视角可看到 MCP/PIP/DIP 90 度弯曲，原 MCP spread 30 度保持不变。
Anchor 示例每手 32 个紫色点，突出 anchor 0 及其 barycentric 插值来源三角形。
Error Correction 在 optimizer update 后重新计算渲染姿态／loss，修正旧图中迭代标签与状态错位的问题。
脚本 docstring、scripts/README.md 和主 README 的生成命令及图注均已更新。

### UpSampleLayer 缓存：已实现，收益仅限重复细分

旧实现每次 forward 将 faces 搬到 CPU，逐 batch 用 Python 重建边字典，再上传拓扑。
新实现缓存唯一边和四个子面的索引；共享 expand 拓扑只建立一次，用 nonpersistent long buffers 随设备移动。
原双参数接口按 faces identity／普通 in-place version／vertex_count 自动失效；prepare 快照后可调用 layer(vertices)。
快照后源 faces 编辑不改变结果，需主动重新 prepare；.data／外部存储编辑绕过 version，也需显式更新。
没有 version counter 的 inference faces 使用 prepare 快照。缓存只保留一个拓扑，可 clear_cache，checkpoint 不包含缓存。
返回 faces 拥有独立存储；保持共享边顺序、子面 winding 和可微分 midpoint interpolation。

RTX 4090／PyTorch 2.11／4 CPU threads，MANO 右手 open-wrist 778 vertices／1538 faces；
五组交替计时，输出与 c936b59 逐值一致。下表单位 ms，计时不包含 MANO forward：

| Device | Batch | 旧实现 | 自动缓存 | prepare 快照 |
| --- | --- | --- | --- | --- |
| CPU | 1 | 27.491 | 0.437 | 0.432 |
| CPU | 32 | 876.458 | 2.537 | 2.541 |
| CPU | 128 | 3515.005 | 9.455 | 9.438 |
| CUDA | 1 | 27.628 | 0.228 | 0.223 |
| CUDA | 32 | 874.356 | 0.239 | 0.227 |
| CUDA | 128 | 3539.932 | 0.241 | 0.226 |

首次共享拓扑 prepare 20.7–22.9 ms；单次／不断改变拓扑时仍需支付这笔开销。
大 batch 的极大倍率主要消除了旧实现对相同拓扑逐 batch 重复遍历，不代表 MANO GPU 数学计算加速。
检索当前仓库和已审计的下游，未发现 UpSampleLayer 在既有 forward／fitting 中使用，因此普通 fitting 直接收益为零。
这项实现难度低，适合重复渲染／高密度网格／依赖细分的任务，已经落地。
数据及 hash 为 data/benchmarks/upsample_cache.json；API 见 doc/features.md，复现说明见 scripts/README.md。

### Triton 与 C++/CUDA：原型与评估完成，生产后端未接入

独立 scripts/benchmark_kernels.py 实现两个 contiguous CUDA float32 Triton 原型：
skin forward 保持原 PyTorch backward；axis-angle rotation forward 仅用于 no-grad inference，没有 backward。
它们不被核心包导入，不改变默认 eager，也没有新增强制依赖。CUDA profile 验证 rotation launches 28→1、skin 3→1。

相同冻结 MANO_Poses targets／初始化／Adam／reference basis，两轮六组交替测量 B1/128/1024，
覆盖局部 eager／compile／Triton forward、完整 MANO forward／backward／inference、有无 anatomy 的完整 100 步 fitting。
计时排除编译／JIT 启动、加载、构造、reset、warmup 和质量检查，保留 Python launch 开销。

| Batch | 第一轮完整 inference eager／rotation Triton（ms） | 第二轮（ms） | 延迟降低 |
| --- | --- | --- | --- |
| 1 | 3.720／2.787 | 4.010／3.045 | 24.1–25.1% |
| 128 | 1.303／0.971 | 3.968／3.009 | 24.2–25.5% |
| 1024 | 1.302／0.973 | 1.307／0.973 | 25.3–25.5% |

Rotation 局部约 11–13× 的提速不能外推到完整模型，完整 inference 实测约 1.32–1.34×。
其最大 vertex 差为 4.47e-5 mm；已测零点、近零、两个 Taylor 阈值及较大角度的 forward。
这不构成任何训练／高阶梯度兼容证据，原型没有 rotation backward。

Skin geometry／pose与shape gradients 完全相同；所有 measured fitting cases 的最终 RMSE／prior／objective 完全相同。
Skin 完整 forward 在 B1/128 约快 1.4–1.6%，B1024 约慢 2.3–2.4%，完整 fitting 无稳定收益。
稳定的 B128＋anatomy，两轮 eager/Triton 为 0.819/0.822 s 和 0.820/0.824 s，即略慢约 0.5%。
共享机器发生明显 host-load transitions；第一轮 B1＋anatomy 各组 eager 范围 1.426–2.337 s、Triton 1.426–2.381 s。
这些组中 median 差约 19% 不能解释为可复现退化；第二轮 B1024＋prior 也存在同类波动。
因此不对当前 skin kernel 宣称 fitting 提速，也不凭跨 batch 的绝对时间推断 batch scalability。

当前建议：保留原生默认；可选 Triton rotation inference 有收益，列为 P2，需真实下游 inference 热路径支持。
训练 rotation kernel 需要单独 backward 与零点 gradgrad／torch.func／compile 验证，收益尚未测定。
仅将 Python 包装改为 C++ 不会自动融合 GPU 运算；真正 fused CUDA 可以后续评估，但没有实测证明优于 Triton。
C++/CUDA 目前暂缓，原因是拟合收益不足、构建／分发／维护成本更高，而不是实现不可能。

投入粗估（熟悉本代码的开发者，非承诺）：窄范围可选 Triton inference＋fallback 测试 3–5 工作日；
训练与高阶／torch.func／多版本覆盖 1–3 周；C++/CUDA backend 与分发 2–4 周。
困难主要是保留 CPU/MPS、dtype/layout、最低 PyTorch 2.0.1、autograd／JVP／vmap／二阶及编译路径。
依据 PyTorch [torch.func 扩展要求](https://docs.pytorch.org/docs/2.11/notes/extending.func.html)；
[官方 C++/CUDA 教程](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) 的稳定 ABI 示例最低为 2.10，
不能覆盖我们 2.0.1 的整个支持区间。没有实际实现 C++/CUDA，TODO 的生产接入仍未勾选。

原始数据及 hash：data/benchmarks/kernel_feasibility.json、kernel_feasibility_repeat.json；
单独 profile：kernel_launch_probe.json；详细英文方法与结果：doc/benchmark.md。

### 本轮验证

- PyTorch 2.11 全量回归：168 passed，18 个已核实的上游首次 JVP 初始化 warning；当时尚未新增单独 compile/JVP 用例。
- 最终全部 12 项 upsample 专项：12 passed，覆盖共享边与独立 face winding golden、identity/version 失效、snapshot、输出 ownership、
  distinct batch、空拓扑、输入检查、long buffer、gradcheck／gradgradcheck／JVP、Dynamo fullgraph 及 CUDA 无 warm CPU transfer。
- 最终最低环境 PyTorch 2.0.1 全量：163 passed／6 skipped（3 个 CUDA FP16 参数、3 个 CUDA 专项）。
- prepared UpSample 的 CUDA Inductor fullgraph 实际 forward/backward probe：输出逐值相同，梯度最大差 7.28e-12。
  这不等于 compiled 二阶／JVP 组合已经验证；那些组合仍需按需测试。
- 两个新 benchmark 的 CLI help、完整 CPU/CUDA subdivision、两轮 CUDA kernel/model/fitting 数值检查通过。
- Ruff／diff 检查通过；源代码／公开文档保持英文。所有 licensed data 与原始输出继续留在 ignored data。

18 个 warning 仍是 Torch 内部 JVP 初始化，不是新缓存或新 kernel 引入 manotorch 的弃用调用。
首次编译 probe 仍可见已独立复现的 Inductor torch._prims_common.check FutureWarning；未屏蔽或修改 site-packages。
本轮没有改变已发布 tag／master，也没有触发新的远程发布或 CI。

## 2026-10-08：原始 MANO_Poses 注册验证、脚本审计与 QC

### 输入约定与独立准确性

已读取 data/MANO_Poses/README.md 和原始 README/examples，逐个审计 1554 PKL/对应 PLY：31 subjects，895 right、659 left-mirrored，原始注册均为 MANO_RIGHT、ncomps=0。
使用 48 维 absolute axis-angle（含非零 global rotation）、每注册 10 betas 和 scanner translation；use_pca=False、flat_hand_mean=True、center_idx=None。
16 个 J_transformed 按 MANO 顺序映射，不能拿 out.joints[:,:16] 代替。本地数据说明原“ManoLayer 没有 translation 输入”已修正。
原始 v/J_transformed 是独立目标，不是先前 c936b59 生成的冻结 targets。R.npy 与排序 PKL 完全对应；L.npy 用旋转矩阵验证，避免等价 axis-angle 跨 pi 表示造成假差异。

| Device/dtype | Vertex max Euclidean mm | Joint max Euclidean mm | 样本数 |
| --- | --- | --- | --- |
| CPU/float32 | 0.00007823 | 0.00006005 | 1554 |
| CUDA/float32 | 0.00012732 | 0.00010156 | 1554 |
| CPU/float64 | 0.00001029 | 0.000001831 | 1554 |
| CUDA/float64 | 0.00001029 | 0.000001831 | 1554 |

全部通过 0.001 mm 门槛。float64 模型 buffer 仍在构造时经历 float32 rounding；不是原始模型数组的全双精度路径。
独立 NumPy sinc Rodrigues、顺序 FK、homogeneous LBS 重建所有原始注册，最大 coordinate error 1.94e-13 mm。
CPU/CUDA 全部 float32 forward/backward finite；34 个代表/极值样本的 batching、permutation、shared betas、joints-only outputs/gradients、跨设备检查通过。
8 个真实姿态的 double 方向有限差分最大差 1.68e-9；零点、Taylor 两个边界、近 pi 的模型二阶导均 finite。
左模型两种 shapedirs 政策都与各自独立 NumPy 参考一致；corrected 镜像 coordinate residual 0.008993 mm、uncorrected 44.796 mm。
镜像残差反映左/右模型资产差别，与原始右模型注册准确性分开。此数据不含独立原始左 scanner targets，不替代 FoundationEgo limits/tips 迁移验证。

用户询问为什么不一次输入全部样本后，额外完成 B=1554 单个 batch 的 accuracy/finite backward。CUDA float32 vertex/joint max 为 0.00012794/0.00011515 mm，仍通过。
最初 chunk=128 是诊断/存储选择，不是 MANOLayer 的 batch 上限。

### 推理与 fitting

28 个真实独立样本 inference cases 已完成：B1/8/32/128/512/1024/1554 × full/joints-only × eager/compile。
真 inference_mode，十次 warmup，六组旋转顺序、每组 30 calls；构造/加载/transfer/first compile 不计 warm time，另记 source/model hashes 与 CUDA event spans。
B1554 full mesh eager/compile 为 3.730/1.493 ms；joints-only 为 3.887/0.253 ms；first compiled forward+sync 为各 case 5.66–9.94 s，受磁盘 cache 影响。
这些是 shared-machine 参考值，不与 README 旧的 autograd/different-input layer workload 混比，也不代表 compiled fitting/二阶兼容已由本次验证。

已做 8 样本 B1/B8 smoke，并完成固定 128 个独立注册 bank 的 16 个正式 eager fitting case：
B32/B128 × vertices/16 joints × weight 0/1e-4 × 100/300 steps，三组 reset 计时。Bank 覆盖 subjects、极值 shape/articulation；seed=20261008。
从原参数加 pose 0.05 rad / betas 0.1 / translation 0.005 m 正态扰动；Adam LR .01/.02/.001，foreach=False。
独立 per-hand coordinate MSE 与 mean anatomy penalty 先逐手计算再 sum，避免 batch mean 改变梯度相对 Adam epsilon 的尺度。
计时和诊断 replay 分开；threshold 的 actual timestamps 含诊断/host 开销，不冒充纯优化耗时。

B128/300 steps 的 mean per-hand metrics：

| 目标 | Anatomy weight | 目标 RMSE mm | Mesh RMSE mm | 曾达到门槛比例 | Final anatomy mean rad |
| --- | --- | --- | --- | --- | --- |
| vertices | 0 | 0.07157 | 0.07157 | 81.25% @ 0.1mm | 0.4601 |
| 16 joints | 0 | 0.03739 | 0.94249 | 100% @ 1mm | 0.4660 |
| vertices | 1e-4 | 3.23511 | 3.23511 | 0% @ 0.1mm | 0.01034 |
| 16 joints | 1e-4 | 2.16763 | 6.62401 | 3.125% @ 1mm | 0.00654 |

B32 质量接近；16 joints 能拟合良好，不代表完整 mesh/latent parameters 唯一恢复。
1e-4 prior 明显降低 penalty，同时明显损害几何准确性；需先用小 bank 标定 weights，不能把旧 synthetic benchmark 默认权重直接作为全量 fitting 建议。
B128/300 complete fits vertices 无/有 prior 为 3.319/4.884 s，joints 为 3.216/5.089 s；峰值 extra allocation 最大 17.13 MiB（排除已存活输入/模型/Adam）。
原始组/分块计时存在明显 shared-host 波动，这些耗时仅参考，不宣称受控的 batch speedup 或 prior overhead。
用户告知 GPU 被其他 task 占用后，停止新增 GPU 测试；没有后台 GPU benchmark 继续运行。
新增 B1554 full-bank fitting 尚未启动，等资源空闲再补；现有误差/finite 结果可使用，performance 应在受控资源下重测。

### QC 文档与图册设计

已生成 data/qc/MANO_Poses/registration_qc.png + selection JSON：6 hands × target/reconstruction/error 三列，不透明 solid surface、逐行 matched camera、共享色标。
用户确认观感；原 scanner-frame errors 先计算，之后仅为显示消除 wrist/global rotation。小 batch 的 float32 rounding 与完整 chunk128 验证略有不同。
已生成 qc_summary.pdf（覆盖全部1554的统计、稳定性/单批确认、QC preview、128样本fitting），qc_metrics.csv 与 qc_statistics.json。
CSV 每个注册含全部device/dtype误差、source side、error rank、拟议 atlas 页/行；JSON 含 distributions、fitting geometry/penalty、环境与 SHA-256 references。
PDF 排版由 Poppler 渲染检查；全文、文件数量和 CSV coverage 检查。
全量 QC 推荐 A3 portrait PDF，259 个六手×三列 grid pages，另外加统计/索引；filename 排序、subject bookmarks、error-ranked jump index、统一色标、native ≥9pt 标注。
PNG 按需导出。全量图册目前只完成设计/索引，未渲染259页；synthetic 38 sequences/18947 frames 因没有原始mesh targets，要另做temporal QC。

### scripts 修正与验证

新增 benchmark_registrations.py、render_registration_qc.py、summarize_registration_qc.py 和三项 asset-independent script regressions。
旧 fitting 曲线补最后一次 optimizer update、threshold schema version2、peak allocation 在diagnostics之前捕获；compare 拒绝混用旧schema。
indexed cuda:0 同步、旧Torch marker API guard、seeded/interleaved anatomy benchmark、PCA solve 代替逆矩阵、空dataset/invalidhand诊断、notebook工作路径与demo镜头说明已修正。
最新 scripts/README.md 逐项列出输入/输出/依赖、复现命令与计时边界；旧chumpy/notebook数值执行仍需要独立legacy环境，不宣称已执行notebook。

初次 licensed MANO+CUDA 全量：172 passed/32 warnings。其中18个为已有torch.func/JVP内部 torch.jit.script；新增14个是脚本eager测量无必要调用cudagraph marker、引入Inductor再导入torch.utils.mkldnn内部script_method。
不涉及manotorch的JIT弃用调用；用仅import torch + cudagraph_mark_step_begin 的独立probe已复现14类warning的路径。
已限制marker仅用于compiled CUDA；最新三项script regressions在Torch2.11和最低2.0.1均3passed/no warnings；CPU eager fitting CLI/schema2 self-comparison和registration runtime smoke通过。
原核心18个首次JVP初始化warning仍属PyTorch内部；未屏蔽warnings或修改site-packages。
所有已有standalone scripts的--help审计通过；新summary CLI在没有reportlab时也能--help。Ruff/diff与publicEnglish检查通过。
本轮资料留在optimize工作区；Chinese review report仍为development-only，raw licensed data均在ignored data；未更改已发布master/tag或触发远程CI。

### 全量 QC atlas 完成（2026-10-08，后续 GPU 渲染授权）

用户明确要求执行全量图册并使用 GPU，已新增 scripts/render_registration_atlas.py，实际后端为 NVIDIA RTX4090/OpenGL 4.5。
完成 data/qc/MANO_Poses/registration_atlas.pdf：A3 portrait，259 个六手×三列 grid pages，另含 cover/index 共261页，约99MiB；覆盖所有1554个原始注册。
固定全量global error scale，opaque lit surfaces、matched orthographic cameras、9–12pt native PDF captions、31 subject bookmarks、top20 error index，共51个可点击index links。
独立accuracy CSV与本次重建metrics最大差1.36e-20 mm；vertex/joint最大误差仍为0.00012732/0.00010156 mm。

QA 发现两个窗口复用问题并已修正：clear会移除lights，需逐页重新启用light kit；screenshot只在首次自动render，后续需在所有camera设置完显式render，避免最后viewport捕获旧相机buffer。
已完整重新渲染并逐页强制nonempty/unclipped/matched-bbox gate；4662个模型视窗全部通过，最小边界余量48 pixels，三列最大bbox差2pixels（颜色/抗锯齿）。
最终PDF261页全部解析，259个grid均6样本/18native filename captions/1张完整1800×2226图像，逐个文件与原始1554 PKL映射一致；31 bookmarks/51links目标页核验通过。
人工复查cover/index以及grid1/61/130/144/241/259：首/中/末、最大shape/pose及最差误差，无文字/模型裁切与相机错位。
输出atlas_pages/grid_*.png六张，可用于快速分享；registration_atlas_index.csv包含实际pdf_page/grid_page/row与逐文件hash，registration_atlas.json包含GPU/backend/source/model/PDF/index hashes和渲染检查记录。
qc_summary.pdf、qc_metrics.csv、qc_statistics.json已更新为completed atlas与实际PDF页号。TODO、README、scripts说明、CHANGELOG、英文benchmark文档同步。
本次GPU授权用于图册生成，尚未恢复全量fitting或受控performance复测；这些仍在TODO。所有licensed data与PDF保持ignored data，未进行新发布/推送。

### MANO_Poses 左右手自包含 QC（2026-10-08）

按用户要求，完整左/右 poses 各 1554 个保存在 data/qc/MANO_Poses，附 full global axis-angle/betas/scanner trans、目标/重建网格与 joints、NumPy 参考、raw 模型、代码快照、来源与 README。使用官方 L.npy（115 行 canonical vectors 不同，但 rotation matrix delta≤1.99e-14），镜像 global/trans；fix_left_shapedirs=True 且 raw 模型不提前修正。

左右 CPU/CUDA float32/64 全量参考、全部梯度 finite、B1554/chunk/单样本/排列/joints-only 通过。左 CUDA float32 independent-reference vertex / joint max 0.000105672 / 0.000090054 mm；full scanner-frame corrected model mirror residual 0.010575863 mm，未修正 shapedirs 为 50.449993 mm。该口径为 Euclidean 距离，与此前 canonical maximum-coordinate 残差不同；镜像数据不是独立左手扫描 GT。

完整左 atlas 261 页、1554 hands/4662 面板，RTX4090 OpenGL；边界余量≥49px，camera bbox 差≤1px；31 subject bookmarks/51 links。全部页 caption/image 结构验证、首/中/末/最差/shape/pose 极值目视检查通过。两页 left/qc_summary.pdf 区分 implementation/reference 和 asset mirror residual，附 CDF/稳定性与重现说明。原右 atlas/报告保留。

Cached renderer 无 Torch/原始目录依赖；bundle source/model/pose fresh CPU left inference 验证通过（vertex / joint max 0.000086358 / 0.000062730 mm）。使用相对路径 hash 清单核对整包；公开 README/scripts/CHANGELOG 和本地 TODO 已同步。未新增 fitting 或受控 GPU timing，相关 TODO 保留。

### QC 目录与合并报告更正

按用户要求，left/right 数据与结果采用同样子目录；三个最终 PDF 均为 data/qc/MANO_Poses 根目录普通文件。qc_summary.pdf 合并为四页，包含双方准确性/稳定性、CDF、首样本/最大 mirror residual/最大 beta 的成对 QC images，以及原有 right fitting 16-case 表。left previews 仅作180度画面 roll 来对齐指尖方向，不改变数据或测量；两份完整 atlas 字节保持不变，页面索引和 source hashes 保留。旧 summary/provenance 已备份到 ignored data/benchmarks/qc_layout_archive，右原始 audit JSON/CSV 留在 right/。所有读取/重渲染/发布路径与目录 README 已更新为对称结构，旧 right-only summary 工具默认写入独立 benchmark audit 目录以避免覆盖 combined PDF。

### 左手图集展示方向更正

用户发现左 atlas 全部手朝下。原因是右手显示 basis 将模型 x 反射映射为显示 y 反射，左右沿用同一 basis 导致 left 朝下；与 MANO pose 正确性无关。改为 left display-z 轴的 180° proper rotation（det=1），保留 handedness，方向与 right 对齐。所有 1554 left/right 非拇指 MCP center 均在 wrist 上方，min高度约69.85mm。完整 left atlas 采用 GPU 原生重渲染；combined report 新 atlas 不再额外 image-plane roll。原始参数、scanner targets/reference 和量化指标不变。

最终重渲染与核验完成：left atlas 261 页、1554 poses、4662 面板；31 bookmarks/51 links，最小边界49px、最大 matched bbox 差2px。全部1554行误差指标和index CSV字节与重渲染前完全相同，right PDF hash不变。逐页PDF结构检查、六张首/中/末/极值PNG及combined四页报告目视检查通过；新版combined直接使用朝上left原图，不再额外旋转。

### 最终 code-check 与开发分支提交（2026-10-08）

最终检查修正两个问题：UpSampleLayer 在 inference_mode 下建立/迁移的缓存可能无法用于随后训练的 gather backward，现使用普通索引 buffers，新增 CPU/CUDA warmup-to-training 回归；right-only 统计 publisher 的 indexed CUDA CSV 名称现与 benchmark 写入规则一致，cuda:0 的四案例/1554样本 integration 通过。更新 bundle preparation 的后续提示为双 atlas + combined report。

最终本地 licensed MANO + CUDA 全量回归：174 passed，18 个已知上游 PyTorch 首次 JVP 初始化 DeprecationWarning；FutureWarning 作为 error 未失败。没有屏蔽 warning 或修改 PyTorch；全局 -W error 的单独探针在同一上游 JVP 初始化告警处失败，不能宣称整个测试 suite warning-free。Ruff、diff check、QC bundle 60 文件 checksum、wheel build 检查通过。仅提交/推送 optimize 开发分支；模型、数据、PDF 与 TODO 不进入版本控制，开发 report 继续保留在 optimize，未升级版本或新建 release。
