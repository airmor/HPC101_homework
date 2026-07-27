# HPC101 Lab2 夜间 Claude Code 验证循环设计

日期：2026-07-27
状态：已获用户批准，等待书面规格复核
目标分支：`lab2_AMX`
当前基线提交：`35a7a27 Optimize MoE router and intra-FFN paths`

## 1. 目标与范围

本项目需要一个可在无人值守的夜间时段运行的 Claude Code 验证循环。白天由用户与主 agent 提出优化假设和候选路线，夜间系统负责在隔离环境中逐项实现或恢复候选，并在真实 ZJU HPC101 集群上完成严格验证。

夜间系统以验证为主，不以自主产生大量新方案为目标。它必须做到：

1. 从 `IDEAS.md` 领取白天准备好的候选；
2. 一轮只处理一个候选；
3. 隔离主工作区和既有候选文件；
4. 完整读取项目 memory 快照；
5. 在真实集群完成四场景长循环正确性和性能测试；
6. 用 paired benchmark 控制节点、NUMA 和执行顺序噪声；
7. 对重复 load、workspace 复用、动态申请、AVX-512 register spill 做专门审计；
8. 对达到性能门槛的候选执行四场景 VTune hotspots；
9. 达标才提交到 nightly 分支，不自动合入或 push；
10. 在次日 07:00 停止领取新实验并生成早晨报告。

本规格只设计夜间验证器。它不改变 Lab2 算法评分规则，不自动创建大量优化路线，不自动修改主分支，也不自动清理用户保留的候选或远端实验目录。

## 2. 已确认项目基线

设计时核对到的本地状态如下：

- 当前分支：`lab2_AMX`；
- 当前 HEAD：`35a7a27`；
- 当前源码：`lab2/student/moe_opt.cpp`；
- 当前源码 SHA-256：`c64c7cc413f29bcb08e784a984ffc772b41c816bf2884c2e6af09a2fcaa62c3e`；
- 该哈希与 memory 中经过真实集群验证并集成 Top-M=16 worst-root heap 的权威版本一致；
- 当前分支相对 `origin/lab2_AMX` 领先两个提交；
- 主工作区存在用户保留的未跟踪 worktree、候选源码、计划和 memory。

以下内容不得被夜间控制器删除或覆盖：

```text
.claude/worktrees/
lab2/OPTIMIZATION_PLAN.md
lab2/candidates/
lab2/student/moe_opt.cpp.best_so_far
lab2/student/moe_opt.cpp.gateup_2way_live
lab2/student/moe_opt.cpp.router_score_topm_fused
lab2/student/moe_opt.cpp.router_topm_heap
lab2/student/moe_opt.cpp.streamload
lab2/student/moe_opt.cpp.stride_only
memory_local/
```

## 3. Memory 硬约束

夜间每个 Claude 主会话必须完整读取启动时复制的 `memory_local` 快照。快照至少包括：

```text
MEMORY.md
hpc101-lab2-amx-permission.md
hpc101-lab2-cluster-access.md
hpc101-lab2-moe-impl.md
lab2_moe_optimization_knowledge.md
```

控制器保存各文件 SHA-256，以便早晨确认每一轮使用的知识基线。memory 中第 19 节存在原文件字符损坏，但第 18 节已经完整保存对应权威结论。

### 3.1 不得回退的正确实现

候选必须保留：

- AMX 的 `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILE_DATA)` 权限请求；
- `xq_workspace` 的 `MAX_D_MODEL` 正确步幅；
- `(token, slot)` 扁平并行；
- 小 N 路径使用 VNNI；
- 已验证有效的 VNNI 展开；
- S2-only intra-FFN 触发条件；
- 每个 token 只计算一次并复用的 `x_sum_workspace`；
- router int8 coarse Top-M=16；
- 16 个候选的 FP32 rescore；
- exact stable worst-root heap；
- Top-M/Top-K 的稳定 tie-breaking 语义。

### 3.2 不得无差别重复的失败方案

除非 `IDEAS.md` 明确写出与旧实验的实质差异和重新验证理由，夜间不得重复：

- expert-major AMX batching；
- 单 token AMX expert FFN；
- 只减少线程数、不真正拆分 intra-FFN；
- stride-only output workspace；
- direct int8 router Top-K；
- router W8A8 后对全部专家重新做 FP32 logits；
- intra down four-chain；
- FFN stream-load/load-use adjacency rewrite；
- 单纯把 gate/up 4-way 改成 2-way以降低源码 live set；
- router score generation 与 Top-M heap 直接融合；
- 直接使用 coarse score 决定最终 Top-K；
- 将 Sapphire Rapids 错误地视为只有 16 个 ZMM。

Sapphire Rapids 有 32 个 architectural ZMM。源码中同时出现超过 16 个向量值不能直接判定为 register spill，必须审计 GCC 最终汇编。

## 4. 总体架构

系统使用单一持久化夜间 worktree，所有候选串行执行：

```text
主工作区：保持不变
    │
    ├── run.bat
    │     └── nightly/controller.ps1
    │
    └── .claude/worktrees/nightly-YYYYMMDD/
          ├── 唯一源码写入者：Claude 主会话
          ├── 只读子 agent A/B/C
          └── 串行真实集群验证
```

接受的候选在 nightly 分支累积独立 commit。早晨由用户审查后决定是否 cherry-pick；系统不自动 push、merge 或修改 `lab2_AMX`。

## 5. 文件结构与职责

```text
PROMPT.md
IDEAS.md
run.bat

.claude/
└── agents/
    ├── assembly-auditor.md
    ├── correctness-auditor.md
    └── benchmark-auditor.md

nightly/
├── controller.ps1
├── README.md
├── schemas/
│   └── round-result.schema.json
├── state/
│   ├── NIGHTLY_STATE.md
│   ├── accepted.tsv
│   ├── rejected.tsv
│   └── current-round.env
└── logs/
    └── YYYY-MM-DD/
        ├── controller.log
        ├── MORNING_REPORT.md
        └── round-NNN/
            ├── claude.stdout.json
            ├── claude.stderr.log
            ├── claude.debug.log
            ├── result.json
            ├── round-progress.json
            ├── candidate.patch
            ├── candidate-source.cpp
            ├── assembly-audit.txt
            ├── cluster-raw.log
            ├── correctness.tsv
            ├── benchmark.tsv
            └── profiles/
```

职责边界：

- `run.bat` 是用户入口；
- `controller.ps1` 管理循环、时间、worktree、Claude 进程和运行时状态；
- `PROMPT.md` 是 Claude 每轮必须遵守的验证协议；
- `IDEAS.md` 是白天到夜间的任务队列；
- Claude 主会话是唯一源码写入者；
- PowerShell 控制器是唯一权威 ledger 和状态转换写入者；
- Claude 主会话只可写源码、实验产物和本轮 `round-progress.json` 恢复日志；
- 子 agent 全部只读；
- 真实计算节点是唯一性能和热点判据。

## 6. 参考 run.bat 的取舍

参考项目 `D:\ZJU\homework\HPC\test\run.bat` 的结构为：

```bat
@echo off
:loop
powershell -NoProfile -Command "claude --dangerously-skip-permissions -p (Get-Content -Path 'PROMPT.md' -Raw)"
timeout /t 1 /nobreak >nul
goto loop
```

本项目保留它的两个优点：

1. `run.bat` 足够简单，用户可直接双击启动；
2. Claude 使用 `-p` 非交互模式读取 `PROMPT.md`。

本项目不沿用以下行为：

- 无限 `goto loop`；
- 每秒无条件重启；
- `--dangerously-skip-permissions`；
- 不检查 Git、worktree、截止时间和状态；
- 不保存结构化结果；
- 不限制集群并发；
- 不区分候选失败和基础设施失败。

因此本项目的 `run.bat` 只定位项目根目录并调用 `nightly/controller.ps1`。循环逻辑放在 PowerShell 中，以便可靠处理日期、JSON、PID、超时、日志和恢复。

## 7. IDEAS.md 验证队列

夜间默认：

```text
AUTONOMOUS_DISCOVERY=false
```

每个候选使用稳定 ID，例如：

```markdown
## IDEA-20260727-001

- Status: READY
- Priority: P0
- Base-Commit: 35a7a27
- Target-Scenarios: S2, S3
- Files: lab2/student/moe_opt.cpp
- Hypothesis:
- Intended-Change:
- Expected-Effect:
- Correctness-Risks:
- Required-Static-Audits:
- Required-Cluster-Tests:
- Forbidden-Deviations:
- Notes:
```

实际模板中的每个字段都必须填写完整；信息不足的任务标记为 `BLOCKED`，Claude 不得自行扩大范围。

状态机：

```text
READY
  ↓
RUNNING
  ├── ACCEPTED
  ├── REJECTED
  ├── BLOCKED
  └── ABORTED
```

领取规则：

1. 选择最高优先级的 `READY`；
2. 同优先级按文件顺序；
3. 一次只允许一个 `RUNNING`；
4. 每轮只实现和验证一个假设；
5. 不把两个 idea 合并成不可归因的实验；
6. 队列为空时恢复状态、整理报告并安全退出；
7. 队列为空时不强行产生新方案。

## 8. 子 agent 分工与防冲突

### 8.1 Assembly Auditor

只读检查：

- 热循环重复 load；
- 循环不变量是否被重复加载；
- ZMM stack spill/reload；
- 源码 live set 与最终汇编的实际差异；
- 动态申请和不必要的临时 workspace；
- 已有 workspace 是否可以复用；
- 候选是否破坏已有 VNNI 指令级并行。

### 8.2 Correctness Auditor

只读检查：

- Top-M/Top-K tie-breaking；
- FP32 rescore；
- workspace 步幅；
- AMX permission；
- OpenMP data race；
- token/slot 写入边界；
- 边界尺寸和四场景风险。

### 8.3 Benchmark Auditor

只读检查：

- base/candidate 是否同一作业内 paired；
- 原始 trial 是否完整；
- 中位数和回退比例；
- 执行顺序是否交替；
- CPU affinity 与 `Cpus_allowed_list`；
- 是否把基础设施波动误判为候选性能。

三个子 agent 的工具仅为 `Read`、`Glob`、`Grep`。它们没有 Edit、Write、Bash、Git 写操作或 SSH 权限。只有主 Claude 可以修改源码、执行受控命令和操作集群。

## 9. Night key、worktree 与截止时间

夜间标识按睡眠周期计算：

- `00:00–06:59` 启动或恢复时使用前一天日期；
- 其他时间使用当天日期。

示例：

```text
branch:   nightly/20260727
worktree: .claude/worktrees/nightly-20260727
start:    2026-07-27 evening
stop-admission: 2026-07-28 07:00:00 Asia/Shanghai
```

控制器创建或复用相同 night key 的 branch/worktree。创建前和复用前都必须验证解析后的绝对路径仍位于项目的 `.claude/worktrees` 内。

到次日 07:00：

- 不再领取新的 `READY`；
- 已经进入验证阶段的候选可以完成既定验证；
- 不因时间不足省略四场景、第五轮复测或汇编审计；
- 收尾阶段不得产生新实现变体；
- 截止后 90 分钟仍无法收尾则保存证据并标记 `ABORTED`，防止无限挂起。

## 10. Claude Code 运行模式

设计时本机 Claude Code 版本为：

```text
2.1.132
```

每轮使用非交互模式，核心参数为：

```text
-p
--permission-mode dontAsk
--effort high
--output-format json
--json-schema <round-result.schema.json>
--debug-file <round debug log>
--no-session-persistence
--strict-mcp-config
--allowed-tools <explicit allowlist>
--disallowed-tools <explicit denylist>
--name <night and round identity>
```

不使用：

```text
--dangerously-skip-permissions
--worktree
WebSearch
WebFetch
git push
git reset
git clean
git rebase
force checkout
```

不设置 `--max-budget-usd`。每轮使用新 Claude 会话，恢复依据磁盘状态和结构化结果，不依赖对话历史。

主会话允许：

- Read、Glob、Grep、Edit、Write；
- 调用三个只读子 agent；
- 受限的 Git status/diff/add/commit；
- 编译和汇编审计；
- 固定端口、固定账户的 SCP；
- `BatchMode=yes` 的固定集群 SSH；
- 固定远端实验目录中的构建和作业提交。

## 11. 每轮状态机

```text
PRECHECK
→ RECOVER
→ SELECT_IDEA
→ STATIC_AUDIT
→ CREATE_CANDIDATE
→ BUILD
→ FOUR_SCENARIO_CORRECTNESS
→ PAIRED_BENCHMARK ×3
→ 边界时扩展到 ×5
→ ASSEMBLY_AUDIT
→ 达标候选 FOUR_SCENARIO_VTUNE
→ ACCEPT_AND_COMMIT 或 REJECT_AND_RECORD
→ UPDATE_STATE
→ NEXT ROUND
```

正确性失败立即拒绝，不继续性能测试。基础设施失败不计入性能失败。

## 12. 单轮超时和恢复

每个 Claude 进程默认超时 90 分钟。

控制器必须：

1. 只终止自己启动并记录的 PID；
2. 不按模糊进程名批量终止；
3. 保存 stdout、stderr、debug、patch 和源码副本；
4. 启动一次恢复会话；
5. 第二次仍失败则标记 `ABORTED`；
6. 不自动取消身份不明的远端作业。

控制器在启动 Claude 前创建权威的 `current-round.env`，其中包含 idea ID、round ID、base commit、启动阶段和 Claude PID。Claude 不直接改写该 ledger。

在上传或提交集群作业前，主 Claude 必须使用临时文件加原子 rename 更新本轮目录中的 `round-progress.json`，记录：

- idea ID；
- round ID；
- candidate SHA-256；
- remote experiment path；
- 当前阶段；
- 已知或待获取的 job ID；
- 预期 job 目的。

若 Claude 异常退出，恢复会话读取 `current-round.env` 和 `round-progress.json`。控制器只在验证结构化结果后更新权威 ledger。所有状态和恢复日志都使用临时文件加原子 rename，避免断电留下半行内容。

## 13. 真实集群环境

SSH：

```bash
ssh -o BatchMode=yes -p 443 \
  h3250105245+lab2+hpc101@clusters.zju.edu.cn
```

作业：

```bash
cd /home/h3250105245/work
hpc submit -p lab2 -c 16 -- bash <script>
```

固定环境：

```bash
OMP_NUM_THREADS=16
OMP_DYNAMIC=FALSE
OMP_PROC_BIND=close
OMP_PLACES=cores
```

每份结果记录：

- job ID；
- `Cpus_allowed_list`；
- 编译器和编译参数；
- base/candidate 源码 SHA-256；
- 二进制 SHA-256；
- 四场景 correctness；
- 每次 paired trial 原始时间；
- VTune result path 和导出报告。

## 14. 远端隔离和可复现构建

每轮目录：

```text
/home/h3250105245/work/lab2-nightly/YYYYMMDD/round-NNN/
├── base/
├── candidate/
├── artifacts/
└── run-paired.sh
```

流程：

1. 从冻结 base commit 生成本地 `lab2` archive；
2. 上传 archive；
3. base 和 candidate 从同一 archive 解压；
4. 仅将候选 `moe_opt.cpp` 覆盖到 candidate；
5. 分别构建两个二进制；
6. 记录 source/binary hash；
7. 由同一计算作业交替运行两个二进制。

远端旧实验目录不自动删除。一次只允许一个候选占用集群测试通道；同一候选可以按顺序使用 benchmark job 和达标后的 profile job，但不得与另一候选并行。

## 15. 四场景长循环正确性

所有候选必须执行：

```text
S1: 1 256 128 16 4 10000 --benchmark
S2: 1 1024 512 16 4 3000 --benchmark
S3: 128 256 128 16 4 1000 --benchmark
S4: 1024 512 128 512 2 100 --benchmark
```

`--benchmark` 只跳过耗时的 scalar baseline 循环，仍会：

- 对 optimized implementation 执行长循环；
- 在计时后生成 fresh verification batch；
- 调用 optimized 和 reference；
- 执行 `check_result`。

candidate 任一场景错误立即拒绝。如果 base 失败，则视为归档、构建或基础设施异常，不归责于候选。

## 16. Paired benchmark 和验收门槛

四场景都做 base/candidate paired benchmark，不只测目标场景。

三轮顺序：

```text
Trial 1: base → candidate
Trial 2: candidate → base
Trial 3: base → candidate
```

临界或波动结果增加到五轮并继续交替。所有 trials 位于同一个 `-c 16` 计算作业，使用相同节点、CPU 集和 affinity。

接受门槛：

1. 四场景全部正确；
2. 目标场景中位数耗时改善至少 1.0%；
3. 其他任一场景中位数回退不得超过 0.5%；
4. 边界结果经过五轮复测；
5. 汇编和静态审计没有新增明显风险；
6. 达标候选完成四场景 VTune；
7. 结果、job ID 和 hash 完整归档。

候选比较基准是当前已接受优化版本，不使用 scalar reference 的 Speedup 作为 nightly acceptance 值。

## 17. 重复 load、申请复用和 AVX-512 专项审计

每个候选必须回答：

1. forward 热路径是否出现 `new`、`malloc`、`free` 或容器 resize；
2. 是否每次 forward 重复初始化可复用的数据；
3. 新临时数组是否确有必要；
4. 是否可以复用现有静态 workspace；
5. 循环不变量是否被反复加载；
6. 热点 load 是否因为地址计算、布局或缓存而重复；
7. 最终汇编是否有 ZMM stack spill/reload；
8. 是否错误地以“超过 16 个源码 AVX-512 值”为由降低展开度；
9. 修改 load lifetime 后是否真正减少 load，而不是仅改变源码顺序；
10. 是否损害现有的独立 `dpbusd` accumulator chains。

汇编审计需识别 `%rsp`、`%rbp` 及其他 frame-relative 地址中的 ZMM load/store，并区分普通栈访问与真正 spill。只有最终汇编证据可以支持“寄存器压力造成回写”的结论。

候选优先复用已存在的申请和 workspace。若必须增加全局缓冲区，idea 必须说明大小、生命周期、对 cache/NUMA 的影响和不能复用旧缓冲区的原因。

## 18. 四场景 VTune hotspots

普通候选完成已有权威 hotspot 基线对照、静态审计和四场景 paired benchmark。达到性能门槛的候选再执行四场景 VTune：

| 场景 | Profile iterations |
|---|---:|
| S1 | 1,000,000 |
| S2 | 150,000 |
| S3 | 40,000 |
| S4 | 1,000 |

全部使用 `--benchmark`。报告至少导出函数和 source-line CSV/text 数据，并关注：

- S1 的 OpenMP/libgomp 开销；
- S2 的 intra-FFN outlined region、reduction 和 barrier；
- S3 的 FFN weight loads、broadcast 和 VNNI；
- S4 的 Top-M heap、16-candidate FP32 rescore 和 FFN；
- 候选声称要减少的重复 load 或申请；
- 新出现或明显转移的热点。

VTune 本身不以单一百分比决定接受，但实验报告必须说明热点证据是否支持候选假设。

## 19. 接受、拒绝和精确恢复

### 19.1 ACCEPTED

接受后：

1. 保存完整实验结果；
2. 更新运行状态；
3. 只添加明确允许的文件；
4. 创建独立 commit：

```text
nightly(opt): accept IDEA-YYYYMMDD-NNN <summary>
```

5. 下一轮以该 commit 作为新 base；
6. 不 push，不自动合入 `lab2_AMX`。

### 19.2 REJECTED

拒绝前保存：

```text
candidate.patch
candidate-source.cpp
assembly-audit.txt
cluster-raw.log
correctness.tsv
benchmark.tsv
profiles/（若已生成）
```

随后只对已经验证位于 nightly worktree 内的目标源码执行精确恢复。不执行 `git reset --hard`、`git clean` 或 `git checkout .`。候选虽然不留在活动工作树，完整源码和 patch 仍被保留。

## 20. 基础设施错误

以下情况不算候选性能失败：

- SSH 临时断开；
- `hpc submit` 暂时不可用；
- 作业排队或节点启动失败；
- NFS 临时错误；
- base/candidate 同时异常；
- base correctness 失败。

单项基础设施操作最多重试两次。连续三次基础设施失败后整夜安全停止。基础设施失败记录为 `BLOCKED` 或 `ABORTED`，不写入候选性能拒绝结论。

## 21. 结构化结果

每轮 Claude 必须返回符合 JSON Schema 的结果，至少包含：

```text
night_key
round_id
idea_id
status
base_commit
candidate_commit
candidate_sha256
changed_files
summary
correctness[4]
paired_benchmarks[4]
assembly_audit
allocation_audit
load_audit
cluster_jobs
profile_results
infrastructure_failures
artifacts
recommended_next_action
```

控制器验证 JSON 后才更新 ledger。JSON 缺失或不符合 schema 时启动一次恢复会话，不根据自由文本猜测接受状态。

## 22. 早晨报告与人工审核

停止时生成 `MORNING_REPORT.md`，包括：

1. 起止时间和截止时间；
2. 起始 commit 和 nightly 分支；
3. Claude Code 版本；
4. memory 快照哈希；
5. accepted/rejected/blocked/aborted 列表；
6. accepted commits；
7. 所有真实集群 job ID；
8. 四场景 correctness；
9. 每个原始 paired trial；
10. 中位数、提升和回退；
11. 四场景热点变化；
12. 重复 load、申请复用和 ZMM spill 结论；
13. 未完成实验和精确恢复方式；
14. 白天值得继续讨论的问题。

报告可以建议：

```bash
git show <accepted-commit>
git cherry-pick <accepted-commit>
```

但不会自动执行。

## 23. 失败安全原则

系统始终遵循：

- 不确定时停止，不猜测；
- 正确性优先于性能；
- 真实集群优先于本地推测；
- 原始 trial 优先于汇总数字；
- 最终汇编优先于源码 live-set 猜测；
- 候选隔离优先于并发吞吐；
- 保存失败证据，不清理历史；
- 队列为空即结束，不为了跑满夜间而发明方案。

## 24. 实施完成标准

后续实现只有满足以下条件才算完成：

1. `run.bat` 可以从项目根目录启动控制器；
2. 重复启动被 lock 阻止；
3. 能创建和恢复指定 nightly worktree；
4. 主工作区保留文件不被修改；
5. `IDEAS.md` 能稳定选择单个 READY idea；
6. 子 agent 工具权限确实为只读；
7. Claude CLI 不使用 bypass permissions；
8. 单轮超时和一次恢复路径可测试；
9. 07:00 后不再领取新任务；
10. 队列为空安全退出；
11. 四场景长循环命令正确；
12. paired 三轮和五轮统计正确；
13. 真实集群命令、affinity 和远端隔离路径正确；
14. 结构化结果能通过 schema 验证；
15. accepted 候选形成独立 nightly commit；
16. rejected 候选保留 patch、源码和原始结果；
17. 四场景 VTune 报告可归档；
18. 早晨报告包含全部审计证据；
19. 系统不 push、不自动合入、不删除旧实验；
20. 故障注入测试能证明恢复和安全停止机制有效。
