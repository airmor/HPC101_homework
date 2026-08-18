# Lab4 任务一：ABE CPU 演化优化 — 设计文档

## 目标与约束

### 任务
在华为鲲鹏 920B（ARM aarch64）上，保证数值正确性的前提下，优化
`TwoPunctureABE + ABE` 的端到端运行时间（`AMSS_NCKU_Program.py` 输出的
`This Program Cost`）。

### 评分锚点
- 340s = 100 分，500s = 60 分（曲线 `y = a·x^b`，满分 100）
- 正确性是前置门槛：不通过正确性检查的运行不进入性能计分

### 正确性约束（来自实验文档）
- `bssn_BH.dat` 六个黑洞坐标列（BH1_x..BH2_z）相对 RMS 误差 ≤ 0.1%
- `bssn_constraint.dat` 中 Grid Level 0 的 H / Px / Py / Pz constraint 最大值 ≤ 2.0
- 校验：`./check.sh`（对 `golden/`）

### 修改范围（允许）
`src/lab4/src/` 下 C++/Fortran/CUDA 源文件、`CMakeLists.txt`、`compile.sh`、
`run.sh`、必要辅助文件。算法必须等价，禁止用低精度替换高精度来换速度。

### `AMSS_NCKU_Input.py` 修改限制
正式测试只允许改 MPI/OpenMP/GPU 相关参数；物理参数、网格、演化时间
（`Final_Evolution_Time=40.0`）、输出间隔严禁改。调试可临时缩短演化时间，
但短时长结果不作评分依据。

## 总体策略

严格遵循实验文档反复强调的优化循环：

```
跑通 → 计时 baseline → profile 找热点 → 做一个小而明确的优化
     → 验证正确性 → 重新计时 → 记录收益 → 重复
```

每一步都要有数据支撑，不盲目套用优化。报告需展示 profiling 证据，
不仅是最终加速比。

分阶段推进，每个阶段是独立可验证的闭环：

| Phase | 目标 | 触发条件 |
|-------|------|----------|
| 0 | baseline + profile | 必做，所有后续依据 |
| 1 | 编译/工具链 | 必做，低成本高收益 |
| 2 | OpenMP + MPI + 绑核 | 必做，最大未开发空间 |
| 3 | Fortran kernel | 必做，profile 驱动定具体位置 |
| 4 | 通信优化 | 条件触发：MPI 占比 > 15–20% |
| 5 | 收尾固化 + 提交 | 必做 |

## 环境事实（已确认）

- **DevPod**: `ssh h3250105245+lab4-1+hpc101@clusters.zju.edu.cn -p 443`
  （arm64-920B 预设）
- **计算节点**: `hpc submit -p lab4 -c N -t <walltime> -- bash ./job_run.sh`
  - lab4 分区：1–60 核，100GB，**walltime 硬上限 30min（1800s）**，
    镜像 `hpc101-lab4:4`
  - **关键坑 1**：计算节点默认 cwd 不是 devpod 的提交 cwd（容器初始 cwd
    指向 `/workspace/lab4` 而非 NFS 家目录），导致 `run.sh` 的
    `ROOT_DIR=$(pwd)` 解析错误、找不到 `build/ABE`。`job_run.sh`
    包装器显式 `cd` 到家目录 lab4 并 `export AMSS_BUILD_DIR` 等绝对路径。
  - **关键坑 2**：OpenMPI 5/PRRTE 在容器内读不到 cgroup cpuset（容器
    `nproc=30` 但 PRRTE 默认 slot 数远少于 30，实际分配到的 CPU 列表是
    `64-93` 这 30 核），`mpiexec -n 30 ./ABE` 报 "not enough slots"。
    `job_run.sh` 用 `AMSS_MPIEXEC="mpiexec --allow-run-as-root --oversubscribe"`
    绕过。后续改 MPICH 时此坑不同（MPICH 无 PRRTE slot 概念）。

### Baseline 实测（关键，决定策略）

- 每个物理 timestep（Δt=1.0）≈ **60.3 秒**（30 rank、`-O3 -g`、纯 MPI，
  无 OpenMP）。从 `ABE_out.log` 的 `Timestep #N: ... Computer used X seconds`
  实测，t=9..15 稳定在 60s/步。
- TwoPuncture + setup ≈ 20s。
- **baseline 端到端（外推 t=40）≈ 2432s** = 60.3 × 40 + 20。
- **完整 baseline 跑不完**：walltime 硬上限 1800s < baseline ~2432s，
  实测 t=15 时被 Timeout 杀死。不是排队问题，是硬上限。
- 评分对照：340s=100 分（需 ~7.2× 加速），500s=60 分（需 ~4.9× 加速），
  baseline ~2432s。优化空间巨大。

### 计时与验证协议（因 walltime 限制）

- **性能基准**：用每步秒数 × 40 + TwoPuncture 外推 t=40 总时间。每步秒数
  从 `ABE_out.log` 的 `Computer used X seconds` 取（稳定值）。
- **优化迭代**：用 **t=5 短跑**（~5 分钟，安全在 walltime 内）测每步秒数，
  外推比较。文档："大多数优化手段在不同演化时间的 Scaling 效果非常好"。
- **正确性验证**：`./check.sh` 支持不同步数比对（对 `golden/`），用短时长
  验证。调试可临时缩短 `Final_Evolution_Time`，但正式评分必须恢复 40.0。
- **最终确认**：最优配置若总时间 < 30min，跑一次完整 t=40 确认。
- **CPU**: TaiShan-v120（鲲鹏 920B），aarch64，256 核，4 NUMA node
  （每 node 64 核）。任务限 30 核。
- **默认工具链**: GNU 14 + OpenMPI 5（`gfortran`/`mpicxx`/`mpifort`/`perf`）
- **可选工具链**: Arm Compiler for Linux
  - `armclang++`/`armflang` 位于 `/opt/arm/arm-linux-compiler-24.10.1_Ubuntu-22.04/`
  - 激活方式：`source /etc/profile.d/acfl.sh`（注意：环境里没有 `module`
    命令，README 提到的 `module load acfl/24.10.1` 实际通过 acfl.sh）
  - 可选 MPICH：`mpicxx.mpich`/`mpiexec.mpich`
- **SSH 限流**：集群 SSH 代理对短时间高频连接限流（连续多次连接会被
  `Permission denied`）。需控制连接频率，必要时用 Monitor 等待恢复。

## Phase 0: Baseline + Profile

### 目标
拿到 baseline 时间 + 热点 profile，作为所有后续优化的依据和报告素材。

### Baseline（已完成）
- 每步 ~60.3s，外推 t=40 ≈ 2432s（详见"环境事实"）。
- 完整 baseline 受 walltime 30min 限制跑不完，用外推值作基准。

### Profile（进行中）

#### perf stat 结果（t=3 实跑 191.3s，cache 命中跳过 TwoPuncture）
```
task-clock:u        5,718,158 msec    # 28.294 CPUs utilized
instructions:u      9,103,483,180,716 # IPC 1.28  insn/cycle  ← 偏低
cycles:u            7,092,576,951,655  # 1.240 GHz
branches:u          1,228,402,171,500  # branch-miss 1.33% (正常)
L1-dcache-loads:u    3,630,268,114,257 # L1-dcache-miss 2.58% (正常)
LLC-loads:u             53,472,158,238 # LLC-load-miss 52.59% ← 严重
user time 2676s : sys time 3043s       # sys > user ← 通信/sys 开销大
```

**瓶颈判断（初步，待 perf record 热点函数确认）**：
1. **访存瓶颈为主**：LLC miss 52.6% 过半未命中 → 数据搬运重，不是纯计算。
   → Phase 3 访存优化（数据布局、cache 友好、消除临时数组）是关键收益点。
2. **sys 时间 > user 时间**：30 rank MPI 通信 + kernel 过渡（fork/调度）开销大。
   → 印证 MPI 通信占比可能高，Phase 4 通信优化大概率需要做（非条件触发，
   而是实锤了 sys 占比高）。但也可能是 OpenMP 缺失导致单 rank 串行 +
   MPI barrier 等待放大了 sys。
3. **IPC 1.28**：远未到计算瓶颈（健康 >2），向量化/OpenMP 还有空间。
4. branch-miss 和 L1 miss 正常，不是分支/局部性微观问题。

#### perf record 热点函数表（t=3，582K samples，`-e cycles:u -m1 --call-graph fp`）

| 占比 | 函数 | 来源 | 类型 |
|------|------|------|------|
| **25.57%** | `0xf13c4` opal | libopen-pal.so (OpenMPI) | **MPI 进度/通信** |
| **21.36%** | `compute_rhs_bssn_` | bssn_rhs.f90 | BSSN 右端项计算 |
| **17.07%** | `[k] 0xffff...` | 内核态 | sys/kernel（通信/syscall） |
| 3.30% | `fdderivs_` | diff_new.f90 | 二阶差分 |
| 3.27% | `__memcpy_sve` | libc | 内存拷贝 |
| 2.82% | `lopsided_` | lopsidediff.f90 | 单侧差分 |
| 2.66% | `polint_` | ABE | 插值（prolong） |
| 2.39% | `__memset_sve_zva64` | libc | 内存清零 |
| 2.11% | `kodis_` | kodiss.f90 | Kreiss-Oliger 耗散 |
| 1.70% | `fderivs_` | diff_new.f90 | 一阶差分 |
| 1.32% | `prolong3_` | prolongrestrict | AMR 延拓 |
| ~1.10% | `cfree`/`malloc` | libc | 动态内存分配/释放 |
| 0.66% | `rungekutta4_rout_` | rungekutta4_rout.f90 | RK4 时间推进 |

**瓶颈结论（perf record 实锤，修正 perf stat 初判）**：
1. **MPI 通信是第一大头（~45%+）**：`libopen-pal` opal_progress 系列 25.6% +
   kernel 态 17% + 多个 opal 小项。rank 间大量时间在 MPI barrier 等待 / 消息
   进度轮询。与 perf stat 的 sys>user 完全吻合。
   → **Phase 4 通信优化不是条件触发，而是第一优先级之一**。
2. **`compute_rhs_bssn_` 21.36%** 是计算热点第一（bssn_rhs.f90），内部大量
   整体数组语法 + fderivs/fdderivs 调用。
3. **差分类合计 ~8.2%**（fdderivs+lopsided+fderivs+kodis）：stencil 计算，
   OpenMP + 向量化天然落点。
4. **内存操作 ~6.8%**（memcpy+memset+malloc/cfree）：临时数组分配/拷贝/清零，
   对应整体数组语法产生的中间临时数组。消除临时数组能吃下这部分。
5. **AMR prolong/restrict ~4%**：不是大头。

#### 策略调整（基于热点表）
原 Phase 顺序（编译→OpenMP→kernel→通信）改为：
- Phase 1 编译/工具链
- Phase 2 OpenMP + MPI 平衡（降 rank 数 → 直接减通信量，OpenMP 填补单 rank 算力）
- Phase 3 通信优化（提前，针对 opal_progress / barrier / ghost exchange）
- Phase 4 Fortran kernel（compute_rhs + 差分 + 消除临时数组）
- Phase 5 收尾固化

**理由**：通信 45%+ 是第一瓶颈，但通信优化的前提是先用 OpenMP 降 rank 数
（纯 MPI 30 rank 通信开销 inherent 高）。OpenMP 同时给 kernel 向量化铺路。

#### perf profile 操作要点（记入 spec 供复现）
- 计时/profile 用 `--twop-cache`，先手工 seed `twopuncture_cache/<sha1>/`
  （用上一次 run 的 `Ansorg.psid` + `TwoPunctureinput.par` 的 sha1）。
- perf 在容器内 mmap 受限，必须 `-m 1`（或 `-m 8`），且 `--call-graph fp`
  替代 `dwarf`。`-e cycles:u` 或 `task-clock:u` 采样用户态热点。
- perf stat 不受 mmap 限制，已成功拿到硬件计数器。

### 产出（报告核心素材）
- 端到端时间分解表（TwoPuncture / ABE 演化 / setup+plot）
- ABE 内部 top-10 热点函数表（文件、计算类型：差分/逐点/通信/分配）
- 瓶颈性质判断：计算 / 访存 / 通信
- MPI rank 间负载均衡情况
- IPC、cache miss、TLB miss、branch miss 异常项

**没有这张表，不开始任何优化。** Phase 1–4 的具体清单由 profile 数据定。

## Phase 1: 编译 / 工具链

### 目标
不碰算法、不改源码逻辑，先榨取编译器和工具链的免费收益。

### 方向（按预期收益排序）
1. **`-O3` + 架构选项**：当前只有 `-O3`，无 `-march/-mtune/-mcpu`。
   鲲鹏 920B 是 ARMv8.2 + NEON/SVE。先 `lscpu` 确认指令集，加
   `-mcpu=native`（或 TaiShan-v120 对应的 `-mcpu`/`-mtune`）开启 NEON
   自动向量化。用 `-fopt-info-vec` 确认向量化是否生效。
2. **工具链对比**（用户已确认愿意投入）：
   - GNU gfortran/mpicxx（baseline）
   - Arm `armflang`/`armclang++`（`source /etc/profile.d/acfl.sh`）
   - 每次对比用不同 `BUILD_DIR`（CMake 缓存编译器，不可复用 cache）
   - 固定 rank/输入，记录正确性 + 端到端时间
   - 注意：C++/Fortran/MPI wrapper 要成套选，CMake 显式链接 gfortran
     runtime，切 Fortran 编译器需同步处理 runtime
3. **谨慎 `-ffast-math` / `-Ofast`**：可能重排浮点表达式，影响 BSSN 收敛
   或数值结果。**必须重新过端到端正确性验证**。先标"高风险"，不默认开。
   TwoPuncture（初值，不参与演化）可单独尝试 `-Ofast`，风险低些——但仍
   profile 后定。
4. **MPICH vs OpenMPI**（次要）：默认 OpenMPI，可试 MPICH
   （`mpicxx.mpich`/`mpiexec.mpich` + `AMSS_MPIEXEC`），需新 BUILD_DIR。

### 正确性 + 计时协议
每个工具链配置：先短时长（如 t=2）验证正确性，再全时长（40.0）计时。
固定 `MPI=30, OMP=1` 作此阶段基线。

### 预期
编译/工具链通常给 5–20% 端到端收益，最便宜的优化。

## Phase 2: OpenMP + MPI + 绑核

### 目标
baseline 只有 MPI 30 rank、CMake 未开 OpenMP。这是最大未开发空间。
在 `MPI rank × OMP thread`、绑核、NUMA 间找平衡。

### 子步骤
1. **启用 OpenMP 编译**：`AMSS_ENABLE_OPENMP=ON`（改 `compile.sh` 或命令行）
2. **找并行插入点**（profile 后定具体位置，候选）：
   - `bssn_rhs.f90` / `compute_rhs_bssn`：大量整体数组语法（`gxx = dxx + ONE`
     等）。向量化好，但 OpenMP 需改显式 `do` 循环才能加 `!$omp parallel do`
   - `diff_new.f90` / `fderivs`：已有显式 `do k/j/i` 三重循环，OpenMP 天然
     落点；带 `if` 分支（边界/内部），可分支剥离提升向量化
   - TwoPuncture 计算密集循环
3. **配置空间扫描**（固定 30 核总量，网格搜索）：
   - 纯 MPI `30×1`（baseline）
   - `15×2`、`10×3`、`6×5`、`3×10`
   - 每组配绑核：`OMP_PROC_BIND=close`、`OMP_PLACES=cores`、
     `numactl --cpunodebind`/`--membind`
4. **NUMA**：`lscpu`/`numactl --hardware`/`hwloc-ls` 看拓扑，4 NUMA node，
   确保 rank+thread 不跨 NUMA 访问远端内存。

### 关键判断（思考题 2 预演）
MPI 和 OMP 能否只用一个？MPI 负责 patch 间分布式、OMP 负责 patch 内共享。
纯 MPI 通信开销大、纯 OMP 无法跨节点且受单进程内存限制。结论大概率"混合
最优"，但需 profile 数据支撑。

## Phase 3: Fortran kernel 优化（profile 驱动）

### 前提
Phase 0 perf 报告定具体热点函数。已读代码，候选方向：

### `bssn_rhs.f90` / `compute_rhs_bssn`（980 行）
BSSN 右端项，大量整体数组运算。如果它是热点：
- `-fopt-info-vec` 确认是否已自动向量化
- 整体数组语法可能产生中间临时数组 → 改显式 `do` 循环，消除临时数组，
  改善 cache 局部性
- 开头 `sum()` 做 NaN sanity check：每步 sum 所有数组，可能是不必要开销
  （profile 会显示），考虑用编译宏关闭或降频

### `diff_new.f90` / `fderivs`（1017 行）
4 阶差分，已有显式 `do k/j/i` 循环，带 `if` 分支。如果它是热点：
- OpenMP 落点（循环规整后）
- 分支剥离：内部点与边界点拆成两个循环，内部无分支利于向量化
- `fh` padded 临时数组每次调用重填（`symmetry_bd`），考虑复用

### `prolongrestrict_cell.f90`（3649 行）
AMR 层间插值，体积最大。通信/插值占比高才深入。

### `rungekutta4_rout.f90`（246 行）
RK4 时间推进，通常大循环 + 数组运算。

### 原则
每个 kernel 改动，单独计时 + 正确性验证，记录"改了什么 + 收益多少 + 是否
影响精度"。报告素材。

## Phase 4: 通信优化（条件触发）

### 触发条件
perf / mpi timing 显示 MPI 时间（`MPI_Allreduce`/`MPI_Sendrecv`/ghost
exchange）占总时间 > 15–20%。不盲目优化。

### 如果触发的方向
1. ghost zone exchange：看 `Parallel.C`/`MPatch.C` 通信量，合并小消息、
   减少同步点
2. 通信-计算重叠：非阻塞 `MPI_Isend/Irecv` + 等待时做独立计算
3. 负载均衡：各 rank 计算时间差异大时重新划分 patch（AMR 层级天然不均衡）

### 如果不触发
跳过此 phase，精力全投 Phase 3。

## Phase 5: 收尾固化 + 提交

### 步骤
1. 固定最优配置（工具链/编译选项/MPI×OMP/绑核），跑正式 t=40.0，记录
   `This Program Cost`
2. `./check.sh` 全过
3. 第二次计时确认稳定（排除抖动）
4. 清理：确认 `Final_Evolution_Time=40.0`、删 `twopuncture_cache/`、
   删 `binary_output/` 大文件
5. 提交物：
   - 实验代码压缩包（`src/`、`CMakeLists.txt`、`compile.sh`、`run.sh`、
     `AMSS_NCKU_Input.py` 只含允许的 MPI/OMP 参数）
   - 实验报告 PDF
   - `GW250118/AMSS_NCKU_output/` 关键 .dat + 日志 + `GW250118/figure/` 图
   - 不提交：`binary_output/`、`twopuncture_cache/`、大 .bin、profiler 原始目录

## 报告与思考题策略

### 报告
优化全程收集真实 profile 数据、配置对比、失败尝试记录，作为报告事实素材。
报告需说明：运行环境、baseline 性能、正确性结果、profiling 结论、主要优化
过程（每项：针对什么瓶颈、改了哪些位置、端到端收益、是否影响正确性）、
最终运行配置（MPI rank、OMP thread、绑核/NUMA、编译参数）。失败尝试也写。

### 思考题（禁止 AI 生成，会检测痕迹）
提供事实素材（profile 数据、配置对比、失败记录），但**论述和结论由用户自己
写**，只提供数据和不涉及代写的澄清。思考题方向：
1. 主要热点更接近 stencil / 稠密线性代数 / 通信调度？结合 profile
2. CPU 上 MPI rank × OMP thread 权衡 + 能否只用一个
3. 优化后程序在哪些并行结构上实现并行
4. 精度小幅差异：合理浮点误差 vs 程序错误的判断
5. 单 MIG 实例 MPI_processes > 1 的问题（任务二相关，可前置思考）
6. Stencil 是否一味用 shared memory 更好（任务二相关）
7. A100 40GB vs 80GB 理论差异 + 引用
8. (Bonus) 实验体验与改进建议

## 风险与回退

- **正确性失败**：任一优化导致 `check.sh` 不通过 → `git stash`/回退该改动，
  记录为失败尝试（报告素材），继续下一方向
- **`-ffast-math` 破坏数值**：立即回退，标记"高风险已验证不适用"
- **walltime 超 30min**：baseline 可能就接近上限，长运行需关注；优化后
  应远低于 30min
- **SSH 限流**：控制连接频率，必要时等恢复，不硬撞
- **家目录不共享**：arm64 与 x86 家目录不同，任务二需在 x86 DevPod
  重新 clone；任务一仅在 arm64，当前已就绪
