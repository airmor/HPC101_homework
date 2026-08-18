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
- **计算节点**: `hpc submit -p lab4 -c N -- bash ./job_run.sh`
  - lab4 分区：1–60 核，100GB，walltime 30min，镜像 `hpc101-lab4:4`
  - **关键坑**：计算节点默认 cwd 不是 devpod 的提交 cwd（容器初始 cwd
    指向 `/workspace/lab4` 而非 NFS 家目录），导致 `run.sh` 的
    `ROOT_DIR=$(pwd)` 解析错误、找不到 `build/ABE`。已用 `job_run.sh`
    包装器解决：显式 `cd` 到家目录 lab4 并 `export AMSS_BUILD_DIR` 等
    绝对路径。
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
拿到真 baseline 时间 + 热点 profile，作为所有后续优化的依据和报告素材。

### 步骤
1. 恢复 `Final_Evolution_Time=40.0`（smoke test 改过 1.0，必须改回）
2. 正式 baseline 计时（计算节点，非 devpod）：
   - `hpc submit -p lab4 -c 30 -- bash ./job_run.sh`
   - 默认 `MPI_processes=30, OMP_threads=1, GPU=no`（纯 MPI baseline）
   - 记录 `This Program Cost = X Seconds`
   - 另用 `--twop-cache` 跑一次，分离 TwoPuncture 时间 vs ABE 演化时间
3. 正确性验证：`./check.sh` 对 `golden/`，确认 baseline 本身通过
4. perf profile（计算节点）：
   - 编译加 `-g`（已做：`AMSS_OPT="-O3 -g"`）让热点对应源文件/行号
   - `perf stat -d ./run.sh` — 完整 IPC/cache miss/分支/TLB
   - `perf record --call-graph dwarf -- ./run.sh` — 采样热点+调用栈
   - `perf report` 找 top 函数
   - 确认报告采样到 `ABE`（run.sh 会启动 Python→TwoPuncture→mpirun 子进程；
     必要时直接对 `mpirun -n 30 ./ABE` 采样）

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
