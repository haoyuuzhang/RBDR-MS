# 分层弹性重调度策略

## 1. 问题背景与动机

在 `baseline-test` 仿真框架中，所有重调度策略共享一个公共的 $t=0$ 时刻 MILP 最优初始方案，策略之间的差异**仅在于如何响应后续事件**（新工件到达、机器扰动）。

现有策略的局限：

| 策略 | 事件响应方式 | 不足 |
|:---|:---|:---|
| Full MILP | 全规模 MILP 重优化 | 计算代价高、难以扩展 |
| Right-Shift | 平移受影响工序 | 不重优化，偏离最优 |
| 分派规则 (FIFO/EDD/SPT/ATC/CR) | 局部贪心选择 | 缺乏全局协同 |

本文提出 **分层弹性重调度策略（Hierarchical Rescheduling Strategy）**，作为 `BaseStrategy` 的一种新实现。核心思路：将重调度决策分解为三层 —— 车间层负责工序→服务单元的全局重分配（粗粒度），服务单元层负责单元内工序→机器的局部重排产（细粒度），机器层负责执行与扰动感知。三层通过 **周期驱动 + 事件驱动** 混合机制协同，兼顾全局优化与局部响应效率。

---

## 2. 策略架构：三层事件响应流程

策略实现 `BaseStrategy` 接口，由 `SimulationEngine` 在以下关键时刻调用：

- **`on_job_arrival(job)`**：新工件到达
- **`on_disruption(disruption)`**：机器扰动发生
- **`select_operation(machine, ready_queue, current_time)`**：机器空闲，选择下一道工序

```
事件触发
  │
  ├── 新工件到达 ──→ 车间层：纳入全局工序-单元重分配
  │                        │
  │                        ▼
  │                 服务单元层：更新受影响的单元内排产
  │
  ├── 机器扰动 ──→ 机器层：量化偏差
  │                  │
  │                  ▼
  │            服务单元层：单元内排产是否仍可行?
  │                  ├── 可行 → 局部重调度（不惊动车间层）
  │                  └── 不可行 → 上报车间层，触发全局重分配
  │
  └── 机器空闲 ──→ 服务单元层：从该单元排产计划中选择下一道工序
```

```
┌─────────────────────────────────────────────────────────────────┐
│                      车间层 (Shop Level)                        │
│  周期驱动 · 响应外部事件（新订单）· 工序→服务单元全局重分配       │
│  决策: X_{i,j,u} ∈ {0,1}                                        │
│  目标: min max 各单元平均负载                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 分配指令 X_{i,j,u}
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  服务单元层 (Service Unit Level)                  │
│  事件驱动 · 响应内部扰动 · 单元内工序→机器局部重排产               │
│  决策: x_{i,j,u,k}, y_{i,j',k}, S_{i,j}                          │
│  目标: min 单元内 makespan                                       │
│  对外接口: select_operation(machine, ready_queue, t) → Operation  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 偏差不可吸收时上报
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    机器层 (Machine Level)                        │
│  工序执行 · 加工时间偏差感知 · 扰动量上传                         │
│  感知量: Δp = |p_actual - p̄|                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 车间层：事件驱动的全局工序-单元重分配

车间层仅在**外部事件触发**时介入（新工件到达、或服务单元层上报不可吸收的扰动），对**未开始加工**的工序进行全局工序→服务单元重分配。已开始或已完工的工序保持冻结。

**触发条件与响应：**

| 触发事件 | 响应方式 | 说明 |
|:---|:---|:---|
| 常规新工件到达 | 延迟至下一周期边界处理 | 与周期扫描合并，减少重分配频次 |
| 紧急工件到达 | 即时重分配 | 以当前时刻为新调度起点 |
| 服务单元层上报扰动不可吸收 | 即时重分配 | 局部无法消化，升级到全局 |

> **与初始调度的区别**：初始 $t=0$ 方案由公共 MILP 生成（不在本策略范围内）。车间层的重分配仅针对**剩余未完成工序**，已冻结工序（`fixed=True`）不可更改。

**决策变量：**

$$X_{i,j,u} \in \{0, 1\}, \quad \forall \text{ 未冻结工序 } O_{i,j}$$

**输入 — 单元聚合加工时间**（供粗粒度负载估算）：

\begin{equation}
\tilde{P}_{i,j,u} = \frac{\sum_{k \in \mathcal{K}_u} \alpha_{u,k} \cdot \bar{p}_{i,j,u,k}}{\sum_{k \in \mathcal{K}_u} \alpha_{u,k}}
\end{equation}

**优化目标 — 最小化最大单元平均负载：**

\begin{equation}
f_{\text{shop}} = \min \; \max_{u \in \mathcal{U}} \; \frac{\sum_{i,j} X_{i,j,u} \cdot \tilde{P}_{i,j,u}}{m_u}
\end{equation}

**约束条件：**

| # | 约束 | 公式 | 说明 |
|:---:|:---|:---|:---|
| C1 | 工序唯一分配 | $\sum_{u} X_{i,j,u} = 1$ | 每道未冻结工序只分配至一个服务单元 |
| C2 | 机器适配 | $X_{i,j,u}=0$，若单元 $u$ 不含工序所需机器类型 | 由工序的可选机器集合决定 |
| C3 | 迁移代价 | $\delta_{i,j,u,u'} \geq X_{i,j,u'} - X_{i,j-1,u}$ | 标记跨单元转运需求，为服务单元层提供约束输入 |
| C4 | 冻结保持 | $X_{i,j,u} = X_{i,j,u}^{\text{prev}}$，若工序已冻结 | 已固定工序不参与重分配 |

---

## 4. 服务单元层：局部重排产与机器选择

服务单元层是策略的核心 —— 它既响应车间层下发的（重）分配指令生成单元内排产计划，又直接对接 `SimulationEngine` 的 `select_operation` 调用。

**两类运行时行为：**

| 场景 | 行为 |
|:---|:---|
| 收到车间层新分配指令 | 对分配至本单元的未冻结工序，求解单元内 MILP 排产（工序→机器 + 加工顺序 + 开始时间） |
| 收到机器层扰动上报 | 校验现有排产是否仍可行：若偏差在容忍范围内 → 局部调整；若不可行 → 上报车间层 |
| 机器空闲 (`select_operation`) | 从单元排产计划中返回该机器下一道应加工的工序 |

**决策变量：**

| 变量 | 类型 | 含义 |
|:---|:---|:---|
| $x_{i,j,u,k}$ | binary | 工序 $O_{i,j}$ 在单元 $u$ 的机器 $k$ 上加工 |
| $y_{i,j,i',j',k}$ | binary | 机器 $k$ 上 $O_{i,j}$ 先于 $O_{i',j'}$ |
| $S_{i,j}$ | continuous | 工序开始时间 |
| $C_{i,j}$ | continuous | 工序完工时间 |

**优化目标：**

\begin{equation}
f_{\text{unit}}(u) = \min \; \max_{i,j: X_{i,j,u}=1} C_{i,j}
\end{equation}

**约束条件：**

| # | 约束 | 公式 | 说明 |
|:---:|:---|:---|:---|
| C5 | 层级一致性 | $\sum_{k \in \mathcal{K}_u} x_{i,j,u,k} = X_{i,j,u}$ | 机器分配服从车间层单元分配 |
| C6 | 工艺顺序 | $S_{i,j+1} \geq C_{i,j}$ | 同工件工序严格串行 |
| C7 | 转运时间 | $S_{i,j+1} \geq C_{i,j} + L_{u,u'} \cdot \delta_{i,j,u,u'}$ | 跨单元转运需额外耗时 |
| C8 | 机器独占 | $S_{i,j} + p_{i,j,u,k} \leq S_{i',j'} + M(1-y_{i,j,i',j',k})$ | 析取约束，大 M 法 |
| C9 | 机器类型 | $x_{i,j,u,k}=0$，若 $k \notin \mathcal{M}(O_{i,j})$ | 机器须在工序可选集合内 |
| C10 | 释放时间 | $S_{i,1} \geq r_i$ | 工件释放后方可开始加工 |

---

## 5. 机器层：扰动感知与偏差反馈

机器层在仿真中以 **SimPy Process** 形式运行，与实际加工过程交织：

1. 按服务单元层的排产计划在指定机器上执行工序（`timeout(p)`）；
2. 实际加工时间 $p_{\text{actual}}$ 与标称值对比，偏差超过阈值 $\tau$ 时触发回调；
3. 将偏差信息上传至服务单元层。

**加工时间不确定性模型：**

\begin{equation}
p_{i,j,u,k}(\xi) = \bar{p}_{i,j,u,k} + \xi_{i,j,u,k} \cdot \hat{p}_{i,j,u,k}, \quad \xi \in [-1, 1]
\end{equation}

> **当前实现**：机器层扰动感知接口预留。仿真中先通过 `Disruption` 事件（特定时刻特定机器加工时间倍率变化）注入扰动，验证上层重调度逻辑的响应能力。

---

## 6. 层级耦合：事件驱动的重调度决策链

### 6.1 事件→层级→响应 映射

```
事件类型                    响应层级                    最终动作
──────────────────────────────────────────────────────────────────
新工件到达 (常规)     →    车间层 (周期批量)      →    全局工序-单元重分配
                                                    →    各服务单元层重排产

新工件到达 (紧急)     →    车间层 (即时)          →    同上，立即执行

机器加工时间波动      →    机器层感知              →    服务单元层校验
                         (可吸收) 服务单元层       →    局部重排产
                         (不可吸收) 服务单元层     →    上报车间层 → 全局重分配

机器空闲              →    服务单元层              →    select_operation → 下一道工序
```

### 6.2 与其他策略的对比

| 维度 | Full MILP | Right-Shift | 分派规则 | **分层重调度** |
|:---|:---|:---|:---|:---|
| 全局优化 | 是 | 否 | 否 | 车间层提供全局负载均衡 |
| 局部优化 | N/A | N/A | 是（贪心） | 服务单元层 MILP 局部优化 |
| 计算代价 | 高（全规模 MILP） | 低 | 低 | 中（分层分解降低维度） |
| 扰动响应粒度 | 全局 | 平移 | 局部 | 可吸收扰动局部处理，不可吸收升级全局 |
| 可扩展性 | 差 | 好 | 好 | 好（单元内问题规模可控） |

---

## 7. 接入 baseline-test 框架

本策略作为 `baseline-test/strategies/` 下的一个 `BaseStrategy` 子类实现：

```python
class HierarchicalReschedulingStrategy(BaseStrategy):
    """
    三层分层弹性重调度策略。

    车间层 (ShopLevelAllocator):    工序→服务单元全局重分配
    服务单元层 (UnitScheduler):      单元内工序→机器排产
    机器层 (embedded in SimEngine):  扰动感知与反馈
    """

    def on_job_arrival(self, job: Job, current_time: float):
        """新工件到达 → 车间层标记待重分配."""
        self.shop_level.enqueue_job(job, current_time)

    def on_disruption(self, disruption: Disruption, current_time: float):
        """机器扰动 → 服务单元层校验可行性."""
        affected_unit = machine_to_unit(disruption.machine_id)
        feasible = self.unit_schedulers[affected_unit].check_feasibility(disruption)
        if not feasible:
            self.shop_level.request_reallocation(affected_unit, disruption, current_time)

    def select_operation(self, machine: str, ready_queue: list, current_time: float) -> Operation | None:
        """机器空闲 → 从该单元排产计划中取下一道工序."""
        unit = machine_to_unit(machine)
        return self.unit_schedulers[unit].next_operation(machine, ready_queue, current_time)
```

策略内部维护：

- **`ShopLevelAllocator`**：以固定周期 $\Delta t$ 检查待重分配队列，求解车间层 MILP
- **`UnitScheduler`**（每服务单元一个实例）：接收分配指令求解单元内 MILP，接收扰动校验可行性
- 已冻结工序（`ScheduleEntry.fixed == True`）在所有层级的优化中保持不变
