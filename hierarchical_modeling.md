

# 柔性作业车间弹性调度问题的三层数学模型

## 1. 问题背景与描述
在离散制造环境中，柔性作业车间面临内部扰动（如机器故障、加工耗时波动）和外部扰动（如紧急订单、原材料延迟）的挑战。本模型旨在通过**车间层 (Shop Level)**、**服务单元层 (Service Unit Level)** 和 **机器层 (Machine Level)** 的三层分层调度架构，利用冗余库存、冗余能力及提前期缓冲，最大化系统的**弹性 (Resilience)**。

### 建模假设
1. 工序不可中断且机器同一时间只能加工一道工序。
2. 属于同一工件的工序具有严格的先后顺序约束。
3. 跨服务单元的任务迁移会产生转运时长 $L_{u,u'}$ 和协调开销 $\varepsilon_{i,j}$。
4. 机器能力具有异质性，通过重要程度系数 $\alpha_{u,k}$ 进行量化。

---

## 2. 弹性度量指标 (Resilience Metric)
系统弹性 $R(\xi)$ 通过资源可用性与恢复时间进行度量：
\begin{equation} 
R(\xi) = e^{-(\phi(\xi) + \eta(\xi))} 
\end{equation}
其中：
*   **资源受损比例 $\phi(\xi)$**：反映受扰动机器及其重要性的加权比例。
    \begin{equation} \phi(\xi) = \frac{\sum_{u,k} \alpha_{u,k} \Phi_{u,k}(\xi)}{\sum_{u,k} \alpha_{u,k}} \end{equation}
*   **完工延迟比例 $\eta(\xi)$**：反映场景 $\xi$ 下完工时间相对于标称值的偏差。
    \begin{equation} \eta(\xi) = \frac{C_{max}'(\xi) - C_{max}}{C_{max}} \end{equation}

---

## 3. 三层调度优化模型

### 3.1 车间层 (Shop Level)：全局任务分配
车间层负责将工序 $O_{i,j}$ 分配至最合适的服务单元 $U_u$，核心目标是**平均负载强度均衡**。

*   **决策变量**：$X_{i,j,u} \in \{0,1\}$，若工序分配至单元 $u$ 则为 1。
*   **输入参数**：加权聚合加工时间 $\tilde{P}_{i,j,u}$。
    \begin{equation} \tilde{P}_{i,j,u} = \frac{\sum_{k=1}^{m_u} \alpha_{u,k} \cdot \bar{p}_{i,j,u,k}}{\sum_{k=1}^{m_u} \alpha_{u,k}} \end{equation}
*   **优化目标**：最小化最大单元平均负载强度。
    \begin{equation} f_1(\xi) = \min \max_u \frac{\sum_{i,j} X_{i,j,u} \cdot \tilde{P}_{i,j,u}}{m_u} \end{equation}
*   **核心约束**：
    1. **唯一分配**：$\sum_{u=1}^m X_{i,j,u} = 1$。
    2. **不确定性预算控制**：$\sum_{u=1}^m \Gamma_u \leq \Gamma_s$。
    3. **迁移一致性**：$\delta_{i,j,u,u'} \geq X_{i,j,u'} - X_{i,j,u}$。

### 3.2 服务单元层 (Service Unit Level)：局部鲁棒排产
作为中间管理层，负责单元内部的机器选择与排序，目标是最小化单元内的**局部总完工时间**。

*   **决策变量**：$x_{i,j,u,k}$ (机器选择), $y_{i,j,u,k,i',j'}$ (加工顺序), $S_{i,j}$ (开始时间)。
*   **优化目标**：$f_u(\xi) = \min \max_{i,j} C_{i,j}(\xi)$。
*   **核心约束**：
    1. **层级一致性**：$\sum_{k=1}^{m_u} x_{i,j,u,k} = X_{i,j,u}$。
    2. **加工时长约束（考虑转运）**：$S_{i,j+1}(\xi) \geq C_{i,j}(\xi) + L_{u,u'}(\xi) \cdot \delta_{i,j,u,u'}$。
    3. **机器不重叠**：通过 disjunctive 约束确保同一时间一台机器只加工一道工序。

### 3.3 机器层 (Machine Level)：实时执行与扰动感知
作为底层执行级，存储细粒度数据并感知场景变量 $\xi$。

*   **不确定性模型**：实际加工时间 $p_{i,j,u,k}(\xi)$ 在标称值 $\bar{p}$ 与最大偏差 $\hat{p}$ 之间波动。
    \begin{equation} p_{i,j,u,k}(\xi) = \bar{p}_{i,j,u,k} + \xi_{i,j,u,k} \cdot \hat{p}_{i,j,u,k} \end{equation}
*   **扰动状态**：记录机器实时状态 $\Phi_{u,k}(\xi) \in \{0, 1\}$。

---

## 4. 层级耦合逻辑说明
1.  **自上而下 (Top-down)**：车间层根据各单元反馈的聚合能力 $\tilde{P}$，下达 $X_{i,j,u}$ 指令，为服务单元层划定任务空间。
2.  **自下而上 (Bottom-up)**：机器层感知的扰动通过不确定性预算 $\Gamma_u$ 向上传递。若单元层无法通过局部重排产收敛，则触发车间层的全局重分配。
3.  **负载与弹性的平衡**：车间层通过牺牲部分协调成本 $f_4$，利用负载均衡 $f_1$ 为底层预留“冗余能力缓冲”，从而实现系统整体弹性 $R$ 的最大化。

---
