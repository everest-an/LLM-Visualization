import os

content = '''import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera, phaseTools, DimStyle } from "./WalkthroughTools";
import { Vec3 } from "@/src/utils/vector";

export function walkthroughMTLNN(args: IWalkthroughArgs) {
    let { state, walkthrough: wt } = args;
    let { breakAfter, c_str } = phaseTools(args.state);

    if (wt.phase === Phase.MTLNN_Intro) {
        setInitialCamera(state, new Vec3(-120000, 800, -1200), new Vec3(270, 15, 3000));
        commentary(wt, null, 0)\欢迎来到 **多时间尺度液态神经网络 (MT-LNN)** 架构可视化解析。
        
受生物学微管（Microtubules）机理的启发，MT-LNN 通过连续时间微分方程（ODEs）与稀疏记忆瓶颈动态处理序列数据。\
        breakAfter();

        commentary(wt, null, 0)\在此视图中，我们将自底向上完整拆解 MT-LNN 的核心组件。
        
它的架构被设计为极具生物学特性的拓扑结构，主要表现为：
- 🔬 **13根原纤维 (Protofilaments)**: 每一层都被分为13个并行的多维计算轨道。
- ⏳ **5种时间尺度 (Time Scales)**: 在液态状态演化过程中，时间尺度 \\\$\\tau\\\$ 通过闭式 LTC 动态更新。
- ⚡ **GWTB 记忆瓶颈**: 所有的信息都要经过一个狭窄的 1/8 维度工作区压缩，逼迫网络产生强烈的特征提纯。\
        breakAfter();

        commentary(wt, null, 0)\现在，模型已全尺寸展开在您眼前！右侧为 MT-LNN 的全局参数展板，接下来我们将一步步走进它的内部流转机制。\;
    }

    if (wt.phase === Phase.MTLNN_Tokens) {
        setInitialCamera(state, new Vec3(-120000, 0, -1200), new Vec3(270, 20, 1200));
        let tokenStr = c_str('Token', 0, DimStyle.Token);
        commentary(wt, null, 0)\最底层负责读取输入的 \，并生成带有位置编码的高维几何向量（**词嵌入层 / Embeddings**）。\
        breakAfter();

        commentary(wt, null, 0)\相较于传统依赖 \\\^2\\\$ 复杂度的全局注意力机制，MT-LNN 能以极高的效率处理具有长上下文的序列。

输入数据在这里首先完成序列边界的基础对齐，为接下来步入 **液态时序处理管道 (Liquid Node)** 做好一切特征准备。\;
    }

    if (wt.phase === Phase.MTLNN_Architecture) {
        setInitialCamera(state, new Vec3(-120000, 1500, -1200), new Vec3(350, 25, 2000));
        commentary(wt, null, 0)\架构的视觉中心与计算核心是：**微管动态层 (Microtubule Dynamic Layer, MT-DL)**。
        
这里的数组被沿径向展开为 **13条原纤维 (Protofilaments)**。每一条原纤维都独立维系着 **5种离散的时间尺度**。系统通过 **ODE 权重 (\\\$ \\tau \\\$ 和 decay)** 让隐状态能够跨时序地连续演化。\;
        
        breakAfter();
        commentary(wt, null, 0)\原纤维之间并非孤立，而是通过 **侧向耦合连接 (Lateral Coupling)** 相互通讯。

这允许全模型建立起高度并行的隐含结构，极大地拓展了时间接受野，让 MT-LNN 在没有全序列 Attention 的情况下也能拥有甚至超越 Transformer 的长序列记忆力。\;
    }

    if (wt.phase === Phase.MTLNN_Gates) {
        setInitialCamera(state, new Vec3(-120000, 2000, -800), new Vec3(300, 20, 1200));
        commentary(wt, null, 0)\进一步放大 MT-DL 的原纤维微观结构，我们可以看到其内部复杂的门控处理机制。
        
**Kappa 门 (\\\$ \\kappa \\\$)** 利用 Sigmoid 阈值动态决定当前时刻哪些特征通道需要被长期保留；随后的 **MAP 门**（类似微型 MLP）则负责处理非线性特征平移。\;
        breakAfter();

        commentary(wt, null, 0)\这种纯局部、稀疏并基于连续时间的门控，彻底平替了标准多头自注意力层 (Multi-Head Self-Attention)。

而且它在推理时只占用 \\\(N)\\\$ 的递推时间，属于真正的 **Position-Free** 递归架构。\;
    }

    if (wt.phase === Phase.MTLNN_Gwtb) {
        setInitialCamera(state, new Vec3(-120000, 3100, -1200), new Vec3(270, 15, 1200));
        commentary(wt, null, 0)\架构继续向上，液态计算的中间结果会汇总交由 **全局工作区任务瓶颈 (Global Workspace Task Bottleneck, GWTB)**。
        
这一层基于巴尔斯（Baars）的全局工作区理论构建：数据会被施加极其激进的 **特征压缩**（将其缩减到了 \\\{model}/8\\\$ 的隐层）。\
        breakAfter();

        commentary(wt, null, 0)\注意力机制被强迫仅在这狭窄且密集的知识通道中进行演算 (Task Bottleneck)。处理完毕后再将高价值的决策 **广播 (Broadcast)** 恢复到全量维度，实现参数上的极大优化。\;
    }

    if (wt.phase === Phase.MTLNN_Wm) {
        setInitialCamera(state, new Vec3(-120000, 3400, -1200), new Vec3(270, 10, 1100));
        commentary(wt, null, 0)\在当前层即将结束推演之时，MT-LNN 引入了全局致密相干层——附带 **O(1) 工作记忆缓冲区 (Working Memory, WM)**。
        
通过使用 Top-K 极值稀疏化（通常仅保留最核心的 10% 激活信号），模型故意剥离次要特征。\;
        breakAfter();

        commentary(wt, null, 0)\这就完成了一次类似生物神经学意义上的 "Orch-OR 坍缩"，确保只有至关重要、凝练一致的上下文信息才会被注入进长期有效的工作记忆区中保存。这样不仅节省带宽，更从物理上保证了抗噪能力。\;
    }

    if (wt.phase === Phase.MTLNN_Output) {
        setInitialCamera(state, new Vec3(-120000, 3800, -1200), new Vec3(270, 5, 1400));
        commentary(wt, null, 0)\最终，所有经过时间尺度与原纤维液态混合后的统合表征，会通过最顶部的 **Language Model Head** 权重矩阵，投影回自然语言的词汇表空间中。\;
        breakAfter();

        commentary(wt, null, 0)\视图最高处生成的 Logits 矩阵元将严格决定模型生成的极大概率分布，指导 MT-LNN 输出序列末尾的下一个预测词。\;
    }

}
'''

with open(r'C:\Users\admin\AppData\Local\Temp\llm-viz\src\llm\walkthrough\Walkthrough10_MTLNN.ts', 'w', encoding='utf-8') as f:
    f.write(content)
