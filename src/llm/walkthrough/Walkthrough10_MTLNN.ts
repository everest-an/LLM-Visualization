import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera, phaseTools, DimStyle } from "./WalkthroughTools";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IMTLNNLayout } from "../MtLnnModelLayout";
import { addSourceDestCurveLine } from "../Annotations";
import { IBlkDef } from "../GptModelLayout";

export function walkthroughMTLNN(args: IWalkthroughArgs) {
    let { state, walkthrough: wt } = args;
    let { breakAfter, c_str, c_blockRef } = phaseTools(args.state);
    let layout = args.layout as unknown as IMTLNNLayout;
    let render = state.render;

    if (render && wt.phase >= Phase.MTLNN_Intro) {
        let getBlk = (n: string) => layout.cubes.find(c => c.name === n);
        let connect = (src: IBlkDef | undefined, dest: IBlkDef | undefined, alpha = 0.6) => {
            if (!src || !dest) return;
            let cSrc = new Vec3(Math.floor(src.cx/2), Math.floor(src.cy/2), 0);
            let cDest = new Vec3(Math.floor(dest.cx/2), Math.floor(dest.cy/2), 0);
            let color = new Vec4(0.5, 0.5, 1.0, alpha);
            addSourceDestCurveLine(render, layout, src, dest, cSrc, cDest, color);
        }
        
        connect(getBlk('Tokens'), getBlk('Token Embed'));
        connect(getBlk('Tokens'), getBlk('Pos Embed'));
        connect(getBlk('Token Embed'), getBlk('Embed'));
        connect(getBlk('Pos Embed'), getBlk('Embed'));
        connect(getBlk('Embed'), getBlk('LN 1'));
        
        let targetLayers = 3;
        for (let l = 1; l <= targetLayers; l++) {
            let ln = getBlk(`LN ${l}`);
            let resid = getBlk(`Residual ${l}`);
            if (ln && resid) {
                // Find all proto nodes for this layer
                let layerProtos = layout.cubes.filter(c => c.name.startsWith('P') && c.y >= ln!.y && c.y <= resid!.y);
                for (let p of layerProtos) {
                    connect(ln, p, 0.2);
                    connect(p, resid, 0.2);
                }
            }
            if (l < targetLayers) connect(resid, getBlk(`LN ${l+1}`));
        }
        connect(getBlk(`Residual ${targetLayers}`), getBlk('GWTB'));
        connect(getBlk('GWTB'), getBlk('O(1) WM'));
        connect(getBlk('O(1) WM'), getBlk('LM Head'));
        connect(getBlk('O(1) WM'), getBlk('Logits'));
    }

    if (wt.phase === Phase.MTLNN_Intro) {
        setInitialCamera(state, new Vec3(-119753, 0, -540), new Vec3(235, 18, 20.0));

        commentary(wt, null, 0)`欢迎来到 _多时间尺度液态神经网络_ (Multi-Timescale Liquid Neural Network, MT-LNN) 的可视化解析。

MT-LNN 是一类受 _生物微管_ (Microtubules) 启发的连续时间神经架构，目标是用 [闭式 LTC ODE  +  全局工作区瓶颈  +  O(1) 工作记忆] 三件套，在大幅压缩参数和算力的前提下，逼近甚至超过 Transformer 在长上下文 / 可解释推理上的表现。`;
        breakAfter();

        commentary(wt, null, 0)`右侧 3D 视图自下而上完整展示了一个 demo 配置 (L=2 层，d_model=48，T=6，V=3) 的全部张量与权重几何：

  ·  _底部_  —  词元与位置嵌入  E_tok ∈ ℝ^(V×d) ， E_pos ∈ ℝ^(T×d)
  ·  _中段_  —  每层 MT-DL 微管核：13 根原纤维 × 5 个时间尺度的 ODE 阵列
  ·  _GWTB 瓶颈_  —  把 d_model 压到 d_model/8 上做稀疏注意力，再广播回来
  ·  _O(1) 工作记忆_  —  Orch-OR 启发的固定尺寸相干缓冲
  ·  _顶端_  —  LM Head  W_lm ∈ ℝ^(d×V)  与  Logits`;
        breakAfter();

        commentary(wt, null, 0)`生物学映射 (与代码中的命名一一对应)：

  ·  13 根原纤维 (protofilaments)    →   并行计算轨道   ( n_proto = 13 )
  ·  GTP 水解动力学                  →   闭式 LTC decay  ( τ ∈ [τ_min, ∞) )
  ·  α/β-tubulin 极性                →   Polarity attention  ( κ-gate )
  ·  MAP 蛋白调控                    →   per-proto MLP   ( MAP-Gate )
  ·  横向 B-晶格键                    →   Lateral coupling  ( RMC-SA )
  ·  Orch-OR 坍缩                     →   Top-K 稀疏注意 + 衰减 KV  ( k = 0.1·T )

下面我们将按数据从底向上的流向，逐层拆开看。`;
        breakAfter();

        commentary(wt, null, 0)`_参数尺寸感_：本 demo 仅 ~5.8×10⁴ 参数；在真实 MT-LNN-large 配置中 d_model = 832 = 13 × 64，L = 12，可达 ~2×10⁸ 参数。同等表征能力下相比标准 Transformer-LM 节省 ~30% FLOPs (参考 BENCHMARKS.md)。`;
    }

    if (wt.phase === Phase.MTLNN_Tokens) {
        setInitialCamera(state, new Vec3(-119753, 0, -60), new Vec3(235, 28, 4.0));
        let targetCubes = layout.cubes.filter(c => c.name.includes("Embed") || c.name.includes("Tokens"));
        if (targetCubes.length > 0) wt.dimHighlightBlocks = targetCubes;

        let tokenStr = c_str('Token IDs', 0, DimStyle.Token);
        let embedRef = c_blockRef('词嵌入 + 位置嵌入', layout.cubes.filter(c => c.name.includes("Embed")), DimStyle.Token);

        commentary(wt, null, 0)`输入是长度 T 的整数序列 ${tokenStr}  x ∈ {0, …, V-1}^T。

第一步：${embedRef}

    h⁽⁰⁾_t  =  E_tok[ x_t ]  +  E_pos[ t ]              h⁽⁰⁾ ∈ ℝ^(T×d)

其中  E_tok ∈ ℝ^(V×d)、E_pos ∈ ℝ^(T×d)  都是可学习矩阵。在视图中 E_tok 就是底部那块横向的蓝色 weight cube，E_pos 是与之并排的较小蓝块。`;
        breakAfter();

        commentary(wt, null, 0)`_为什么需要显式 PosEmbed？_

MT-DL 是 position-free 的递推核 (仅靠 τ 与隐状态时序传递信息)，所以位置先验必须通过 E_pos 注入。这与 RoPE / ALiBi 不同——后者直接把位置编码进 attention 的 dot-product，而 MT-LNN 的注意力只发生在 GWTB 瓶颈这一层。`;
        breakAfter();

        commentary(wt, null, 0)`_形状对账_：

  ·  Token IDs          shape = [T]            输入 token id
  ·  E_tok              shape = [V × d]        词表查找
  ·  E_pos              shape = [T × d]        时序先验
  ·  h⁽⁰⁾               shape = [T × d]        进入 L1 的初始残差流

视图中 h⁽⁰⁾ 是上方那条贯穿全模型的绿色 residual spine，它会被每一层的 MT-DL + GWTB 反复读出和写入。`;
    }

    if (wt.phase === Phase.MTLNN_Architecture) {
        setInitialCamera(state, new Vec3(-119753, 0, -260), new Vec3(235, 28, 6.5));

        let mtdlCubes = layout.cubes.filter(c => /^P\d+/.test(c.name) || c.name.includes("LN"));
        let mtdlRef   = c_blockRef('微管动态层 (Microtubule Dynamic Layer)', mtdlCubes);
        if (mtdlCubes.length > 0) wt.dimHighlightBlocks = mtdlCubes;

        commentary(wt, null, 0)`架构的视觉中心与计算核心是  ${mtdlRef}。

每一层 MT-DL 把当前残差流  h⁽ˡ⁻¹⁾ ∈ ℝ^(T×d)  重塑为 13 根 原纤维 (protofilament) 上的子向量：

    x⁽ᵖ⁾  =  h⁽ˡ⁻¹⁾ · W_in⁽ᵖ⁾  ∈ ℝ^(T × d_proto)        p = 1, …, 13

视图中你能看到每层有 13 列紫色立方体横向排开——那就是 13 根 protofilament 的输入投影 W_in。`;
        breakAfter();

        commentary(wt, null, 0)`_闭式 LTC_ (Closed-form Liquid Time-Constant) —— MT-DL 的灵魂方程。

对每根 protofilament  p、每个时间尺度  s ∈ {1, …, 5}：

    τ_(p,s)    =  softplus( log τ_(p,s) )  +  τ_min
    α_(p,s)    =  exp( -Δt / τ_(p,s) )
    h⁽ᵖ,ˢ⁾_t   =  α_(p,s) · h⁽ᵖ,ˢ⁾_(t-1)  +  (1 - α_(p,s)) · σ( W_in⁽ᵖ,ˢ⁾ · x⁽ᵖ⁾_t  +  b )

这是一阶 ODE   ḣ = -h/τ + f(x)   在欧拉离散下的闭式解，无需任何数值积分器。`;
        breakAfter();

        commentary(wt, null, 0)`_多时间尺度的几何_：右侧每根 protofilament 列里上下堆叠的 5 个青绿色小块就是  τ_1 < τ_2 < … < τ_5  五条独立的 ODE 隐状态。

  ·  τ_1 ≈ 1     —  快通道  (高频细节、语法结构)
  ·  τ_3 ≈ 10    —  中通道  (短语级语义)
  ·  τ_5 ≈ 100   —  慢通道  (篇章级主题)

不同 τ 让同一根 protofilament 同时具备多个"记忆衰减率"，模仿微管内 GTP 水解的多时间尺度动力学。`;
        breakAfter();

        commentary(wt, null, 0)`_Parallel Scan 优化_：朴素 RNN 是 O(T) 串行，但闭式 LTC 形如  h_t = α · h_(t-1) + u_t ，可用 prefix-scan 在 O(log T) 深度内并行 (实现见 mt_lnn/parallel_scan.py)。

实测在 A100 上：T = 8192 时朴素循环 ≈ 1.2 s，parallel scan ≈ 38 ms —— 30× 加速。`;
    }

    if (wt.phase === Phase.MTLNN_Gates) {
        setInitialCamera(state, new Vec3(-119753, 0, -300), new Vec3(225, 30, 4.5));
        let gateCubes = layout.cubes.filter(c =>
            c.name.includes("Kappa") || c.name.includes("MAP") || c.name.includes("ODE"));
        if (gateCubes.length > 0) wt.dimHighlightBlocks = gateCubes;

        commentary(wt, null, 0)`_门控_ (Gating) 子模块 —— 决定 MT-DL 每一步要"听"哪些时间尺度，要保留哪些通道。`;
        breakAfter();

        commentary(wt, null, 0)`_Kappa 门_  (κ-gate) —— 类比 α/β-tubulin 极性。

    κ_(t,p,s)  =  σ( W_κ⁽ᵖ⁾ · x⁽ᵖ⁾_t  +  b_κ )   ∈ (0, 1)
    w_(p,s)    =  softmax_s( β_(p,s) )  ⊙  κ_(t,p,s)

β_(p,s) 是 per-proto 可学习的 blend weight；与动态 κ 相乘后，每个 token 都能在 5 个时间尺度间自适应地分配权重。

跨尺度融合：

    h⁽ᵖ⁾_t  =  Σ_{s=1..5}  w_(p,s) · h⁽ᵖ,ˢ⁾_t`;
        breakAfter();

        commentary(wt, null, 0)`_MAP 门_  (MAP-Gate) —— 类比 MAP 蛋白对微管的稳定调控。

它是 per-protofilament 的轻量 MLP (2 层，隐藏维 d_proto)：

    g⁽ᵖ⁾_t      =  MLP_MAP⁽ᵖ⁾ ( h⁽ᵖ⁾_t )
    h̃⁽ᵖ⁾_t      =  h⁽ᵖ⁾_t  ⊙  σ( g⁽ᵖ⁾_t )

把 h̃⁽ᵖ⁾ 重新拼回 d_model 维并加回残差流：

    h⁽ˡ⁾  =  h⁽ˡ⁻¹⁾  +  concat_p( h̃⁽ᵖ⁾ · W_out⁽ᵖ⁾ )`;
        breakAfter();

        commentary(wt, null, 0)`_麻醉验证_ (Anesthesia Probe) —— κ-gate 让 MT-LNN 拥有标准 Transformer 没有的可调"意识参数"。

外部干预  κ ← (1 - a) · κ   (a ∈ [0,1] 是麻醉强度)，观察整合信息量 Φ̂ 的塌缩：

  ·  Transformer       ΔΦ̂  =  0.000     (无响应)
  ·  Vanilla LNN       ΔΦ̂  =  0.000     (无响应)
  ·  MT-LNN            ΔΦ̂  =  +7.578    ← 显著响应

只有 MT-LNN 对"麻醉"产生显著响应 (详见 demo_awareliquid_v2.py)。`;
    }

    if (wt.phase === Phase.MTLNN_Gwtb) {
        setInitialCamera(state, new Vec3(-119753, 0, -460), new Vec3(225, 28, 5));
        let bottleneckCubes = layout.cubes.filter(c => c.name.includes("GWTB"));
        if (bottleneckCubes.length > 0) wt.dimHighlightBlocks = bottleneckCubes;

        let gwtbRef = c_blockRef('全局工作区瓶颈 (Global Workspace Theory Bottleneck)', bottleneckCubes);

        commentary(wt, null, 0)`MT-DL 输出之后，每一层还要过一道 ${gwtbRef}。

灵感来自 Baars 的 _全局工作区理论_ (Global Workspace Theory)：意识 = 在一个狭窄的全脑共享缓冲区里，让多个专家模块竞争广播权。`;
        breakAfter();

        commentary(wt, null, 0)`_三步走_：

[1] _Compress_  —  把 d_model 压到 d_gw = d_model / 8

        z  =  h⁽ˡ⁾ · W_c           W_c ∈ ℝ^(d × d/8)

[2] _Workspace Sparse Attention_  —  在瓶颈空间里做 4-head 稀疏自注意

        A  =  softmax( Q·K^T / √d_h   ⊙   M_topk ) · V

[3] _Broadcast_  —  用 γ-gated 投影把信息广播回 d_model

        h⁽ˡ⁾  ←  h⁽ˡ⁾  +  γ · A · W_b           γ_init = 0.01 (learnable)`;
        breakAfter();

        commentary(wt, null, 0)`_为什么是 8× 压缩？_

  ·  _参数_：QKV 投影从  O(d²)  降到  O(d²/64)  ——  省 98% attention 参数
  ·  _算力_：注意力矩阵从  T²·d  降到  T²·(d/8)  ——  FLOPs 减 8×
  ·  _表达力_：消融实验表明 d/8 是 PPL 的甜点 (更深则欠拟合，更宽则边际收益消失，见 benchmarks/sparse_resonance_ablation.md)`;
        breakAfter();

        commentary(wt, null, 0)`_γ 慢启动_：初始 γ = 0.01 让 GWTB 在训练前期几乎"不发声"，模型先把 MT-DL 学好，再逐步让 GWTB 接管全局协调——避免了瓶颈层在随机初始化时压坏底层信号。

在 demo 视图中你能看到 GWTB Compress 是一块明显比 MT-DL 窄的紫色立方体 (d/8 = 6 通道)，紧跟在 LayerNorm 之后。`;
    }

    if (wt.phase === Phase.MTLNN_Wm) {
        setInitialCamera(state, new Vec3(-119753, 0, -560), new Vec3(235, 25, 5));
        let wmCubes = layout.cubes.filter(c =>
            c.name.includes("Coherence") || c.name.includes("h_prev"));
        if (wmCubes.length === 0) wmCubes = [layout.cubes[0]]; // safety
        if (wmCubes.length > 0) wt.dimHighlightBlocks = wmCubes;

        let wmRef = c_blockRef('O(1) 工作记忆 (h_prev + Global Coherence/Orch-OR)', wmCubes, DimStyle.Token);

        commentary(wt, null, 0)`${wmRef} —— MT-LNN 最反直觉的一块。

传统 Transformer KV-cache 随 T 线性增长 (T = 128k 时一个 7B 模型要 ~30 GB)，MT-LNN 维护一个 _固定尺寸_ 的 K / V 缓冲，无论上下文多长，内存恒定。

                       Transformer        MT-LNN
  ·  KV-cache 大小         O(T)            O(1)  ←
  ·  注意力密度            100%            10%  (Top-K)
  ·  长文本内存           线性增长         指数衰减保留`;
        breakAfter();

        commentary(wt, null, 0)`_Top-K 稀疏注意力_ —— 只保留每个 query 最强的  k = 0.1·T  个 key：

    mask_(ij)  =  1[ s_(ij) ∈ TopK_j( s_(:,j) ) ]
    A_(ij)     =  softmax_j( s_(ij)  |  mask_(ij) )

剩下 90% 的 attention pair 直接跳过——既节省算力又强制网络挑出最相关的少数 token。`;
        breakAfter();

        commentary(wt, null, 0)`_指数衰减 KV_ —— 旧 token 自然褪色，类似生物短时记忆：

    K̃_t  =  ρ^t · K_t        Ṽ_t  =  ρ^t · V_t        ρ  =  σ(θ_ρ)  ≈ 0.99

距离当前  Δt = 100  的 token 权重已衰减到  0.99¹⁰⁰ ≈ 0.37 ；
Δt = 500 时   ≈ 0.007   ——  实质上把"无限上下文"压到一个有效窗口里。`;
        breakAfter();

        commentary(wt, null, 0)`_Collapse Gate_ (Orch-OR 启发) —— Penrose-Hameroff 的"客观坍缩"被简化为一个 sigmoid：

    c_t  =  σ( W_c · h_t )
    y_t  =  c_t · Attn_t  +  (1 - c_t) · h_t

c_t → 1  =  此刻"意识介入"，全局信息覆写本地；
c_t → 0  =  透明传递，残差走捷径。

视图中 WM 块是淡蓝色的——左侧 bracket 标记 O(1) Working Memory (Orch-OR)。`;
    }

    if (wt.phase === Phase.MTLNN_Output) {
        setInitialCamera(state, new Vec3(-119753, 0, -1080), new Vec3(235, 20, 4.5));
        let outCubes = layout.cubes.filter(c => c.name.includes("LM Head") || c.name.includes("Logits"));
        if (outCubes.length > 0) wt.dimHighlightBlocks = outCubes;

        let headRef = c_blockRef('LM Head + Logits', outCubes, DimStyle.TokenIdx);

        commentary(wt, null, 0)`经过 L 层 MT-DL + GWTB + WM 之后，最终残差 h⁽ᴸ⁾ ∈ ℝ^(T×d) 通过 ${headRef} 投影回词表：

    z  =  h⁽ᴸ⁾ · W_lm                    W_lm ∈ ℝ^(d×V)

    p( x_(t+1) = v  |  x_(1:t) )  =   exp( z_(t,v) )  /  Σ_v' exp( z_(t,v') )`;
        breakAfter();

        commentary(wt, null, 0)`_权重绑定_ (weight tying)：实现里  W_lm = E_tok^T  —— LM head 复用 token embedding 矩阵，节省  V × d  个参数 (在 V = 50257, d = 832 时省 ~4.2×10⁷ 参数)。视图中 LM Head 与底部的 Token Embed 是同色的蓝色立方体。`;
        breakAfter();

        commentary(wt, null, 0)`_Deliberation Router_ (推理时) —— 训练之外，MT-LNN 在推理时还有一个外挂的"慢思考"路由器：

  [1] 计算每个 token 的 logit 熵   H_t  =  - Σ_v  p_v · log p_v
  [2] 若  H_t < θ_1   →   LOCAL 直出
  [3] 若  H_t ≥ θ_1   →   采样 N 个 trace，做语义熵检查
  [4] 若发散          →   REVISE 自我修正
  [5] 若仍发散        →   Cloud Oracle 检索注入

这是 AWARELIQUID_SYSTEM_MVP.md 里 AwareLiquid 系统的核心环路——MT-LNN 提供 快通道，Oracle 提供 慢通道。`;
        breakAfter();

        commentary(wt, null, 0)`_到这里 MT-LNN 的全部 7 章就走完了。_

完整流向回顾：

    x  →[E_tok+E_pos]→  h⁽⁰⁾
        →[MT-DL₁]→  →[GWTB₁]→  →[WM]→
        →     ⋯     →
        →[MT-DLₗ]→  →[GWTBₗ]→  →[WM]→  h⁽ᴸ⁾
        →[W_lm]→  z  →[softmax]→  p

可继续切到 nano-gpt 或 GPT-2 small tab 对比传统 Transformer 的几何形态——你会注意到 MT-LNN 没有典型 Transformer 的  T × T  attention 大矩阵，所有"全局通讯"都被压在 GWTB 那一小块瓶颈里。`;
    }

}
