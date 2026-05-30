import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Phase } from '../walkthrough/Walkthrough';
import { jumpToPhase } from '../Commentary';
import { useProgramState } from '../Sidebar';

/* ============================================================
 * MT-LNN 架构图 (nano-gpt 风格的左侧 chapter 面板)
 * - 纵向堆叠的彩色层矩形
 * - 左侧绿色"张量竖线" + 维度标签
 * - 右侧黑色数据流箭头, 残差用灰色虚线
 * - GWTB Core 有一条红色 h_{t-1} 循环箭头
 * - 鼠标悬停: 弹出层说明 + 数学公式
 * - 点击层: 跳到对应 walkthrough 章节
 * - 底部: 进度条 / 播放暂停 / 步进; 空格键 = 下一步
 * - 当前 walkthrough.phase 变化时, 对应层自动高亮
 * ============================================================ */

interface LayerDef {
    id: string;
    name: string;
    dim: string;
    color: string;
    group: string;
    desc: string;
    formula: string;
    phase: Phase;       // 该层映射到的 walkthrough 章节
}

const LAYERS: LayerDef[] = [
    { id:'input',   name:'Input Tokens',          dim:'[T, 3]',   color:'#fff7e0', group:'输入',
      desc:'输入 token 序列, 每个 token 用 3 维表示 (codebook id).',
      formula:'x ∈ ℝ^{T×3}',                                             phase: Phase.MTLNN_Tokens },
    { id:'embed',   name:'Embedding',             dim:'[T, 48]',  color:'#ffe4b5', group:'嵌入',
      desc:'词嵌入: 将离散 token 映射到 48 维连续空间; 同时叠加位置嵌入.',
      formula:'h⁽⁰⁾_t = E_tok[x_t] + E_pos[t]',                          phase: Phase.MTLNN_Tokens },

    { id:'b1-ln1',  name:'LayerNorm 1',           dim:'[T, 48]',  color:'#d9edff', group:'Block 1',
      desc:'Block1 第一个层归一化, 稳定注意力前激活分布.',
      formula:'y = (x − μ) / σ · γ + β',                                  phase: Phase.MTLNN_Architecture },
    { id:'b1-att',  name:'Self-Attention 1',      dim:'[T, 48]',  color:'#c1d9ff', group:'Block 1',
      desc:'多头自注意力, 在 token 序列上建立局部依赖.',
      formula:'A = softmax(QKᵀ / √d) · V',                                phase: Phase.MTLNN_Architecture },
    { id:'b1-res1', name:'Residual Connection',   dim:'[T, 48]',  color:'#eef3ff', group:'Block 1',
      desc:'残差相加: x ← x + Attn(LN(x)).',
      formula:'x ← x + Attn(LN(x))',                                      phase: Phase.MTLNN_Architecture },
    { id:'b1-ln2',  name:'LayerNorm 2',           dim:'[T, 48]',  color:'#d9edff', group:'Block 1',
      desc:'Block1 第二个层归一化, 位于 MT-DL 之前.',
      formula:'y = (x − μ) / σ · γ + β',                                  phase: Phase.MTLNN_Architecture },
    { id:'b1-mtdl', name:'MT-DL 1 · 13×5 微管动力学', dim:'[T,48]', color:'#c8f0c8', group:'Block 1',
      desc:'微管动力学层: 13 条原丝 × 5 个时间尺度的液态状态, 按指数衰减更新.',
      formula:'h_t = decay · h_{t−1} + (1 − decay) · A_t',                phase: Phase.MTLNN_Architecture },
    { id:'b1-res2', name:'Residual Connection',   dim:'[T, 48]',  color:'#eef3ff', group:'Block 1',
      desc:'残差相加: x ← x + MTDL(LN(x)).',
      formula:'x ← x + MTDL(LN(x))',                                      phase: Phase.MTLNN_Gates },

    { id:'b2-ln1',  name:'LayerNorm 1',           dim:'[T, 48]',  color:'#d9edff', group:'Block 2',
      desc:'Block2 第一个层归一化.',
      formula:'y = (x − μ) / σ · γ + β',                                  phase: Phase.MTLNN_Architecture },
    { id:'b2-att',  name:'Self-Attention 2',      dim:'[T, 48]',  color:'#c1d9ff', group:'Block 2',
      desc:'第二层多头自注意力.',
      formula:'A = softmax(QKᵀ / √d) · V',                                phase: Phase.MTLNN_Architecture },
    { id:'b2-res1', name:'Residual Connection',   dim:'[T, 48]',  color:'#eef3ff', group:'Block 2',
      desc:'残差相加.',
      formula:'x ← x + Attn(LN(x))',                                      phase: Phase.MTLNN_Architecture },
    { id:'b2-ln2',  name:'LayerNorm 2',           dim:'[T, 48]',  color:'#d9edff', group:'Block 2',
      desc:'Block2 第二个层归一化.',
      formula:'y = (x − μ) / σ · γ + β',                                  phase: Phase.MTLNN_Architecture },
    { id:'b2-mtdl', name:'MT-DL 2 · 13×5 微管动力学', dim:'[T,48]', color:'#c8f0c8', group:'Block 2',
      desc:'第二个微管动力学层, 进一步累积多时间尺度上下文.',
      formula:'h_t = decay · h_{t−1} + (1 − decay) · A_t',                phase: Phase.MTLNN_Architecture },
    { id:'b2-res2', name:'Residual Connection',   dim:'[T, 48]',  color:'#eef3ff', group:'Block 2',
      desc:'残差相加.',
      formula:'x ← x + MTDL(LN(x))',                                      phase: Phase.MTLNN_Gates },

    { id:'fln',     name:'Final LayerNorm',       dim:'[T, 48]',  color:'#d9edff', group:'输出归一',
      desc:'最终层归一化, 准备进入 GWTB 全局工作空间瓶颈.',
      formula:'y = (x − μ) / σ · γ + β',                                  phase: Phase.MTLNN_Gwtb },

    { id:'gwtb-e',  name:'GWTB Encode (48→10)',   dim:'48 → 10',  color:'#ffd6e0', group:'GWTB',
      desc:'全局工作空间编码: 将 48 维表征压缩到 10 维瓶颈, 作为稀疏注意输入.',
      formula:'z = W_enc · x,  W_enc ∈ ℝ^{10×48}',                       phase: Phase.MTLNN_Gwtb },
    { id:'gwtb-c',  name:'GWTB Core (O(1) h_prev)', dim:'[10]',   color:'#ff9fb6', group:'GWTB',
      desc:'GWTB 核心循环: 以常数代价维护 h_prev, 跨时间步传递全局上下文.',
      formula:'h_t = f(z_t, h_{t−1})    // 时间步循环',                   phase: Phase.MTLNN_Wm },
    { id:'gwtb-d',  name:'GWTB Decode (10→48)',   dim:'10 → 48',  color:'#ffd6e0', group:'GWTB',
      desc:'全局工作空间解码: 将 10 维瓶颈展开回 48 维表征, γ-gated 广播.',
      formula:'x̂ = γ · W_dec · h_t',                                     phase: Phase.MTLNN_Gwtb },

    { id:'coh',     name:'Global Coherence Layer', dim:'[T, 48]', color:'#e0d4ff', group:'相干',
      desc:'全局相干层: 融合 GWTB 输出与残差流, 保证 token 间全局一致性.',
      formula:'x ← LN(x + α · x̂)',                                       phase: Phase.MTLNN_Wm },

    { id:'lm',      name:'LM Head Logits',        dim:'[T, 3]',   color:'#ffe4b5', group:'输出',
      desc:'语言模型头: 投影回词表维度, 得到未归一化 logits (权重与 E_tok 绑定).',
      formula:'z = h⁽ᴸ⁾ · W_lm,  W_lm = E_tokᵀ',                         phase: Phase.MTLNN_Output },
    { id:'soft',    name:'Softmax Probs',         dim:'[T, 3]',   color:'#fff7e0', group:'输出',
      desc:'对 logits 应用 softmax, 得到每个 token 的预测概率分布.',
      formula:'p_t = softmax(z_t)',                                       phase: Phase.MTLNN_Output },
];

/* ===== 布局常量 ===== */
const CANVAS_W = 420;
const BOX_W    = 240;
const BOX_H    = 30;
const GAP      = 10;
const TOP      = 16;
const CX       = CANVAS_W / 2;

interface PositionedLayer extends LayerDef {
    x: number; y: number; w: number; h: number; cx: number; cy: number;
}
const nodes: PositionedLayer[] = LAYERS.map((L, i) => {
    const y = TOP + i * (BOX_H + GAP);
    return { ...L, x: CX - BOX_W/2, y, w: BOX_W, h: BOX_H, cx: CX, cy: y + BOX_H/2 };
});
const SVG_H = TOP + LAYERS.length * (BOX_H + GAP) + 16;

/* 哪些层属于残差跳连组 (start, end) */
const RESIDUALS: Array<[string, string]> = [
    ['embed',   'b1-res1'],
    ['b1-res1', 'b1-res2'],
    ['b1-res2', 'b2-res1'],
    ['b2-res1', 'b2-res2'],
];

export const MtLnnToc: React.FC<{ activePhase: Phase | null }> = ({ activePhase }) => {
    const progState = useProgramState();
    const { walkthrough } = progState;

    // 当前动画游标 (高亮哪一层)
    const [step, setStep] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [speed, setSpeed] = useState(700);   // ms per step
    const [hover, setHover] = useState<{ n: PositionedLayer; x: number; y: number } | null>(null);
    const timerRef = useRef<number | null>(null);
    const wrapRef = useRef<HTMLDivElement | null>(null);

    /* 当 walkthrough 章节切换时, 把游标对齐到第一个属于该章节的层 */
    useEffect(() => {
        if (activePhase == null) return;
        const idx = nodes.findIndex(n => n.phase === activePhase);
        if (idx >= 0) setStep(idx);
    }, [activePhase]);

    /* 自动播放 */
    useEffect(() => {
        if (!playing) {
            if (timerRef.current) { window.clearInterval(timerRef.current); timerRef.current = null; }
            return;
        }
        timerRef.current = window.setInterval(() => {
            setStep(s => {
                if (s >= nodes.length - 1) { setPlaying(false); return s; }
                return s + 1;
            });
        }, speed);
        return () => { if (timerRef.current) window.clearInterval(timerRef.current); };
    }, [playing, speed]);

    /* 空格 / 方向键 单步 */
    useEffect(() => {
        const el = wrapRef.current;
        if (!el) return;
        const onKey = (e: KeyboardEvent) => {
            if (e.code === 'Space')      { e.preventDefault(); setPlaying(false); setStep(s => Math.min(nodes.length-1, s+1)); }
            else if (e.code === 'ArrowRight') { setPlaying(false); setStep(s => Math.min(nodes.length-1, s+1)); }
            else if (e.code === 'ArrowLeft')  { setPlaying(false); setStep(s => Math.max(0, s-1)); }
        };
        el.addEventListener('keydown', onKey);
        return () => el.removeEventListener('keydown', onKey);
    }, []);

    const onLayerClick = (n: PositionedLayer) => {
        setStep(nodes.indexOf(n));
        jumpToPhase(walkthrough, n.phase);
    };

    const onLayerHover = (e: React.MouseEvent, n: PositionedLayer) => {
        const rect = wrapRef.current!.getBoundingClientRect();
        setHover({ n, x: e.clientX - rect.left + 12, y: e.clientY - rect.top + 12 });
    };

    /* GWTB Core 循环箭头几何 */
    const loopPath = useMemo(() => {
        const c = nodes.find(n => n.id === 'gwtb-c')!;
        const right = c.x + c.w;
        return `M ${right} ${c.cy - 6}
                Q ${right + 40} ${c.cy - 22}, ${right + 30} ${c.cy + 14}
                Q ${c.x - 24} ${c.cy + 40}, ${c.x - 28} ${c.cy + 4}
                L ${c.x} ${c.cy + 4}`;
    }, []);

    return (
        <div
            ref={wrapRef}
            tabIndex={0}
            style={{
                position: 'relative',
                padding: '8px 6px 4px',
                color: '#111',
                fontFamily: '-apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
                outline: 'none',
            }}
        >
            <div style={{ fontSize: 13, fontWeight: 600, textAlign: 'center', marginBottom: 4 }}>
                MT-LNN 架构 · 交互动画
            </div>
            <div style={{ fontSize: 10, color: '#777', textAlign: 'center', marginBottom: 6 }}>
                点击任意层跳转章节 · 悬停查看公式 · 空格键单步
            </div>

            <svg width={CANVAS_W} height={SVG_H} style={{ display: 'block', margin: '0 auto' }}>
                <defs>
                    <marker id="mtlnn-ah" viewBox="0 0 10 10" refX="9" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M0,0 L10,5 L0,10 z" fill="#111"/>
                    </marker>
                    <marker id="mtlnn-ah-red" viewBox="0 0 10 10" refX="9" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M0,0 L10,5 L0,10 z" fill="#c0392b"/>
                    </marker>
                </defs>

                {/* 主流箭头 (黑色, 层间) + 绿色张量线 */}
                {nodes.slice(0, -1).map((a, i) => {
                    const b = nodes[i+1];
                    const flowed = i < step;
                    return (
                        <g key={`flow-${i}`}>
                            <line
                                x1={a.x + a.w + 14} y1={a.y + a.h}
                                x2={a.x + a.w + 14} y2={b.y}
                                stroke="#111" strokeWidth={1} markerEnd="url(#mtlnn-ah)"
                            />
                            <line
                                x1={a.x - 14} y1={a.y + a.h}
                                x2={a.x - 14} y2={b.y}
                                stroke="#21a35a" strokeWidth={3} strokeLinecap="round"
                                opacity={flowed ? 1 : 0.12}
                                style={{ transition: 'opacity .25s' }}
                            />
                            <text
                                x={a.x - 18} y={(a.y + a.h + b.y)/2 + 3}
                                fontSize={8} fill="#2a6b2a" textAnchor="end"
                                fontFamily="Consolas, monospace"
                            >{b.dim}</text>
                        </g>
                    );
                })}

                {/* 输入层之上也画一条短张量线 */}
                <line
                    x1={nodes[0].x - 14} y1={4}
                    x2={nodes[0].x - 14} y2={nodes[0].y}
                    stroke="#21a35a" strokeWidth={3} strokeLinecap="round"
                    opacity={step >= 0 ? 1 : 0.12}
                />
                <text x={nodes[0].x - 18} y={14}
                      fontSize={8} fill="#2a6b2a" textAnchor="end"
                      fontFamily="Consolas, monospace">{nodes[0].dim}</text>

                {/* 残差跳连 (右侧灰虚线) */}
                {RESIDUALS.map(([fromId, toId], i) => {
                    const a = nodes.find(n => n.id === fromId)!;
                    const b = nodes.find(n => n.id === toId)!;
                    const x = a.x + a.w + 38 + (i % 2) * 6;
                    return (
                        <path
                            key={`res-${i}`}
                            d={`M ${a.x + a.w} ${a.cy} L ${x} ${a.cy} L ${x} ${b.cy} L ${b.x + b.w} ${b.cy}`}
                            stroke="#888" strokeWidth={1} fill="none"
                            strokeDasharray="3 2" markerEnd="url(#mtlnn-ah)"
                        />
                    );
                })}

                {/* GWTB Core 循环 (红色) */}
                <path d={loopPath} stroke="#c0392b" strokeWidth={1.4} fill="none"
                      markerEnd="url(#mtlnn-ah-red)" />
                <text x={(nodes.find(n=>n.id==='gwtb-c')!.x + BOX_W) + 46}
                      y={nodes.find(n=>n.id==='gwtb-c')!.cy - 10}
                      fontSize={9} fill="#c0392b" fontWeight={600}>h_{'{t-1}'}</text>

                {/* 层矩形 */}
                {nodes.map((n, i) => {
                    const active = i === step;
                    return (
                        <g key={n.id}
                           style={{ cursor: 'pointer' }}
                           onClick={() => onLayerClick(n)}
                           onMouseEnter={(e) => onLayerHover(e, n)}
                           onMouseMove={(e) => onLayerHover(e, n)}
                           onMouseLeave={() => setHover(null)}
                        >
                            <rect
                                x={n.x} y={n.y} width={n.w} height={n.h} rx={4}
                                fill={n.color}
                                stroke={active ? '#ff4e4e' : '#222'}
                                strokeWidth={active ? 2.2 : 1}
                                style={{
                                    transition: 'stroke .15s, stroke-width .15s',
                                    filter: active ? 'drop-shadow(0 0 4px rgba(255,80,80,.5))' : undefined,
                                }}
                            />
                            <text x={n.cx} y={n.cy - 1} textAnchor="middle"
                                  fontSize={10} fontWeight={600} fill="#111" pointerEvents="none">
                                {n.name}
                            </text>
                            <text x={n.cx} y={n.cy + 10} textAnchor="middle"
                                  fontSize={8} fill="#555" pointerEvents="none">
                                {n.group} · {n.dim}
                            </text>
                        </g>
                    );
                })}
            </svg>

            {/* 控制条 */}
            <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '6px 4px', fontSize: 11,
                borderTop: '1px solid #eee', marginTop: 4,
            }}>
                <button onClick={() => setPlaying(p => !p)}
                        style={btnStyle}>{playing ? '⏸' : '▶'}</button>
                <button onClick={() => { setPlaying(false); setStep(s => Math.max(0, s-1)); }}
                        style={btnStyle}>◀</button>
                <button onClick={() => { setPlaying(false); setStep(s => Math.min(nodes.length-1, s+1)); }}
                        style={btnStyle}>▶</button>
                <input
                    type="range" min={0} max={nodes.length - 1} value={step}
                    onChange={(e) => { setPlaying(false); setStep(parseInt(e.target.value, 10)); }}
                    style={{ flex: 1 }}
                />
                <span style={{ minWidth: 32, textAlign: 'right', color: '#555' }}>
                    {step + 1}/{nodes.length}
                </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '0 4px 6px', fontSize: 10, color: '#666' }}>
                速度
                <input type="range" min={200} max={1500} step={50} value={speed}
                       onChange={e => setSpeed(parseInt(e.target.value, 10))}
                       style={{ flex: 1 }} />
                <span style={{ minWidth: 70 }}>{nodes[step].name}</span>
            </div>

            {/* 悬浮提示 */}
            {hover && (
                <div style={{
                    position: 'absolute', left: hover.x, top: hover.y,
                    background: '#fff', border: '1px solid #888',
                    boxShadow: '0 4px 12px rgba(0,0,0,.12)',
                    padding: '6px 8px', borderRadius: 4,
                    fontSize: 11, lineHeight: 1.5, maxWidth: 280, zIndex: 10,
                    pointerEvents: 'none',
                }}>
                    <div style={{ fontWeight: 700, color: '#2b7cff', marginBottom: 2 }}>
                        {hover.n.name} <span style={{ color:'#888', fontWeight:400 }}>[{hover.n.group}]</span>
                    </div>
                    <div>维度: <b style={{ color:'#21a35a' }}>{hover.n.dim}</b></div>
                    <div style={{ margin: '2px 0' }}>{hover.n.desc}</div>
                    <div style={{
                        marginTop: 4, padding: '4px 6px',
                        background:'#f6f8fc', borderLeft:'3px solid #2b7cff',
                        fontFamily:'Consolas, monospace', fontSize: 10, color: '#224',
                    }}>{hover.n.formula}</div>
                </div>
            )}
        </div>
    );
};

const btnStyle: React.CSSProperties = {
    border: '1px solid #ccc',
    background: '#fff',
    padding: '2px 8px',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 11,
};
