import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Phase } from '../walkthrough/Walkthrough';
import { jumpToPhase } from '../Commentary';
import { useProgramState } from '../Sidebar';
import { mtLnnSpreadState, SPREAD_COLLAPSED, SPREAD_EXPANDED } from '../MtLnnModel';

/* ============================================================================
 * MT-LNN 架构图  (nano-gpt 教程风格 · 左侧 chapter 面板)
 *
 * 设计要点:
 *   - 纯白底, 1px 浅灰边框圆角矩形, 14px 黑色无衬线居中
 *   - 配色严格对齐 nano-gpt (LN/Attn/MLP 等)
 *   - 左侧绿色"张量主干" spine (#28a745, 3px), 维度标签放主干左侧, 格式 [T, D]
 *   - 残差不再用单独矩形, 改为右侧灰色虚线箭头从子块入口跨到出口
 *   - 每层有 1px Z 偏移阴影, 体现轻微深度感
 *   - MT-DL 内部: 13 原丝沿 180° 半弧 (P1-P13) + 底部 5 个 τ 立方体 (τ1-τ5)
 *   - 交互: 播放/暂停/上一步/下一步/进度条/速度 + 空格单步 + 悬停公式 + 点击跳转章节
 *   - 当 walkthrough.phase 切换时, 对应层自动高亮 (双向联动)
 * ============================================================================ */

interface LayerDef {
    id: string;
    name: string;
    dim: string;
    color: string;
    group: string;
    desc: string;
    formula: string;
    phase: Phase;
    kind?: 'mtdl';
}

/* nano-gpt 配色 */
const C = {
    input:  '#fff3cd',
    embed:  '#ffeeba',
    ln:     '#d1ecf1',
    attn:   '#bee5eb',
    mtdl:   '#d4edda',
    gwtb:   '#f8d7da',
    coh:    '#e2d9f3',
    lm:     '#ffeeba',
    soft:   '#fef9e7',
};

const LAYERS: LayerDef[] = [
    { id:'input',   name:'Input Tokens',           dim:'[T, 3]',  color:C.input, group:'输入',
      desc:'输入 token 序列, 每个 token 用 3 维表示 (codebook id).',
      formula:'x ∈ ℝ^{T×3}',                                            phase: Phase.MTLNN_Tokens },
    { id:'embed',   name:'Embedding',              dim:'[T, 48]', color:C.embed, group:'嵌入',
      desc:'词嵌入 + 位置嵌入: 离散 token 映射到 48 维连续空间.',
      formula:'h⁽⁰⁾_t = E_tok[x_t] + E_pos[t]',                         phase: Phase.MTLNN_Tokens },

    { id:'b1-ln1',  name:'LayerNorm 1',            dim:'[T, 48]', color:C.ln,    group:'Block 1',
      desc:'Block1 第一个层归一化, 稳定注意力前激活分布.',
      formula:'y = (x − μ) / σ · γ + β',                                 phase: Phase.MTLNN_Architecture },
    { id:'b1-att',  name:'Self-Attention 1',       dim:'[T, 48]', color:C.attn,  group:'Block 1',
      desc:'多头自注意力: 在 token 序列上建立局部依赖关系.',
      formula:'A = softmax(QKᵀ / √d) · V',                               phase: Phase.MTLNN_Architecture },
    { id:'b1-ln2',  name:'LayerNorm 2',            dim:'[T, 48]', color:C.ln,    group:'Block 1',
      desc:'Block1 第二个层归一化, 位于 MT-DL 之前.',
      formula:'y = (x − μ) / σ · γ + β',                                 phase: Phase.MTLNN_Architecture },
    { id:'b1-mtdl', name:'MT-DL 1',                dim:'[T, 48]', color:C.mtdl,  group:'Block 1',
      desc:'微管动力学层: 13 条原丝 (P1–P13) × 5 个时间尺度 (τ1–τ5) 的液态状态, 闭式 LTC 指数衰减更新.',
      formula:'h_t = decay · h_{t−1} + (1 − decay) · A_t',               phase: Phase.MTLNN_Architecture, kind:'mtdl' },

    { id:'b2-ln1',  name:'LayerNorm 1',            dim:'[T, 48]', color:C.ln,    group:'Block 2',
      desc:'Block2 第一个层归一化.',
      formula:'y = (x − μ) / σ · γ + β',                                 phase: Phase.MTLNN_Architecture },
    { id:'b2-att',  name:'Self-Attention 2',       dim:'[T, 48]', color:C.attn,  group:'Block 2',
      desc:'第二层多头自注意力.',
      formula:'A = softmax(QKᵀ / √d) · V',                               phase: Phase.MTLNN_Architecture },
    { id:'b2-ln2',  name:'LayerNorm 2',            dim:'[T, 48]', color:C.ln,    group:'Block 2',
      desc:'Block2 第二个层归一化.',
      formula:'y = (x − μ) / σ · γ + β',                                 phase: Phase.MTLNN_Architecture },
    { id:'b2-mtdl', name:'MT-DL 2',                dim:'[T, 48]', color:C.mtdl,  group:'Block 2',
      desc:'第二个微管动力学层, 进一步累积多时间尺度上下文.',
      formula:'h_t = decay · h_{t−1} + (1 − decay) · A_t',               phase: Phase.MTLNN_Gates, kind:'mtdl' },

    { id:'fln',     name:'Final LayerNorm',        dim:'[T, 48]', color:C.ln,    group:'输出归一',
      desc:'最终层归一化, 准备进入 GWTB 全局工作空间瓶颈.',
      formula:'y = (x − μ) / σ · γ + β',                                 phase: Phase.MTLNN_Gwtb },

    { id:'gwtb-e',  name:'GWTB Encode',            dim:'[T, 10]', color:C.gwtb,  group:'GWTB',
      desc:'全局工作空间编码: 将 48 维表征压缩到 10 维瓶颈.',
      formula:'z = W_enc · x,  W_enc ∈ ℝ^{10×48}',                      phase: Phase.MTLNN_Gwtb },
    { id:'gwtb-c',  name:'GWTB Core (O(1) h_prev)', dim:'[10]',   color:C.gwtb,  group:'GWTB',
      desc:'GWTB 核心循环: 以常数代价维护 h_prev, 跨时间步传递全局上下文.',
      formula:'h_t = f(z_t, h_{t−1})    // 时间步循环',                  phase: Phase.MTLNN_Wm },
    { id:'gwtb-d',  name:'GWTB Decode',            dim:'[T, 48]', color:C.gwtb,  group:'GWTB',
      desc:'全局工作空间解码: 10 维瓶颈展开回 48 维, γ-gated 广播.',
      formula:'x̂ = γ · W_dec · h_t',                                    phase: Phase.MTLNN_Gwtb },

    { id:'coh',     name:'Global Coherence',       dim:'[T, 48]', color:C.coh,   group:'相干',
      desc:'全局相干层: 融合 GWTB 输出与残差流, 保证 token 间全局一致性.',
      formula:'x ← LN(x + α · x̂)',                                      phase: Phase.MTLNN_Wm },

    { id:'lm',      name:'LM Head',                dim:'[T, 3]',  color:C.lm,    group:'输出',
      desc:'语言模型头: 投影回词表维度, 得到未归一化 logits (权重与 E_tok 绑定).',
      formula:'z = h⁽ᴸ⁾ · W_lm,  W_lm = E_tokᵀ',                        phase: Phase.MTLNN_Output },
    { id:'soft',    name:'Softmax Probs',          dim:'[T, 3]',  color:C.soft,  group:'输出',
      desc:'对 logits 应用 softmax, 得到每个 token 的预测概率分布.',
      formula:'p_t = softmax(z_t)',                                      phase: Phase.MTLNN_Output },
];

/* 残差跳连 (灰色虚线): 每对 [from, to] 绕过中间子块 */
const RESIDUALS: Array<[string, string]> = [
    ['b1-ln1',  'b1-att'],
    ['b1-ln2',  'b1-mtdl'],
    ['b2-ln1',  'b2-att'],
    ['b2-ln2',  'b2-mtdl'],
];

/* 布局常量 */
const CANVAS_W = 460;
const BOX_W    = 240;
const BOX_H    = 32;
const MTDL_H   = 130;
const GAP      = 16;
const TOP      = 16;
const SPINE_X  = 92;
const BOX_X    = 130;
const Z_DEPTH  = 1;

interface PositionedLayer extends LayerDef {
    x: number; y: number; w: number; h: number; cx: number; cy: number;
}
function buildNodes(): PositionedLayer[] {
    let y = TOP;
    const out: PositionedLayer[] = [];
    for (const L of LAYERS) {
        const h = L.kind === 'mtdl' ? MTDL_H : BOX_H;
        out.push({
            ...L, x: BOX_X, y, w: BOX_W, h,
            cx: BOX_X + BOX_W / 2, cy: y + h / 2,
        });
        y += h + GAP;
    }
    return out;
}
const nodes: PositionedLayer[] = buildNodes();
const SVG_H = nodes[nodes.length - 1].y + nodes[nodes.length - 1].h + 20;

/* MT-DL 内部子图形: 13 原丝 (半弧) + 5 τ (底部一行) */
interface MtdlGeom {
    arcCX: number; arcCY: number; R: number;
    protos: Array<{ id: string; x: number; y: number; cx: number; cy: number }>;
    taus:   Array<{ id: string; x: number; y: number; w: number; h: number; cx: number }>;
}
function mtdlGeom(n: PositionedLayer): MtdlGeom {
    const arcCX = n.cx;
    const arcCY = n.y + 64;
    const R = 70;
    const protos = Array.from({ length: 13 }, (_, i) => {
        const ang = Math.PI * (1 - i / 12);
        const cx = arcCX + R * Math.cos(ang);
        const cy = arcCY - R * Math.sin(ang);
        const w = 10;
        return { id: `P${i+1}`, x: cx - w/2, y: cy - w/2, cx, cy };
    });
    const tauW = 16, tauGap = 6;
    const total = 5 * tauW + 4 * tauGap;
    const startX = n.cx - total / 2;
    const taus = Array.from({ length: 5 }, (_, i) => ({
        id: `τ${i+1}`,
        x: startX + i * (tauW + tauGap),
        y: n.y + n.h - 24,
        w: tauW, h: 14,
        cx: startX + i * (tauW + tauGap) + tauW / 2,
    }));
    return { arcCX, arcCY, R, protos, taus };
}

export const MtLnnToc: React.FC<{ activePhase: Phase | null }> = ({ activePhase }) => {
    const progState = useProgramState();
    const { walkthrough } = progState;

    const [step, setStep] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [speed, setSpeed] = useState(700);
    const [, setSpreadTick] = useState(0); // re-render the spread button label on toggle
    const [hover, setHover] = useState<{ n: PositionedLayer; x: number; y: number } | null>(null);
    const timerRef = useRef<number | null>(null);
    const wrapRef = useRef<HTMLDivElement | null>(null);

    /* 章节切换 → 把游标对齐到该章节的第一个层 */
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
            if (e.code === 'Space')           { e.preventDefault(); setPlaying(false); setStep(s => Math.min(nodes.length-1, s+1)); }
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

    /* 主干各段 (用于动画进度) */
    const spineSegments = useMemo(() => {
        const segs: Array<{ y1: number; y2: number; active: boolean }> = [];
        segs.push({ y1: 4, y2: nodes[0].y, active: step >= 0 });
        for (let i = 0; i < nodes.length - 1; i++) {
            segs.push({
                y1: nodes[i].y + nodes[i].h,
                y2: nodes[i+1].y,
                active: i < step,
            });
        }
        return segs;
    }, [step]);

    return (
        <div
            ref={wrapRef}
            tabIndex={0}
            style={{
                position: 'relative',
                background: '#ffffff',
                padding: '8px 6px 4px',
                color: '#000',
                fontFamily: '-apple-system, "Segoe UI", "Helvetica Neue", "PingFang SC", "Microsoft YaHei", sans-serif',
                outline: 'none',
            }}
        >
            <div style={{ fontSize: 13, fontWeight: 600, textAlign: 'center', marginBottom: 4 }}>
                MT-LNN 架构 · 交互动画
            </div>
            <div style={{ fontSize: 10, color: '#777', textAlign: 'center', marginBottom: 6 }}>
                点击层跳转章节 · 悬停查看公式 · 空格键单步
            </div>

            <svg width={CANVAS_W} height={SVG_H} style={{ display: 'block', margin: '0 auto', background:'#fff' }}>
                <defs>
                    <marker id="mtlnn-ah" viewBox="0 0 10 10" refX="9" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M0,0 L10,5 L0,10 z" fill="#666"/>
                    </marker>
                    <marker id="mtlnn-ah-red" viewBox="0 0 10 10" refX="9" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                        <path d="M0,0 L10,5 L0,10 z" fill="#c0392b"/>
                    </marker>
                </defs>

                {/* 绿色张量主干 (分段) */}
                {spineSegments.map((s, i) => (
                    <line key={`sp-${i}`}
                        x1={SPINE_X} y1={s.y1} x2={SPINE_X} y2={s.y2}
                        stroke="#28a745" strokeWidth={3} strokeLinecap="round"
                        opacity={s.active ? 1 : 0.15}
                        style={{ transition: 'opacity .25s' }}
                    />
                ))}

                {/* 每层: 左侧维度标签 + 主干到层的短连接 */}
                {nodes.map((n, i) => (
                    <g key={`tap-${i}`}>
                        <text x={SPINE_X - 8} y={n.cy + 3}
                              textAnchor="end" fontSize={10} fill="#28a745"
                              fontFamily="Consolas, monospace">{n.dim}</text>
                        <line x1={SPINE_X} y1={n.cy} x2={n.x} y2={n.cy}
                              stroke="#28a745" strokeWidth={2}
                              opacity={i <= step ? 1 : 0.2}
                              style={{ transition: 'opacity .25s' }} />
                    </g>
                ))}

                {/* 残差跳连 (右侧灰色虚线箭头) */}
                {RESIDUALS.map(([fromId, toId], i) => {
                    const a = nodes.find(n => n.id === fromId)!;
                    const b = nodes.find(n => n.id === toId)!;
                    const rx = a.x + a.w + 22 + (i % 2) * 8;
                    return (
                        <path key={`res-${i}`}
                            d={`M ${a.x + a.w} ${a.y + 6} L ${rx} ${a.y + 6} L ${rx} ${b.y + b.h - 6} L ${b.x + b.w} ${b.y + b.h - 6}`}
                            stroke="#999" strokeWidth={2} fill="none"
                            strokeDasharray="5 5" markerEnd="url(#mtlnn-ah)"
                        />
                    );
                })}

                {/* GWTB Core 循环 (红色, h_{t-1} 跨时间步) */}
                {(() => {
                    const c = nodes.find(n => n.id === 'gwtb-c')!;
                    const right = c.x + c.w;
                    const d = `M ${right} ${c.cy - 6}
                               Q ${right + 36} ${c.cy - 24}, ${right + 28} ${c.cy + 14}
                               Q ${c.x - 20} ${c.cy + 36}, ${c.x - 24} ${c.cy + 4}
                               L ${c.x} ${c.cy + 4}`;
                    return (
                        <g>
                            <path d={d} stroke="#c0392b" strokeWidth={1.4} fill="none"
                                  markerEnd="url(#mtlnn-ah-red)" />
                            <text x={right + 42} y={c.cy - 12}
                                  fontSize={10} fill="#c0392b" fontWeight={600}>h_{'{t-1}'}</text>
                        </g>
                    );
                })()}

                {/* 层矩形 (含 Z 偏移阴影) */}
                {nodes.map((n, i) => {
                    const active = i === step;
                    const hovered = hover?.n.id === n.id;
                    return (
                        <g key={n.id}
                           style={{ cursor: 'pointer' }}
                           onClick={() => onLayerClick(n)}
                           onMouseEnter={(e) => onLayerHover(e, n)}
                           onMouseMove={(e) => onLayerHover(e, n)}
                           onMouseLeave={() => setHover(null)}
                        >
                            <rect x={n.x + Z_DEPTH} y={n.y + Z_DEPTH} width={n.w} height={n.h} rx={6}
                                  fill="#eaeaea" stroke="none" />
                            <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={6}
                                fill={n.color}
                                stroke={active ? '#ff4e4e' : (hovered ? '#333' : '#bbb')}
                                strokeWidth={active || hovered ? 2 : 1}
                            />

                            {n.kind === 'mtdl' ? (
                                <>
                                    <text x={n.cx} y={n.y + 16} textAnchor="middle"
                                          fontSize={14} fontWeight={600} fill="#000" pointerEvents="none">
                                        {n.name}
                                    </text>
                                    {(() => {
                                        const g = mtdlGeom(n);
                                        return (
                                            <g pointerEvents="none">
                                                <path
                                                    d={`M ${g.arcCX - g.R} ${g.arcCY} A ${g.R} ${g.R} 0 0 1 ${g.arcCX + g.R} ${g.arcCY}`}
                                                    stroke="#9ec9a3" strokeWidth={0.6} fill="none" strokeDasharray="2 2" />
                                                {g.protos.map(p => (
                                                    <g key={p.id}>
                                                        <rect x={p.x} y={p.y} width={10} height={10} rx={1.5}
                                                              fill="#7fbf8a" stroke="#3f7a4a" strokeWidth={0.5} />
                                                        <text x={p.cx + 6} y={p.cy - 3}
                                                              fontSize={7} fill="#3f7a4a"
                                                              fontFamily="Consolas, monospace">{p.id}</text>
                                                    </g>
                                                ))}
                                                <text x={n.x + 8} y={n.y + n.h - 28} fontSize={9} fill="#3f7a4a">
                                                    时间尺度:
                                                </text>
                                                {g.taus.map(t => (
                                                    <g key={t.id}>
                                                        <rect x={t.x} y={t.y} width={t.w} height={t.h} rx={2}
                                                              fill="#b8e0c2" stroke="#3f7a4a" strokeWidth={0.6} />
                                                        <text x={t.cx} y={t.y + 10} textAnchor="middle"
                                                              fontSize={8} fill="#1f4d2a"
                                                              fontFamily="Consolas, monospace">{t.id}</text>
                                                    </g>
                                                ))}
                                            </g>
                                        );
                                    })()}
                                </>
                            ) : (
                                <>
                                    <text x={n.cx} y={n.cy + 1} textAnchor="middle"
                                          fontSize={14} fontWeight={500} fill="#000" pointerEvents="none">
                                        {n.name}
                                    </text>
                                    <text x={n.cx} y={n.cy + 13} textAnchor="middle"
                                          fontSize={8} fill="#666" pointerEvents="none">
                                        {n.group}
                                    </text>
                                </>
                            )}
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
                <button onClick={() => setPlaying(p => !p)} style={btnStyle}>
                    {playing ? '⏸' : '▶'}
                </button>
                <button onClick={() => { setPlaying(false); setStep(s => Math.max(0, s-1)); }} style={btnStyle}>◀</button>
                <button onClick={() => { setPlaying(false); setStep(s => Math.min(nodes.length-1, s+1)); }} style={btnStyle}>▶</button>
                <button
                    onClick={() => {
                        const expanded = mtLnnSpreadState.target >= (SPREAD_COLLAPSED + SPREAD_EXPANDED) / 2;
                        mtLnnSpreadState.target = expanded ? SPREAD_COLLAPSED : SPREAD_EXPANDED;
                        progState.markDirty();
                        setSpreadTick(t => t + 1);
                    }}
                    style={{ ...btnStyle, minWidth: 56 }}
                    title="切换 MT-LNN 3D 视图各子层 Y 间距 (展开 / 收起)"
                >
                    {mtLnnSpreadState.target >= (SPREAD_COLLAPSED + SPREAD_EXPANDED) / 2 ? '收起' : '展开'}
                </button>
                <input
                    type="range" min={0} max={nodes.length - 1} value={step}
                    onChange={(e) => { setPlaying(false); setStep(parseInt(e.target.value, 10)); }}
                    style={{ flex: 1 }}
                />
                <span style={{ minWidth: 36, textAlign: 'right', color: '#555' }}>
                    {step + 1}/{nodes.length}
                </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '0 4px 6px', fontSize: 10, color: '#666' }}>
                速度
                <input type="range" min={200} max={1500} step={50} value={speed}
                       onChange={e => setSpeed(parseInt(e.target.value, 10))}
                       style={{ flex: 1 }} />
                <span style={{ minWidth: 80 }}>{nodes[step].name}</span>
            </div>

            {/* 悬浮提示 */}
            {hover && (
                <div style={{
                    position: 'absolute', left: hover.x, top: hover.y,
                    background: '#fff', border: '1px solid #888',
                    boxShadow: '0 4px 12px rgba(0,0,0,.12)',
                    padding: '6px 8px', borderRadius: 4,
                    fontSize: 11, lineHeight: 1.5, maxWidth: 300, zIndex: 10,
                    pointerEvents: 'none',
                }}>
                    <div style={{ fontWeight: 700, color: '#2b7cff', marginBottom: 2 }}>
                        {hover.n.name} <span style={{ color:'#888', fontWeight:400 }}>[{hover.n.group}]</span>
                    </div>
                    <div>维度: <b style={{ color:'#28a745' }}>{hover.n.dim}</b></div>
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
