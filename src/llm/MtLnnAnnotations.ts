import { IBlkDef } from "./GptModelLayout";
import { IMTLNNLayout } from "./MtLnnModel";
import { addLine, ILineOpts, makeLineOpts } from "./render/lineRender";
import { measureTextWidth, writeTextToBuffer } from "./render/fontRender";
import { IRenderState } from "./render/modelRender";
import { addQuad } from "./render/triRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { clamp } from "@/src/utils/data";
import { IProgramState } from "./Program";
import { cameraToMatrixView } from "./Camera";
import { drawLineRect } from "./components/ModelCard";

// ---------------------------------------------------------------------------
// Local helper: draw a semi-transparent white background quad behind every
// piece of text we emit, then draw the text. The background sits in the same
// model matrix as the text so it stays attached during camera moves. This is
// the closest equivalent to "renderOrder=10 + rounded box behind label" in
// this WebGL2 engine — there is no per-object renderOrder, so we rely on the
// fact that addQuad/writeTextToBuffer issue draws in call order within a
// frame, with text drawn after (and therefore visually on top of) the bg.
// ---------------------------------------------------------------------------
const TEXT_BG_COLOR    = new Vec4(1.0, 1.0, 1.0, 0.85);
const TEXT_BG_BORDER   = new Vec4(0.0, 0.0, 0.0, 0.10);

function drawTextWithBg(
    render: IRenderState,
    text: string,
    color: Vec4,
    x: number, y: number,
    fontSize: number,
    mtx: Mat4f,
) {
    const w     = measureTextWidth(render.modelFontBuf, text, fontSize);
    const padX  = Math.max(2, fontSize * 0.30);
    const padY  = Math.max(1, fontSize * 0.18);
    // Slight negative z so the bg sits *behind* the text plane and never
    // accidentally hides it after perspective foreshortening.
    const tl    = new Vec3(x - padX,       y - padY,             -0.05);
    const br    = new Vec3(x + w + padX,   y + fontSize + padY,  -0.05);
    addQuad(render.triRender, tl, br, TEXT_BG_COLOR, mtx);
    // 1-px-equivalent border via four thin line segments (no rounded corners
    // — engine has no curve primitive). Color = rgba(0,0,0,0.1).
    const lo: ILineOpts = { color: TEXT_BG_BORDER, mtx, thick: 0.5, n: new Vec3(0, 0, 1) };
    drawLineRect(render, tl, br, lo);
    writeTextToBuffer(render.modelFontBuf, text, color, x, y, fontSize, mtx);
}

// ---------------------------------------------------------------------------
// Section labels on the LEFT side of the MT-LNN spine, plus simple
// straight arrows between consecutive spine blocks. Mirrors nano-gpt's
// drawBlockLabels + drawAllArrows so MT-LNN doesn't look bare.
// ---------------------------------------------------------------------------

const SPINE_NAMES_PRIMARY = ['Tokens', 'Embed'];

export function drawMTLNNAnnotations(render: IRenderState, layout: IMTLNNLayout, offset: Vec3 = new Vec3()) {
    const cubes = layout.cubes;
    const baseColor = new Vec4(0.30, 0.30, 0.36, 1.0);
    const accent    = new Vec4(0.20, 0.35, 0.85, 1.0);
    const protoCol  = new Vec4(0.55, 0.25, 0.65, 1.0);
    const tauCol    = new Vec4(0.20, 0.55, 0.45, 1.0);
    const find = (n: string) => cubes.find(c => c.name === n);

    const shape = (layout as any).shape ?? {};
    const nLayers: number = shape.nLayers ?? 4;
    const nProto:  number = shape.nProtofilaments ?? 13;
    const nTau:    number = shape.nTimeScales ?? 5;
    const dModel:  number = shape.dModel ?? shape.C ?? 0;
    const gwtbDim          = dModel ? Math.floor(dModel / 8) : 0;

    // ---- Section labels (left-side brackets, nano-gpt style) ----
    const tokens = find('Tokens');
    const embed  = find('Embed');
    if (tokens && embed) {
        labelLeft(render, 'Embedding', tokens, embed, baseColor, 18, offset);
    }

    // Color palette per the user's request — green I/O, purple Attn, blue MT-DL,
    // orange GWTB, gray LN/residual, red Coherence
    const colIO    = new Vec4(0.18, 0.55, 0.28, 1.0);  // green
    const colAttn  = new Vec4(0.45, 0.22, 0.62, 1.0);  // purple
    const colMtdl  = new Vec4(0.20, 0.35, 0.85, 1.0);  // blue
    const colGwtb  = new Vec4(0.88, 0.48, 0.10, 1.0);  // orange
    const colCoh   = new Vec4(0.78, 0.20, 0.20, 1.0);  // red
    const colNorm  = new Vec4(0.45, 0.45, 0.50, 1.0);  // gray

    for (let l = 1; l <= nLayers; l++) {
        // Sub-layer 1: Microtubule Attention
        const attnLn  = find(`L${l} Attn LN`);
        const attnRes = find(`L${l} Attn Residual`);
        if (attnLn && attnRes) {
            labelLeft(render, `Block ${l} · Sub-layer 1: Microtubule Attention`,
                attnLn, attnRes, colAttn, 16, offset);
        }
        // Sub-layer 2: MT-DL (Liquid)
        const lnnLn = find(`L${l} LNN LN`);
        const mtdlRes = find(`L${l} MT-DL Residual`);
        if (lnnLn && mtdlRes) {
            labelLeft(render, `Block ${l} · Sub-layer 2: MT-DL  ( ${nProto} Protofilaments × ${nTau} τ )`,
                lnnLn, mtdlRes, colMtdl, 16, offset);
        }
        // Sub-layer 3: per-block GWTB
        const gLn  = find(`L${l} GWTB LN`);
        const gRes = find(`L${l} GWTB Residual`);
        if (gLn && gRes) {
            const gwtbTxt = gwtbDim
                ? `Block ${l} · Sub-layer 3: GWTB  (d=${dModel} → ${gwtbDim} bottleneck)`
                : `Block ${l} · Sub-layer 3: GWTB (d/8 Bottleneck)`;
            labelLeft(render, gwtbTxt, gLn, gRes, colGwtb, 13, offset);
        }
        // h_prev recurrent state tag
        const hp = find(`L${l} h_prev (recurrent)`);
        if (hp) {
            labelLeft(render, `h_prev · O(1) recurrent state (LNN, no KV growth)`,
                hp, hp, colMtdl.mul(0.7), 11, offset);
        }
    }

    // Global Coherence (Orch-OR) — once, after all blocks
    const coh = find('Global Coherence (Orch-OR)');
    if (coh) {
        labelLeft(render, 'Global Coherence Layer  ·  Orch-OR collapse', coh, coh, colCoh, 14, offset);
    }
    // Final LN
    const finalLn = find('Final LN');
    if (finalLn) {
        labelLeft(render, 'Final LayerNorm', finalLn, finalLn, colNorm, 12, offset);
    }

    const lmHead = find('LM Head W');
    const logits = find('Logits');
    if (lmHead && logits) {
        labelLeft(render, 'LM Head → Logits  (weight-tied to E_tok)', lmHead, logits, colIO, 18, offset);
    }

    // ---- Microtubule ring annotations (one per block) ----
    // The protofilament cubes share the same name across blocks (P1..P13), so
    // group them by y-coordinate to recover the per-block ring.
    const protoCubes = cubes.filter(c => /^P\d+ W_in$/.test(c.name));
    if (protoCubes.length > 0) {
        // Cluster by y (cubes within nProto items sharing the same y form one ring).
        const rings: IBlkDef[][] = [];
        const sortedByY = [...protoCubes].sort((a, b) => a.y - b.y);
        let currentRing: IBlkDef[] = [];
        let lastY: number | null = null;
        for (const c of sortedByY) {
            if (lastY === null || Math.abs(c.y - lastY) < 1.0) {
                currentRing.push(c);
            } else {
                rings.push(currentRing);
                currentRing = [c];
            }
            lastY = c.y;
        }
        if (currentRing.length > 0) rings.push(currentRing);

        rings.forEach((ring, blockIdx) => {
            // Big floating label above the ring center (spine z ≈ 0)
            const meanY = ring.reduce((s, c) => s + c.y, 0) / ring.length;
            const ringLabel = `Microtubule (Block ${blockIdx + 1})  ·  ${nProto} protofilaments × ${nTau} τ-scales  ·  closed-form LTC`;
            const mtx = new Mat4f(); mtx[14] = 0 + offset.z;
            const txtSize = 7;
            const tw = measureTextWidth(render.modelFontBuf, ringLabel, txtSize);
            drawTextWithBg(render, ringLabel, colMtdl,
                -tw / 2, meanY - txtSize - 14 + offset.y, txtSize, mtx,
            );

            // Per-protofilament index labels at every PF position (around the ring)
            ring.forEach((pf, pi) => {
                const lbl = `P${pi + 1}`;
                const lblMtx = new Mat4f(); lblMtx[14] = pf.z + pf.dz / 2 + offset.z;
                const lblFs = 3.5;
                const lblW  = measureTextWidth(render.modelFontBuf, lbl, lblFs);
                drawTextWithBg(render, lbl, colMtdl.mul(0.8),
                    pf.x + pf.dx / 2 - lblW / 2 + offset.x,
                    pf.y - lblFs - 1.5 + offset.y,
                    lblFs, lblMtx,
                );
            });
        });
    }

    // ---- τ-scale labels for ONE representative protofilament per block (P1) ----
    // The tau cubes are named `P1 tau0` .. `P1 tau4` per block. Annotate them
    // with the time-constant role so the multi-timescale liquid nature is obvious.
    const tauLabels = ['τ₀ ≈ 0.01  fast', 'τ₁', 'τ₂', 'τ₃', 'τ₄ ≈ 10.0  slow'];
    const tauCubesP1 = cubes.filter(c => /^P1 tau\d+$/.test(c.name));
    // Group by y-block (one set per block)
    const tauByBlock = new Map<number, IBlkDef[]>();
    for (const t of tauCubesP1) {
        const yKey = Math.round(t.y / 50) * 50;
        if (!tauByBlock.has(yKey)) tauByBlock.set(yKey, []);
        tauByBlock.get(yKey)!.push(t);
    }
    for (const tauGroup of Array.from(tauByBlock.values())) {
        tauGroup.sort((a, b) => a.y - b.y);
        tauGroup.forEach((tc, i) => {
            if (i >= tauLabels.length) return;
            const mtx = new Mat4f(); mtx[14] = tc.z + tc.dz / 2 + offset.z;
            const tauFs = 4;
            drawTextWithBg(render, tauLabels[i], colMtdl.mul(0.9),
                tc.x + tc.dx + 2 + offset.x,
                tc.y + tc.dy / 2 - tauFs / 2 + offset.y,
                tauFs, mtx,
            );
        });
    }

    // ---- LTC formula floating tag at the first microtubule ring ----
    if (protoCubes.length > 0) {
        const firstPF = protoCubes.reduce((m, c) => (c.y < m.y ? c : m), protoCubes[0]);
        const mtx = new Mat4f(); mtx[14] = firstPF.z + offset.z;
        const formula = 'h⁽ᵖ,ˢ⁾_t = α·h⁽ᵖ,ˢ⁾_(t-1) + (1-α)·σ(W_in·x+b),   α = exp(-Δt/τ)';
        const formulaFs = 5;
        const fw = measureTextWidth(render.modelFontBuf, formula, formulaFs);
        drawTextWithBg(render, formula, colMtdl.mul(0.85),
            -fw / 2, firstPF.y - formulaFs - 22 + offset.y, formulaFs, mtx,
        );
    }

    // ---- Per-block inline tags on the RIGHT side ----
    const tag = (block: IBlkDef | undefined, txt: string, col: Vec4 = baseColor, sz = 10) => {
        if (block) labelRight(render, txt, block, col, sz, offset);
    };
    tag(tokens,                                 `T tokens`,                                   colIO,   10);
    tag(embed,                                  `x  [T × d_model=${dModel}]`,                colIO,   10);
    tag(find('Token Embed'),                    `E_tok  [V × C]`,                             colIO,    7);
    tag(find('Pos Embed'),                      `E_pos  [T × C]`,                             colIO,    7);
    tag(find('Global Coherence (Orch-OR)'),     `coherence · Φ̂ collapse`,                    colCoh,  10);
    tag(find('Final LN'),                       `LayerNorm`,                                  colNorm, 9);
    tag(find('Logits'),                         `logits  [T × V]`,                            colIO,   10);

    // ---- Per-block inner-component tags (one per block, picked via y-grouping) ----
    // Lateral coupling cubes share the same name across blocks; pick all and tag each.
    for (const lc of cubes.filter(c => c.name === 'Lateral Coupling SA')) {
        labelRight(render, `Lateral Coupling · cross-PF τ-mixing`, lc, colMtdl.mul(0.85), 5, offset);
    }
    for (const gc of cubes.filter(c => c.name === 'GWTB Compress')) {
        labelRight(render, `Compress  d → d/8`, gc, colGwtb.mul(0.9), 6, offset);
    }
    for (const gb of cubes.filter(c => c.name === 'GWTB Broadcast')) {
        labelRight(render, `Broadcast  d/8 → d`, gb, colGwtb.mul(0.9), 6, offset);
    }

    // ---- Straight spine arrows (data flow) ----
    const spine: (IBlkDef | undefined)[] = [];
    spine.push(find('Tokens'));
    spine.push(find('Embed'));
    for (let l = 1; l <= nLayers; l++) {
        spine.push(find(`L${l} Attn LN`));
        spine.push(find(`L${l} Attn Residual`));
        spine.push(find(`L${l} LNN LN`));
        spine.push(find(`L${l} MT-DL Residual`));
        spine.push(find(`L${l} GWTB LN`));
        spine.push(find(`L${l} GWTB Residual`));
    }
    spine.push(find('Global Coherence (Orch-OR)'));
    spine.push(find('Final LN'));
    spine.push(find('Logits'));

    let prev: IBlkDef | undefined = undefined;
    for (const cur of spine) {
        if (prev && cur && prev !== cur) {
            verticalArrow(render, prev, cur, accent, offset);
        }
        if (cur) prev = cur;
    }
}

// Bracket-style label sitting to the LEFT of a vertical span between
// `top` (high block) and `bot` (low block).
function labelLeft(
    render: IRenderState,
    text: string,
    top: IBlkDef,
    bot: IBlkDef,
    color: Vec4,
    fontSize: number,
    offset: Vec3,
) {
    const z = top.z + top.dz / 2 + offset.z;
    const mtx = new Mat4f();
    mtx[14] = z;

    const pad = 10;
    const leftX = Math.min(top.x, bot.x) - top.dx * 0.35 - 8 + offset.x;
    const yTop  = top.y + offset.y;
    const yBot  = bot.y + bot.dy + offset.y;
    const midY  = (yTop + yBot) / 2;

    const tw = measureTextWidth(render.modelFontBuf, text, fontSize);
    drawTextWithBg(render, text, color,
        leftX - tw - 2 * pad, midY - fontSize / 2, fontSize, mtx,
    );

    const lineColor = color.mul(0.55);
    const p0 = new Vec3(leftX, yTop, z);
    const p1 = new Vec3(leftX, yBot, z);
    const inward = new Vec3(1, 0, 0);

    addLine(render.lineRender, 1.0, lineColor,
        p0.mulAdd(inward, -pad), p1.mulAdd(inward, -pad));
    addLine(render.lineRender, 1.0, lineColor,
        p0.mulAdd(inward, -pad), p0);
    addLine(render.lineRender, 1.0, lineColor,
        p1.mulAdd(inward, -pad), p1);
}

// Small tag sitting just to the RIGHT of a block (single short line).
function labelRight(
    render: IRenderState,
    text: string,
    block: IBlkDef,
    color: Vec4,
    fontSize: number,
    offset: Vec3,
) {
    const z   = block.z + block.dz / 2 + offset.z;
    const mtx = new Mat4f(); mtx[14] = z;
    const x   = block.x + block.dx + offset.x + 6;
    const y   = block.y + block.dy / 2 + offset.y - fontSize / 2;
    drawTextWithBg(render, text, color, x, y, fontSize, mtx);
}

// ---------------------------------------------------------------------------
// MT-LNN model card — sits to the left of the model, shows title, key dims,
// param count, and source attribution (repo / paper / biological citations).
// Mimics nano-gpt's drawModelCard but tailored for MT-LNN's extra structure.
// ---------------------------------------------------------------------------
export function drawMTLNNModelCard(state: IProgramState, layout: IMTLNNLayout, offset: Vec3) {
    const { render } = state;
    const { camPos } = cameraToMatrixView(state.camera);
    const dist = camPos.dist(new Vec3(0, 0, -30).add(offset));
    const scale = clamp(dist / 1500.0, 0.30, 800.0);

    const shape: any = (layout as any).shape ?? {};
    const weightCount: number = (layout as any).weightCount ?? 0;

    const pinY = 5;
    const mtx = Mat4f.fromScaleTranslation(new Vec3(scale, scale, scale), new Vec3(0, pinY, 0).add(offset))
        .mul(Mat4f.fromTranslation(new Vec3(0, -pinY, 0)));

    const thick = 1.0 / 10.0 * scale;
    const borderColor = Vec4.fromHexColor("#3b3b66", 0.85);
    const backgroundColor = Vec4.fromHexColor("#dbeafe", 0.55);
    const titleColor = Vec4.fromHexColor("#11224a", 1.0);
    const labelColor = Vec4.fromHexColor("#334155", 1.0);
    const linkColor  = Vec4.fromHexColor("#1d4ed8", 1.0);
    const sectionColor = Vec4.fromHexColor("#6b6b8e", 1.0);
    const n = new Vec3(0, 0, 1);
    const lineOpts: ILineOpts = { color: borderColor, mtx, thick, n };

    const tl = new Vec3(-58, -43, 0);
    const br = new Vec3( 58,   5, 0);
    drawLineRect(render, tl, br, lineOpts);
    addQuad(render.triRender, new Vec3(tl.x, tl.y, -0.1), new Vec3(br.x, br.y, -0.1), backgroundColor, mtx);

    const midX = (tl.x + br.x) / 2;
    let y = tl.y + 1.5;

    // ---- Title ----
    const titleFs = 7;
    const title = "Microtubule Liquid Neural Network (MT-LNN)";
    const titleW = measureTextWidth(render.modelFontBuf, title, titleFs);
    drawTextWithBg(render, title, titleColor, midX - titleW / 2, y, titleFs, mtx);
    y += titleFs * 1.2;

    // ---- Subtitle (paper line) ----
    const subFs = 2.8;
    const sub = "Brain-inspired continuous-time LM | Orch-OR + Liquid Time-Constant";
    const subW = measureTextWidth(render.modelFontBuf, sub, subFs);
    drawTextWithBg(render, sub, sectionColor, midX - subW / 2, y, subFs, mtx);
    y += subFs * 1.6;

    // ---- Key dims (two columns) ----
    const rowFs = 3;
    const rowH  = rowFs * 1.4;
    const colLeftX  = tl.x + 3;
    const colRightX = midX + 3;

    const nParams = weightCount > 0 ? numComma(weightCount) : "n/a";
    const left: [string, string][] = [
        ["n_params", nParams],
        ["n_layers (L)", String(shape.nBlocks ?? shape.nLayers ?? "?")],
        ["d_model (C)", String(shape.C ?? shape.dModel ?? "?")],
        ["seq_len (T)", String(shape.T ?? shape.maxSeqLen ?? "?")],
    ];
    const right: [string, string][] = [
        ["n_protofilaments", String(shape.nProtofilaments ?? 13)],
        ["n_timescales (tau)", String(shape.nTimeScales ?? 5)],
        ["n_heads (Attn)", String(shape.nHeads ?? "?")],
        ["d_head (A)", String(shape.A ?? shape.dHead ?? "?")],
    ];
    for (let i = 0; i < left.length; i++) {
        writeRow(render, mtx, colLeftX,  y + i * rowH, left[i][0],  left[i][1],  labelColor, titleColor, rowFs);
        writeRow(render, mtx, colRightX, y + i * rowH, right[i][0], right[i][1], labelColor, titleColor, rowFs);
    }
    y += left.length * rowH + 2;

    // ---- Divider ----
    addLine(render.lineRender, thick * 0.6, sectionColor,
        new Vec3(tl.x + 2, y, 0).mulAdd(new Vec3(0, 0, 1), 0).add(new Vec3()),
        new Vec3(br.x - 2, y, 0));
    // (line ignored if perf issues — kept simple)
    y += 1.5;

    // ---- Source / attribution ----
    const srcFs = 2.6;
    const srcs: [string, Vec4][] = [
        ["Source: github.com/everest-an/M1  |  mt_lnn/model.py", linkColor],
        ["Spec: ARCHITECTURE.md  |  MT_LNN_ARCHITECTURE_VISUAL.md", labelColor],
        ["Bio: Hameroff & Penrose (1996) Orch-OR  |  Hasani et al. (2021) LTC", sectionColor],
        ["Viz fork: github.com/everest-an/LLM-Visualization (from bbycroft/llm-viz)", linkColor],
    ];
    for (const [text, color] of srcs) {
        drawTextWithBg(render, text, color, tl.x + 3, y, srcFs, mtx);
        y += srcFs * 1.3;
    }
}

function writeRow(
    render: IRenderState, mtx: Mat4f,
    x: number, y: number,
    label: string, value: string,
    labelColor: Vec4, valueColor: Vec4, fs: number,
) {
    drawTextWithBg(render, label + " =", labelColor, x, y, fs, mtx);
    const lw = measureTextWidth(render.modelFontBuf, label + " =", fs);
    drawTextWithBg(render, " " + value, valueColor, x + lw, y, fs, mtx);
}

function numComma(a: number) {
    const s = a.toString();
    let out = "";
    for (let i = 0; i < s.length; i++) {
        if (i > 0 && (s.length - i) % 3 === 0) out += ",";
        out += s[i];
    }
    return out;
}

// Simple downward arrow from bottom-center of `src` to top-center of `dest`.
function verticalArrow(
    render: IRenderState,
    src: IBlkDef,
    dest: IBlkDef,
    color: Vec4,
    offset: Vec3,
) {
    const cx = (src.x + src.dx / 2 + dest.x + dest.dx / 2) / 2 + offset.x;
    const z  = (src.z + src.dz / 2 + dest.z + dest.dz / 2) / 2 + offset.z;
    const yStart = src.y + src.dy + 1 + offset.y;
    const yEnd   = dest.y - 1 + offset.y;
    if (yEnd <= yStart) return;

    const a = new Vec3(cx, yStart, z);
    const b = new Vec3(cx, yEnd, z);
    addLine(render.lineRender, 1.5, color, a, b);

    // simple arrow head
    const head = Math.min(6, (yEnd - yStart) * 0.15);
    addLine(render.lineRender, 1.5, color,
        b, new Vec3(cx - head * 0.6, yEnd - head, z));
    addLine(render.lineRender, 1.5, color,
        b, new Vec3(cx + head * 0.6, yEnd - head, z));
}
