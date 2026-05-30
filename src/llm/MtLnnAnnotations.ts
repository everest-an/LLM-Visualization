import { IBlkDef } from "./GptModelLayout";
import { IMTLNNLayout } from "./MtLnnModel";
import { addLine } from "./render/lineRender";
import { measureTextWidth, writeTextToBuffer } from "./render/fontRender";
import { IRenderState } from "./render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";

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

    // ---- Per-block inline tags on the RIGHT side ----
    const tag = (block: IBlkDef | undefined, txt: string, col: Vec4 = baseColor, sz = 10) => {
        if (block) labelRight(render, txt, block, col, sz, offset);
    };
    tag(tokens,                                 `T tokens`,                                   colIO,   10);
    tag(embed,                                  `x  [T × d_model=${dModel}]`,                colIO,   10);
    tag(find('Global Coherence (Orch-OR)'),     `coherence · Φ̂ collapse`,                    colCoh,  10);
    tag(find('Final LN'),                       `LayerNorm`,                                  colNorm, 9);
    tag(find('Logits'),                         `logits  [T × V]`,                            colIO,   10);

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
    writeTextToBuffer(
        render.modelFontBuf, text, color,
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
    writeTextToBuffer(render.modelFontBuf, text, color, x, y, fontSize, mtx);
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
