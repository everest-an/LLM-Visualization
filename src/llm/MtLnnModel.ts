import { IBlkDef, BlkSpecial } from "./GptModelLayout";
import { Vec3 } from "@/src/utils/vector";
import { DimStyle } from "./walkthrough/WalkthroughTools";
import { isNil } from "@/src/utils/data";

export interface MTLNNConfig {
    vocabSize: number;
    maxSeqLen: number;
    dModel: number;
    nLayers: number;
    nHeads: number;
    dHead: number;
    nProtofilaments: number;
    nTimeScales: number;
    dProto: number;
    tauMin: number;
    tauMax: number;
    dt: number;
    gammaInit: number;
    gtpPeriod: number;
    gwtbCompressionRatio: number;
    gwtbNHeads: number;
    coherenceSparsity: number;
    useDecayWM: boolean;
}

export const DEFAULT_MTLNN_CONFIG: MTLNNConfig = {
    vocabSize: 50257,
    maxSeqLen: 512,
    dModel: 512,
    nLayers: 4,
    nHeads: 8,
    dHead: 64,
    nProtofilaments: 13,
    nTimeScales: 5,
    dProto: 64,
    tauMin: 0.01,
    tauMax: 10.0,
    dt: 1.0,
    gammaInit: 0.1,
    gtpPeriod: 256,
    gwtbCompressionRatio: 8,
    gwtbNHeads: 4,
    coherenceSparsity: 0.1,
    useDecayWM: true,
};

// Tiny demo config (analogous to nano-gpt's scale) so the whole MT-LNN architecture
// fits on screen with visible cubes & relations.
export const DEMO_MTLNN_CONFIG: MTLNNConfig = {
    vocabSize: 3,
    maxSeqLen: 6,
    dModel: 48,
    nLayers: 2,
    nHeads: 3,
    dHead: 16,
    nProtofilaments: 13,
    nTimeScales: 5,
    dProto: 4,
    tauMin: 0.01,
    tauMax: 10.0,
    dt: 1.0,
    gammaInit: 0.1,
    gtpPeriod: 256,
    gwtbCompressionRatio: 8,
    gwtbNHeads: 4,
    coherenceSparsity: 0.1,
    useDecayWM: true,
};

export interface IMTLNNShape {
    B: number;
    T: number;
    C: number;
    vocabSize: number;
    nLayers: number;
    nProtofilaments: number;
    nTimeScales: number;
    dProto: number;
}

import { IModelLayout } from "./GptModelLayout";
export interface IMTLNNLayout extends IModelLayout {
    shape: any;
    cubes: IBlkDef[];
}

function mk(args: any, offset: Vec3): IBlkDef {
    let xDef = [args.xL, args.xR, args.xM].map((a: any) => +!isNil(a)).reduce((a: number, b: number) => a + b, 0);
    let yDef = [args.zF, args.zB, args.zM].map((a: any) => +!isNil(a)).reduce((a: number, b: number) => a + b, 0);

    if (xDef !== 1 || yDef !== 1) {
        throw new Error('Must supply 1 x arg and 1 z arg');
    }

    let cell = 1.5;
    let dx = args.cx * cell;
    let dy = args.cz * cell;
    let x = !isNil(args.xL) ? args.xL : !isNil(args.xR) ? args.xR - dx : args.xM - dx / 2;
    let z = !isNil(args.zB) ? args.zB : !isNil(args.zF) ? args.zF - dy : args.zM - dy / 2;

    x += offset.x;
    let y = args.y + offset.y;
    z += offset.z;

    return {
        dx: args.cx * cell,
        dy: args.cy * cell,
        dz: args.cz * cell,
        t: args.t,
        x: x,
        y: y,
        z: z,
        cx: args.cx,
        cy: args.cy,
        cz: args.cz,
        dimX: args.dimX,
        dimY: args.dimY,
        name: args.name || 'Block',
        opacity: 1.0,
        highlight: 0.0,
        small: args.small || false,
        special: args.special || BlkSpecial.None,
        idx: -1,
    } as any;
}

function genMTLNNLayout(shape: IMTLNNShape, offset: Vec3): IMTLNNLayout {
    let layout: any = { shape };
    let { B, T, C, vocabSize, nLayers, nProtofilaments, nTimeScales, dProto } = shape;
    let cubes: IBlkDef[] = [];
    let y = 0;
    let cell = 1.5;
    let margin = Math.max(12, C / 10);
    let leftX = -T * cell / 2 - margin;
    let rightX = T * cell / 2 + margin;

    let m = (args: any) => mk(args, offset);

    cubes.push(m({ t: 'i', cx: T, cz: B, cy: 1, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.None, name: 'Tokens' }));
    y += cell + margin;

    cubes.push(m({ t: 'w', cx: vocabSize, cz: 1, cy: C, y: y, xR: leftX, zM: 0, dimX: DimStyle.n_vocab, dimY: DimStyle.C, name: 'Token Embed' }));
    cubes.push(m({ t: 'w', cx: T, cz: 1, cy: C, y: y, xL: rightX, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Pos Embed' }));
    cubes.push(m({ t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Embed' }));
    y += C * cell + margin * 2;

    // Microtubule cylinder geometry �� pick radius so the 13 protofilaments
    // visually form a wall (adjacent PFs nearly touch in the tangential direction).
    // PF lateral footprint �� B*cell (each cube is B-cells deep, which sits tangent
    // to the cylinder), so circumference 2��R �� nPF * B*cell * fillFactor.
    let pfTangential = B * cell + margin / 4;
    let protoRadius = nProtofilaments * pfTangential / (2 * Math.PI) * 1.15;
    let pfWidth = Math.max(2, Math.floor(T / 2));   // chunkier PF cubes (was T/4)

    for (let layerIdx = 0; layerIdx < nLayers; layerIdx++) {
        // ===== Sub-layer 1 :  Microtubule Attention  (pre-norm + residual, multi-head spread along z) =====
        cubes.push(m({ t: 'w', cx: C, cz: 1, cy: C, y: y, xR: leftX, zM: 0, dimX: DimStyle.C, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' Attn LN W' }));
        cubes.push(m({ t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' Attn LN' }));
        y += C * cell + margin;

        // multi-head fan: spread each head along z (same pattern as nano-gpt)
        const nHeads = (shape as any).nHeads ?? 1;
        const A = (shape as any).dHead ?? Math.max(1, Math.floor(C / Math.max(1, nHeads)));
        const headWidth = 3 * B * cell + margin * 0.6 + (C * cell) / 16;
        const attnQkvY = y;
        const attnMatrixY = attnQkvY + A * cell + margin;
        const headOutY   = attnMatrixY + T * cell + margin;
        for (let h = 0; h < nHeads; h++) {
            const headZ = headWidth * h - (nHeads - 1) * headWidth / 2;
            // Q, K, V weights (each [C �� A])
            cubes.push(m({ t: 'w', cx: C, cz: 1, cy: A, y: attnQkvY, xR: leftX, zM: headZ - B * cell - margin / 3, dimX: DimStyle.C, dimY: DimStyle.C, name: `L${layerIdx + 1} H${h + 1} Q W` }));
            cubes.push(m({ t: 'w', cx: C, cz: 1, cy: A, y: attnQkvY, xR: leftX, zM: headZ,                          dimX: DimStyle.C, dimY: DimStyle.C, name: `L${layerIdx + 1} H${h + 1} K W` }));
            cubes.push(m({ t: 'w', cx: C, cz: 1, cy: A, y: attnQkvY, xR: leftX, zM: headZ + B * cell + margin / 3,  dimX: DimStyle.C, dimY: DimStyle.C, name: `L${layerIdx + 1} H${h + 1} V W` }));
            // Q, K, V vectors (each [T �� A])
            cubes.push(m({ t: 'i', cx: T, cz: B, cy: A, y: attnQkvY, xM: 0, zM: headZ - B * cell - margin / 3, dimX: DimStyle.T, dimY: DimStyle.C, name: `L${layerIdx + 1} H${h + 1} Q vec` }));
            cubes.push(m({ t: 'i', cx: T, cz: B, cy: A, y: attnQkvY, xM: 0, zM: headZ,                          dimX: DimStyle.T, dimY: DimStyle.C, name: `L${layerIdx + 1} H${h + 1} K vec` }));
            cubes.push(m({ t: 'i', cx: T, cz: B, cy: A, y: attnQkvY, xM: 0, zM: headZ + B * cell + margin / 3,  dimX: DimStyle.T, dimY: DimStyle.C, name: `L${layerIdx + 1} H${h + 1} V vec` }));
            // attention matrix [T �� T] per head
            cubes.push(m({ t: 'i', cx: T, cz: B, cy: T, y: attnMatrixY, xM: 0, zM: headZ, dimX: DimStyle.T, dimY: DimStyle.T, name: `L${layerIdx + 1} H${h + 1} Attn`, special: BlkSpecial.Attention }));
            // per-head value-out (output vectors [T �� A])
            cubes.push(m({ t: 'i', cx: T, cz: B, cy: A, y: headOutY, xM: 0, zM: headZ, dimX: DimStyle.T, dimY: DimStyle.C, name: `L${layerIdx + 1} H${h + 1} Out vec` }));
        }
        y = headOutY + A * cell + margin;
        // output projection (concat-heads �� C)
        cubes.push(m({ t: 'w', cx: C, cz: 1, cy: C, y: y, xR: leftX, zM: 0, dimX: DimStyle.C, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' Attn Out W' }));
        cubes.push(m({ t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' Attn Residual' }));
        y += C * cell + margin * 2;

        // ===== Sub-layer 2 :  MT-DL (Liquid)  (pre-norm + residual) =====
        cubes.push(m({ t: 'w', cx: C, cz: 1, cy: C, y: y, xR: leftX, zM: 0, dimX: DimStyle.C, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' LNN LN W' }));
        cubes.push(m({ t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' LNN LN' }));
        y += C * cell + margin;

        let protoStartY = y;
        
        let lateralY = protoStartY + (nTimeScales * (dProto * cell + margin / 2)) / 2;
        cubes.push(m({ t: 'w', cx: dProto * 2, cz: 1, cy: dProto * 2, y: lateralY, xR: -margin, zM: 0, dimX: DimStyle.C, dimY: DimStyle.C, name: 'Lateral Coupling W' }));
        cubes.push(m({ t: 'i', cx: T, cz: B, cy: dProto * 2, y: lateralY, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Lateral Coupling SA', special: BlkSpecial.Attention }));

        for (let protoIdx = 0; protoIdx < nProtofilaments; protoIdx++) {
            let angle = (protoIdx / nProtofilaments) * Math.PI * 2;
            let protoX = protoRadius * Math.cos(angle);
            let protoZ = protoRadius * Math.sin(angle);

            cubes.push(m({
                t: 'w', cx: C, cz: 1, cy: dProto * nTimeScales, y: protoStartY,
                xR: protoX - (pfWidth * cell) / 2 - margin / 2, zM: protoZ, dimX: DimStyle.C, dimY: DimStyle.C, name: 'P' + (protoIdx + 1) + ' W_in'
            }));

            let endTimeScaleY = protoStartY;
            for (let scaleIdx = 0; scaleIdx < nTimeScales; scaleIdx++) {
                let scaleY = protoStartY + scaleIdx * (dProto * cell + margin / 2);
                endTimeScaleY = scaleY + dProto * cell;
                
                cubes.push(m({
                    t: 'i', cx: pfWidth, cz: B, cy: dProto, y: scaleY,
                    xM: protoX, zM: protoZ, dimX: DimStyle.T, dimY: DimStyle.C,
                    name: 'P' + (protoIdx + 1) + ' tau' + scaleIdx
                }));
                
                cubes.push(m({
                    t: 'w', cx: pfWidth, cz: 1, cy: dProto, y: scaleY,
                    xM: protoX, zM: protoZ - B * cell - margin / 2, dimX: DimStyle.C, dimY: DimStyle.C, name: 'ODE W'
                }));
            }
            
            let combY = endTimeScaleY + margin;
            cubes.push(m({
                 t: 'i', cx: pfWidth, cz: B, cy: dProto, y: combY,
                 xM: protoX, zM: protoZ, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Kappa Gate', special: BlkSpecial.Attention
            }));
            cubes.push(m({
                 t: 'w', cx: dProto, cz: 1, cy: dProto, y: combY,
                 xL: protoX + (pfWidth * cell) / 2 + margin / 2, zM: protoZ, dimX: DimStyle.C, dimY: DimStyle.C, name: 'MAP Gate (MLP)'
            }));

            cubes.push(m({
                t: 'w', cx: dProto * nTimeScales, cz: 1, cy: C, y: protoStartY,
                xL: protoX + (pfWidth * cell) / 2 + margin * 2 + dProto * cell, zM: protoZ, dimX: DimStyle.C, dimY: DimStyle.C, name: 'P' + (protoIdx + 1) + ' W_out'
            }));
        }

        y += nTimeScales * (dProto * cell + margin / 2) + margin * 4 + dProto * cell;
        cubes.push(m({ t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' MT-DL Residual' }));
        y += C * cell + margin * 2;

        // ===== Sub-layer 3 (optional, gwtb_per_block=True) :  per-block GWTB =====
        let gwtbSize = Math.floor(C / 8);
        cubes.push(m({ t: 'w', cx: C, cz: 1, cy: C, y: y, xR: leftX, zM: 0, dimX: DimStyle.C, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' GWTB LN W' }));
        cubes.push(m({ t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' GWTB LN' }));
        y += C * cell + margin;

        cubes.push(m({ t: 'w', cx: C, cz: 1, cy: gwtbSize, y: y, xR: leftX, zM: 0, dimX: DimStyle.C, dimY: DimStyle.C, name: 'GWTB Compress' }));
        cubes.push(m({ t: 'i', cx: T, cz: B, cy: gwtbSize, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'GWTB Attn', special: BlkSpecial.Attention }));
        y += gwtbSize * cell + margin;
        
        cubes.push(m({ t: 'w', cx: gwtbSize, cz: 1, cy: C, y: y, xR: leftX, zM: 0, dimX: DimStyle.C, dimY: DimStyle.C, name: 'GWTB Broadcast' }));
        cubes.push(m({ t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'L' + (layerIdx + 1) + ' GWTB Residual' }));
        y += C * cell + margin * 3;

        // h_prev recurrent state tag (a thin slab that lives between this layer's
        // MT-DL output and the NEXT layer's Attn LN �� it is what the code calls
        // `h_prev` and what gets passed into both Attn and MT-DL sub-layers).
        cubes.push(m({
            t: 'i', cx: pfWidth, cz: B, cy: Math.floor(C / 4),
            y: y, xR: leftX, zM: 0,
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'L' + (layerIdx + 1) + ' h_prev (recurrent)',
        }));
    }

    // ===== Global GlobalCoherenceLayer (Orch-OR) �� ONCE, after all blocks =====
    let gwtbGlobal = Math.floor(C / 8);
    cubes.push(m({
        t: 'w', cx: gwtbGlobal, cz: 1, cy: Math.floor(C / 2), y: y, xR: leftX, zM: 0,
        dimX: DimStyle.C, dimY: DimStyle.C, name: 'Coherence Update W',
    }));
    cubes.push(m({
        t: 'i', cx: Math.floor(T / 2), cz: B, cy: Math.floor(C / 2), y: y, xM: 0, zM: 0,
        dimX: DimStyle.T, dimY: DimStyle.C, name: 'Global Coherence (Orch-OR)',
        special: BlkSpecial.Attention,
    }));
    y += Math.floor(C / 2) * cell + margin * 2;

    // ===== Final LayerNorm =====
    cubes.push(m({ t: 'w', cx: C, cz: 1, cy: C, y: y, xR: leftX, zM: 0, dimX: DimStyle.C, dimY: DimStyle.C, name: 'Final LN W' }));
    cubes.push(m({ t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Final LN' }));
    y += C * cell + margin * 2;

    cubes.push(m({ t: 'w', cx: C, cz: 1, cy: vocabSize, y: y, xM: 0, zM: 0, dimX: DimStyle.C, dimY: DimStyle.n_vocab, name: 'LM Head W' }));
    cubes.push(m({ t: 'i', cx: T, cz: B, cy: vocabSize, y: y, xL: rightX, zM: 0, dimX: DimStyle.T, dimY: DimStyle.n_vocab, name: 'Logits' }));

    cubes.forEach((cube, idx) => { cube.idx = idx; });
    let weightCount = cubes.reduce((acc, c) => acc + (c.t === 'w' ? c.cx * c.cy * c.cz : 0), 0);
    return { shape, cubes, cell, height: y, margin, weightCount } as any;
}

export function createMTLNNLayout(config: MTLNNConfig, offset: Vec3 = new Vec3(0, 0, 0)): IMTLNNLayout {
    let shape: any = { A: config.dModel / config.nHeads, nBlocks: config.nLayers, nHeads: config.nHeads, dHead: config.dHead,
        B: 4,
        T: config.maxSeqLen,
        C: config.dModel,
        vocabSize: config.vocabSize,
        nLayers: config.nLayers,
        nProtofilaments: config.nProtofilaments,
        nTimeScales: config.nTimeScales,
        dProto: config.dProto,
    };
    return genMTLNNLayout(shape, offset);
}