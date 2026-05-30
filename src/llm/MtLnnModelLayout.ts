/**
 * MT-LNN 3D Layout for llm-viz
 */

import { IBlkDef, BlkSpecial } from "./GptModelLayout";
import { Vec3 } from "@/src/utils/vector";
import { DimStyle } from "./walkthrough/WalkthroughTools";
import { isNil } from "@/src/utils/data";

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

import { IModelLayout } from './GptModelLayout';

export interface IMTLNNLayout extends IModelLayout {
    cubes: IBlkDef[];
    protoNodes: IProtoNode[];
    lateralEdges: ILateralEdge[];
}

export interface IProtoNode {
    layerIdx: number;
    protoIdx: number;
    scaleIdx: number;
    position: Vec3;
    cube: IBlkDef;
}

export interface ILateralEdge {
    from: IProtoNode;
    to: IProtoNode;
}

interface IBlkDefArgs {
    t: 'w' | 'i' | 'a';
    xL?: number;
    xR?: number;
    xM?: number;
    zF?: number;
    zB?: number;
    zM?: number;
    y: number;
    cx: number;
    cy: number;
    cz: number;
    dimX: DimStyle;
    dimY: DimStyle;
    name?: string;
    small?: boolean;
    special?: BlkSpecial;
}

function mk(args: IBlkDefArgs): IBlkDef {
    let xDef = [args.xL, args.xR, args.xM].map(a => +!isNil(a)).reduce((a, b) => a + b, 0);
    let yDef = [args.zF, args.zB, args.zM].map(a => +!isNil(a)).reduce((a, b) => a + b, 0);
    if (xDef !== 1 || yDef !== 1) {
        throw new Error('Must supply exactly 1 x arg and 1 z arg');
    }
    let cell = 1.5;
    let dx = args.cx * cell;
    let dy = args.cz * cell;
    let x = !isNil(args.xL) ? args.xL : !isNil(args.xR) ? args.xR - dx : args.xM! - dx / 2;
    let z = !isNil(args.zB) ? args.zB : !isNil(args.zF) ? args.zF - dy : args.zM! - dy / 2;
    return {
        dx: args.cx * cell,
        dy: args.cy * cell,
        dz: args.cz * cell,
        t: args.t,
        x: x,
        y: args.y,
        z: z,
        cx: args.cx,
        cy: args.cy,
        cz: args.cz,
        dimX: args.dimX,
        dimY: args.dimY,
        name: args.name ?? 'Block',
        opacity: 1.0,
        highlight: 0.0,
        small: args.small ?? false,
        special: args.special ?? BlkSpecial.None,
        idx: -1,
    };
}

export function genMTLNNLayout(shape: IMTLNNShape): IMTLNNLayout {
    let layout: any = { shape };
    let { B, T, C, vocabSize, nLayers, nProtofilaments, nTimeScales, dProto } = shape;
    let cubes: IBlkDef[] = [];
    let protoNodes: IProtoNode[] = [];
    let lateralEdges: ILateralEdge[] = [];
    let y = 0;
    let cell = 1.5;
    let margin = Math.max(12, C / 10);
    let leftX = -T * cell / 2 - margin;
    let rightX = T * cell / 2 + margin;

    cubes.push(mk({t: 'i', cx: T, cz: B, cy: 1, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.None, name: 'Tokens'}));
    y += cell + margin;

    cubes.push(mk({t: 'w', cx: vocabSize, cz: 1, cy: C, y: y, xR: leftX, zM: 0, dimX: DimStyle.n_vocab, dimY: DimStyle.C, name: 'Token Embed'}));
    cubes.push(mk({t: 'w', cx: T, cz: 1, cy: C, y: y, xL: rightX, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Pos Embed'}));
    cubes.push(mk({t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'Embed'}));
    y += C * cell + margin * 2;

    let protoRadius = T * cell * 0.8;
    for (let layerIdx = 0; layerIdx < nLayers; layerIdx++) {
        cubes.push(mk({t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: `LN ${layerIdx + 1}`}));
        y += C * cell + margin;
        let layerProtoNodes: IProtoNode[] = [];
        for (let protoIdx = 0; protoIdx < nProtofilaments; protoIdx++) {
            let angle = (protoIdx / nProtofilaments) * Math.PI * 2;
            let protoX = protoRadius * Math.cos(angle);
            let protoZ = protoRadius * Math.sin(angle);
            for (let scaleIdx = 0; scaleIdx < nTimeScales; scaleIdx++) {
                let scaleY = y + scaleIdx * (dProto * cell + margin / 2);
                let protoBlock = mk({t: 'i', cx: Math.max(1, Math.floor(T / 4)), cz: B, cy: dProto, y: scaleY, xM: protoX, zM: protoZ, dimX: DimStyle.T, dimY: DimStyle.C, name: `P${protoIdx + 1}`, small: true});
                cubes.push(protoBlock);
                layerProtoNodes.push({layerIdx, protoIdx, scaleIdx, position: new Vec3(protoX, scaleY, protoZ), cube: protoBlock});
            }
        }
        protoNodes.push(...layerProtoNodes);
        for (let protoIdx = 0; protoIdx < nProtofilaments; protoIdx++) {
            let nextProtoIdx = (protoIdx + 1) % nProtofilaments;
            for (let scaleIdx = 0; scaleIdx < nTimeScales; scaleIdx++) {
                let fromNode = layerProtoNodes.find(n => n.protoIdx === protoIdx && n.scaleIdx === scaleIdx);
                let toNode = layerProtoNodes.find(n => n.protoIdx === nextProtoIdx && n.scaleIdx === scaleIdx);
                if (fromNode && toNode) {
                    lateralEdges.push({ from: fromNode, to: toNode });
                }
            }
        }
        y += nTimeScales * (dProto * cell + margin / 2) + margin * 2;
        cubes.push(mk({t: 'i', cx: T, cz: B, cy: C, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: `Residual ${layerIdx + 1}`}));
        y += C * cell + margin * 2;
    }

    let gwtbSize = Math.floor(C / 8);
    cubes.push(mk({t: 'i', cx: T, cz: B, cy: gwtbSize, y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'GWTB', special: BlkSpecial.Attention}));
    y += gwtbSize * cell + margin * 2;

    cubes.push(mk({t: 'i', cx: Math.floor(T / 2), cz: B, cy: Math.floor(C / 2), y: y, xM: 0, zM: 0, dimX: DimStyle.T, dimY: DimStyle.C, name: 'O(1) WM', special: BlkSpecial.Attention}));
    y += Math.floor(C / 2) * cell + margin * 2;

    cubes.push(mk({t: 'w', cx: C, cz: 1, cy: vocabSize, y: y, xM: 0, zM: 0, dimX: DimStyle.C, dimY: DimStyle.n_vocab, name: 'LM Head'}));
    cubes.push(mk({t: 'i', cx: T, cz: B, cy: vocabSize, y: y, xL: rightX, zM: 0, dimX: DimStyle.T, dimY: DimStyle.n_vocab, name: 'Logits'}));

    cubes.forEach((cube, idx) => { cube.idx = idx; });
    let weightCount = vocabSize * C + T * C + nLayers * (C * C + nProtofilaments * nTimeScales * (dProto * dProto));
    return { cubes, protoNodes, lateralEdges, cell, height: y, margin, weightCount, shape } as any;
}
