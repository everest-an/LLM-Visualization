'use client';
import React, { useLayoutEffect, useRef, useState } from 'react';
import { Vec3 } from '@/src/utils/vector';
import { Mat4f } from '@/src/utils/matrix';
import { IModelLayout, IBlkDef } from './GptModelLayout';
import { initRender } from './render/modelRender';
import { fetchFontAtlasData } from './render/fontRender';
import { renderBlocksSimple } from './render/blockRender';
import { DimStyle } from './walkthrough/WalkthroughTools';

import { createMTLNNLayout, DEFAULT_MTLNN_CONFIG } from './MtLnnModel';

export function MtLnnLayerView() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [cubeCount, setCubeCount] = useState(0);

    useLayoutEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        let mounted = true;
        let animationId: number;
        
        fetchFontAtlasData().then(fontData => {
            if (!mounted) return;
            
            const state = initRender(canvas, fontData);
            if (!state) {
                console.error('WebGL init failed');
                return;
            }

            // 鎵嬪姩鍒涘缓绠€鍗曠殑娴嬭瘯甯冨眬
            const cubes: IBlkDef[] = [];
            const cell = 1.5;
            let y = 0;
            
            // Input tokens
            cubes.push({
                t: 'i', cx: 128, cz: 1, cy: 1, x: 0, y: y, z: 0,
                dx: 128 * cell, dy: cell, dz: cell,
                dimX: DimStyle.T, dimY: DimStyle.None,
                name: 'Input', opacity: 1, highlight: 0, small: false,
                special: 0, idx: 0
            });
            
            y += 50;
            
            // Embedding
            cubes.push({
                t: 'i', cx: 128, cz: 1, cy: 832, x: 0, y: y, z: 0,
                dx: 128 * cell, dy: 832 * cell, dz: cell,
                dimX: DimStyle.T, dimY: DimStyle.C,
                name: 'Embed', opacity: 1, highlight: 0, small: false,
                special: 0, idx: 1
            });
            
            y += 832 * cell + 50;
            
            // 13 Protofilaments in circle
            const protoRadius = 300;
            for (let i = 0; i < 13; i++) {
                const angle = (i / 13) * Math.PI * 2;
                const px = protoRadius * Math.cos(angle);
                const pz = protoRadius * Math.sin(angle);
                
                cubes.push({
                    t: 'i', cx: 32, cz: 1, cy: 64, 
                    x: px, y: y, z: pz,
                    dx: 32 * cell, dy: 64 * cell, dz: cell,
                    dimX: DimStyle.T, dimY: DimStyle.C,
                    name: `P${i+1}`, opacity: 1, highlight: 0, small: true,
                    special: 0, idx: 2 + i
                });
            }
            
            y += 100;
            
            // Output
            cubes.push({
                t: 'i', cx: 128, cz: 1, cy: 200, x: 0, y: y, z: 0,
                dx: 128 * cell, dy: 200 * cell, dz: cell,
                dimX: DimStyle.T, dimY: DimStyle.n_vocab,
                name: 'Output', opacity: 1, highlight: 0, small: false,
                special: 0, idx: 15
            });

            const layout: IModelLayout = {
                cell: cell,
                height: y + 300,
                margin: 12,
                cubes: cubes
            };

            setCubeCount(cubes.length);
            console.log('MT-LNN cubes created:', cubes.length);

            const camera = {
                angle: new Vec3(-0.3, 0, 0),
                center: new Vec3(0, 200, 0)
            };

            let rotation = 0;

            function render() {
                if (!state || !canvas || !mounted) return;

                const gl = state.gl;
                const dpr = window.devicePixelRatio || 1;
                const w = Math.floor(canvas.clientWidth * dpr);
                const h = Math.floor(canvas.clientHeight * dpr);

                if (canvas.width !== w || canvas.height !== h) {
                    canvas.width = w;
                    canvas.height = h;
                }

                gl.viewport(0, 0, w, h);
                gl.clearColor(0.05, 0.05, 0.1, 1.0);
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                gl.enable(gl.DEPTH_TEST);
                gl.enable(gl.BLEND);
                gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

                rotation += 0.002;
                camera.angle = new Vec3(-0.3, rotation, 0);

                const modelMtx = Mat4f.fromAxisAngle(new Vec3(0, 1, 0), 0);
                
                try {
                    renderBlocksSimple(state.blockRender, layout.cubes);
                } catch (e) {
                    console.error('Render error:', e);
                }

                animationId = requestAnimationFrame(render);
            }

            animationId = requestAnimationFrame(render);
        }).catch(err => {
            console.error('Init error:', err);
        });

        return () => {
            mounted = false;
            if (animationId) cancelAnimationFrame(animationId);
        };
    }, []);

    return (
        <div style={{width: '100%', height: '100vh', position: 'relative', background: '#0a0e1a'}}>
            <canvas ref={canvasRef} style={{width: '100%', height: '100%', display: 'block'}} />
            <div style={{position: 'absolute', top: 20, left: 20, color: 'white', background: 'rgba(0,0,0,0.9)', padding: 20, borderRadius: 8, fontFamily: 'monospace', fontSize: 14, border: '2px solid #64c8ff'}}>
                <h2 style={{margin: 0, marginBottom: 15, fontSize: 24, color: '#64c8ff'}}>MT-LNN Architecture</h2>
                <div>馃К 13 Protofilaments (circular)</div>
                <div>鈴憋笍 5 Time Scales (multiscale)</div>
                <div>鈿?GWTB Bottleneck (8脳)</div>
                <div>馃 O(1) Working Memory</div>
                <div style={{marginTop: 15, paddingTop: 15, borderTop: '1px solid rgba(255,255,255,0.2)'}}>
                    <div>Blocks: {cubeCount}</div>
                    <div style={{fontSize: 12, opacity: 0.8, marginTop: 8}}>
                        WebGL 3D Rendering - Rotating View
                    </div>
                </div>
            </div>
        </div>
    );
}
