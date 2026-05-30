import React from 'react';
import { MtLnnLayerView } from '@/src/llm/MtLnnLayerView';

export const metadata = {
  title: 'MT-LNN 3D Visualization',
  description: 'MT-LNN (Microtubule Liquid Neural Network) - 3D WebGL Visualization',
};

export default function Page() {
    return <MtLnnLayerView />;
}
