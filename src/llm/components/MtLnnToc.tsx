import React from 'react';
import s from './TocDiagram.module.scss';
import { Phase } from '../walkthrough/Walkthrough';
import { jumpToPhase } from '../Commentary';
import { useProgramState } from '../Sidebar';

export const MtLnnToc: React.FC<{ activePhase: Phase | null }> = ({ activePhase }) => {
    const progState = useProgramState();
    const { walkthrough } = progState;

    const entries = [
        { id: Phase.MTLNN_Intro, title: 'Introduction' },
        { id: Phase.MTLNN_Tokens, title: '1. Tokens & Setup' },
        { id: Phase.MTLNN_Architecture, title: '2. Architecture' },
        { id: Phase.MTLNN_Gates, title: '3. Gate Weights' },
        { id: Phase.MTLNN_Gwtb, title: '4. GWTB Memory' },
        { id: Phase.MTLNN_Wm, title: '5. WM Attention' },
        { id: Phase.MTLNN_Output, title: 'Conclusion' },
    ];

    return (
        <div style={{ padding: '20px', color: '#111' }}>
            <div className={s.tocTitle}>Table of Contents</div>
            <div className={s.toc}>
                {entries.map((entry, idx) => {
                    const isActive = entry.id === activePhase;
                    return (
                        <div
                            key={idx}
                            onClick={() => jumpToPhase(walkthrough, entry.id)}
                            style={{
                                cursor: 'pointer',
                                padding: '8px 0',
                                fontWeight: isActive ? 'bold' : 'normal',
                                color: isActive ? '#000' : '#666',
                                fontSize: '14px',
                            }}
                        >
                            {entry.title}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
