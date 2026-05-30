// Standalone enum module to break the import cycle between Walkthrough.ts
// and MtLnnToc.tsx (the latter is reachable from Walkthrough's transitive
// imports via Commentary -> Sidebar -> ...). Keeping enums in a dependency-
// free leaf module guarantees they're fully initialized at first read.

export enum PhaseGroup {
    Intro,
    Detailed_Input,
    MTLNN,
}

export enum Phase {
    None,

    Intro_Intro,
    Input_First,
    Input_Detail_Tables,
    Input_Detail_TokEmbed,
    LayerNorm1,
    Intro_Prelim,
    Input_Detail_Embedding,
    Input_Detail_LayerNorm,
    Input_Detail_SelfAttention,
    Input_Detail_Softmax,
    Input_Detail_Projection,
    Input_Detail_Mlp,
    Input_Detail_Transformer,
    Input_Detail_Output,

    MTLNN_Intro,
    MTLNN_Tokens,
    MTLNN_Architecture,
    MTLNN_Gates,
    MTLNN_Gwtb,
    MTLNN_Wm,
    MTLNN_Output,
}
