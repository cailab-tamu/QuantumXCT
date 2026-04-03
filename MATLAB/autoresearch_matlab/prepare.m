% prepare.m  –  shared setup for train.m.
% Run this script first; it populates the workspace with all common
% variables and defines the cost function used in every experiment.
%
% Outputs
%   targettop      – ground-truth interaction topology  [K x 2]
%   FibroblastGenes, CancerGenes, FCGenes  – gene name vectors
%   pt_f_mo, pt_c_mo  – monoculture probability distributions
%   pt_f_co, pt_c_co  – co-culture  probability distributions
%   f0_f_mo, f0_c_mo  – per-gene expression frequencies (monoculture)
%   cg1_mapped, cg2_mapped  – state-initialisation composite gates
%   states_f, states_c      – qubit state labels (no entanglement)
%   po_f_mo,  po_c_mo       – baseline state probabilities (no entanglement)
%   extragates  – fixed intracellular CX(4,5) gate
%   costfn      – @(po_f, po_c) one-way KL cost  (scalar)

% make +hlp/ package visible from this subdirectory
addpath('..');

import hlp.*

%% 1. Load dataset
a = pwd;
targettop = [7 2; 1 6; 3 5];
cd('../../manuscript/Dataset_8_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824/new_analysis/');
run('s0_merge_subunit_genes.m')
cd(a);

%% 2. Gene sets
FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];
CancerGenes     = ["STAT3", "IL6RorST", "TGFBR1or2", "PDGFB"];
FCGenes         = [FibroblastGenes CancerGenes];

%% 3. Pattern frequency distributions from single-cell data
[pt_f_mo, ~, ~, f0_f_mo] = hlp.getn("Fibroblasts (Mo)",  FibroblastGenes, sce, false);
[pt_c_mo, ~, ~, f0_c_mo] = hlp.getn("Cancer Cells (Mo)", CancerGenes,     sce, false);
[pt_f_co, ~, ~, ~]        = hlp.getn("Fibroblasts (Co)",  FibroblastGenes, sce, false);
[pt_c_co, ~, ~, ~]        = hlp.getn("Cancer Cells (Co)", CancerGenes,     sce, false);

%% 4. Quantum state initialisation (monoculture amplitudes)
% qubits 1-3: fibroblast   |   qubits 4-7: cancer
cg_f = initGate(1:3, sqrt(pt_f_mo));   % CompositeGate for 3 qubits
cg_c = initGate(1:4, sqrt(pt_c_mo));   % CompositeGate for 4 qubits

cg1_mapped = compositeGate(cg_f, [1 2 3]);
cg2_mapped = compositeGate(cg_c, [4 5 6 7]);

%% 5. Baseline simulation (no inter-cellular entanglement)
C = quantumCircuit([cg1_mapped; cg2_mapped]);
S = simulate(C);
[states_f, po_f_mo] = querystates(S, [1 2 3]);
[states_c, po_c_mo] = querystates(S, [4 5 6 7]);

%% 6. Fixed intracellular gate (STAT3 – IL6RorST within cancer cells)
extragates = cxGate(4, 5);

%% 7. Cost function: one-way KL divergence (monoculture -> co-culture)
% Usage:  Y(idx) = costfn(po_f, po_c)
%   po_f  – simulated fibroblast state probabilities  (querystates output)
%   po_c  – simulated cancer cell state probabilities (querystates output)
costfn = @(po_f, po_c) hlp.i_kldiverg(pt_f_co, po_f) + ...
                        hlp.i_kldiverg(pt_c_co, po_c);
