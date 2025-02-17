load 'lite_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824.mat'

CancerGenes = ["TGFBR2","SMAD3","HIF1A","PDGFB"];
FibroblastGenes = ["TGFB1", "IL6", "PDGFRB"];

hx=gui.myFigure;

nexttile
batquery = "Cancer Cells (Mo)";
genquery = CancerGenes;
[patn, X] = e_getnX(batquery, genquery, sce);

assert(sum(patn)==size(X,2))

f0 = sum(X,2)./size(X,2);   % per gene initial activate freq.
pt = patn./sum(patn);             % target cell state freq.    


% ------------


n = length(f0);
layer1 = [];
for k=1:n, layer1 = [layer1; ryGate(k,2*asin(sqrt(f0(k))))]; end

% theta0 = pi*((rand(n,1)*2)-1);
theta0 = zeros(n,1);


layer2 = [];
for k=1:n, layer2 = [layer2; rxGate(k, theta0(k))]; end

C = quantumCircuit([layer1; layer2]);

nexttile
plot(C);


S = simulate(C);
[states,po] = querystates(S);

nexttile
bar(po)
set(gca,'XTick',1:length(states));
set(gca,'XTickLabel',states);
ylabel('# of cells');
xlabel('Expression pattern');

% nexttile
% batquery = "Cancer Cells (Co)";
% genquery = CancerGenes;
% [patn, X] = e_getnX(batquery, genquery, sce);

hx.FigureHandle.Position = [2.4650    0.2077    1.2700    0.4200]*1000;
hx.show;