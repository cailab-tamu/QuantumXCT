run('s0_merge_subunit_genes.m')
CancerGenes = ["STAT3","IL6RorST","TGFBR1or2","PDGFB"];
FibroblastGenes = ["TGFB1", "PDGFRB", "IL6"];

batquery = "Cancer Cells (Mo)";
genquery = CancerGenes;


[pt, ~, ~, f0] = getn(batquery, genquery, sce, false);

%f0 = sum(X,2)./size(X,2);         % per gene initial activate freq.
%pt = patn./sum(patn);             % target cell state freq.    


figure;

testgenes = genquery;
nexttile
bar(f0)
set(gca,'XTick',1:length(f0));
set(gca,'XTickLabel', testgenes);
title('Frequency of Gene')

nexttile
p=e_patnfreq(f0);
bar(p);
set(gca,'XTick',1:length(p));
title('Theoretical Frequencies')

nexttile
bar(pt-p);
set(gca,'XTick',1:length(p));
title('Observed-Theoretical')


% ------------

n = length(f0);
layer1 = [];
for k=1:n, layer1 = [layer1; ryGate(k,2*asin(sqrt(f0(k))))]; end
%theta0 = pi*((rand(n,1)*2)-1);
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

[kl]=i_kldiverg(pt, po);

title(sprintf('Quantum Simulation - KL = %f',kl));


% hx.Position = [2.4650    0.2077    1.2700    0.4200]*1000;
% hx.show;

nexttile
bar(pt-po);
set(gca,'XTick',1:length(p));
title('Observed-QSimulated')


nexttile
bar(p-po);
set(gca,'XTick',1:length(p));
title('Theoretical-QSimulated')
ylim([-0.025 0.025]);
