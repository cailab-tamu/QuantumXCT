load 'lite_PDGF_signaling_slows_ovarian_cancer GSM7019822_GSM7019823_GSM7019824.mat'

CancerGenes = ["TGFBR2","SMAD3","HIF1A","PDGFB"];
FibroblastGenes = ["TGFB1", "IL6", "PDGFRB"];

figure;
nexttile
batquery = "Cancer Cells (Mo)";
genquery = CancerGenes;
[n1] = getn(batquery, genquery, sce);
nexttile
batquery = "Cancer Cells (Co)";
genquery = CancerGenes;
[n2] = getn(batquery, genquery, sce);

%figure;
nexttile
batquery = "Fibroblasts (Mo)";
genquery = FibroblastGenes;
[n3] = getn(batquery, genquery, sce);

nexttile
batquery = "Fibroblasts (Co)";
genquery = FibroblastGenes;
[n4] = getn(batquery, genquery, sce);

function [n] = getn(batquery, genquery, sce)
    idxc = sce.c_batch_id == batquery;
    [y, idxg] = ismember(genquery, sce.g);
    assert(all(y))
    X = 0+(sce.X(idxg, idxc)>0);
    n = frebar(X, batquery+" - "+strtrim(sprintf("%s, ",genquery)));
end

function [n]=frebar(X, titx)
    % [M]=qtm.permn([0 1],size(X,1));
    [M] = permn([0 1], size(X,1));
    y=0+(X>0).';
    [~,idx]=ismember(y,M,'rows');
    t=tabulate(idx);
    n=zeros(size(M,1),1);
    n(t(:,1))=t(:,2);
    %figure;
    %subplot(2,2,1)
    bar(n)
    for k=1:size(M,1)
        txt{k}=sprintf('%d',M(k,:));
    end
    txt=string(txt);
    set(gca,'XTick',1:size(M,1));
    set(gca,'XTickLabel',txt);
    ylabel('# of cells');
    xlabel('Expression pattern');
    title(titx)
end