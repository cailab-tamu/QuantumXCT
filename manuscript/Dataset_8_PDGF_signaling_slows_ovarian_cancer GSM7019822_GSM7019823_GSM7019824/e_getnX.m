function [patn, X] = e_getnX(batquery, genquery, sce, plotit)
    if nargin<4, plotit = true; end
    idxc = sce.c_batch_id == batquery;
    [y, idxg] = ismember(genquery, sce.g);
    assert(all(y))
    X = 0+(sce.X(idxg, idxc)>0);
    patn = frebar(X, batquery+" - "+strtrim(sprintf("%s, ",genquery)), plotit);
end

function [n] = frebar(X, titx, plotit)
    [M]=qtm.permn([0 1],size(X,1));
    % [M] = permn([0 1], size(X,1));
    y=0+(X>0).';
    [~,idx]=ismember(y,M,'rows');
    t=tabulate(idx);
    n=zeros(size(M,1),1);
    n(t(:,1))=t(:,2);
    %figure;
    %subplot(2,2,1)
    if plotit
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
end