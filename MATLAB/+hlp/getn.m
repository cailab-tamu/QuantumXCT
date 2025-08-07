function [n, X, n_expected, f0obs] = getn(batquery, genquery, sce, plotit, realx)
if nargin<5, realx = false; end
if nargin<4, plotit = true; end

    idxc = sce.c_batch_id == batquery;
    [y, idxg] = ismember(genquery, sce.g);
    assert(all(y))
    if ~realx
        X = 0+(sce.X(idxg, idxc)>0);
    else
        X = sce.X(idxg, idxc);
    end
    
    f0obs = sum(X,2)./size(X,2);
    n_expected = e_patnfreq(f0obs);
    n = frebar(X, batquery+" - "+strtrim(sprintf("%s, ",genquery)), plotit);
    % n = n./sum(n);
end

function [n]=frebar(X, titx, plotit)

    if nargin<3, plotit = true; end
    f0obs = sum(X,2)./size(X,2);
    n_expected = e_patnfreq(f0obs);

    % [M]=qtm.permn([0 1],size(X,1));
    [M] = permn([0 1], size(X,1));
    y=0+(X>0).';
    [~,idx]=ismember(y,M,'rows');
    t=tabulate(idx);
    n=zeros(size(M,1),1);
    n(t(:,1))=t(:,2);
    %figure;
    %subplot(2,2,1)
    n = n./sum(n);

    if plotit
        bar([n, n_expected]);
        for k=1:size(M,1)
            txt{k}=sprintf('%d',M(k,:));
        end
        txt=string(txt);
        set(gca,'XTick',1:size(M,1));
        set(gca,'XTickLabel',txt);
        ylabel('# of cells');
        % xlabel('Expression pattern');
        title(titx)
        legend({'Observed','Theoretical'})
        [kl]=i_kldiverg(n, n_expected);
        xlabel(sprintf('KL = %g',kl));
    end
end

