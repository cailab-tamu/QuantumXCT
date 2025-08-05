function [kl]=i_kldiverg(pt,po,rm0)

    if nargin<3, rm0=false; end   % remove all-zero state or not
    if rm0
        pt=pt(2:end);
        po=po(2:end);
        pt=pt./sum(pt);
        po=po./sum(po);
    end
    KL1 = sum(po .* (log(po)-log(pt)));
    KL2 = sum(pt .* (log(pt)-log(po)));
    % kl = max([KL1 KL2]);
    kl=10*mean([KL1 KL2]);
end