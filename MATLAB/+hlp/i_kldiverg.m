function [kl]=i_kldiverg(pt,po,rm0)

    if nargin<3, rm0 = false; end   % remove all-zero state or not

    if rm0
        %disp('000 removed.')
        pt=pt(2:end);
        po=po(2:end);
    end

    epsilon = 1e-12;
    if any(po==0), po = po + epsilon; end
    if any(pt==0), pt = pt + epsilon; end


    pt = pt ./ sum(pt);  % Normalize pt
    po = po ./ sum(po);  % Normalize po

   

    KL1 = sum(po .* (log(po)-log(pt)));
    KL2 = sum(pt .* (log(pt)-log(po)));
    % kl = max([KL1 KL2]);
    kl=10*mean([KL1 KL2]);
end