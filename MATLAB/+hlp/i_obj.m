function [y] = i_obj(theta, pt_f, pt_c, C)
    n = length(theta);
    for k = 1:n
        C.Gates(end-(k-1)).Angles = theta(k);
    end

    S = simulate(C);
    [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
    kl1 = i_kldiverg(pt_f, po_f, false);
    kl2 = i_kldiverg(pt_c, po_c, false);
    % y = 3*kl1/7 + 4*kl2/7;
    y = kl1 + kl2;
end
