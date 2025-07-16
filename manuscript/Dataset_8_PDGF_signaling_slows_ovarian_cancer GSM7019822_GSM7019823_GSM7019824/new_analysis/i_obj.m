function [y] = i_obj(theta, pt_f, pt_c, C)
    C.Gates(end).Angles = theta(1);
    C.Gates(end-1).Angles = theta(2);
    C.Gates(end-2).Angles = theta(3);
    C.Gates(end-3).Angles = theta(4);
    S = simulate(C);
    [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
    kl1 = i_kldiverg(pt_f, po_f);
    kl2 = i_kldiverg(pt_c, po_c);
    y = kl1+kl2;
end
