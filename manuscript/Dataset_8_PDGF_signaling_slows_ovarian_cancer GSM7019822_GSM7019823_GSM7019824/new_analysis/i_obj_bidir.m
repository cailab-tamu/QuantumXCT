function [y] = i_obj_bidir(theta, pt_f, pt_c, C)
    n = length(theta);
    for k = 1:n
        C{1}.Gates(end-(k-1)).Angles = theta(k);
        C{2}.Gates(end-(k-1)).Angles = theta(k);
        %C.Gates(end-1).Angles = theta(2);
        %C.Gates(end-2).Angles = theta(3);
        %C.Gates(end-3).Angles = theta(4);
    end

    S = simulate(C{1});
    [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
    kl1 = i_kldiverg(pt_f{1}, po_f);
    kl2 = i_kldiverg(pt_c{1}, po_c);
    % y = 3*kl1/7 + 4*kl2/7;
    y1 = kl1 + kl2;

    S = simulate(C{2});
    [~, po_f] = querystates(S,[1 2 3]);     % observed state pattern in fibroblast
    [~, po_c] = querystates(S,[4 5 6 7]);   % observed state pattern in cancer   
    kl1 = i_kldiverg(pt_f{2}, po_f);
    kl2 = i_kldiverg(pt_c{2}, po_c);
    % y = 3*kl1/7 + 4*kl2/7;
    y2 = kl1 + kl2;

    y = y1 + y2;
    
end
