function t = fun_drawreshisto(topk, Cc, configsK, ...
    pt_c_mo, pt_c_co, pt_f_mo, pt_f_co, ...
    FCGenes, states_c, states_f, targettop)

    C = Cc{topk};
    S = simulate(C);

    n1=log2(length(pt_f_mo));
    n2=log2(length(pt_c_mo));
    [states_f2, po_f2] = querystates(S,1:n1);         % observed state pattern in fibroblast
    [states_c2, po_c2] = querystates(S,n1+1:n1+n2);   % observed state pattern in cancer   
    
    figure;
    t = tiledlayout(2, 2);
    nexttile
    plot(C)
    
    nexttile
    for k = 1:length(configsK{topk})
        
        if ismember(configsK{topk}(k,:), targettop, 'rows')
            prefixa = "*";
        else
            prefixa = "";
        end
        x = prefixa + " " + configsK{topk}(k, 1) +" "+ FCGenes(configsK{topk}(k, 1))+...
            " -> "+configsK{topk}(k, 2)+" "+FCGenes(configsK{topk}(k, 2));

        text(0, 50 - k*10, x);
        ylim([0 50]);
        axis off
    end
    %text(0, 50 - k*10 - 10, '------------------------');
    %text(0, 50 - k*10 - 20, '[7->2; 1->6; 3->5; 5->4]');
    
    nexttile
    % pt_f_mo'
    % pt_f_co'
    % po_f2'

    if isequal(size(pt_f_mo), size(pt_f_co)) && isequal(size(pt_f_mo), size(po_f2))
        bar([pt_f_mo pt_f_co po_f2])
        set(gca,'XTick',1:length(states_f));
        set(gca,'XTickLabel',states_f);
        ylabel('Freq. of cells');
        xlabel('Expression pattern');
        title('Cell Type 1')
    end

    nexttile
    bar([pt_c_mo pt_c_co po_c2])
    set(gca,'XTick',1:length(states_c));
    set(gca,'XTickLabel',states_c);
    ylabel('Freq. of cells');
    xlabel('Expression pattern');
    title('Cell Type 2')
