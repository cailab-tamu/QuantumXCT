function [layer] = in_12layers(f0)
    n = length(f0);
    layer1 = ryGate(1:n, 2*asin(sqrt(f0)));
    % layer2 = rxGate(1:n, zeros(n,1));
    layer = [layer1];
end
