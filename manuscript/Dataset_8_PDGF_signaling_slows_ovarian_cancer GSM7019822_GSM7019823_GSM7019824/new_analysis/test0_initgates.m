f=[0.3 0.2 0.5];
[layer_base] = in_12layers(f);
C1 = quantumCircuit(layer_base);
C2 = quantumCircuit(initGate(1:length(f), sqrt(e_patnfreq(f))));

norm(C1.getMatrix - C2.getMatrix)

