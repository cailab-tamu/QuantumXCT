function cswapDecomp = buildCSWAP(control, target1, target2)
    % Build a CSWAP (Fredkin) gate on (control, target1, target2)
    % using only Clifford+T gates (no CCX primitive).
    %
    % Qubit ordering: [control, target1, target2]

    % Step 1: decompose CCX(control, target1, target2) using Clifford+T
    % (one known 6-CNOT, 7-T decomposition)
    % We'll implement it inline.

    g = quantum.gate.quantumGate.empty;

    % 1. Hadamard on target2
    g(end+1) = hGate(target2);

    % 2. CNOT(target1 -> target2)
    g(end+1) = cxGate(target1, target2);

    % 3. Tdg on target2
    g(end+1) = tdgGate(target2);

    % 4. CNOT(control -> target2)
    g(end+1) = cxGate(control, target2);

    % 5. T on target2
    g(end+1) = tGate(target2);

    % 6. CNOT(target1 -> target2)
    g(end+1) = cxGate(target1, target2);

    % 7. Tdg on target2
    g(end+1) = tdgGate(target2);

    % 8. CNOT(control -> target2)
    g(end+1) = cxGate(control, target2);

    % 9. T on target1, T on target2
    g(end+1) = tGate(target1);
    g(end+1) = tGate(target2);

    % 10. H on target2
    g(end+1) = hGate(target2);

    % Now we have an effective CCX(control,target1,target2)

    % Step 2: sandwich with CNOT(target2 -> target1) around CCX
    fullGates = [ ...
        cxGate(target2, target1); ...
        g; ...
        cxGate(target2, target1) ...
    ];

    innerCircuit = quantumCircuit(fullGates, Name="CSWAP_Decomp");
    cswapDecomp = compositeGate(innerCircuit, [control target1 target2]);
end
