% swaptest_matlab_example.m
% Estimate swap-test cost C(theta) for |psi> = |0>, |phi> = |+>, U(theta)=RY(theta)
% Requires: MATLAB Support Package for Quantum Computing (quantumCircuit, hGate, ryGate, ccxGate, compositeGate, simulate)

%function swaptest_matlab_example()
    % parameters
    thetas = linspace(0, 2*pi, 13);   % sample points
    nThetas = numel(thetas);
    shots = 0;                        % 0 -> use exact statevector (no sampling). Set >0 to simulate sampling.

    % Pre-construct the controlled-SWAP (CSWAP / Fredkin) as an inner circuit:
    % Known decomposition: CSWAP can be implemented using 3 Toffoli (ccx) gates
    % inner qubit ordering: [control, target1, target2]
    innerGates = [ ...
        ccxGate(1,2,3); ...
        ccxGate(1,3,2); ...
        ccxGate(1,2,3) ...
    ];
    innerCircuit = quantumCircuit(innerGates, Name="CSWAP_inner");
    % Map inner circuit (3 qubits) onto outer circuit qubits when used as composite
    % We'll map [1 2 3] directly (ancilla=1, dataA=2, dataB=3)
    cswapComposite = compositeGate(innerCircuit, [1 2 3]);

    % Prepare storage
    C_sim = zeros(1, nThetas);
    C_analytic = zeros(1, nThetas);

    % Loop over thetas
    for k = 1:nThetas
        theta = thetas(k);

        % Build the 3-qubit swap-test circuit:
        % qubit indices: ancilla=1, dataA=2, dataB=3
        gates = [ ...
            % Prepare |phi> on qubit 2: |+> = H|0>
            hGate(2); ...
            % Prepare U(theta)|psi> on qubit 3: RY(theta) on |0>
            ryGate(3, theta); ...
            % Prepare ancilla superposition
            hGate(1); ...
            % Controlled-SWAP with ancilla as control and data qubits 2 and 3 as swap targets
            cswapComposite; ...
            % Final H on ancilla
            hGate(1) ...
        ];

        qc = quantumCircuit(gates);

        % Simulate exact final statevector
        finalState = simulate(qc);

        % finalState.Amplitudes is a 2^3 vector ordered as |q1 q2 q3> with q1=top index
        amps = finalState.Amplitudes;

        % Compute probability ancilla == 0 by summing amplitudes where top (ancilla) bit = 0
        % Basis ordering: |q1 q2 q3> with q1 the most significant bit.
        nQ = 3;
        p0 = 0;
        for idx = 1:numel(amps)
            % zero-based index:
            zidx = idx - 1;
            % extract ancilla bit (most significant bit)
            ancilla_bit = bitget(zidx, nQ); % bitget uses 1=LSB; ancilla is MSB (=bit nQ)
            if ancilla_bit == 0
                p0 = p0 + abs(amps(idx))^2;
            end
        end

        % Cost from swap test: |<phi|U|psi>|^2 = 2 p0 - 1, so C = 1 - overlap^2 = 2(1 - p0)
        overlap2 = 2*p0 - 1;
        C_sim(k) = 1 - overlap2;
        % analytic expression for our chosen states:
        % |psi> = |0>, |phi> = |+>, U = RY(theta) -> overlap^2 = (1 + sin(theta))/2
        C_analytic(k) = (1 - sin(theta))/2;
    end

    % Print table
    fprintf(' theta (rad)   C_sim       C_analytic   p0 (ancilla=0)\n');
    for k = 1:nThetas
        theta = thetas(k);
        % recover p0 from C_sim: p0 = 1 - C_sim/2
        p0_est = 1 - C_sim(k)/2;
        fprintf('%8.3f     %8.5f    %8.5f     %8.5f\n', theta, C_sim(k), C_analytic(k), p0_est);
    end

    % Plot comparison
    figure;
    plot(thetas, C_sim, 'o-', 'LineWidth', 1.2);
    hold on;
    plot(thetas, C_analytic, 'x--', 'LineWidth', 1.2);
    xlabel('\theta (rad)');
    ylabel('C(\theta)');
    legend('Simulated (exact statevector)','Analytic','Location','Best');
    title('Swap-test cost: simulated vs analytic');
    grid on;
%end
