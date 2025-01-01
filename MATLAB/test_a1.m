hyperParams.numQubitsTotal = 6;
numAngles = 2*hyperParams.numQubitsTotal; 

options = optimoptions("surrogateopt",...
    "MaxFunctionEvaluations",10, ...
    "PlotFcn","optimplotfval",...
    "InitialPoints",pi*ones(numAngles,1));

lb = repmat(-pi,numAngles,1);
ub = repmat(pi,numAngles,1);
% [angles,minEnergy] = surrogateopt(objFcn,lb,ub,[],[],[],[],[],options);

% cg = initGate(1:6,'+++---');

% rng default


thetai = (2 * pi) * rand(1,6) - pi;
phii = (2 * pi) * rand(1,4) - pi;

gates = [hGate(1:6); ryGate(1:6,thetai); ...
         cryGate(1,4,phii(1)); cryGate(6,5,phii(4)); ...
         cryGate(5,2,phii(2)); cryGate(3,6,phii(3))];

C = quantumCircuit(gates);
%C.plot

S = simulate(C);
formula(S);
histogram(S);

[states,P] = querystates(S);

p = probability(S,2,"1");
M = randsample(S,50);

T = table(M.Counts,M.Probabilities,M.MeasuredStates, ...
    VariableNames=["Counts","Probabilities","States"]);


