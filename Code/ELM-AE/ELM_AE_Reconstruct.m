function [R] = ELM_AE_Reconstruct(testP, model)
Weights = model.W;
Hn = model.Hn;
sig = model.sig;
no_of_Layers =  model.no_hidden_Layers;
InputData = testP;
for i=1:1:no_of_Layers
    tempH_test=(Weights{i})*(InputData);
    if Hn(i+1) == Hn(i)
        InputData = tempH_test;
    else
        InputData =  1 ./ (1 + exp(-sig(i)*tempH_test));
    end
    clear tempH_test;    
end
R=(InputData' * Weights{no_of_Layers+1})';
end