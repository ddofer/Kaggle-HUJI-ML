X = rand(20,100); 
param.numHiddenLayers = 2;
param.Hn = [40,40];
param.C = [10000,0,100000,0,0,0];    
sg = 0.1;   
param.sig = [sg,sg,sg,sg,sg];    
model = ELM_AE(X,X, param);
p = rand(20,1);
TY= ELM_AE_Reconstruct(p, model);
    
