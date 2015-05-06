function [model] = ELM_AE(X, T, param)
%%%X (dxN) samples are in columns
[num_InputNeurons, num_TrainingSamples] = size(X);
num_outputNeurons = num_InputNeurons;
no_hidden_Layers = param.numHiddenLayers;
Weights = cell(no_hidden_Layers+1,1);
Hn = [num_InputNeurons,param.Hn];
C = param.C;
sig = param.sig;
InputData = X;
rng(1);
for i=1:1:no_hidden_Layers    
    InputW=rand(Hn(i+1),Hn(i))*2 -1;    %InputWeight=rand(HN(i+1),HN(i));
    if Hn(i+1) > Hn(i)
        InputW = orth(InputW);
    else
        InputW = orth(InputW')'; 
    end    
    Bias=rand(Hn(i+1),1)*2 -1;
    Bias=orth(Bias);    
    tempH=InputW*InputData;      
    clear InputWeight;
    ind=ones(1,num_TrainingSamples);
    BiasMatrix=Bias(:,ind); 
    tempH=tempH+BiasMatrix;    
    clear BiasMatrix Bias;  
    H = 1 ./ (1 + exp(-sig(i)*tempH));
    clear tempH;                                   
    %%%%%%%%%%% Calculate output weights (beta_i)
    if Hn(i+1) == Hn(i)
        [~,Weights{i},~] = fun_Procrust(InputData',H');
    else
        if C(i) == 0
            Weights{i} =pinv(H') * InputData';                        
        else          
            Weights{i}=  ((1/C(i)*eye(size(H,1)))+(H*H'))\(H * InputData');            
        end
    end    
    tempH=(Weights{i}) *(InputData);
    clear InputData;    
    if Hn(i+1) == Hn(i)
        InputData = tempH;
    else        
        InputData =  1 ./ (1 + exp(-sig(i)*tempH));
    end    
    clear tempH H;
end  %%%%End of Autoencoder

%%Final Layer
if C(no_hidden_Layers+1) == 0
    Weights{no_hidden_Layers+1}=pinv(InputData') * T';
else
    Weights{no_hidden_Layers+1}=(eye(size(InputData,1))/C(no_hidden_Layers+1)+InputData * InputData') \ ( InputData * T');
end
clear InputData H;
model.no_hidden_Layers = no_hidden_Layers;
model.W = Weights;
model.Hn = Hn;
model.sig = sig;
end

