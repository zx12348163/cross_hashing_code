function [B_query B_base] = Coding(X_train, X_query, X_base, bit, method, sparsity)

% training by X_train
% encoding X_query/X_base into B_query/B_base
% B_query and B_base are compact codes: 8 bits in one uint8
%
% ''method'' can be
% LSH  (Locality Sensitive Hashing)
% ITQ  (Gong et al. CVPR 2011)
% BP (Gong et al. CVPR 2013)
% CBE (Yu et al. ICML 2014)
%
% Note that methods implemented here are for comparing the map
% not for evaluting the computational time

if(nargin < 6)
    sparsity = 0.9;
end

fprintf('coding by %s to %d bits\n', method, bit);

dim = size(X_train, 2);
switch(method)
    case 'LSH'
        R = randn(dim, bit);
        
        % coding
        B_query = (X_query*R >= 0);
        B_base = (X_base*R >=0);                
        B_query = compactbit(B_query);
        B_base = compactbit(B_base);
        
    case 'ITQ'        
        if (bit > dim)
            fprintf('the bit num for ITQ must no greater than dim\n');
            return;
        end
        
        % PCA
        [pc, ~] = eigs(cov(X_train),bit);
        X_train_pc = X_train * pc;     
        % train
        [~, R] = ITQ(X_train_pc,50);
        R = pc * R;
        
        % coding
        B_query = (X_query*R >= 0);
        B_base = (X_base*R >=0); 
        B_query = compactbit(B_query);
        B_base = compactbit(B_base);
        
        fprintf('non-zero elements %.2f%%\n', 100*nnz(R)/numel(R));
           
    case 'BP'
        % determine b1 and b2 for bit
        d = bit;
        n = 1: d;
        m = d./n;
        idx = find(abs(m - round(m)) > 0.000001);
        m(idx) = [];
        n(idx) = [];
        [tmp, idx] = min(abs(m-n));
        b2 = m(idx);
        b1 = n(idx);
        
        % reshape and train
        X_train2 = TensorFV(X_train);
        [R1,R2] = BilinearITQ_low(X_train2, b1, b2, 20);
        
        % coding
        B_query = BilinearITQ_coding(X_query, R1, R2);
        B_base = BilinearITQ_coding(X_base, R1, R2);         
        B_query = compactbit(B_query);
        B_base = compactbit(B_base);
            
    case 'CBE'
        % train
        % for all bit<dim, CBE use one model. so just train it once
        model_file = 'results\cbe.model';
        if(exist(model_file,'file'))
            model = importdata(model_file);
        else
            para.lambda = 1;
            para.verbose = 0;
            train_size = min(size(X_train,1), 5000);
            [~, model] = circulant_learning(X_train(1:train_size,:), para);
            save(model_file, 'model');
        end
        
        % coding
        B_query = CBE_prediction(model, X_query);
        B_base = CBE_prediction(model, X_base);        
        if (bit < dim)
            B_query = B_query (:, 1:bit);
            B_base = B_base (:, 1:bit);
        end
        B_query = compactbit(B_query);
        B_base = compactbit(B_base);
    
    case 'SP'        
        % train
        R = SP(X_train, bit, sparsity, 50);
                
        % coding
        B_query = (X_query*R' >= 0);
        B_base = (X_base*R' >=0); 
        B_query = compactbit(B_query);
        B_base = compactbit(B_base);
       
        fprintf('non-zero elements %.2f%%\n', 100*nnz(R)/numel(R));
        
    otherwise
        fprintf('not recognized method %s\n', method);
end

fprintf('coding done\n');