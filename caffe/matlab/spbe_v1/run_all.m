

%%% set method and bit range

if( strcmp(method,'ITQ') || strcmp(method,'BP') || strcmp(method,'CBE'))
    bits = [2.^(5:1:9) 960];
else
    bits = 2.^(5:1:12);
end

%%% do
vecMAP = zeros(length(bits), 1);
for i=1:length(bits)
    nbit = bits(i);
    % train and encoding
    [B_query B_base] = Coding(X_train, X_query, X_base, nbit, method);
    % evaluate mAP by hamming ranking. for one million data, time-consuming
    vecMAP(i) = Evaluate(B_query, B_base, truth_nn);
end

result.bits = bits;
result.map = vecMAP;
save(['results\' method '.mat'], 'result');

