function mAP = Evaluate(B_query, B_base, truth_nn)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B_query and B_base are the compressed binray codes (8 bits compressed to 1 uint8)
% B_query: query_num * (bit/8)
% B_base: base_num * (bit/8)
% truth_nn: query_num * base_num, groudtruth neighbors from the database,
% indexes are based on 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

query_num = size(B_query,1);
base_num = size(B_base,1);

% time and memory consuming
neighbors = hamm_nns(B_base', B_query', base_num);
fprintf('compute hamming distance ok\n');

vecAP = zeros(query_num,1);
parfor i=1:query_num
    vecAP(i) = AveragePrecision(neighbors(:,i), truth_nn(i,:));
end
mAP = mean(vecAP);
fprintf('mAP = %.4f\n', mAP);

