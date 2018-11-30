function ap = AveragePrecision(rank_id, truth_id)
%%%%%%%%%%%%%%%%%%%%%%%
% compute the average precision of one query
% rank_id: the ranked idx
% truth_id: the groud-truth idx
%%%%%%%%%%%%%%%%%%%%%%%

truth_num = length(truth_id);
truth_pos = zeros(truth_num,1);
for j=1:truth_num
    truth_pos(j) = find(rank_id == truth_id(j));
end
truth_pos = sort(truth_pos, 'ascend');

% compute average precision as the area below the recall-precision curve
ap = 0;
delta_recall = 1/truth_num;
for j=1:truth_num
    p = j/truth_pos(j);
    ap = ap + p*delta_recall;
end
