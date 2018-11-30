%%% suppose the mAP of all methods have been stored
methods = {'LSH','ITQ','BP', 'CBE', 'SP'};
colors = {'b','k','m','g','r','c'};

figure;
bit_min = 2^20;
bit_max = 0;
map_max = 0;
for i=1:length(methods)
    method = methods{i};
    file = ['results\' method '.mat'];
    load(file); % load to result
    semilogx(result.bits, result.map, '*-', 'Color', colors{i});hold on;
    bit_min = min(min(result.bits), bit_min);
    bit_max = max(max(result.bits), bit_max);
    map_max = max(max(result.map), map_max);
    fprintf('method:');disp(method);
    fprintf('bits:');disp(result.bits);
    fprintf('mAP:');disp(result.map');
    fprintf('------------------------------------\n');
end
hlg = legend(methods);
set(hlg, 'Location', 'northwest');

x_bit = 2.^(log2(bit_min):1:log2(bit_max));
set(gca, 'XTick', x_bit);
xlim([bit_min bit_max]);
ylim([0 map_max]);
xlabel('bits');
ylabel('k-nn mAP');