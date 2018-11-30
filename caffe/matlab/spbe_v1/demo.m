%%%% open matlab pool to run multithread code
matlabpool
% 
%%%% load data and settings. the same for all the methods. 
%%%% comment the following line to avoid redundant loading
LoadDataGist;

method = 'lsh';
run_all;

method = 'itq';
run_all;

method = 'bp';
run_all;

method = 'cbe';
run_all;

method = 'SP';
run_all;

draw_figure;