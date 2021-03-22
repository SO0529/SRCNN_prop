function make_PSNR_curve_parallel_x4()

% close all;
% clear all;
% 
% run_parallel_x4('Set5','CPU',4900000,5000000);
% 
% close all;
% clear all;
% 
% run_parallel_x4('Set14','CPU',4900000,5000000);
% 
% close all;
% clear all;
% 
% run_parallel_x4('original_images','CPU',4900000,5000000);
% 
% close all;
% clear all;
% 
% run_parallel_x4('Set5','GPU',100000,5000000);
% 
% close all;
% clear all;
% 
% run_parallel_x4('Set14','GPU',100000,5000000);
% 
% close all;
% clear all;
% 
% run_parallel_x4('original_images','GPU',100000,5000000);


close all;
clear all;

run_parallel_x4('Set5','GPU',100000,100000);
