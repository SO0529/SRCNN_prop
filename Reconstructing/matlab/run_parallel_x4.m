function run_parallel_x4(target, CPUorGPU, start, stop)

up_scale = 4;   
shrink_gnd = 8;
shrink_srcnn = 0;

imgFileLocation = sprintf('../test_images/%s/', target);
imgListLocation = sprintf('../test_images/%s/*.bmp', target);
imgFileList = dir(imgListLocation);

imgNum = size(imgFileList);
imgFileNameList = cell(imgNum);
for i = 1 : imgNum(1)            
    imgFileName = char(imgFileList(i).name);
    imgFileNameList{i} = sprintf('%s%s', imgFileLocation, imgFileName);
end;

model_dir = '../../Training/x4';
base_weights_A = sprintf('%s/model_331_A_x4/331_A_x4_iter_', model_dir);
base_weights_B = sprintf('%s/model_331_B_x4/331_B_x4_iter_', model_dir);
base_weights_C = sprintf('%s/model_331_C_x4/331_C_x4_iter_', model_dir);
base_weights_D = sprintf('%s/model_331_D_x4/331_D_x4_iter_', model_dir);
base_weights_E = sprintf('%s/model_331_E_x4/331_E_x4_iter_', model_dir);
base_weights_F = sprintf('%s/model_331_F_x4/331_F_x4_iter_', model_dir);
base_weights_G = sprintf('%s/model_331_G_x4/331_G_x4_iter_', model_dir);
base_weights_H = sprintf('%s/model_331_H_x4/331_H_x4_iter_', model_dir);
base_weights_I = sprintf('%s/model_331_I_x4/331_I_x4_iter_', model_dir);
base_weights_J = sprintf('%s/model_331_J_x4/331_J_x4_iter_', model_dir);
base_weights_K = sprintf('%s/model_331_K_x4/331_K_x4_iter_', model_dir);
base_weights_L = sprintf('%s/model_331_L_x4/331_L_x4_iter_', model_dir);
base_weights_M = sprintf('%s/model_331_M_x4/331_M_x4_iter_', model_dir);
base_weights_N = sprintf('%s/model_331_N_x4/331_N_x4_iter_', model_dir);
base_weights_O = sprintf('%s/model_331_O_x4/331_O_x4_iter_', model_dir);
base_weights_P = sprintf('%s/model_331_P_x4/331_P_x4_iter_', model_dir);
model = '../prop/x4/SRCNN_deploy_331_parallel_x4.prototxt';
save_name = '331_parallel';
folder_name = sprintf('331_parallel/PSNR_331_parallel_%s_%s', target, CPUorGPU);

if ~exist(sprintf('../prop/x4/%s', folder_name), 'dir')
    mkdir(sprintf('../prop/x4/%s', folder_name));
end

average_psnr_file = sprintf('../prop/x4/%s/all_psnr.txt', folder_name);
average_fid = fopen(average_psnr_file, 'wt');

for iter = start:100000:stop
    fprintf('Calculate PSNR : %d\n', iter);  
    
    weights.A = sprintf('%s%d.caffemodel', base_weights_A, iter);
    weights.B = sprintf('%s%d.caffemodel', base_weights_B, iter);
    weights.C = sprintf('%s%d.caffemodel', base_weights_C, iter);
    weights.D = sprintf('%s%d.caffemodel', base_weights_D, iter);
    weights.E = sprintf('%s%d.caffemodel', base_weights_E, iter);
    weights.F = sprintf('%s%d.caffemodel', base_weights_F, iter);
    weights.G = sprintf('%s%d.caffemodel', base_weights_G, iter);
    weights.H = sprintf('%s%d.caffemodel', base_weights_H, iter);
    weights.I = sprintf('%s%d.caffemodel', base_weights_I, iter);
    weights.J = sprintf('%s%d.caffemodel', base_weights_J, iter);
    weights.K = sprintf('%s%d.caffemodel', base_weights_K, iter);
    weights.L = sprintf('%s%d.caffemodel', base_weights_L, iter);
    weights.M = sprintf('%s%d.caffemodel', base_weights_M, iter);
    weights.N = sprintf('%s%d.caffemodel', base_weights_N, iter);
    weights.O = sprintf('%s%d.caffemodel', base_weights_O, iter);
    weights.P = sprintf('%s%d.caffemodel', base_weights_P, iter);
    psnr_dir = sprintf('../prop/x4/%s/%d', folder_name, iter);
    if ~exist(psnr_dir, 'dir')
        mkdir(psnr_dir);
    end
    
    psnr_file = sprintf('%s/psnr_%d.txt', psnr_dir, iter);
    fid = fopen(psnr_file, 'wt');
    
    time_file = sprintf('%s/time_%d.txt', psnr_dir, iter);
    time_fid = fopen(time_file, 'wt');
    
    sum_psnr = 0;
    ave_psnr = 0;
    ave_time = 0;
    
    for data = 1:length(imgFileNameList)
       
        img_path = char(imgFileNameList(data));
        im_y = imread(img_path);
        im_y = modcrop(im_y, up_scale);
        if size(im_y,3)>1
            im_y = rgb2ycbcr(im_y);
            im_y = im_y(:,:,1);
        end
        
        im_l_y = imresize(im_y, 1/up_scale, 'bicubic');
        
        im_y = im2single(im_y);
        im_l_y = im2single(im_l_y);  
        
        if CPUorGPU == 'CPU'
            caffe.set_mode_cpu(); 
        else
            caffe.set_mode_gpu(); 
        end
        net.A = caffe.Net(model, weights.A, 'test');  
        net.B = caffe.Net(model, weights.B, 'test'); 
        net.C = caffe.Net(model, weights.C, 'test'); 
        net.D = caffe.Net(model, weights.D, 'test'); 
        net.E = caffe.Net(model, weights.E, 'test'); 
        net.F = caffe.Net(model, weights.F, 'test'); 
        net.G = caffe.Net(model, weights.G, 'test'); 
        net.H = caffe.Net(model, weights.H, 'test'); 
        net.I = caffe.Net(model, weights.I, 'test');   
        net.J = caffe.Net(model, weights.J, 'test'); 
        net.K = caffe.Net(model, weights.K, 'test'); 
        net.L = caffe.Net(model, weights.L, 'test'); 
        net.M = caffe.Net(model, weights.M, 'test'); 
        net.N = caffe.Net(model, weights.N, 'test'); 
        net.O = caffe.Net(model, weights.O, 'test'); 
        net.P = caffe.Net(model, weights.P, 'test');
        
        ttime = tic;
        tic;
        
        im_h_y = parallel_x4(net, im_l_y);
        
        ttime = toc(ttime);
        toc;
        fprintf(time_fid, '%.5f\n', ttime);
        
        % calc psnr
        sr_sn = single(shave(im_h_y, [shrink_srcnn, shrink_srcnn]));
        ref_sn = single(shave(im_y, [shrink_gnd, shrink_gnd]));
        psnr_matlab = psnr(sr_sn, ref_sn, 1);
        fprintf(fid, '%.3f\n', psnr_matlab);
        
        sum_psnr = sum_psnr + psnr_matlab;

        im_h_y = shave(uint8(im_h_y*255), [shrink_srcnn, shrink_srcnn]);

        image_name = strrep(img_path, imgFileLocation, '');
        image_name = strrep(image_name, '.bmp', '');

        srcnn_name = sprintf('%s/%s_%s_%d.bmp', psnr_dir, image_name, save_name, iter);
        if iter == 100000 || iter == 5000000
        	imwrite(im_h_y, srcnn_name);
        end
        
        ave_psnr = ave_psnr + psnr_matlab;
        ave_time = ave_time + ttime;
     
        caffe.reset_all();
        pause(0.1);
    end
    ave_psnr = ave_psnr/length(imgFileNameList);
    ave_time = ave_time/length(imgFileNameList);
    fprintf(fid, '%.3f\n', ave_psnr);
    fprintf(time_fid, '%.5f\n', ave_time);
    fprintf(average_fid, '%.3f\n', sum_psnr/length(imgFileNameList));
    fclose(fid);
    fclose(time_fid);
    pause(0.1);
end
fclose(average_fid);
