function run_parallel_x2(target, CPUorGPU, start, stop)

up_scale = 2;   
shrink_gnd = 10;
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

model_dir = '../../Training/x2';
base_weights_A = sprintf('%s/model_553_A_x2/553_A_x2_iter_', model_dir);
base_weights_B = sprintf('%s/model_553_B_x2/553_B_x2_iter_', model_dir);
base_weights_C = sprintf('%s/model_553_C_x2/553_C_x2_iter_', model_dir);
base_weights_D = sprintf('%s/model_553_D_x2/553_D_x2_iter_', model_dir);
model = '../prop/x2/SRCNN_deploy_553_parallel_x2.prototxt';
save_name = '553_parallel';
folder_name = sprintf('553_parallel/PSNR_553_parallel_%s_%s', target, CPUorGPU);

if ~exist(sprintf('../prop/x2/%s', folder_name), 'dir')
    mkdir(sprintf('../prop/x2/%s', folder_name));
end

average_psnr_file = sprintf('../prop/x2/%s/all_psnr.txt', folder_name);
average_fid = fopen(average_psnr_file, 'wt');

for iter = start:100000:stop
    fprintf('Calculate PSNR : %d\n', iter);  
    
    weights_A = sprintf('%s%d.caffemodel', base_weights_A, iter);
    weights_B = sprintf('%s%d.caffemodel', base_weights_B, iter);
    weights_C = sprintf('%s%d.caffemodel', base_weights_C, iter);
    weights_D = sprintf('%s%d.caffemodel', base_weights_D, iter);
    psnr_dir = sprintf('../prop/x2/%s/%d', folder_name, iter);
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
        net_A = caffe.Net(model, weights_A, 'test');
        net_B = caffe.Net(model, weights_B, 'test');  
        net_C = caffe.Net(model, weights_C, 'test');  
        net_D = caffe.Net(model, weights_D, 'test');  
        
        ttime = tic;
        tic;
        
        im_in = permute(im_l_y, [2 1]);

        input = {im_in};
        
        [hei, wid] = size(im_in);

        net_A.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
        net_A.reshape(); 
        recon.A = cell2mat(net_A.forward(input));

        net_B.blobs('data').reshape([hei wid 1 1]);
        net_B.reshape(); 
        recon.B = cell2mat(net_B.forward(input));
        
        net_C.blobs('data').reshape([hei wid 1 1]);
        recon.C = cell2mat(net_C.forward(input));
        
        net_D.blobs('data').reshape([hei wid 1 1]);
        net_D.reshape(); 
        recon.D = cell2mat(net_D.forward(input));

        out.A = permute(recon.A, [2 1]);
        out.B = permute(recon.B, [2 1]);
        out.C = permute(recon.C, [2 1]);
        out.D = permute(recon.D, [2 1]);

        % output is double
        im_h_y= composition_x2(out.A, out.B, out.C, out.D);
        
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
