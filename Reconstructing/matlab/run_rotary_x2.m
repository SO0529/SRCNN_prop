function run_rotary_x2(target, CPUorGPU, start, stop)

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
model = '../prop/x2/SRCNN_deploy_553_parallel_x2.prototxt';
save_name = '553_rotary';
folder_name = sprintf('553_rotary/PSNR_553_rotary_%s_%s', target, CPUorGPU);

if ~exist(sprintf('../prop/x2/%s', folder_name), 'dir')
    mkdir(sprintf('../prop/x2/%s', folder_name));
end

average_psnr_file = sprintf('../prop/x2/%s/all_psnr.txt', folder_name);
average_fid = fopen(average_psnr_file, 'wt');

for iter = start:100000:stop
    fprintf('Calculate PSNR : %d\n', iter);  
    
    weights_A = sprintf('%s%d.caffemodel', base_weights_A, iter);
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
        
        ttime = tic;
        tic;
        
        im_in = permute(im_l_y, [2 1]);
        im_in_90 = rot90(im_in, 1);
        im_in_180 = rot90(im_in, 2);
        im_in_270 = rot90(im_in, 3);

        input = {im_in};
        input_90 = {im_in_90}; % B
        input_180 = {im_in_180}; % D
        input_270 = {im_in_270}; % C

        [hei_A, wid_A] = size(im_in);
        [hei_B, wid_B] = size(im_in_90);
        [hei_C, wid_C] = size(im_in_270);
        [hei_D, wid_D] = size(im_in_180);

        net_A.blobs('data').reshape([hei_A wid_A 1 1]); % hei wid ch num
        net_A.reshape(); 
        recon.A = cell2mat(net_A.forward(input));

        net_A.blobs('data').reshape([hei_B wid_B 1 1]); % hei wid ch num
        net_A.reshape();  
        recon.B = cell2mat(net_A.forward(input_90));

        net_A.blobs('data').reshape([hei_C wid_C 1 1]); % hei wid ch num
        net_A.reshape();  
        recon.C = cell2mat(net_A.forward(input_270));

        net_A.blobs('data').reshape([hei_D wid_D 1 1]); % hei wid ch num
        net_A.reshape();  
        recon.D = cell2mat(net_A.forward(input_180));

        out.A = permute(recon.A, [2 1]);
        out.B = permute(recon.B, [2 1]);
        out.C = permute(recon.C, [2 1]);
        out.D = permute(recon.D, [2 1]);

        out.B = rot90(out.B, 1);
        out.C = rot90(out.C, 3);
        out.D = rot90(out.D, 2);

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
