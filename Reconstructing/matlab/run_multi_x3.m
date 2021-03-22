function run_multi_x3(target, CPUorGPU, start, stop)

up_scale = 3;   
shrink_gnd = 9;
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

model_dir = '../../Training/x3';
base_weights = sprintf('%s/model_531_multi_x3/531_multi_x3_iter_', model_dir);
model = '../prop/x3/SRCNN_deploy_531_multi_x3.prototxt';
folder_name = sprintf('531_multi/PSNR_531_multi_%s_%s', target, CPUorGPU);
save_name = '531_multi';

if ~exist(sprintf('../prop/x3/%s', folder_name), 'dir')
    mkdir(sprintf('../prop/x3/%s', folder_name));
end

average_psnr_file = sprintf('../prop/x3/%s/all_psnr.txt', folder_name);
average_fid = fopen(average_psnr_file, 'wt');

for iter = start:100000:stop
    fprintf('Calculate PSNR : %d\n', iter);  
    
    weights = sprintf('%s%d.caffemodel', base_weights, iter);
    psnr_dir = sprintf('../prop/x3/%s/%d', folder_name, iter);
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
        net = caffe.Net(model, weights, 'test');  
        
        ttime = tic;
        tic;

        im_h_y = multi_x3(net, im_l_y);
        
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
