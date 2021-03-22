function bicubic(target)

up_scale = 2;   
shrink_gnd = 10;
shrink_bc = 10;

imgFileLocation = sprintf('../test_images/%s/', target);
imgListLocation = sprintf('../test_images/%s/*.bmp', target);
imgFileList = dir(imgListLocation);

imgNum = size(imgFileList);
imgFileNameList = cell(imgNum);
for i = 1 : imgNum(1)            
    imgFileName = char(imgFileList(i).name);
    imgFileNameList{i} = sprintf('%s%s', imgFileLocation, imgFileName);
end;

base_dir = '../Bicubic';

folder_name = sprintf('%s/x2/PSNR_bc_%s', base_dir, target);

if ~exist(sprintf('%s/%s', base_dir, folder_name), 'dir')
    mkdir(sprintf('%s/%s', base_dir, folder_name));
end

psnr_file_y = sprintf('%s/%s/psnr_bc.txt', base_dir, folder_name);
time_file_y = sprintf('%s/%s/time_bc.txt', base_dir, folder_name);

fid_y = fopen(psnr_file_y, 'wt');
fid_time_y = fopen(time_file_y, 'wt');

for data = 1:length(imgFileNameList)
       
    img_path = char(imgFileNameList(data));
    im_y  = imread(img_path); 
    im_y = modcrop(im_y, up_scale);
    im_l_y = area_average_reduction_2x(im_y, up_scale);

    if size(im_y,3)>1
        im_y = rgb2ntsc(im_y);
        im_y = im_y(:, :, 1);
    else
        im_y = double(im_y)/255;
    end

    if size(im_l_y,3)>1
        im_l_y = rgb2ntsc(im_l_y);
        im_l_y = im_l_y(:, :, 1);
    else
        im_l_y = single(im_l_y)/255;
    end   
    
    tic;
    
    im_h_y = imresize(im_l_y, up_scale, 'bicubic');
    
    fprintf(fid_time_y, '%.5f\n', toc);
    
    bc_sn_y = single(shave(im_h_y, [shrink_bc, shrink_bc]));
    ref_sn_y = single(shave(im_y, [shrink_gnd, shrink_gnd]));

    psnr_y = psnr(bc_sn_y, ref_sn_y, 1);
    
    fprintf(fid_y, '%.3f\n', psnr_y);

    im_h_y = shave(uint8(im_h_y*255), [shrink_bc, shrink_bc]);

    image_name = strrep(img_path, imgFileLocation, '');
    image_name = strrep(image_name, '.bmp', '');

    y_name = sprintf('%s/%s/%s_bc_y.bmp', base_dir, folder_name, image_name);
    imwrite(im_h_y, y_name);
end
fclose(fid_y);
fclose(fid_time_y);