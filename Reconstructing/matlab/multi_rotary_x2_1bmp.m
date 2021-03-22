function multi_rotary_x2_1bmp()

up_scale = 2;   
iter = 5000000;

image_name = './test.bmp';
save_name = './result.bmp';

model_dir = '../../Training/x2';
base_weights = sprintf('%s/model_553_multi_x2/553_multi_x2_iter_', model_dir);
model = '../prop/x2/SRCNN_deploy_553_multi.prototxt';

fprintf('Calculate PSNR : %d\n', iter);  

weights = sprintf('%s%d.caffemodel', base_weights, iter);

im_y = imread(image_name);
im_y = modcrop(im_y, up_scale);
if size(im_y,3)>1
    im_y = rgb2ycbcr(im_y);
    im_y = im_y(:,:,1);
end

im_y = single(im_y)/255; 

caffe.set_mode_gpu(); 
net = caffe.Net(model, weights, 'test');  

tic;

im_l_90 = rot90(im_y, 1);
im_l_180 = rot90(im_y, 2);
im_l_270 = rot90(im_y, 3);

im_h = multi_x2(net, im_y);
im_h_90 = multi_x2(net, im_l_90);
im_h_180 = multi_x2(net, im_l_180);
im_h_270 = multi_x2(net, im_l_270);

im_h_ave = im_h.all + rot90(im_h_90.all,3) + rot90(im_h_180.all,2) + rot90(im_h_270.all,1);
im_h_y = im_h_ave.*0.25;

toc;

im_h_y = uint8(im_h_y*255);

imwrite(im_h_y, save_name);

caffe.reset_all();