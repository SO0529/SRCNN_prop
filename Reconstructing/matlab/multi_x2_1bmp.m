function multi_x2_1bmp()

up_scale = 2;   
iter = 5000000;

image_name = './test.bmp';
save_name = './result.bmp';

model_dir = '../../Training/x2';
base_weights = sprintf('%s/model_553_multi_x2/553_multi_x2_iter_', model_dir);
model = '../prop/x2/SRCNN_deploy_553_multi.prototxt';

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

im_in = permute(im_y, [2 1]);

input = {im_in};

[hei, wid] = size(im_in);

net.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.reshape(); 
cell2mat(net.forward(input));

out.A = permute(net.blobs('recon-A').get_data(), [2 1]);
out.B = permute(net.blobs('recon-B').get_data(), [2 1]);
out.C = permute(net.blobs('recon-C').get_data(), [2 1]);
out.D = permute(net.blobs('recon-D').get_data(), [2 1]);

im_h_y= composition_x2(out.A, out.C, out.B, out.D);

toc;

im_h_y = uint8(im_h_y*255);

imwrite(im_h_y, save_name);

caffe.reset_all();