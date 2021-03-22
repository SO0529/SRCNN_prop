function parallel_x2_1bmp()

up_scale = 2;   
iter = 5000000;

image_name = './test.bmp';
save_name = './result.bmp';

model_dir = '../../Training/x2';
base_weights_A = sprintf('%s/model_553_A_x2/553_A_x2_iter_', model_dir);
base_weights_B = sprintf('%s/model_553_B_x2/553_B_x2_iter_', model_dir);
base_weights_C = sprintf('%s/model_553_C_x2/553_C_x2_iter_', model_dir);
base_weights_D = sprintf('%s/model_553_D_x2/553_D_x2_iter_', model_dir);
model = './prop/SRCNN_deploy_553_parallel.prototxt';

weights_A = sprintf('%s%d.caffemodel', base_weights_A, iter);
weights_B = sprintf('%s%d.caffemodel', base_weights_B, iter);
weights_C = sprintf('%s%d.caffemodel', base_weights_C, iter);
weights_D = sprintf('%s%d.caffemodel', base_weights_D, iter);

im_y = imread(image_name);
im_y = modcrop(im_y, up_scale);
if size(im_y,3)>1
    im_y = rgb2ycbcr(im_y);
    im_y = im_y(:,:,1);
end

im_y = single(im_y)/255;

caffe.set_mode_gpu(); 

net_A = caffe.Net(model, weights_A, 'test');
net_B = caffe.Net(model, weights_B, 'test');  
net_C = caffe.Net(model, weights_C, 'test');  
net_D = caffe.Net(model, weights_D, 'test');  

tic;

im_in = permute(im_y, [2 1]);

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
im_h_y= composition_x2(out.A, out.C, out.B, out.D);

toc;

im_h_y = uint8(im_h_y*255);

imwrite(im_h_y, save_name);

caffe.reset_all();
