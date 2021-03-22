function rotary_x2_1bmp()

up_scale = 2;   
iter = 5000000;

image_name = './test.bmp';
save_name = './result.bmp';

model_dir = '../../Training/x2';
base_weights_A = sprintf('%s/model_553_A_x2/553_A_x2_iter_', model_dir);
model = './prop/SRCNN_deploy_553_parallel.prototxt';

weights_A = sprintf('%s%d.caffemodel', base_weights_A, iter);

im_y = imread(image_name);
im_y = modcrop(im_y, up_scale);
if size(im_y,3)>1
    im_y = rgb2ycbcr(im_y);
    im_y = im_y(:,:,1);
end

im_y = single(im_y)/255;

caffe.set_mode_gpu(); 
net_A = caffe.Net(model, weights_A, 'test');  

tic;

im_in = permute(im_y, [2 1]);
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

toc;

im_h_y = uint8(im_h_y*255);

imwrite(im_h_y, save_name);

caffe.reset_all();
