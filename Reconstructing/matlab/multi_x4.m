function im_h = multi_x4(net, im_l)
    
im_in = permute(im_l, [2 1]);

input = {im_in};

[hei, wid] = size(im_in);

net.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.reshape(); 
cell2mat(net.forward(input));

out = permute(net.blobs('recon').get_data(), [2 1 3]);

% caffe's output pixels are transposed
tmp.A = out(:,:,1);
tmp.E = out(:,:,2);
tmp.I = out(:,:,3);
tmp.M = out(:,:,4);
tmp.B = out(:,:,5);
tmp.F = out(:,:,6);
tmp.J = out(:,:,7);
tmp.N = out(:,:,8);
tmp.C = out(:,:,9);
tmp.G = out(:,:,10);
tmp.K = out(:,:,11);
tmp.O = out(:,:,12);
tmp.D = out(:,:,13);
tmp.H = out(:,:,14);
tmp.L = out(:,:,15);
tmp.P = out(:,:,16);

im_h = composition_x4(tmp);

