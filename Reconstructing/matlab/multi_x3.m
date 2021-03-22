function im_h = multi_x3(net, im_l)
    
im_in = permute(im_l, [2 1]);

input = {im_in};

[hei, wid] = size(im_in);

net.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.reshape(); 
cell2mat(net.forward(input));

out = permute(net.blobs('recon').get_data(), [2 1 3]);

% caffe's output pixels are transposed
tmp.A = out(:,:,1);
tmp.D = out(:,:,2);
tmp.G = out(:,:,3);
tmp.B = out(:,:,4);
tmp.E = out(:,:,5);
tmp.H = out(:,:,6);
tmp.C = out(:,:,7);
tmp.F = out(:,:,8);
tmp.I = out(:,:,9);

im_h = composition_x3(tmp);

