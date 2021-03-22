function im_h = multi_x2(net, im_l)
    
im_in = permute(im_l, [2 1]);

input = {im_in};

[hei, wid] = size(im_in);

net.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.reshape(); 
cell2mat(net.forward(input));

out = permute(net.blobs('recon').get_data(), [2 1 3]);

A = out(:,:,1);
B = out(:,:,2);
C = out(:,:,3);
D = out(:,:,4);

im_h = composition_x2(A, B, C, D);


