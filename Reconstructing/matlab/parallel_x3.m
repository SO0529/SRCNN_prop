function im_h = parallel_x3(net, im_l)
    
im_in = permute(im_l, [2 1]);

input = {im_in};

[hei, wid] = size(im_in);

net.A.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.A.reshape(); 
cell2mat(net.A.forward(input));
out.A = permute(net.A.blobs('recon').get_data(), [2 1 3]);

net.B.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.B.reshape(); 
cell2mat(net.B.forward(input));
out.B = permute(net.B.blobs('recon').get_data(), [2 1 3]);

net.C.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.C.reshape(); 
cell2mat(net.C.forward(input));
out.C = permute(net.C.blobs('recon').get_data(), [2 1 3]);

net.D.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.D.reshape(); 
cell2mat(net.D.forward(input));
out.D = permute(net.D.blobs('recon').get_data(), [2 1 3]);

net.E.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.E.reshape(); 
cell2mat(net.E.forward(input));
out.E = permute(net.E.blobs('recon').get_data(), [2 1 3]);

net.F.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.F.reshape(); 
cell2mat(net.F.forward(input));
out.F = permute(net.F.blobs('recon').get_data(), [2 1 3]);

net.G.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.G.reshape(); 
cell2mat(net.G.forward(input));
out.G = permute(net.G.blobs('recon').get_data(), [2 1 3]);

net.H.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.H.reshape(); 
cell2mat(net.H.forward(input));
out.H = permute(net.H.blobs('recon').get_data(), [2 1 3]);

net.I.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.I.reshape(); 
cell2mat(net.I.forward(input));
out.I = permute(net.I.blobs('recon').get_data(), [2 1 3]);

% caffe's output pixels are transposed
tmp.A = out.A(:,:);
tmp.D = out.B(:,:);
tmp.G = out.C(:,:);
tmp.B = out.D(:,:);
tmp.E = out.E(:,:);
tmp.H = out.F(:,:);
tmp.C = out.G(:,:);
tmp.F = out.H(:,:);
tmp.I = out.I(:,:);

im_h = composition_x3(tmp);

