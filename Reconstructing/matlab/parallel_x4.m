function im_h = parallel_x4(net, im_l)
    
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

net.J.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.J.reshape(); 
cell2mat(net.J.forward(input));
out.J = permute(net.J.blobs('recon').get_data(), [2 1 3]);

net.K.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.K.reshape(); 
cell2mat(net.K.forward(input));
out.K = permute(net.K.blobs('recon').get_data(), [2 1 3]);

net.L.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.L.reshape(); 
cell2mat(net.L.forward(input));
out.L = permute(net.L.blobs('recon').get_data(), [2 1 3]);

net.M.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.M.reshape(); 
cell2mat(net.M.forward(input));
out.M = permute(net.M.blobs('recon').get_data(), [2 1 3]);

net.N.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.N.reshape(); 
cell2mat(net.N.forward(input));
out.N = permute(net.N.blobs('recon').get_data(), [2 1 3]);

net.O.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.O.reshape(); 
cell2mat(net.O.forward(input));
out.O = permute(net.O.blobs('recon').get_data(), [2 1 3]);

net.P.blobs('data').reshape([hei wid 1 1]); % hei wid ch num
net.P.reshape(); 
cell2mat(net.P.forward(input));
out.P = permute(net.P.blobs('recon').get_data(), [2 1 3]);


% caffe's output pixels are transposed
tmp.A = out.A(:,:);
tmp.E = out.B(:,:);
tmp.I = out.C(:,:);
tmp.M = out.D(:,:);
tmp.B = out.E(:,:);
tmp.F = out.F(:,:);
tmp.J = out.G(:,:);
tmp.N = out.H(:,:);
tmp.C = out.I(:,:);
tmp.G = out.J(:,:);
tmp.K = out.K(:,:);
tmp.O = out.L(:,:);
tmp.D = out.M(:,:);
tmp.H = out.N(:,:);
tmp.L = out.O(:,:);
tmp.P = out.P(:,:);

im_h = composition_x4(tmp);

