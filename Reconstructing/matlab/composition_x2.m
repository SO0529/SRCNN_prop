function im_out = composition_x2(A, B, C, D)

% caffe's output pixels are transposed

% color
if size(A,3)>1
    [hei_A, wid_A, ch_A] = size(A);
    [hei_B, wid_B, ch_B] = size(B);
    [hei_C, wid_C, ch_C] = size(C);
    [hei_D, wid_D, ch_D] = size(D);

    im_out = zeros(hei_A + hei_C, wid_A + wid_B, ch_A);

    count_x = 1;
    count_y = 1;

    for c = 1:ch_A
        for y = 1:2:size(im_out, 1)
            for x = 1:2:size(im_out, 2)
                im_out(y,x,c) = A(count_y,count_x,c);
                im_out(y+1,x,c) = B(count_y,count_x,c);
                im_out(y,x+1,c) = C(count_y,count_x,c);
                im_out(y+1,x+1,c) = D(count_y,count_x,c);
                count_x = count_x + 1;
            end
            count_x = 1;
            count_y = count_y + 1;
        end
        count_y = 1;
    end
% gray
else
    [hei_A, wid_A] = size(A);
    [hei_B, wid_B] = size(B);
    [hei_C, wid_C] = size(C);
    [hei_D, wid_D] = size(D);

    im_out = zeros(hei_A + hei_C, wid_A + wid_B);

    count_x = 1;
    count_y = 1;

    for y = 1:2:size(im_out, 1)
        for x = 1:2:size(im_out, 2)
            im_out(y,x) = A(count_y,count_x);
            im_out(y+1,x) = B(count_y, count_x);
            im_out(y,x+1) = C(count_y, count_x);
            im_out(y+1,x+1) = D(count_y, count_x);
            count_x = count_x + 1;
        end
        count_x = 1;
        count_y = count_y + 1;
    end
end
    