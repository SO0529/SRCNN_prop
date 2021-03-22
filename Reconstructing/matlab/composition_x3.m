function im_out = composition_x3(in)
% color
if size(in.A,3)>1
    [hei_A, wid_A, ch_A] = size(in.A);

    im_out = zeros(hei_A*3, wid_A*3, ch_A);

    count_x = 1;
    count_y = 1;

    for c = 1:ch_A
        for y = 1:3:size(im_out, 1)
            for x = 1:3:size(im_out, 2)
                im_out(y,x,c) = in.A(count_y,count_x,c);
                im_out(y,x+1,c) = in.B(count_y,count_x,c);
                im_out(y,x+2,c) = in.C(count_y,count_x,c);
                im_out(y+1,x,c) = in.D(count_y,count_x,c);
                im_out(y+1,x+1,c) = in.E(count_y,count_x,c);
                im_out(y+1,x+2,c) = in.F(count_y,count_x,c);
                im_out(y+2,x,c) = in.G(count_y,count_x,c);
                im_out(y+2,x+1,c) = in.H(count_y,count_x,c);
                im_out(y+2,x+2,c) = in.I(count_y,count_x,c);
                count_x = count_x + 1;
            end
            count_x = 1;
            count_y = count_y + 1;
        end
        count_y = 1;
    end
% gray
else
    [hei_A, wid_A] = size(in.A);

    im_out = zeros(hei_A*3, wid_A*3);

    count_x = 1;
    count_y = 1;

    for y = 1:3:size(im_out, 1)
        for x = 1:3:size(im_out, 2)
            im_out(y,x) = in.A(count_y,count_x);
            im_out(y,x+1) = in.B(count_y,count_x);
            im_out(y,x+2) = in.C(count_y,count_x);
            im_out(y+1,x) = in.D(count_y,count_x);
            im_out(y+1,x+1) = in.E(count_y,count_x);
            im_out(y+1,x+2) = in.F(count_y,count_x);
            im_out(y+2,x) = in.G(count_y,count_x);
            im_out(y+2,x+1) = in.H(count_y,count_x);
            im_out(y+2,x+2) = in.I(count_y,count_x);
            count_x = count_x + 1;
        end
        count_x = 1;
        count_y = count_y + 1;
    end
end
    