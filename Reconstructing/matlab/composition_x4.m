function im_out = composition_x4(in)
% color
if size(in.A,3)>1
    [hei_A, wid_A, ch_A] = size(in.A);

    im_out = zeros(hei_A*4, wid_A*4, ch_A);

    count_x = 1;
    count_y = 1;

    for c = 1:ch_A
        for y = 1:4:size(im_out, 1)
            for x = 1:4:size(im_out, 2)
                im_out(y,x,c) = in.A(count_y,count_x,c);
                im_out(y,x+1,c) = in.B(count_y,count_x,c);
                im_out(y,x+2,c) = in.C(count_y,count_x,c);
                im_out(y,x+3,c) = in.D(count_y,count_x,c);
                im_out(y+1,x,c) = in.E(count_y,count_x,c);
                im_out(y+1,x+1,c) = in.F(count_y,count_x,c);
                im_out(y+1,x+2,c) = in.G(count_y,count_x,c);
                im_out(y+1,x+3,c) = in.H(count_y,count_x,c);
                im_out(y+2,x,c) = in.I(count_y,count_x,c);
                im_out(y+2,x+1,c) = in.J(count_y,count_x,c);
                im_out(y+2,x+2,c) = in.K(count_y,count_x,c);
                im_out(y+2,x+3,c) = in.L(count_y,count_x,c);
                im_out(y+3,x,c) = in.M(count_y,count_x,c);
                im_out(y+3,x+1,c) = in.N(count_y,count_x,c);
                im_out(y+3,x+2,c) = in.O(count_y,count_x,c);
                im_out(y+3,x+3,c) = in.P(count_y,count_x,c);
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

    im_out = zeros(hei_A*4, wid_A*4);

    count_x = 1;
    count_y = 1;

    for y = 1:4:size(im_out, 1)
        for x = 1:4:size(im_out, 2)
            im_out(y,x) = in.A(count_y,count_x);
            im_out(y,x+1) = in.B(count_y,count_x);
            im_out(y,x+2) = in.C(count_y,count_x);
            im_out(y,x+3) = in.D(count_y,count_x);
            im_out(y+1,x) = in.E(count_y,count_x);
            im_out(y+1,x+1) = in.F(count_y,count_x);
            im_out(y+1,x+2) = in.G(count_y,count_x);
            im_out(y+1,x+3) = in.H(count_y,count_x);
            im_out(y+2,x) = in.I(count_y,count_x);
            im_out(y+2,x+1) = in.J(count_y,count_x);
            im_out(y+2,x+2) = in.K(count_y,count_x);
            im_out(y+2,x+3) = in.L(count_y,count_x);
            im_out(y+3,x) = in.M(count_y,count_x);
            im_out(y+3,x+1) = in.N(count_y,count_x);
            im_out(y+3,x+2) = in.O(count_y,count_x);
            im_out(y+3,x+3) = in.P(count_y,count_x);
            count_x = count_x + 1;
        end
        count_x = 1;
        count_y = count_y + 1;
    end
end
    
