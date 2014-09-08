function image_lbp = lbp_uni(image, radio, num_points)
%LBP_UNI Returns a MxN uniform local binary pattern (LBP) imag. Every LBP 
%   label is computed using interpolated neighbors from a neighborhood
%   of variable size "radio "and number of points "num_points".
%
%   LBP_UNI(I, 1, 8) returns a uniform LBP image where every LBP label is 
%   computed using the uniformity measure proposed by Ojala.

%   $Author Rodrigo Nava$
%   $Date: 2011/10/03$

im = double(image);
[I, J] = size(im);
image_lbp = zeros(I, J);
POINTS = 0 : num_points - 1;

% Coordinates for the "num_points" neighbors
% num_points = 8, other values are not implementing yet :)
Y_OFFSET = round(100000*radio*sin(2*pi*POINTS/num_points))/100000;
Y_OFFSET =  [Y_OFFSET(4:-1:1), Y_OFFSET(end:-1:5)];
X_OFFSET = round(100000*radio*cos(2*pi*POINTS/num_points))/100000;
X_OFFSET =  [X_OFFSET(4:-1:1), X_OFFSET(end:-1:5)];

for y = 1:I
    for x = 1:J
        image_lbp(y, x) = get_lbp(im, x, y, num_points, ...
            X_OFFSET, Y_OFFSET);
    end
end

%counts = imhist(uint8(image_lbp));

end

%%%
%%% Function get_lbp
%%%
function bin2dec_value = get_lbp(im, x, y, num_points, X_OFFSET, Y_OFFSET)

neighbors_chain = get_inter_neighbors(im, x, y, num_points, X_OFFSET, Y_OFFSET);
actual_value = im(y, x);

diff_values = neighbors_chain - actual_value;
diff_values(diff_values >= 0) = 1;
diff_values(diff_values < 0) = 0;

POINTS = 0:num_points - 1;
bin2dec_value = diff_values*2.^POINTS';

bin2dec_value = get_uni_chain(bin2dec_value, 8);

if bin2dec_value <= 2
    bin2dec_value = sum(diff_values);
else
    bin2dec_value = 9;
end

end
