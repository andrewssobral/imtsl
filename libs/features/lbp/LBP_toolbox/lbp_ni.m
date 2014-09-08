function [image_lbp, counts] = lbp_ni(image, invariant, radio, num_points)
%LBP_NI Returns a MxN neighbor intensity local binary pattern (LBP) image. Every LBP
%   label is computed using interpolated neighbors from a neighborhood
%   of variable size "radio "and number of points "num_points".
%
%   LBP_NI(I, 0, 1, 8) returns a LBP image where every LBP label is computed using
%   the neighbor mean value propose by Liu. (Generalized Local Binary Patterns for
%   texture dlassification). In addition, "counts" is the column LBP histogram. 
%
%   LBP_NI(I, 1, 1, 8) returns a LBP image from "image". Neighbor pixels
%   are computed from interpolated values and LBP label is computed using 
%   the minimal chain "ROR" by rotating neighboring pixels. In addition, 
%   "counts" is the LBP column histogram.

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
        image_lbp(y, x) = get_lbp(im, x, y, invariant, num_points, ...
            X_OFFSET, Y_OFFSET);
    end
end

counts = imhist(uint8(image_lbp));

end

%%%
%%% Function get_lbp
%%%
function bin2dec_value = get_lbp(im, x, y, invariant, num_points, X_OFFSET, Y_OFFSET)

neighbors_chain = get_inter_neighbors(im, x, y, num_points, X_OFFSET, Y_OFFSET);
actual_value = mean(neighbors_chain);

diff_values = neighbors_chain - actual_value;

diff_values(diff_values >= 0) = 1;
diff_values(diff_values < 0) = 0;

POINTS = 0:num_points - 1;
bin2dec_value = diff_values*2.^POINTS';

if invariant == 1
    bin2dec_value = get_min_chain(bin2dec_value, 8);
end

end
