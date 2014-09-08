function [image_lbp, counts] = lbp(image, invariant)
%LBP Returns a MxN local binary pattern (LBP) image and its histogram.
%
%   LBP(I, 0) returns a LBP image based on the original Ojala's proposal.
%   In addition, "counts" is the column LBP histogram. 
%
%   LBP(I, 1) returns a LBP image based on the original Ojala's 
%   proposal where every LBP label is computed using the minimal chain "ROR" 
%   by rotating neighboring pixels. In addition, "counts" is the LBP column 
%   histogram.

%   $Author Rodrigo Nava$
%   $Date: 2011/10/03$

im = double(image);
[I, J] = size(im);
image_lbp = zeros(I, J);

for y = 1:I
    for x = 1:J
        image_lbp(y, x) = get_lbp(im, x, y, invariant);
    end
end

counts = imhist(uint8(image_lbp));

end
 
%%%
%%% Function get_lbp
%%%
function bin2dec_value = get_lbp(im, x, y, invariant)

neighbors_chain = get_neighbors(im, x, y);
actual_value = im(y, x);
POINTS = 0:7;

diff_values = neighbors_chain - actual_value;
diff_values(diff_values >= 0) = 1;
diff_values(diff_values < 0) = 0;

bin2dec_value = diff_values*2.^POINTS';

if invariant == 1
    bin2dec_value = get_min_chain(bin2dec_value, 8);
end

end
