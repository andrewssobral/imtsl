function neighbors_chain = get_inter_neighbors(im, x, y, num_points, X_OFFSET, Y_OFFSET)
%GET_INTER_NEIGHBORS Returns neighbors' pixel values using bilinearly 
%   interpolation.

%   $Author Rodrigo Nava$
%   $Date: 2011/10/03$

y_p = y - Y_OFFSET;
x_p = x + X_OFFSET;
[M, N] = size(im);

neighbors_chain = zeros(1, num_points);

for index = 1:num_points
    min_x = floor(x_p(index));
    max_x = ceil(x_p(index));
    min_y = floor(y_p(index));
    max_y = ceil(y_p(index));
    
    Q_11 = 0;
    Q_12 = 0;
    Q_21 = 0;
    Q_22 = 0;
    
    if min_x >= 1 && min_x <= N
        if min_y >= 1 && min_y <= M
            Q_11 = im(min_y, min_x);
        end
        if max_y >= 1 && max_y <= M
            Q_21 = im(max_y, min_x);
        end
    end
    
    if max_x >= 1 && max_x <= N
        if min_y >= 1 && min_y <= M
            Q_12 = im(min_y, max_x);
        end
        if max_y >= 1 && max_y <= M
            Q_22 = im(max_y, max_x);
        end
    end
    
    value_Q_1 = (1 - (x_p(index) - min_x))*Q_21 ...
        + (x_p(index) - min_x)*Q_22;
    value_Q_2 = (1 - (x_p(index) - min_x))*Q_11 ...
        + (x_p(index) - min_x)*Q_12;
    neighbors_chain(1, index) = (1 - (y_p(index) - min_y))*value_Q_2 ...
        + (y_p(index) - min_y)*value_Q_1;
end