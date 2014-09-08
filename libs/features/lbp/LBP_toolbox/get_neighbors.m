function neighbors_chain = get_neighbors(im, x, y)
%GET_NEIGHBORS Returns neighbors' pixel values

%   $Author Rodrigo Nava$
%   $Date: 2011/10/03$

NEIGHBORS = [-1,-1; -1,0; -1,1; 0,1; 1,1; 1,0; 1,-1; 0,-1];
neighbors_chain = zeros(1, 8);
[M, N] =  size(im);

for index = 1:8
    actual_neighbor = [y, x] + NEIGHBORS(index,:);
    
    if actual_neighbor(1,1) < 1 || actual_neighbor(1,2)  < 1 ...
            || actual_neighbor(1,1) > M || actual_neighbor(1,2) > N
        neighbors_chain(1, index) = 0;
    else
        neighbors_chain(1, index) = im(actual_neighbor(1,1),...
            actual_neighbor(1,2));
    end
    
end
