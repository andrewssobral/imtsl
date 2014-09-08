function M = getFromMatrix(mat,start,num)
  M = mat(:,start:start+num-1);
end
