%% [K] = compute_similarity2(I,J)
% 
function [K] = compute_similarity2(I,J)
  K = zeros(size(I));
  
  for m = 1:size(I,1)
    for n = 1:size(I,2)
      pI = (I(m,n));
      pJ = (J(m,n));
      pS = 1;
      
      if(pI < pJ)
        pS = (pI/pJ);
      end
      
      if(pI > pJ)
        pS = (pJ/pI);
      end
      
      K(m,n) = pS;
    end
  end
end
