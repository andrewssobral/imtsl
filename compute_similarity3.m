%% [K] = compute_similarity3(I,J)
% 
function [K] = compute_similarity3(I,J)

  absminI = abs(min(min(I)));
  absminJ = abs(min(min(J)));
  absmin = absminI;
  
  if(absminJ > absminI)
    absmin = absminJ;
  end
  
  K = zeros(size(I));
  for m = 1:size(I,1)
    for n = 1:size(I,2)
      pI = absmin+(I(m,n));
      pJ = absmin+(J(m,n));
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
