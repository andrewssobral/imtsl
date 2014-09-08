function [S,k] =              ...
   seqkl_stdpass(S,k,A,m,n,                                 ...
                  verbosity,debug,whch,               ...
                  kstart,lmin,lmax,                         ...
                  kmin,kmax,thresh,ttype,extrak,lambda);
% For 'std/echo', 'error-based refine' or 'error-based expand'

   global SEQKL_U SEQKL_V

   i = 1;
   while i<=n,

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % print header
      if (verbosity > 1) || (debug > 0)
         fprintf(['*********************************** %8s ',...
                  '**************************************\n'],'stdpass');
      end

      if i==1 && k==0,
         
         % init with an SVD of the first kstart columns of A
         ip = k;
         k = kstart;
         if debug, fprintf('DBG  Performing initial QR of first %d columns...\n',k); end
         [SEQKL_U(:,1:k),R] = qr(A(1,k),0);
         [Ur,Sr,Vr] = svd(triu(R));
         S(1:k) = diag(Sr);
         SEQKL_U(:,1:k) = SEQKL_U(:,1:k)*Ur;
         SEQKL_V(1:k,1:k) = Vr;
         % sort singular values
         if (isequal(whch,'S')) 
            [S(1:k),order] = sort(S(1:k),1,'ascend');
         else
            [S(1:k),order] = sort(S(1:k),1,'descend');
         end
         SEQKL_U(:,1:k)   = SEQKL_U(:,order);
         SEQKL_V(1:k,1:k) = SEQKL_V(1:k,order);
         % start partway through the first pass
         i=k+1;

      else

         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % choose the (globally) optimal column size
         lup = clamp( lmin, lmax, round(k/sqrt(2)) );
         % now clamp it
         if i-1+lup > n
            lup = n-i+1;
         end
         ip  =  i+lup-1;
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % printing
         if verbosity > 1,
            fprintf('Expanding with columns %d through %d...\n',i,ip);
         end
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % compute/put update columns into appropriate part of U
         SEQKL_U(1:m,k+1:k+lup) = A(i,lup);
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % Update factorization
         [S,knew] = seqkl_update(whch,m,n,i,k,lup,kmin,kmax,extrak,ttype,thresh,S,debug,lambda);
   
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % update pointers: i,k
         k=knew;
         i=i+lup;
      end

      colsdone = i-1;

   end % while

end % function
