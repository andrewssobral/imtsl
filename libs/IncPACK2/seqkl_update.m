function [S,k] = seqkl_update(whch,m,n,i,k,lup,kmin,kmax,extrak,ttype,thresh,S,debug,lambda);

% [S,knew] = seqkl_update(whch,M,N,I,k,lup,kmin,kmax,extrak,ttype,thresh,S,debug);
%
% Input:
% SEQKL_U is m by k+lup  where the first k columns are U and the next lup are Ai
% SEQKL_V is n by k+lup
% S is k+lup by 1
%
   global SEQKL_U SEQKL_V

   o = k+1;
   op = k+lup;
   ip = i+lup-1;

   % new data Ai is stored in U(1:m,o:op);
   if debug,
      % save a copy for below
      Ai = SEQKL_U(1:m,o:op);
   end

   % find coefficients of new vectors in current space
   C = SEQKL_U(1:m,1:k)' * SEQKL_U(1:m,o:op);
   SEQKL_U(1:m,o:op) = SEQKL_U(1:m,o:op) - SEQKL_U(1:m,1:k)*C;
   C2 = SEQKL_U(1:m,1:k)' * SEQKL_U(1:m,o:op);
   SEQKL_U(1:m,o:op) = SEQKL_U(1:m,o:op) - SEQKL_U(1:m,1:k)*C2;
   C = C + C2;
   Sd = diag(S(1:k));

   if debug, fprintf('DBG  |C|: %16.8e\n',norm(C)); end
   if debug, fprintf('DBG  |U''*Up|: %16.8e\n',norm(SEQKL_U(1:m,1:k)'*SEQKL_U(1:m,o:op))); end

   % build R
   % expand the factorization with A_i
   % V will be expanded and retain orthogonality, so no QL factorizations needed
   % R = [Sd, C; zeros(lup,op) ];
   % lambda = 0.9;
   R = [Sd*lambda^lup, C; zeros(lup,op) ]; % forgetting factor
   % each of these arrangements leaves R strictly upper triangular 
   % (as opposed to block upper triangular)

   % perform QR of remaining data
   [SEQKL_U(1:m,o:op),R(o:op,o:op)] = qr(SEQKL_U(1:m,o:op),0);

   % debugging output: check orthogonality of bases
   if debug,
      fprintf('DBG  U orth: %16.8e\n',norm(SEQKL_U(1:m,1:op)'*SEQKL_U(1:m,1:op)-eye(op)));
      fprintf('DBG  V orth: %16.8e\n',norm(SEQKL_V(1:i-1,1:k)'*SEQKL_V(1:i-1,1:k)-eye(k)));
   end

   % compute the SVD
   [Ur,Sr,Vr] = svd(triu(R));
   Sr = diag(Sr);
   if (isequal(whch,'L'))
      [Sr,order] = sort(Sr,1,'descend');
   else 
      [Sr,order] = sort(Sr,1,'ascend');
   end
   Ur = Ur(:,order);
   Vr = Vr(:,order);

   % debugging output: check factorizations: this one is expensive!!!
   if debug,
      % check that we computed the proper singular values
      % Sr should be the singular value decomposition of [USV' | Ai]
      Sdbg = svd([SEQKL_U(1:m,1:k)*(Sd(1:k,1:k)*SEQKL_V(1:i-1,1:k)'),Ai]);
      fprintf('DBG  R error: %16.8e\n',norm(Sdbg(1:op) - Sr));
   end

   oldk = k;
   % compute new k
   if isequal(ttype,'abs'),
      % absolute threshold
      if isequal(whch,'L')
         k = sum( Sr >= thresh );
      else
         k = sum( Sr <= thresh );
      end 
   elseif isequal(ttype,'rel'),
      % relative threshold
      if isequal(whch,'L')
         k = sum( Sr >= thresh*Sr(1) );
      else 
         k = sum( Sr <= thresh*Sr(1) );
      end 
   end
   k = min(k+extrak,op);
   k = clamp(kmin,kmax,k);

   % info for decoupling
   Usel  = Ur(1:op,1:k);
   Vsel1 = Vr(1:oldk,1:k);
   Vsel2 = Vr(o:op,  1:k);

   % save new S
   S(1:op) = Sr(1:op);

   % compute new U
   SEQKL_U(1:m,1:k) = SEQKL_U(1:m,1:op)*Usel;

   % compute new V
   SEQKL_V(1:i-1,1:k) = SEQKL_V(1:i-1,1:oldk)*Vsel1;
   SEQKL_V(i:ip,1:k) = Vsel2;

return;
