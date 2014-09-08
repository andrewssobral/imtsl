function varargout = seqkl(varargin)
% SEQKL    Dynamic Rank Sequential Karhunen-Loeve
%
% Description:
%  This function implements the dynamic-rank Sequential Karhunen-Loeve,
%  making one pass to approximate the dominant SVD U*S*V'
%
% Synopsis:
%  S = SEQKL(A) returns a vector S with approximations for the largest 6 singular 
%  values of the matrix A. The elements of S are guaranteed to be non-negative 
%  and sorted ascending.
% 
%  [U,S] = SEQKL(A) returns a matrix U with orthonormal columns approximating
%  the left singular vectors corresponding to the approximate singular values in 
%  S. S is a diagonal matrix with non-negative and non-decreasing elements.
%
%  [U,S,V] = SEQKL(A) returns a matrix V with orthonormal columns approximating
%  the right singular vectors.
% 
%  [U,S,V] = SEQKL(A,m,n) computes the SVD where A is an incremental function
%  A(i,l), returning the columns i to i+l-1 of A
%
%  [...] = SEQKL(A,K) computes approximations for the largest K singular values
%  of A.
% 
%  [...] = SEQKL(A,KMAX,THRESH) computes approximations for up to KMAX of the
%  largest singular values satisfying the relative threshhold THRESH, as follows:
%            If OPTS.ttype == 'rel', preserve all singular values such that
%               sigma >= thresh*min(sigma)    if whch == 'L'
%               sigma <= thresh*min(sigma)    if whch == 'S'
%            If OPTS.ttype == 'abs', preserve all singular values such that
%               sigma >= thresh               if whch == 'L'
%               sigma <= thresh               if whch == 'S'
% 
%  [...] = SEQKL(A,K,...,'S') computes approximations for the smallest 
%  singular values.
%
%  [...] = SEQKL(A,K,...,OPTS) allows the specification of other options, as
%  listed below.
%   
% Optional input:
%   opts.thresh      - Tolerance for determining rank at each step. See above.
%   opts.ttype       - Rank threshhold type: 'abs' or 'rel'. See above. 
%                      Default='rel'
%   opts.kstart      - number of columns used to initial decomp (lmax)
%   opts.kmin        - minimum rank tracked by method, default=1
%   opts.extrak      - extra dimension tracked.
%                      applied after thresh-chosen k, before accounting for kmax.
%   opts.lmin        - minimum value for l at each step, default=1
%   opts.lmax        - maximum value for l at each step, default=kmax/sqrt(2)
%   opts.verbosity   - print while running or not, default=0 
%           0  -  silent
%           1  -  one-liners
%           2  -  chatty
%   opts.debug: debugging checks and output [{0} | 1]. Expensive!!!
%   opts.paramtest: parameter test          [{0} | 1]
%           do not actually run the algorithm, but instead test parameters,
%           set the defaults, and return them in a struct
%           Example: params = seqkl(A,kmax,thresh,opts);
%
% Example:
%  data = randn(1000,100);
%  [U,S,V,OPS] = seqkl(data,10,.20,'S');
%

% About: IncPACK - The Incremental SVD Package
% Version 0.1.2
% (C) 2001-2012, Written by Christopher Baker
% <a href="http://www.fsu.edu">The Florida State University</a>
% <a href="http://www.scs.fsu.edu">School of Computational Science</a>

% Modifications:
% 13-aug-2013, CGB (added support for matrix-free pass, remove multipass, new version)
% 18-dec-2012, CGB (added modified BSD license)
% 11-apr-2008, CGB (switched order of inputs: thresh before whch. improved documentation.)
%  8-apr-2008, CGB (got rid of recovery mode, simplified other modes, switched from global
%                   data to handle class)
%  7-jul-2006, CGB (added SDB,SDA; fixed flop counting; renamed auto modes)
% 16-feb-2006, CGB (redid input processing, restructured, rewrote multipass code, 
%                   updated printing, misc)
% 29-jan-2006, CGB 
% 14-sep-2005, CGB (added option to compute dominated SVD)
% 04-aug-2004, CGB (added Gu/GV-based theta/pi estimation)
% 28-oct-2003, CGB

% don't print file/lineno on warnings
warning off backtrace;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we accept at most three output args: U, S, V and Ops
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargout > 4)
    error('Too many output arguments. See ''help seqkl''')
end
if length(varargin) < 1,
    error('Not enough input arguments. See ''help seqkl'' for more.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% grab input arguments: A,kmax,thresh,opts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[A,m,n,kmax,whch,thresh,opts] = get_args(varargin{:});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% strings
ttype = 'rel';
fttype = [];
% ints
lambda = 1;
verbosity = 0;
kmin = 1;
lmin = 1;
lmax = round(kmax/sqrt(2));
kstart = lmax;
fkmin = 1;
fkmax = kmax;
extrak = 0;
paramtest   = 0;
debug = 0;
earlystop = n;
% floats
fthresh = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get parameters from opts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% strings
[whch,opts] = get_string(opts,'which','Target singular values',whch,...
                           {'L','S'});
[ttype,opts] = get_string(opts,'ttype','Threshhold type',ttype,...
                           {'abs','rel'});
[fttype,opts] = get_string(opts,'finalttype','Final threshhold type',ttype,...  % default is ttype
                           {'abs','rel'});
% ints
[verbosity,opts] = get_int(opts,'verbosity','Verbosity level',verbosity);
[kmin,opts] = get_int(opts,'kmin','Minimum rank',kmin,1,kmax);
[lmin,opts] = get_int(opts,'lmin','Minimum update size',lmin,1);     % do lmin first so it takes precedent
[lmax,opts] = get_int(opts,'lmax','Maximum update size',lmax,lmin);  % bound lmax by lmin
[kstart,opts] = get_int(opts,'kstart','Starting rank',kstart,1);
[fkmax,opts] = get_int(opts,'finalkmax','Final maximum rank',fkmax,1,kmax);   % do fkmax first so it takes precedent
[fkmin,opts] = get_int(opts,'finalkmin','Final minimum rank',fkmin,1,fkmax);   % bound fkmin by fkmax
[extrak,opts] = get_int(opts,'extrak','Extra rank modifier',extrak,1);
[paramtest,opts] = get_int(opts,'paramtest','Parameter test flag',paramtest);
[debug,opts] = get_int(opts,'debug','Debug flag',debug);
% scalar floats
if isnan(thresh), % not specified as arg to seqkl(...)
   % these defaults only make sense if ttype == 'rel'
   switch([ttype whch])
   case {'absS','relS'}
      thresh = inf;
   case {'absL','relL'}
      thresh = 0;
   end
   [thresh,opts] = get_float(opts,'thresh','Threshhold',thresh,0,inf,[1 1]);
end
[fthresh,opts] = get_float(opts,'finalthresh','Final threshhold',thresh,0,inf,[1 1]);  % default is thresh

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If performing a parameter test, save params and exit now
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if paramtest,
   theparams.whch       = whch;
   theparams.kmin       = kmin;
   theparams.kmax       = kmax;
   theparams.kstart     = kstart;
   theparams.extrak     = extrak;
   theparams.thresh     = thresh;
   theparams.ttype      = ttype;
   theparams.verbosity    = verbosity;
   theparams.lmin       = lmin;
   theparams.lmax       = lmax;
   theparams.finalttype  = fttype;
   theparams.finalthresh = fthresh;
   theparams.finalkmin  = fkmin;
   theparams.finalkmax  = fkmax;
   theparams.hasInit    = hasInit;
   theparams.debug      = debug;
   
   varargout{1} = theparams;
   return;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Allocate large data structures first, as early as possible
% These are stored globally, so that they can efficiently be modified 
% across routines.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global SEQKL_U SEQKL_V
maxudim = max(kmax,kstart)+lmax;
maxvdim = max(kmax,kstart);
if debug, fprintf('DBG  Allocating space for bases\n'); end
SEQKL_U = zeros(m,maxudim);
SEQKL_V = zeros(n,maxvdim);
S = zeros(maxudim,1);
clear maxudim maxvdim;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Any fields in OPTS that were not consumed should be passed
% on to OPS. Init OPS now.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
OPS = opts;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set initial factorization and Initialize variables
%
% Pointers/counters
% i          points to which column of A we use to update
% vr         denotes the number of valid rows in V
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 0;
vr = 0;

[S,k] = seqkl_stdpass(S,k,A,m,n,                               ...
                      verbosity,debug,whch,            ...
                      kstart,lmin,lmax,                        ...
                      kmin,kmax,thresh,ttype,extrak,lambda);

%%%%%%%%%%%%%%%%%%
% compute final k
oldk = k;
S = S(1:k);
if isequal(fttype,'abs'),
   % absolute threshold
   if isequal(whch,'L')
      k = sum( S >= fthresh );
   else
      k = sum( S <= fthresh );
   end
elseif isequal(fttype,'rel'),
   % relative threshold
   if isequal(whch,'L')
      k = sum( S >= fthresh*S(1) );
   else
      k = sum( S <= fthresh*S(1) );
   end
end
k = clamp(kmin,kmax,k);
if k < oldk,
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Print
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if verbosity > 1 || (debug > 0),
      fprintf(['*********************************** %8s ',...
               '**************************************\n'],'Final K');
   end
   if verbosity > 0,
      fprintf(' rank: %*d   sum(sigma):%13.6e\n', width(kmax), k, sum(S(1:k)));
   end
end


%%%%%%%%%%%%%%%%%%
% collect results
U = SEQKL_U(1:m,1:k);
S = S(1:k);
V = SEQKL_V(1:n,1:k);
clear global SEQKL_U SEQKL_V

% print footer
if (verbosity > 1) || (debug > 0)
    fprintf(['***************************************', ...
             '********************************************\n']);
end

% send out the data
if nargout <= 1,
   varargout{1} = S;
else
   varargout{1} = U;
   varargout{2} = diag(S);
   if nargout >= 3, varargout{3} = V;   end
end

end   % end function seqkl




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  get_string %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret,opts] = get_string(opts,argname,argdesc,def,options)

    % Process inputs and do error-checking 
    errstr = sprintf('%s opts.%s must be: \n',argdesc,argname);
    errstr = [errstr, sprintf('%s  ',options{:})];
    if isfield(opts,argname)
        ret = getfield(opts,argname);
        valid = 0;
        if isstr(ret),
            for i = 1:length(options),
                if isequal(ret,options{i}),
                    valid = 1;
                    break;
                end
            end
        end
        if ~valid,
            error(errstr);
        end
         
        % remove field from opts
        opts = rmfield(opts,argname);
    else
        ret = def;
    end
end   % end function get_string




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  get_int %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret,opts] = get_int(opts,argname,argdesc,def,lb,ub)

    if nargin < 6
        ub = inf;
        if nargin < 5,
            lb = -inf;
        end
    end

    % Process inputs and do error-checking 
    errstr = sprintf('%s opts.%s must be an integer in [%d,%d]',...
                     argdesc,argname,lb,ub);
    if isfield(opts,argname)
        ret = getfield(opts,argname);
        valid = 0;
        % check that it is an int
        if isnumeric(ret),
            ret = floor(ret);
            % check size (1 by 1) and bounds
            if isequal(size(ret),[1 1]) && lb <= ret && ret <= ub,
                valid = 1;
            end
        end
        if ~valid,
            error(errstr);
        end

        % remove field from opts
        opts = rmfield(opts,argname);
    else
        ret = def;
    end
end   % end function get_int




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  get_float %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret,opts] = get_float(opts,argname,argdesc,def,lb,ub,sz)
% only test bounds if sz == [1 1]

    if nargin < 7,
        sz = [];
    end
    if nargin < 6
        ub = inf;
    end
    if nargin < 5,
        lb = -inf;
    end

    % Process inputs and do error-checking 
    if isequal(sz,[1 1]),
        errstr = sprintf('%s opts.%s must be a scalar in [%d,%d]',...
                         argdesc,argname,lb,ub);
    elseif ~isempty(sz),
        errstr = sprintf('%s opts.%s must be an array of dimension %d by %d',...
                         argdesc,argname,sz(1),sz(2));
    % else, there are no tests, and no possible failure
    end

    if isfield(opts,argname)
        ret = getfield(opts,argname);
        valid = 0;
        % check that it is an int
        if isnumeric(ret),
            ret = double(ret);
            % no size request, no checks at all
            if isempty(sz),
                valid = 1;
            % if scalar requested, perform bounds check
            elseif isequal(sz,[1 1]),
                if isequal(sz,size(ret)) && lb <= ret && ret <= ub,
                    valid = 1;
                end
            % if matrix requested, just check size
            elseif isequal(sz,size(ret)),
                valid = 1;
            end
        end
        if ~valid,
            error(errstr);
        end

        % remove field from opts
        opts = rmfield(opts,argname);
    else
        ret = def;
    end
end   % end function get_float

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  get_args   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [A,m,n,kmax,whch,thresh,opts] = get_args(varargin)
% Process inputs and do error-checking 
   
   % possible calling methods
   %  skl = seqkl(A)
   %  skl = seqkl(A,k)
   %  skl = seqkl(A,k,whch)
   %  skl = seqkl(A,k,opts)
   %  skl = seqkl(A,k,whch,opts)
   %  skl = seqkl(A,kmax,thresh)
   %  skl = seqkl(A,kmax,thresh,whch)
   %  skl = seqkl(A,kmax,thresh,opts)
   %  skl = seqkl(A,kmax,thresh,whch,opts)


   % arg 1 must be A
   if isa(varargin{1},'double')
      Amat = varargin{1};
      [m,n] = size(Amat);
      Aargs=1;
      A = @(start,num)getFromMatrix(Amat,start,num);
      %disp('here1');
      %disp(m);
      %disp(n);
   elseif isa(varargin{1},'function_handle')
      A = varargin{1};
      m = varargin{2};
      n = varargin{3};
      Aargs = 3;
      %disp('here2');
      %disp(A);
      %disp(m);
      %disp(n);
   else
      error('A must be a real matrix.');
   end

   % arg Aargs+1 must be k (if it exists)
   if (nargin < Aargs+1)
      kmax = min(n,6);
   else
      kmax = varargin{Aargs+1};
   end
   kstr = ['Requested basis size, k, must be a positive integer <= n.'];
   if ~isa(kmax,'double') || ~isequal(size(kmax),[1,1]) || ~isreal(kmax) || (kmax>n) || (kmax<0),
      error(kstr)
   end
   if issparse(kmax)
      kmax = full(kmax);
   end
   if (ceil(kmax) ~= kmax)
      warning(msgid,['%s\n         ' ...
              'Non-integer k. Taking the ceiling.'],kstr)
      kmax = ceil(kmax);
   end

   % next argument may be either opts or whch or thresh: check if it's thresh
   if (nargin < Aargs+2)
      threshnotthere = 1;
      thresh = nan;
   else
      to = varargin{Aargs+2};
      if isnumeric(to),
         threshnotthere = 0;
         thresh = upper(to);
      else
         threshnotthere = 1;
         thresh = nan;
      end
   end
   errstr = 'Threshhold must be a scalar >= 0.';
   if ~isequal(size(thresh),[1,1]),
      error(errstr);
   end

   % next argument may be either opts or thresh: check if it's thresh
   if (nargin >= Aargs+3-threshnotthere)
      wo = varargin{Aargs+3-threshnotthere};
      if ischar(wo)
         whchnotthere = 0;
         whch = wo;
      else
         whchnotthere = 1;
         whch = 'L';
      end
   else
      whchnotthere = 1;
      whch = 'L'; 
   end
   errstr = 'whch must be ''L'' or ''S''.';
   if ~isequal(whch,'L') && ~isequal(whch,'S'),
      error(errstr);
   end

   if (nargin >= Aargs+4-threshnotthere-whchnotthere)
      opts = varargin{Aargs+4-whchnotthere-threshnotthere};
      if ~isa(opts,'struct')
          error('Options argument must be a structure.')
      end
   else
      % create an empty struct to return
      opts = struct();
   end

   if (nargin > Aargs+4-threshnotthere-whchnotthere),
      % extra arguments sent in
      warning('Too many arguments.');
   end

end   % end function get_args
