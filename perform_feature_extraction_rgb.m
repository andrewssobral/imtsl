function F = perform_feature_extraction_rgb(A)
  [m,n,~,p] = size(A);
  %nFeatures = 1;
  %nFeatures = 2;
  %nFeatures = 4;
  %nFeatures = 6;
  nFeatures = 8;
  F = zeros(p,m*n,nFeatures);
  
  for k = 1:p
    Irgb = A(:,:,:,k);
    
    Ir = Irgb(:,:,1); % imshow(Ir,[],'InitialMagnification','fit');
    Ig = Irgb(:,:,2); % imshow(Ig,[],'InitialMagnification','fit');
    Ib = Irgb(:,:,3); % imshow(Ib,[],'InitialMagnification','fit');
    
    I = (uint8((Ir+Ig+Ib)/3)); % imshow(I,[]);
    
    %Ilbp = lbp(I, 0);
    Ilbp = lbp_median(I, 0, 1, 8); % imshow(Ilbp,[]);
    
    [Gx, Gy] = imgradientxy(I); % imshow(Ilbp,[]);
    [Gmag, ~] = imgradient(Gx, Gy); % imshow(Gmag,[]);
    %[Gmag, Gdir] = imgradient(Gx, Gy);
    
    vIr = reshape(Ir,[],1);
    vIg = reshape(Ig,[],1);
    vIb = reshape(Ib,[],1);
    vI = reshape(I,[],1);
    
    vIlbp = reshape(Ilbp,[],1);
    vImag = reshape(Gmag,[],1);
    %vIdir = reshape(Gdir,[],1);
    vIgx = reshape(Gx,[],1);
    vIgy = reshape(Gy,[],1);
    
    F(k,:,1) = vIr;
    F(k,:,2) = vIg;
    F(k,:,3) = vIb;
    F(k,:,4) = vI;
    F(k,:,5) = vIlbp;
    F(k,:,6) = vImag;
    F(k,:,7) = vIgx;
    F(k,:,8) = vIgy;
    %F(k,:,9) = vIdir;
  end
end
