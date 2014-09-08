function w = width(n)
   w1 = log10(n);
   w = ceil(w1);
   if w == w1,
      w = w + 1;
   end
