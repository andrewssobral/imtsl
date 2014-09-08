function uniform = get_uni_chain(dec_value, length_int)
%GET_UNI_CHAIN Returns neighbors' pixel values using uniformity measure. 

%   $Author Rodrigo Nava$
%   $Date: 2011/10/03$

uniform = abs(bitget(dec_value, length_int) - bitget(dec_value, 1));

for index = 1:length_int-1
    uniform = uniform + abs(bitget(dec_value, index + 1)...
        - bitget(dec_value, index));
end

end
