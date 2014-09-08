function minimal = get_min_chain(dec_value, length_int)
%GET_MIN_CHAIN Returns the minimum decimal value that neighborings may 
%   represent by rotating "dec_value", "length_int" times. 

%   $Author Rodrigo Nava$ 
%   $Date: 2011/10/03$

% lengh_int = 8, other values are not implementing yet :)
shifted_value = uint8(dec_value);
chain = zeros(1, length_int);
chain(1, 1) = shifted_value;

for index = 2:length_int
    msb = bitget(shifted_value, length_int);
    shifted_value = bitshift(shifted_value, 1, length_int) + msb;
    chain(1, index) = shifted_value;
end

minimal = min(chain);

end
