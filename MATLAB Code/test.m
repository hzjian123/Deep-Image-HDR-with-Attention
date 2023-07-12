function [H1,V1,D1,A1] = test(input)
[c,s]=wavedec2(input,2,'haar');
[H1,V1,D1] = detcoef2('all',c,s,1);
A1 = appcoef2(c,s,'haar',1);
end