function z = reluGradient(z)
z(z>0) = 1;
z(z<=0)  = 0;
end
