function [lp,flag] = inverse_range_finder_model(X,mi,z,dim,betha,sigma,MAX_RANGE,lp_0,lp_occ,lp_free)

x = X(1);
y = X(2);
theta = X(3);
sigmaR = sigma(1);
sigmaF = sigma(2);
alpha = dim;

xi = dim*(mi(1)-1) + dim/2;
yi = dim*(mi(2)-1) + dim/2;

r = hypot(xi-x,yi-y);
phi = pi_to_pi(atan2(yi-y,xi-x) - theta);

v = abs(ones(size(z,1),1)*phi - z(:,2));
[~,k] = min(v);
flag = 1;

%fprintf('xi:%.2f  yi:%.2f  - lp:',xi,yi)
if z(k,1) == Inf
%    fprintf('lp free........\n')
    lp = lp_free;
elseif r > min(MAX_RANGE,z(k,1) + alpha/2) || abs(phi-z(k,2)) > betha/2
%    fprintf('lp0\n')
    lp = lp_0;
    flag = 0;
elseif z(k,1) < MAX_RANGE && abs(r-z(k,1)) < alpha/2
    lp = lp_occ;
%    fprintf('lp occ----------------------------\n')    
else
%    fprintf('lp free\n')
    lp = lp_free;

end