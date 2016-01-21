function lp = inverse_range_finder_model(X,mi,z,dim,beta,MAX_RANGE,lp_0,lp_occ,lp_free)

x = X(1);
y = X(2);
theta = X(3);
alpha = dim;

xi = dim*(mi(1)-1) + dim/2;
yi = dim*(mi(2)-1) + dim/2;

r = hypot(xi-x,yi-y);
phi = pi_to_pi(atan2(yi-y,xi-x) - theta);

v = abs(ones(size(z,1),1)*phi - z(:,2));
[~,k] = min(v);

%fprintf('xi:%.2f  yi:%.2f  - lp:',xi,yi)
if abs(phi-z(k,2)) <= beta/2
    if z(k,1) == MAX_RANGE
        lp = lp_free;
    elseif abs(r-z(k,1)) < alpha/2
        lp= lp_occ;
    else
        lp = lp_free;
    end
else
    lp = lp_0;
end

end