function q = likelihood_field_range_finder_model(X,x_sen,zt,N,map,dim,std,Zw)
% retorna probabilidad de medida range finder :)
% X col, zt col, xsen col
[n,m] = size(N);

x = X(1);
y = X(2);
theta = X(3);

theta_sen = zt(2);
phi = pi_to_pi(theta + theta_sen);

rotS = [cos(theta),-sin(theta);sin(theta),cos(theta)];

sigma = std.^2;
sigmaR = sigma(1);
zhit = Zw(1);
zrand = Zw(4);
zmax = Zw(3);

px = floor(x/dim) + 1;
py = floor(y/dim) + 1;

if px<1 || px > n || py<1 || py>n || exp(map(px,py))==1
    %q = 0.02*rand(1);
    q = 0;
    return
end

q = 1;
if zt(1) ~= Inf

    xz = X(1:2) + rotS*x_sen + zt(1)*[cos(phi);
                                      sin(phi)];
    xz(3) = phi;                              
    xi = floor(xz(1)/dim) + 1;
    yi = floor(xz(2)/dim) + 1;
    
    if xi>=1 && xi<=n && yi>=1 && yi<=m
        d = N(xi,yi);
    else
        d = Inf;
    end

    gd   = gauss_1D(0,sigmaR,d);
    %q = zhit*gd + zrand/zmax;
    q = gd ;%+ 0.0001*rand(1);
end

end