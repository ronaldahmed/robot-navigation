function zt = range_finder_grid_model(X, cells,missed,dim,sigma,M)
x = X(1);
y = X(2);
theta = X(3);

n = size(cells,1);
m = size(missed,1);
zt = zeros(n+m,3);

%ruidos sigma
sigmaR = sigma(1);
sigmaB = sigma(2);
sigmaC = sigma(3);

for i=1:n
    px = (cells(i,1)-1) * dim + dim/2;
    py = (cells(i,2)-1) * dim + dim/2;
    
    b = atan2(py - y,px - x);                   %finder ray angle
    phi = pi_to_pi(b - theta + sigmaB*rand(1));
    r = hypot(py - y,px - x) + sigmaR*rand(1);
    c = (cells(i,1)-1)*M + cells(i,2) + sigmaC*rand(1);
    zt(i,:) = [r,phi,c];
end

for i=1:m
    px = (missed(i,1)-1) * dim + dim/2;
    py = (missed(i,2)-1) * dim + dim/2;    
    b = atan2(py - y,px - x);                   %finder ray angle
    phi = pi_to_pi(b - theta + sigmaB*rand(1));
    c = (missed(i,1)-1)*M + missed(i,2) + sigmaC*rand(1);    
    zt(i+n,:) = [Inf,phi,c];
end

end