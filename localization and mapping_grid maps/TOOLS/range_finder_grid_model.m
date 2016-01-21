function zt = range_finder_grid_model(X, cells,missed,dim,std_hit,z_max)
x = X(1);
y = X(2);
theta = X(3);

n = size(cells,1);
m = size(missed,1);
zt = zeros(n+m,2);

% Hit measurements
for i=1:n
    px = (cells(i,1)-1) * dim + dim/2;
    py = (cells(i,2)-1) * dim + dim/2;
    b = atan2(py - y,px - x);                   %finder ray angle
    phi = pi_to_pi(b - theta);
    r = hypot(py - y,px - x) + std_hit*randn(1);
    zt(i,:) = [r,phi];
end


for i=1:m
    px = (missed(i,1)-1) * dim + dim/2;
    py = (missed(i,2)-1) * dim + dim/2;    
    b = atan2(py - y,px - x);                   %finder ray angle
    phi = pi_to_pi(b - theta);
    zt(i+n,:) = [z_max,phi];
end

end