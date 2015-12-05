function flag = isSensable(X,cell,mapa,used,dim)
xx = X(1);
yy = X(2);
theta = X(3);

[n,m] = size(mapa);

px = floor(xx/dim) + 1;
py = floor(yy/dim) + 1;
xi = cell(1);
yi = cell(2);
xini = xi;
yini = yi;

flag = 1;

is_in = (xi>=1 && xi<=n && yi>=1 && yi<=m);
if ~is_in
    flag = 0;
    return
end

phi = atan2(yi - py,xi - px) - theta;
phi = pi_to_pi(phi);
r = hypot(px - xi,py - yi);

%fprintf('analisis para px : %d - py: %d\n',px,py)

while( xi~=px && yi~=py )

%    fprintf('-------xi:%d  yi:%d\n',xi,yi)
    if mapa(xi,yi) && used(xi,yi) && xini~=xi && yini~=yi
        flag = 0;
        break;
    end
    
    if abs(phi) < 20*pi/180
        dif = 1;
    else
        dif = min(abs(1/sin(phi)),2);
    end
    
    r = max(r - dif,0);
    %fprintf('-------dif:%.2f  phi:%.2f  r:%.2f\n---------\n',dif,phi,r)
    
    xi = round(px + r*cos(phi));
    yi = round(py + r*sin(phi));
    
end

end