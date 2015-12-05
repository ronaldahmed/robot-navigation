function [inner,front,missed] = get_cells_range(X,S,map,dim,zmax)
% devuelve todas las celdas en el rango de medicion de los rayos listados
% en S

x = X(1);
y = X(2);
theta = X(3);

map = exp(map);
px = floor(x/dim) + 1;
py = floor(y/dim) + 1;
[n,m] = size(map);

ns = length(S);
map_inner = zeros(size(map));
map_front = zeros(size(map));
map_missed = zeros(size(map));

%fprintf('px:%d  - py:%d \n',px,py)

for i=1:ns
    theta_s = S(i);
    r = 0;
    phi = pi_to_pi(theta_s + theta);
    
    while r < zmax
        len = floor(r/dim);
        
        xi = round(px + len*cos(phi));
        yi = round(py + len*sin(phi));
        
        if xi<1 || yi<1 || xi > n || yi > m
            break
        end
        
        %fprintf('xi:%d  - yi:%d  / r:%.2f  phi:%.2f\n',xi,yi,r,phi)
        
        map_inner(xi,yi) = 1;
        if map(xi,yi)
            map_front(xi,yi) = 1;
            break;
        end
        if xi == n || yi == m
            map_missed(xi,yi) = 1;
            break;
        end
        r = r + dim*.9;
    end
    if r >= zmax
        map_missed(xi,yi) = 1;
    end
end

inner = zeros(sum(sum(map_inner)),2);
front = zeros(sum(sum(map_front)),2);
missed = zeros(sum(sum(map_missed)),2);
in = 1;
for i=1:n
    for j=1:m
        if map_inner(i,j)
            inner(in,:) = [i,j];
            in = in+1;
        end
    end
end

in = 1;
for i=1:n
    for j=1:m
        if map_front(i,j)
            front(in,:) = [i,j];
            in = in+1;
        end
    end
end

in = 1;
for i=1:n
    for j=1:m
        if map_missed(i,j)
            missed(in,:) = [i,j];
            in = in+1;
        end
    end
end

% fprintf('CELLS:%d %d  frontier: %d %d\n',size(inner,1),size(inner,2),size(front,1),size(front,2))
% 
% figure(2)
% mm = ones(size(map))*0.5;
% 
% for i=1:size(inner,1)
%     mm(inner(i,1),inner(i,2)) = 0;
% end
% 
% for i=1:size(front,1)
%     mm(front(i,1),front(i,2)) = 1;
% end
% for i=1:size(missed,1)
%     mm(missed(i,1),missed(i,2)) = 0.6;
% end
% mm = log(mm);
% plot_map(mm,dim)

end