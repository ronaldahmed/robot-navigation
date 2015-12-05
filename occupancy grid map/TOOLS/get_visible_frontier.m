function [cells,front,missed] = get_visible_frontier(X,map,dim,MAX_RANGE)
xx = X(1);
yy = X(2);
theta = X(3);

% recobrar de prob log el mapa real
map = exp(map);

% filas Y, columnas X
[n,m] = size(map);
px = floor(xx/dim) + 1;
py = floor(yy/dim) + 1;
len = floor(MAX_RANGE/dim) + 1;

%fprintf('x:%.3f  y:%.3f  ran:%.2f\n',xx,yy,MAX_RANGE)
%fprintf('px:%d  py:%d  len:%d\n',px,py,len)

dx = [0,-1, 0,1];
dy = [1, 0,-1,0];

used = map;
d = ones(size(map))*Inf;
d(px,py) = 0;

Q = [px py];
cells=[];
front = [];

while size(Q,1)~= 0  
    cells = [cells; Q(end,:)];
    cur_x = Q(end,1);
    cur_y = Q(end,2);
    Q = Q(1:(end-1),:);
    
    if ~used(cur_x,cur_y)
        used(cur_x,cur_y) = 1;
        for i = 1:4
            x = cur_x + dx(i);
            y = cur_y + dy(i);
            phi = atan2(y - py,x - px)-theta;
            r = hypot(px - x,py - y);
            is_in = (x>=1 && x<=n && y>=1 && y<=m);
            
            if is_in && cos(phi)>=0 && r <= len && isSensable(X,[x,y],map,used,dim)
                if (~used(x,y) && (d(x,y) > d(cur_x,cur_y) + 1) ) || map(x,y)
                    d(x,y) = d(cur_x,cur_y) + 1;
                    Q = [Q;x,y];
                end
            end
        end
    elseif map(cur_x,cur_y)         % frontera con pared
        front = [front;cur_x,cur_y];
    end
end
c = size(cells,1);
cells = [cells;front];
missed = [];
for i = 1:c
    x = cells(i,1);
    y = cells(i,2);
    r = hypot(px - x,py - y);
    if abs(r-len) < 1 || x==1 || y == 1 || x==n || y==m
        missed = [missed;x,y];
    end
end

% fprintf('CELLS:%d %d  frontier: %d %d\n',size(cells,1),size(cells,2),size(front,1),size(front,2))
% 
% figure(2)
% mm = ones(size(map))*0.5;
% 
% for i=1:size(cells,1)
%     mm(cells(i,1),cells(i,2)) = 0;
% end
% 
% for i=1:size(front,1)
%     mm(front(i,1),front(i,2)) = 1;
% end
% for i=1:size(missed,1)
%     mm(missed(i,1),missed(i,2)) = 1;
% end
% mm = log(mm);
% plot_map(mm,dim)
end