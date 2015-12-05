function N = preprocess_nearest_wall(map,dim)
% rango 0 a 1
map = exp(map);
[n,m] = size(map);
N = ones(n,m)*Inf;

dx = [0,-1,-1,-1, 0, 1,1,1];
dy = [1, 1, 0,-1,-1,-1,0,1];

border = [];

for i=1:n
    for j=1:m
        for k=1:8
            x = i + dx(k);
            y = j + dy(k);
            is_in = x>=1 && x<=n && y>=1 && y<=m;
            if is_in
                if map(i,j) && ~map(x,y)
                    border = [border;i,j];
                    break;
                end
            end
        end
    end
end

bn = size(border,1);

for i=1:n
    for j=1:m
        for k=1:bn
            u = border(k,1);
            v = border(k,2);
            dist = 1- exp(-4*hypot(i-u,v-j)*dim);
            N(i,j) = min(N(i,j),dist);
        end
        if map(i,j) && N(i,j)~=0
            N(i,j) = N(i,j)*0.1;
        end
    end
end

% mx = max(max(N));
% mm = ones(size(N)) - N./mx;
% mm = log(mm);
% 
% figure(4)
% plot_map(mm,dim)

end