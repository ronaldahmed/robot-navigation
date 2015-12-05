function plot_map(M,dim)
% Plotea mapa como un grid
% la intensidad de cada celda es proporcional a la probabilidad de estar
% ocupada
[n,m] = size(M);

% M = (M -ones(size(M))*mn) / d;
%uno = ones(n,m);
%M = uno - (uno+exp(M)).^(-1);
M = exp(M);
hold on
for i=1:n
    for j=1:m
        x = [(i-1)*dim,i*dim,i*dim,(i-1)*dim];
        y = [(j-1)*dim,(j-1)*dim,j*dim,j*dim];
        temp = (1-M(i,j))*ones(1,3);
        fill(x,y, temp,'EdgeColor','None')
       % fill(x,y,[0.4,0.8, 1-M(i,j)],'EdgeColor','None')
    end
end

axis([0,n*dim,0,m*dim])
xlabel('x(m)')
ylabel('y(m)')

end