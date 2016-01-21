function plot_map(M,dim,print_border,prior,logform)
% Plotea mapa como un grid
% la intensidad de cada celda es proporcional a la probabilidad de estar
% ocupada

if nargin < 3
    print_border = 1;
    prior=0.5;
    logform=0;
end

mapborder_width = 0;
if print_border
    mapborder_width = 5;
end

[n,m] = size(M);
hold on

if logform==1
    M(M>0)=1;
    M = exp(M);
    prior = exp(prior);
else
    M = M - min(M(:));
    if range(M(:))~=0
        M = (M/range(M(:)));
    end
end

for i=1-mapborder_width : n+mapborder_width
    for j=1-mapborder_width : m+mapborder_width
        x = [(i-1)*dim,i*dim,i*dim,(i-1)*dim];
        y = [(j-1)*dim,(j-1)*dim,j*dim,j*dim];
        temp = 0;
        if i<1 || i>n || j<1 || j>m
            temp = [1-prior, 1-prior, 1-prior];
        else
            temp = (1-M(i,j))*ones(1,3);
        end
        fill(x,y, temp,'EdgeColor','None')
    end
end

axis([ -mapborder_width   *dim;
       (n+mapborder_width)*dim;
       -mapborder_width   *dim;
       (m+mapborder_width)*dim]')

% imshow(1-M)
% axis on
% xticklabels = 0:n*dim;
% xticks = linspace(1, m, numel(xticklabels));
% 
% yticklabels = 0:m*dim;
% yticks = linspace(1, n, numel(yticklabels));
% 
% set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
% set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
% 
xlabel('x(m)')
ylabel('y(m)')

end