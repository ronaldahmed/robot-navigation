%% busqueda para G
function k=search(OpenList,x,y)
    l=size(OpenList,1);
    for i=1:l
        if x==OpenList(i,1) && y==OpenList(i,2)
            k=i;
            return;
        end
    end
    k=-1;
end