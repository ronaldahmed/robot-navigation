%% Funcion de insercion de elemento en el heap
function open=push(OpenList,x,y,xp,yp,F,G,H,d_actual)
    new=[x y xp yp Inf G H d_actual];    
    OpenList=[OpenList;new];
    
    heap_s=size(OpenList,1);
    OpenList=decrease_key(OpenList,heap_s,F);

    open=OpenList;    

end