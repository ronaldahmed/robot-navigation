%% modificacion de F (key) y reordenamiento del heap
function open=decrease_key(OpenList,i,F)
    OpenList(i,5)=F;
    
    while i>1 && OpenList(floor(i/2),5)> OpenList(i,5)
        temp=OpenList(i,:);
        OpenList(i,:)=OpenList(floor(i/2),:);
        OpenList(floor(i/2),:)=temp;
        i=floor(i/2);
    end

    open=OpenList;
end