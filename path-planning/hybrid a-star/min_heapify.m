%% Min-heapify puja el top al fondo del heap

function Open=min_heapify(OpenList,i,heap_s)
    l=2*i;
    r=2*i+1;
    
    if r<=heap_s || l<=heap_s
        if l<=heap_s && OpenList(l,5)<OpenList(i,5);
            min=l;
        else
            min=i;
        end
        if r<=heap_s && OpenList(r,5)<OpenList(min,5);
            min=r;
        end
        if min ~=i
            temp=OpenList(min,:);
            OpenList(min,:)=OpenList(i,:);
            OpenList(i,:)=temp;
            OpenList=min_heapify(OpenList,min,heap_s);
        end
    end
    Open=OpenList;
end