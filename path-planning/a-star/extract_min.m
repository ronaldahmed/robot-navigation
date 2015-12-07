%% Extraccion de elemento de minF con maxG: extrae y elimina del openlist

function open=extract_min(OpenList)
    heap_s=size(OpenList,1);
    
    if heap_s>1    
        OpenList(1,:)=OpenList(heap_s,:);

        OpenList=OpenList(1:heap_s-1,:);
        heap_s=heap_s-1;
        OpenList=min_heapify(OpenList,1,heap_s);
        open=OpenList;
    else
        open=[];
    end
        
end