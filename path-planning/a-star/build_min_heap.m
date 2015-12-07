%% Construye min-heap
function Open=build_min_heap(OpenList)
    heap_s=length(OpenList);
    for i=floor(heap_s/2):-1:1
        OpenList=min_heapify(OpenList,i);
        Open=OpenList;
    end
end