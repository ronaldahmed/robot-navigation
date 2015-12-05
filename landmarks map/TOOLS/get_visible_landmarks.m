function [vlm,N] = get_visible_landmarks(x,mapa,MAX_RANGE)
    N = 0;
    nlm = size(mapa,2);
    x2 = [x(1)+cos(x(3)), x(2)+sin(x(3))]';
    u = x2-x(1:2);
    indices = zeros(1,nlm);
   
    for i=1:nlm
        mx = mapa(1,i);
        my = mapa(2,i);
        mt = mapa(3,i);
        dist = hypot( mx-x(1) , my-x(2));
        v = mapa([1,2],i) - x(1:2);
        %pp = sum(u.*v);
        phi = atan2(my-x(2),mx-x(1))-x(3);
             
        
        if cos(phi)>=0 && dist <= MAX_RANGE
        %if pp >= 0 && dist <= MAX_RANGE
            indices(i) = 1;
        end
    end
    vlm = mapa(:,indices==1);
    N = size(vlm,2);
end