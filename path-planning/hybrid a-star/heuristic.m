%% Heuristica XD
function h=heuristic(x,y,dir,xT,yT,dT)
%     h=Inf*ones(mx,my);
%     for i=2:mx-1
%         for j=2:my-1
%             h(i,j)=sqrt( (i-xT)*(i-xT) + (j-yT)*(j-yT) );
%         end
%     end

  h=sqrt( (x-xT)*(x-xT) + (y-yT)*(y-yT) );      %% provisional
end