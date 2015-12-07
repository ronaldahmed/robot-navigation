%%  toExplore()  regresa direcciones de puntos disponibles a explorar segun
%  la estructura no-holonomica del vehiculo.
function adjacents=toExplore(current_dir,dmin)
%% Astar normal
    % variables son puntos cardinales
%     e=dmin*[ 1  0]';
%     w=dmin*[-1  0]';
%     n=dmin*[ 0  1]';
%     s=dmin*[ 0 -1]';
%     ne=dmin*[1  1]';
%     se=dmin*[1 -1]';
%     nw=dmin*[-1 1]';
%     sw=dmin*[-1 -1]';
%     
%     switch current_dir
%         case 0
%             adjacents=[ne e se];
%         case 45
%             adjacents=[n ne e];
%         case 90
%             adjacents=[nw n ne];
%         case 135
%             adjacents=[w nw n];
%         case 180
%             adjacents=[sw w nw];
%         case 225
%             adjacents=[s sw w];
%         case 270
%             adjacents=[se s sw];
%         case 315
%             adjacents=[e se s];
%         otherwise
%             adjacents=[];
%     end
 %% Hybrid-A*
    a=1.01*dmin*sqrt(2);
    b=.7*dmin;
    dir=current_dir*pi/180;                 %correcion a rad
    u=a*14/16;
    v_left=b*(1-sqrt(1-u*u/(a*a) ));
    v_right=-v_left;
    x_right=u*cos(dir)- v_right*sin(dir);
    y_right=u*sin(dir)+ v_right*cos(dir);
    
    x_left=u*cos(dir)- v_left*sin(dir);
    y_left=u*sin(dir)+ v_left*cos(dir);
    
    x_forw=a*cos(dir);
    y_forw=a*sin(dir);
    adjacents= [x_left  x_forw x_right;
                y_left  y_forw y_right];    
end