function printClosed(current_x,current_y,current_dir,dmin)
%% A* normal
    plot(current_x-1+dmin/2,current_y-1+dmin/2,'bd')    
    hold on
    %     0 45 90 135 180 225 270 315
    xdyd=[1 1  0  -1  -1  -1   0   1;
          0 1  1   1   0  -1  -1  -1];
    x=[(current_x-1+dmin/2) , (current_x-1+dmin/2*( xdyd(1,current_dir/45+1)+1) )];
    y=[(current_y-1+dmin/2) , (current_y-1+dmin/2*( xdyd(2,current_dir/45+1)+1) )];
    plot(x,y,'g')

%% Hybrid State A*

end