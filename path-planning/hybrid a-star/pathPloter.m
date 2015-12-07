%% Plot the path regarding kinematic model
function pathPloter(path, action, dirStart,dmin)
disp('plot-----------------------------------------------------')
    path(:,1)=path(end:-1:1,1);
    path(:,2)=path(end:-1:1,2);
    a=1.01*dmin*sqrt(2);
    b=.7*dmin;   
    ur=a*14/16;
    
    n=size(path,1);
    cd=dirStart;        
    
    for i=1:n-1
        x=[];
        y=[];
        u=0:dmin/100 :ur;
        dir=cd*pi/180;
        
        if (action(i+1)-action(i) )==45
            v_left=b*(1-sqrt(1-(u.*u)./(a*a) ));            
            x=u.*cos(dir)- v_left.*sin(dir);
            y=u.*sin(dir)+ v_left.*cos(dir);
            
            
        elseif (action(i+1)-action(i) )==-45
            v_right=-b*(1-sqrt(1-(u.*u)./(a*a) ));
            x=u.*cos(dir)- v_right.*sin(dir);
            y=u.*sin(dir)+ v_right.*cos(dir);
            
        elseif (action(i+1)-action(i) )==0
            x=(16/14)*u.*cos(dir);
            y=(16/14)*u.*sin(dir);
        end
        x=x+ path(i,1)+dmin/2;
        y=y+ path(i,2)+dmin/2;
        plot(x,y,'b','Linewidth',2)
        cd=action(i+1);
    end
end