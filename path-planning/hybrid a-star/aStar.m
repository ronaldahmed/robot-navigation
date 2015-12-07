%%  Hybrid-state A* algorithm

function [path_x path_y]=aStar(MAP,xStart, yStart, dirStart, xTarget, yTarget, dirTarget,MAX_X,MAX_Y)

    %% Format open List
    %% x | y | x_p | y_p | f | g | h | teta
    
    OpenList=[];    
    dmin=(MAX_X-2)/10;           %% longitud de cada recuadro de red
    
    current_x=xStart
    current_y=yStart
    current_dir=dirStart
    xtemp=0;
    ytemp=0;
    hStart=heuristic(xStart,yStart,dirStart,xTarget,yTarget,dirTarget);

disp('enters loop-----------')

    
    alcance=3;  %alcance de exploracion siempre es 3cuadros dentro de los 360°
    
    %% 0:libre, 1:obstaculo,  -1: closed map    2: en open       3: target
    MAP(xStart,yStart)=-1;                      %% add to closed map;
    
    ParentList=[xStart yStart dirStart -1 -1];     %% stop criteria for path_finding function            
    g_actual=0;
    heap_s=1;

    
    %% Loop principal -----------------------------------------------------------------
    
    while (( floor(current_x)~=xTarget || floor(current_y)~=yTarget || current_dir~=dirTarget) && heap_s~=0)
        adjacents=toExplore(current_dir,dmin);                      %% set adjacent points regarding non-holonimic vehicle
   
    fprintf('cx: %d\n',current_x)
    fprintf('cy: %d\n',current_y)     
    fprintf('cdir: %d\n',current_dir)         
            
        for i=1:alcance
            dir_exp=current_dir+(2-i)*45;
            if dir_exp==360
                dir_exp=0;
            end
            if dir_exp<0
                dir_exp=dir_exp+360;
            end
    
            x=adjacents(1,i)+current_x;
            y=adjacents(2,i)+current_y;
                        
            xgrid=floor(x);
            ygrid=floor(y);
            
            if xgrid> MAX_X
                xgrid=MAX_X;
            end
            if ygrid> MAX_Y
                ygrid=MAX_Y;
            end
            if xgrid<=0
                xgrid=1;
            end
            if ygrid<=0
                ygrid=1;
            end
                                               
    assignin('base','x',x);
    assignin('base','y',y);
            
            G=g_actual+1;
            H=heuristic(x,y,dir_exp,xTarget,yTarget,dirTarget);
            F=G+H;
            
            if MAP(xgrid,ygrid)==0 || MAP(xgrid,ygrid)==3   || (xtemp==xgrid && ytemp==ygrid && MAP(xgrid,ygrid)~=1)       %% si esta libre, o si se puede repriorizar
                OpenList=push(OpenList,x,y,current_x,current_y,F,G,H,dir_exp);                
                MAP(xgrid,ygrid)=2;
                
                
            else if MAP(xgrid,ygrid)==2                             %% si esta en el open list
                    k=search(OpenList,x,y);
                    if  k~=-1 && OpenList(k,6)> G
                         OpenList(k,6)=G;                            %% update G
                         OpenList(k,8)=dir_exp;

                         F=OpenList(k,6)+OpenList(k,7);              %% update F
                         OpenList=decrease_key(OpenList,k,F);

                         OpenList(k,3)=current_x;                    %update parents
                         OpenList(k,4)=current_y;                    %update parents                    
                    end  
                 else if MAP(xgrid,ygrid)==-1
                         priority=H/hStart;
                         if priority<=0.2
                            OpenList=push(OpenList,x,y,current_x,current_y,F,G,H,dir_exp);                    
%                           MAP(x,y)=2;
                         end
                      end
                end
            end
            xtemp=xgrid;
            ytemp=ygrid;
        end

        printClosed(current_x,current_y,current_dir,dmin);

        MAP(floor(current_x),floor(current_y) )=-1;                %%ya explorado, added to closed map        
        
        x2=OpenList(1,1);
        y2=OpenList(1,2);
        d2=OpenList(1,8);
        xp=OpenList(1,3);
        yp=OpenList(1,4);
        

disp('---Antes de extract_______________________')
OpenList

        ParentList=[ParentList; x2 y2 d2 xp yp]; %PL( x y , dir )        
         
        current_x=x2;
        current_y=y2;
        current_dir=d2;
      
        OpenList=extract_min(OpenList);     %% pop escogido del Openlist            
        heap_s=size(OpenList,1);
        g_actual=g_actual+1;

              
assignin('base','open',OpenList);    
assignin('base','MAP',MAP);

    end
%%fin loop principal ----------------------------------------------------------------------------------
    
    if heap_s==0
        disp('NO PATH')
    else
        pathBuilder=[current_x-1, current_y-1];        

ParentList

        it=ParentList(end, 4);
        ActionList(1)=ParentList(end , 3);
        k=size(ParentList,1);
        h=k;
        i=2;
        
        while(it~=-1)
            for r=1:h
                if (ParentList(k,4)== ParentList(r,1) && ParentList(k,5)== ParentList(r,2) ) && k~=r
                    k=r;
                    ActionList(i)=ParentList(k, 3);
                    x=ParentList(k,1)-1;
                    y=ParentList(k,2)-1;
                    pathBuilder=[pathBuilder; x y];
                    it=ParentList(k, 4);
            
                    i=i+1;        
                    break;
                end
            end                        
        end
                
%        plot(pathBuilder(:,1)+dmin/2,pathBuilder(:,2)+dmin/2,'b','Linewidth',2)

        %% Lista de acciones de giro:
ActionList
        
        n=length(ActionList);
        ActionList=ActionList(end:-1:1);
        
        pathPloter(pathBuilder, ActionList,dirStart,dmin);
        
        for i=1:n-1
            ActionList(i)=ActionList(i+1)-ActionList(i);
        end       

ActionList=ActionList(1:end-1)
pathBuilder


    end

end
