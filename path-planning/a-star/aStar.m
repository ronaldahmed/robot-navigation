%%  A* algorithm

function [path_x path_y]=aStar(MAP,xStart, yStart, dirStart, xTarget, yTarget, dirTarget,MAX_X,MAX_Y,obstac)

    %% Format open List
    %% x | y | x_p | y_p | f | g | h | teta
    
    OpenList=[];        
    
    dmin=(MAX_X-2)/10;           %% longitud de cada recuadro de red

    ParentList=[];
    
    current_x=xStart
    current_y=yStart
    current_dir=dirStart
    hStart=heuristic(xStart,yStart,dirStart,xTarget,yTarget,dirTarget);

disp('enters loop-----------')

    
    alcance=3;  %alcance de exploracion siempre es 3cuadros dentro de los 360°
    
    %% 0:libre, 1:obstaculo,  -1: closed map    2: en open
    MAP(xStart,yStart)=-1;                      %% add to closed map;
    
    ParentList((xStart-1)*MAX_X+yStart , 1)=-1;     %% stop criteria for path_finding function    
    ParentList((xStart-1)*MAX_X+yStart , 2)=dirStart;
    
    g_actual=0;

    heap_s=1;

    
    %% Loop principal -----------------------------------------------------------------
    
    while ((current_x~=xTarget || current_y~=yTarget || current_dir~=dirTarget) && heap_s~=0)
        adjacents=toExplore(current_dir,dmin);   %% set adjacent points regarding non-holonimic vehicle
   
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
                
    assignin('base','x',x);
    assignin('base','y',y);
            
            G=g_actual+1;
            H=heuristic(x,y,dir_exp,xTarget,yTarget,dirTarget);
            F=G+H;
            
            if MAP(x,y)==0  || MAP(x,y)==3  % || MAP(x,y)==-1    %% si esta libre, o si es target
                OpenList=push(OpenList,x,y,current_x,current_y,F,G,H,dir_exp);                
                MAP(x,y)=2;
                
            else if MAP(x,y)==2 %|| MAP(x,y)==-1 %% si esta en el open list
                    k=search(OpenList,x,y);
                    if  k~=-1 && OpenList(k,6)> G
                        OpenList(k,6)=G;                            %% update G
                        OpenList(k,8)=dir_exp;

                        F=OpenList(k,6)+OpenList(k,7);              %% update F
                        OpenList=decrease_key(OpenList,k,F);

                        OpenList(k,3)=current_x;                    %update parents
                        OpenList(k,4)=current_y;                    %update parents                    
                    end  
                    
                 else if MAP(x,y)==-1
%                         priority=H/hStart;                         
%                         p2=hStart/(sqrt(2)*(MAX_X-2));
                         if H<=(sqrt(2)*dmin*2)
disp('reprioriza')                             
                            OpenList=push(OpenList,x,y,current_x,current_y,F,G,H,dir_exp);                    
%                             MAP(x,y)=2;
                         end
                      end
                 end

            end            
        end

        printClosed(current_x,current_y,current_dir,dmin);

        MAP(current_x,current_y)=-1;                %%ya explorado, added to closed map        
        
        x2=OpenList(1,1);
        y2=OpenList(1,2);
        d2=OpenList(1,8);
        xp=OpenList(1,3);
        yp=OpenList(1,4);
        

disp('---Antes de extract_______________________')
OpenList

        ParentList((x2-1)*MAX_X+y2 , 1)=(xp-1)*MAX_X+yp;   %PL( x y , dir )
        ParentList((x2-1)*MAX_X+y2 , 2)=d2;
         
        current_x=x2;
        current_y=y2;
        current_dir=OpenList(1,8);
      
        OpenList=extract_min(OpenList);     %% pop escogido del Openlist            
        heap_s=size(OpenList,1);
        g_actual=g_actual+1;

              
assignin('base','open',OpenList);    
assignin('base','MAP',MAP);

    end  %% fin loop principal ----------------------------------------------------------------------------------
    
    if heap_s==0
        disp('NO PATH')
    else
        pathBuilder=[xTarget-1 yTarget-1];        

ParentList        

        it=ParentList((xTarget-1)*MAX_X+yTarget , 1);        
        ActionList(1)=ParentList((xTarget-1)*MAX_X+yTarget , 2);

        i=2;
        
        while(it~=-1)
            y=mod(it,MAX_X) -1;
            x=floor(it/MAX_X)+1 -1;
            pathBuilder=[pathBuilder; x y];
            
            ActionList(i)=ParentList(it , 2);
            it=ParentList(it , 1);
            
            i=i+1;
        end
                
        plot(pathBuilder(:,1)+dmin/2,pathBuilder(:,2)+dmin/2,'b','Linewidth',2)
        
        %% Lista de acciones de giro:
   ActionList
        n=length(ActionList);
        ActionList=ActionList(end:-1:1);
        for i=1:n-2
           ActionList(i)=ActionList(i+1)-ActionList(i);
        end       
        %%
        ActionList=ActionList(1:end-1)
    end


end
