%%a-star ugv: MAIN: ADQUISICION DE MAPA DE OBSTACULOS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%obstacle- map pickup (plancha)
clc
clear all

%DEFINE THE 2-D MAP ARRAY
MAX_X=10;
MAX_Y=10;
MAX_VAL=10;
paso=MAX_X/20;

obstac=[];

% OPEN_COUNT=0;
% CLOSED_COUNT=0;

%This array stores the coordinates of the map and the 
%Objects in each coordinate

% Obtain Obstacle, Target and Robot Position
% Initialize the MAP with input values
% Obstacle=-1,Target = 0,Robot=1,Space=2
i=0;j=0;
MAP=zeros(MAX_X,MAX_Y);

%0: libre, 1:obstacylo, -1: closed

x_val = 1;
y_val = 1;
axis([1 MAX_X+1 1 MAX_Y+1])
grid on;
hold on;
n=0;%Number of Obstacles

pause(1);
h=msgbox('Please Select the Target using the Left Mouse button');
uiwait(h,5);
if ishandle(h) == 1
    delete(h);
end
xlabel('Please Select the Target using the Left Mouse button','Color','black');
but=0;
while (but ~= 1) %Repeat until the Left button is not clicked
    [xval,yval,but]=ginput(1);
end
varx=floor(xval);        %X Coordinate of the Target
vary=floor(yval);        %Y Coordinate of the Target
xval=varx-mod(varx,paso*2);
yval=vary-mod(vary,paso*2);
xTarget=xval;
yTarget=yval;


MAP(xTarget,yTarget)=3;                     %Initialize MAP with location of the target
plot(xTarget+paso,yTarget+paso,'bd','Linewidth',2.5);
text(xTarget+paso,yTarget+paso,'Target')

pause(2);
h=msgbox('Select Obstacles using the Left Mouse button,to select the last obstacle use the Right button');
  xlabel('Select Obstacles using the Left Mouse button,to select the last obstacle use the Right button','Color','blue');
uiwait(h,10);
if ishandle(h) == 1
    delete(h);
end
while but == 1
    [xval,yval,but] = ginput(1);
    varx=floor(xval);
    vary=floor(yval);
    temp=varx-mod(varx,paso*2);
    if temp~=0
        xval=temp;
    else
        xval=varx;
    end
    temp=vary-mod(vary,paso*2);
    if temp~=0
        yval=temp;
    else
        yval=vary;
    end
    if MAP(xval,yval)~=1
        MAP(xval,yval)=1;           %pone obstaculos
        obstac=[obstac; xval+1 yval+1];
    end
    plot(xval+paso,yval+paso,'ro','Linewidth',2);
 end%End of While loop
 
pause(1);

h=msgbox('Please Select the Vehicle initial position using the Left Mouse button');
uiwait(h,5);
if ishandle(h) == 1
    delete(h);
end
xlabel('Please Select the Vehicle initial position ','Color','black');
but=0;
while (but ~= 1) %Repeat until the Left button is not clicked
    [xval,yval,but]=ginput(1);
    varx=floor(xval);
    vary=floor(yval);
    xval=varx-mod(varx,paso*2);
    yval=vary-mod(vary,paso*2);
end
xStart=xval;%Starting Position
yStart=yval;%Starting Position
MAP(xStart,yStart)=3;                     %Initialize MAP with location of the start

plot(xval+paso,yval+paso,'bo','Linewidth',2.5);
hold on
%End of obstacle-Target pickup

%% Astar:  borde de 1s a MAP para exploracion
MAP=[ones(1,MAX_X+2); ones(MAX_Y,1) MAP ones(MAX_Y,1); ones(1,MAX_X+2)];

aStar(MAP,xStart+1,yStart+1,0,xTarget+1,yTarget+1,0,MAX_X+2,MAX_Y+2,obstac);