function plot_robot(x,L,color,Dim)
% extraemos la data
xx=x(1);
yy=x(2);
theta=x(3);
a=Dim(1);
b=Dim(2);
d=Dim(3);
N=300;
% condiciones de los parametros
if nargin<2
    error('PLOT_ROBOT:numero de ')
end
if nargin<3
    error('PLOT_ROBOT:falta ingresar el color del plot ')
end
if(theta>pi||theta<=-pi)
    error('PLOT_ROBOT:el angulo theta no esta en el rango ')
end
% Dibujamos el robot
x1=xx+L*sin(theta);
y1=yy-L*cos(theta);

x2=x1+b*cos(theta);
y2=y1+b*sin(theta);

x4=xx-L*sin(theta);
y4=yy+L*cos(theta);

x3=x4+b*cos(theta);
y3=y4+b*sin(theta);
if color=='r'
patch([x1 x2 x3 x4],[y1 y2 y3 y4],[1 0 0]);
%plot([x1 x2 x3 x4 x1],[y1 y2 y3 y4 y1],color,'lineWidth',2);
elseif color=='b'
    patch([x1 x2 x3 x4],[y1 y2 y3 y4],[0 0 1]);
end

% ploteo de las ruedas
% plot right back wheel
xrbw1=x1-d/2*cos(theta);
yrbw1=y1-d/2*sin(theta);

xrbw2=xrbw1+d/4*sin(theta);
yrbw2=yrbw1-d/4*cos(theta);

xrbw3=xrbw2+d*cos(theta);
yrbw3=yrbw2+d*sin(theta);

xrbw4=xrbw1+d*cos(theta);
yrbw4=yrbw1+d*sin(theta);
patch([xrbw1 xrbw2 xrbw3 xrbw4],[yrbw1 yrbw2 yrbw3 yrbw4],[0 0 0]);
% plot left back wheel
xlbw1=x4-d/2*cos(theta);
ylbw1=y4-d/2*sin(theta);

xlbw2=xlbw1+d*cos(theta);
ylbw2=ylbw1+d*sin(theta);

xlbw4=xlbw1-d/4*sin(theta);
ylbw4=ylbw1+d/4*cos(theta);

xlbw3=xlbw4+d*cos(theta);
ylbw3=ylbw4+d*sin(theta);
patch([xlbw1 xlbw2 xlbw3 xlbw4],[ylbw1 ylbw2 ylbw3 ylbw4],[0 0 0]);
% plot right front wheel
xrfw1=x2-d/2*cos(theta);
yrfw1=y2-d/2*sin(theta);

xrfw2=xrfw1+d/4*sin(theta);
yrfw2=yrfw1-d/4*cos(theta);

xrfw3=xrfw2+d*cos(theta);
yrfw3=yrfw2+d*sin(theta);

xrfw4=xrfw1+d*cos(theta);
yrfw4=yrfw1+d*sin(theta);
patch([xrfw1 xrfw2 xrfw3 xrfw4],[yrfw1 yrfw2 yrfw3 yrfw4],[0 0 0]);
% plot left front wheel
xlfw1=x3-d/2*cos(theta);
ylfw1=y3-d/2*sin(theta);

xlfw2=xlfw1+d*cos(theta);
ylfw2=ylfw1+d*sin(theta);

xlfw4=xlfw1-d/4*sin(theta);
ylfw4=ylfw1+d/4*cos(theta);

xlfw3=xlfw4+d*cos(theta);
ylfw3=ylfw4+d*sin(theta);
patch([xlfw1 xlfw2 xlfw3 xlfw4],[ylfw1 ylfw2 ylfw3 ylfw4],[0 0 0]);
end