clc;
clear all;
close all;
mapa_real = load('../maps/map3.csv');
mapa_real = mapa_real';
grid_dim = .04;
% mapa_real = make_map(grid_dim);
mapa_real = log(mapa_real);
[N,M] = size(mapa_real);

figure(1)
plot_map(mapa_real,grid_dim)


%----------------------------------------------------------------------%
% 3. PARAMETROS DEL SENSOR "RANGE-FINDER"
%----------------------------------------------------------------------%
MAX_RANGE = 2;                      % Rango maximo
sigmaR = 0.05;                      % Desviaciones estandar
sigmaB = (2.0*pi/180);
sigmaC = 0.05;
std_sensor = [sigmaR,sigmaB,sigmaC];
betha = (2.0*pi/180)*10;            % beam angle

zhit = 0.5;
zshort = 0.1;
zmax = 500;
zrand = 1;
Z_weights = [zhit,zshort,zmax,zrand];   % pesos de ponderacion de las fuentes de error
S = (-pi/2:(5*pi/180):pi/2);            % angulos de 18 rayos laser
xsensor = [0.3,0]';                     % pos relativa del sensor en el robot

%  HANDLERS PARA ANIMAR LAS MEDIDAS DE LOS SENSORES
K = 500;
handle_sensor_ray = zeros(1,K);
for f=1:K
    handle_sensor_ray(f) = plot(0,0,'y');
end
handle_sensor_frontier = plot(0,0,'ro');
handle_sensor_missed = plot(0,0,'go');


%----------------------------------------------------------------------%
% 5. PARAMETROS DEL ROBOT
%----------------------------------------------------------------------%
%  PARAMETROS MECANICOS
D = 0.1;        % Diametro de las ruedas
L = 0.15;       % Distancia de las ruedas al punto central del robot
Wr = 0.2;       % ancho robot
Hr = 0.6;       % largo robot
Dim = [Wr Hr D];

%  PARAMETROS DE ERROR
alpha1 = 1e-2;  alpha2 = 1e-3;  
alpha3 = 1e-3;  alpha4 = 1e-2;
alpha5 = 1e-3;  alpha6 = 1e-3;
alpha_VELOCITY = [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6];

%  CONFIGURACION INICIAL
%x = [0.25;1;0];
x = [1;0.8;0];
plot_robot(x, L, 'r',Dim)

[cells,frontier,missed] = get_cells_range(x,S,mapa_real,grid_dim,MAX_RANGE);
%[cells,frontier,missed] = get_visible_frontier(x,mapa_real,grid_dim,MAX_RANGE);

zt = range_finder_grid_model(x, frontier, missed,grid_dim,std_sensor, size(mapa_real,1));

% -> Iteramos en todas las mediciones y ploteamos
end_points=[];
missed_points=[];

for k=1:size(zt,1)
    if zt(k,2)>0
        d = -grid_dim/2;
    else
        d = grid_dim/2;
    end
    
    if zt(k,1) ~= Inf
        ppx = [x(1),x(1) + zt(k,1)*cos(zt(k,2) + x(3)) + d];
        ppy = [x(2),x(2) + zt(k,1)*sin(zt(k,2) + x(3)) + d];
        end_points = [end_points;ppx(2),ppy(2)];        
    else
        dist = MAX_RANGE + std_sensor(1)*rand(1);
        ppx = [x(1),x(1) + dist*cos(zt(k,2) + x(3)) + d];
        ppy = [x(2),x(2) + dist*sin(zt(k,2) + x(3)) + d];
        missed_points = [missed_points;ppx(2),ppy(2)];
    end
    set(handle_sensor_ray(k),'xdata', ppx, 'ydata', ppy)
end
if size(end_points,1)~=0
    set(handle_sensor_frontier,'xdata', end_points(:,1), 'ydata', end_points(:,2))
end
if size(missed_points,1)~=0
    set(handle_sensor_missed,'xdata', missed_points(:,1), 'ydata', missed_points(:,2))
end
