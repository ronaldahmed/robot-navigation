% DEMO EKF_SLAM con mapa grid
% usando S laser beams y likelihood field
clc
clear all
close all

% VARIABLES GLOBALES
global NN       % Numero de celdas
global done     % Flags de las celdas ocupadas visitadas
addpath('../maps/')
addpath('../TOOLS/')
%----------------------------------------------------------------------%
% 1. CONFIGURACION DEL MAPA GRID DE OCUPACION
%----------------------------------------------------------------------%
%  LECTURA DEL MAPA
%lectura mapa  ... %1: ocupado , 0 : libre

mapa_real = load('map4.csv');
mapa_real = mapa_real';
grid_dim = .04;
%mapa_real = make_map(grid_dim);
NN = sum(sum(mapa_real==1));
mapa_real = log(mapa_real);
[N,M] = size(mapa_real);

%----------------------------------------------------------------------%
% 2. CONFIGURACION DEL MAPA DE PUNTOS DE PASO DEL ROBOT "VIA_POINTS"
%----------------------------------------------------------------------%
via_points =...
   [.5,1;
    3 ,1;
    3.3,1.5;
    3.3,2.7;
    3  , 3;
    0.7,3;
    0.4,2.7];
via_points = via_points';   % Cada punto es una columna


%----------------------------------------------------------------------%
% 2. PLOTEAMOS EL MAPA REAL Y EL ENTORNO INICIAL
%----------------------------------------------------------------------%
figure(1)
plot_map(mapa_real,grid_dim)
%plot(via_points(1,:),via_points(2,:), 'b+','MarkerSize',10,'LineWidth',2)
%hold on
xlabel('x(m)')
ylabel('y(m)')

%----------------------------------------------------------------------%
% 3. PARAMETROS DEL SENSOR "RANGE-FINDER"
%----------------------------------------------------------------------%
MAX_RANGE = 1.5;                      % Rango maximo
sigmaR = 0.01;                      % Desviaciones estandar
sigmaB = (1.0*pi/180);
sigmaC = 0.05;
std_sensor = [sigmaR,sigmaB,sigmaC];
betha = (2.0*pi/180)*10;            % beam angle

zhit = 0.5;
zshort = 0.1;
zmax = 500;
zrand = 1;
Z_weights = [zhit,zshort,zmax,zrand];   % pesos de ponderacion de las fuentes de error
S = (-pi/2:(5*pi/180):pi/2);            % angulos de 18 rayos laser
xsensor = [0,0]';                       % pos relativa del sensor en el robot

%  HANDLERS PARA ANIMAR LAS MEDIDAS DE LOS SENSORES
K = 100;
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
D = 0.2;        % Diametro de las ruedas
L = 0.15;       % Distancia de las ruedas al punto central del robot
Wr = 0.3;       % ancho robot
Hr = 0.6;       % largo robot
Dim = [Wr Hr D];

%  PARAMETROS DE ERROR
alpha1 = 1e-2;  alpha2 = 1e-3;  
alpha3 = 1e-3;  alpha4 = 1e-2;
alpha5 = 1e-3;  alpha6 = 1e-3;
alpha_VELOCITY = [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6];

%  CONFIGURACION INICIAL
x = [0.5;.5;pi/2];
% x = [0.1;0.8;0];
plot_robot(x, L, 'r',Dim)

%----------------------------------------------------------------------%
% 6. PARAMETROS DE CONTROL
%----------------------------------------------------------------------%
%  INTERVALO DE TIEMPO
dt = 0.5;

%  VELOCIDAD MAXIMA
VMAX = 0.2;

%  INDICE DEL PRIMER PUNTO DE PASO
iwp   = 1;


%----------------------------------------------------------------------%
% 7. CONFIGURACION INICIAL DEL 'BELIEF' INICIAL
%----------------------------------------------------------------------%
%  VALORES INICIALES DE LA MEDIA Y LA MATRIZ DE COVARIANZA
dim = 3+3*NN;
mu = zeros(dim,1);
P = 1000*eye(dim); % var inf para marcas

%  MODIFICAMOS LA MEDIA Y COVARIANZA
mu(1:3) = x;                    % Para trabajar con el mismo sistema coord. 
P(1:3,1:3) = 1e-8*ones(3);      % Por definicion - pos ini conocida

%  HANDLES PARA PLOTEAR LAS ELIPSES DEL ROBOT Y DE LAS MARCAS
handle_P  = plot(0,0,'b'); 
handle_lm = zeros(NN,1);
for kk = 1:NN
    handle_lm(kk) = plot(0,0,'b');
end


%----------------------------------------------------------------------%
% 7. LAZO CENTRAL
%----------------------------------------------------------------------%
disp('SLAM USANDO EL FILTRO EXTENDIDO DE KALMAN')
disp('-----------------------------')
disp(' ')
disp(' Presionar una tecla')
disp(' ')
pause

DEBUG_PREDICCION = 0;
DEBUG_CORRECCION = 0;
step = 1;
LOOPS = 2;

while (iwp ~= 0 && LOOPS~=0)    
    % 8.1. CALCULO DE LOS CONTROLES CON EL FIN QUE EL ROBOT SIGA LOS
    %      PUNTOS DE PASO
    [v,w,iwp] = compute_controls(mu(1:3), via_points, iwp, VMAX);
    if (iwp==0)
        % Si "iwp" es zero hemos terminado una vuelta
        iwp = 1;
        LOOPS = LOOPS - 1;
    end   
    u = [v; w];
    
    % 7.2 SIMULAMOS EL MOVIMIENTO DEL ROBOT
    x = sample_motion_model_velocity(x,u,Hr,dt, alpha_VELOCITY);
    
    % 7.3. OBTENEMOS LAS CELDAS QUE ESTAN AL ALCANCE DEL ROBOT
    [cells,frontier,missed] = get_cells_range(x,S,mapa_real,grid_dim,MAX_RANGE);
    %[cells,frontier,missed] = get_visible_frontier(x,mapa_real,grid_dim,MAX_RANGE);
    
    % 7.4. FILTRO EXTENDIDO DE KALMAN
    %   a. Paso de Prediccion
    [mu, P] = EKF_Prediction_step(mu, P, u, alpha_VELOCITY, dt);    
    % -> Simulamos la medida del sensor LIDAR
    zt = range_finder_grid_model(x, frontier, missed,grid_dim, std_sensor,size(mapa_real,2));
    
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
            
            % -> Compute the update step
            [mu, P] = EKF_Update_step(mu, P, zt(k,:)', frontier(k,:)', std_sensor,size(mapa_real,2),grid_dim);
        else
            ppx = [x(1),x(1) + MAX_RANGE*cos(zt(k,2) + x(3)) + d];
            ppy = [x(2),x(2) + MAX_RANGE*sin(zt(k,2) + x(3)) + d];
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

    
    
    % 7.6. PLOTEAMOS RESULTADOS    plot(x(1), x(2), '.r')
    %   -> TRAYECTORIA DEL ROBOT
    plot(x(1), x(2), '.r')    
    %   -> MEDIA Y COVARIANZA DEL 'BELIEF' DEL ROBOT
    plot(mu(1), mu(2), '.b')
    ellipse_points = sigma_ellipse(mu(1:2), P(1:2,1:2), 2);
    set(handle_P, 'xdata', ellipse_points(1,:), 'ydata', ellipse_points(2,:))
    
    
    
    %plot_map(Mp,grid_dim)    
    %plot_robot(x,L,'r',Dim)
    
    pause(0.1)
end
