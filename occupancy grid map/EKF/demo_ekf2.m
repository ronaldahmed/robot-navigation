% DEMO EKF
% usando S laser beams y likelihood field
clc
clear all
close all

%----------------------------------------------------------------------%
% 1. CONFIGURACION DEL MAPA GRID DE OCUPACION
%----------------------------------------------------------------------%
%  LECTURA DEL MAPA
%lectura mapa  ... %1: ocupado , 0 : libre

mapa_real = load('map4.csv');
mapa_real = mapa_real';
grid_dim = .04;
mapa_real = log(mapa_real);
[N,M] = size(mapa_real);


%----------------------------------------------------------------------%
% 2. PLOTEAMOS EL MAPA REAL Y EL ENTORNO INICIAL
%----------------------------------------------------------------------%
figure(1)
plot_map(mapa_real,grid_dim)

%----------------------------------------------------------------------%
% 3. PARAMETROS DEL SENSOR "RANGE-FINDER"
%----------------------------------------------------------------------%
MAX_RANGE = 1.5;                      % Rango maximo
sigmaR = 0.08;                      % Desviaciones estandar
sigmaB = (1.0*pi/180);
sigmaC = 0.06;
std_sensor = [sigmaR,sigmaB,sigmaC];
betha = (2.0*pi/180)*10;            % beam angle

S = (-3*pi/4:(5*pi/180):3*pi/4);            % angulos de 18 rayos laser
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
% 4. PARAMETROS DEL ROBOT
%----------------------------------------------------------------------%
%  PARAMETROS MECANICOS
D = 0.2;        % Diametro de las ruedas
L = 0.15;       % Distancia de las ruedas al punto central del robot
Wr = 0.3;       % ancho robot
Hr = 0.4;       % largo robot
Dim = [Wr Hr D];
%  PARAMETROS DE ERROR
alpha1 = 1e-2;  alpha2 = 1e-3;  
alpha3 = 1e-3;  alpha4 = 1e-2;
alpha5 = 1e-3;  alpha6 = 1e-3;
alpha_VELOCITY = [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6];

%  INTERVALO DE TIEMPO
dt = 0.2;


%----------------------------------------------------------------------%
% 6. CONFIGURACION INICIAL DEL ROBOT
%----------------------------------------------------------------------%
%  CONFIGURACION INICIAL
x = [0.5;0.5;pi/2];
plot_robot(x, L, 'r',Dim)

%  DISTRIBUCION INICIAL - Initial belief
mu = x; %[0.03*randn(2,1); 0.05*randn(1)];          % Mean
P = 1e-2*eye(3);                                    % Covarianze
plot_robot(mu, L, 'b',Dim)
ellipse_points = sigma_ellipse(mu(1:2), P(1:2,1:2), 2);
handle_P  = plot(ellipse_points(1,:),ellipse_points(2,:),'m'); 



%----------------------------------------------------------------------%
% 7. LAZO CENTRAL
%----------------------------------------------------------------------%
disp('FILTRO DE PARTICULAS')
disp('-----------------------------')
disp('Condiciones iniciales')
fprintf('  x = [%2.4f, %2.4f, %2.4f]\n', x)
fprintf('  mu = [%2.4f, %2.4f, %2.4f]\n', mu)
disp(' ')

N = 55;     % Numero de pasos
for n=1:N
    % 7.1 ESTABLECEMOS LOS CONTROLES
    if(n<6)
        v = 0.5;                % Velocidad translacional
        phi = 0;                  % Velocidad rotacional
    elseif(n>=6 && n<12)
        v = 0.5;
        phi = -1*pi/4;
    elseif(n>=11 && n<43)
        v = 0.5;
        phi = 0*pi/8;
    elseif(n>=43 && n<48)
        v = 0.5;
        phi = 1*pi/4;
    else
        v = 0.5;
        phi = 0;
    end
    u = [v; phi];
    
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
    
    
    %plot_robot(x,L,'r',Dim)
    
    pause(0.1)
end
