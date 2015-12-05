% prueba 2 de mapeo de occupancy grid map
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


% Probabilidades iniciales en las celdas
pm_occ = sum(sum(mapa_real==0)) / (N*M);
lp_0 = log(pm_occ/(1 - pm_occ));
lp_0 = -abs(lp_0);
lp_occ = 0.2*lp_0;
lp_free = 2.5*lp_0;
Mp = ones(size(mapa_real))*lp_0;

%----------------------------------------------------------------------%
% 2. PLOTEAMOS EL MAPA REAL Y EL ENTORNO INICIAL
%----------------------------------------------------------------------%
figure(3)
plot_map(mapa_real,grid_dim)

figure(1)
plot_map(Mp,grid_dim)


%----------------------------------------------------------------------%
% 3. PARAMETROS DEL SENSOR "RANGE-FINDER"
%----------------------------------------------------------------------%
MAX_RANGE = 3;                      % Rango maximo
sigmaR = 0.08;                      % Desviaciones estandar
sigmaB = (1.0*pi/180);
sigmaC = 0.06;
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
handle_sensor = zeros(1,K);
for f=1:K
    handle_sensor(f) = plot(0,0, 'y');
end

%----------------------------------------------------------------------%
% 5. PARAMETROS DEL ROBOT
%----------------------------------------------------------------------%
%  PARAMETROS MECANICOS
D = 0.1;        % Diametro de las ruedas
L = 0.15;       % Distancia de las ruedas al punto central del robot
Wr = 0.3;       % ancho robot
Hr = 0.6;       % largo robot
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

N = 55;     % Numero de pasos
for n=1:N
    % 7.1 ESTABLECEMOS LOS CONTROLES
    if(n<6)
        v = 0.5;                % Velocidad translacional
        phi = 0;                  % Velocidad rotacional
    elseif(n>=6 && n<15)
        v = 0.5;
        phi = -1*pi/4;
    elseif(n>=15 && n<43)
        v = 0.5;
        phi = 0*pi/8;
    elseif(n>=43 && n<51)
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
    
    zt = range_finder_grid_model(x, frontier, missed,grid_dim, std_sensor,size(mapa_real,2));
    
    % -> Ploteamos la medida de los sensores
%     for k=1:size(zt,1)
%         if zt(k,1) == Inf
%             r = MAX_RANGE;
%         else
%             r = zt(k,1);
%         end
%         ppx = [x(1)  x(1) + r*cos(zt(k,2) + x(3))];
%         ppy = [x(2)  x(2) + r*sin(zt(k,2) + x(3))];
%         set(handle_sensor(k),'xdata', ppx, 'ydata', ppy)
%         
%     end
   % figure(1)
    for j=1:size(cells,1)
        mi = cells(j,:);
        [lp,flag] = inverse_range_finder_model(x, mi,zt,grid_dim, betha,std_sensor,...
                                                MAX_RANGE,lp_0,lp_occ,lp_free);
        
        if flag
            Mp(mi(1),mi(2)) = min(Mp(mi(1),mi(2)) + lp - lp_0,0);
        end
    end
    
    plot(x(1), x(2), '.r')
    %plot_map(Mp,grid_dim)
    %plot_robot(x,L,'r',Dim)
    %pause
end

plot_map(Mp,grid_dim)