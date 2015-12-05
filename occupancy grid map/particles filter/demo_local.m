% demo 1.2: prueba de mapeo occupancy grid
% usando S laser beams y likelihood field
clc
clear all
close all

%----------------------------------------------------------------------%
% 1. CONFIGURACION DEL MAPA GRID DE OCUPACION
%----------------------------------------------------------------------%
%  LECTURA DEL MAPA
%lectura mapa  ... %1: ocupado , 0 : libre

% mapa_real = load('map3.csv');
% mapa_real = mapa_real';
grid_dim = .04;
mapa_real = make_map(grid_dim);
mapa_real = log(mapa_real);
[N,M] = size(mapa_real);


% Probabilidades iniciales en las celdas
pm_occ = sum(sum(mapa_real==0)) / (N*M);
lp_0 = log(pm_occ/(1 - pm_occ));
lp_occ = 0.2*lp_0;
lp_free = 2*lp_0;
Mp = ones(size(mapa_real))*lp_0;

%----------------------------------------------------------------------%
% 2. PLOTEAMOS EL MAPA REAL Y EL ENTORNO INICIAL
%----------------------------------------------------------------------%
% figure(3)
% plot_map(mapa_real,grid_dim)
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

zhit = 0.5;
zshort = 0.1;
zmax = 600;
zrand = 1;
Z_weights = [zhit,zshort,zmax,zrand];   % pesos de ponderacion de las fuentes de error
S = (-3.5*pi/4:(10*pi/180):3.5*pi/4);            % angulos de 18 rayos laser
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
% 4. PREPROCESADO DE LIKELIHOOD FIELD
%----------------------------------------------------------------------%
nearest_wall = preprocess_nearest_wall(mapa_real,grid_dim);

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

%  INTERVALO DE TIEMPO
dt = 0.2;


%----------------------------------------------------------------------%
% 6. CONFIGURACION INICIAL DEL ROBOT
%----------------------------------------------------------------------%
%  CONFIGURACION INICIAL
%x = [0.25;1;0];
x = [0.1;0.9;0];
plot_robot(x, L, 'r',Dim)


%  CONJUNTO DE PARTICULAS
num_part = 200;            % Numero de particulas
W = ones(num_part,1)/num_part;    % Pesos -----------> en prob normal

%  Inicializamos el conjunto
X = zeros(num_part,3);              
idx = 1;
while(idx~=num_part)
    ind = randi(N*M);
    xi = floor(ind/M) + (mod(ind,M)~=0);
    yi = mod(ind,M)*(mod(ind,M)~=0) + M*(mod(ind,M)==0);
    
    %fprintf('xi:%d  yi:%d\n',xi,yi)
    
    if mapa_real(xi,yi) == log(0)
        xi = (xi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
        yi = (yi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);        
        X(idx,:) = [xi;yi;pi_to_pi(0.5*randn(1))];
        idx = idx + 1;
    end
end

% for i=1:num_part
%     ind = randi(N*M);
%     xi = floor(ind/M)+1;
%     yi = mod(ind,M);
%     xi = (xi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
%     yi = (yi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
%     X(i,:) = [xi;yi;pi_to_pi(0.5*randn(1))];
% end
% for i=1:num_part
%     X(i,:) = [x(1:2)+0.03*randn(2,1);  x(3)+0.05*randn(1)];
% end


%  HALLAMOS LA APROXIMACION GAUSIANA DE LAS PARTICULAS
[mu, P] = compute_gaussian_from_samples(X);

%  HANDLER PARA PLOTEAR LAS PARTICULAS
handler_particles = plot(X(:,1),X(:,2),'r.','markersize',4);
handle_P  = plot(0,0,'b', 'erasemode', 'xor'); 



%----------------------------------------------------------------------%
% 7. LAZO CENTRAL
%----------------------------------------------------------------------%
disp('FILTRO DE PARTICULAS')
disp('-----------------------------')
disp('Condiciones iniciales')
fprintf('  x = [%2.4f, %2.4f, %2.4f]\n', x)
fprintf('  mu = [%2.4f, %2.4f, %2.4f]\n', mu)
disp(' ')

N = 66;     % Numero de pasos
for n=1:N
    % 7.1 ESTABLECEMOS LOS CONTROLES
    if(n<28)
    %if n<25
        v = 0.5;                % Velocidad translacional
        phi = 0;                  % Velocidad rotacional
    elseif(n>=28 && n<37)
        v = 0.5;
        phi = 1*pi/4;
    elseif(n>=37 && n<46)
        v = 0.5;
        phi = 0*pi/8;
    elseif(n>=46 && n<55)
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
    
    % 7.4. FILTRO DE PARTICULAS
    %   a. Paso de Prediccion        
    for i=1:num_part
         X(i,:) = sample_motion_model_velocity(X(i,:), u,Hr,dt, alpha_VELOCITY)';
    end
    
    % -> Simulamos la medida del sensor LIDAR
    zt = range_finder_grid_model(x, frontier, missed,grid_dim, std_sensor,size(mapa_real,2));
    
    % -> Iteramos en todas las mediciones y ploteamos
    for j=1:num_part
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
                
                wm = likelihood_field_range_finder_model(X(j,:)',xsensor,...
                           zt(k,:)',nearest_wall, mapa_real,grid_dim, std_sensor,Z_weights);
                W(j) = W(j) * wm;
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
    end
    
    figure(1)
    

    % c. PASO FINAL DE REMUESTRO
    index = simple_resample(W);   % IMPLEMENTAR ESTA FUNCION
    X = X(index,:);
    W = ones(1,num_part)/num_part;
    
    % d. resetear particulas si todas son iguales (perdido y baja prob)
    temp = X(1,:);
    cont = 1;
%     for i=2:num_part
%         if temp(1) == X(i,1) && temp(2) == X(i,2) && temp(3) == X(i,3)
%             cont = cont +1;
%         end
%     end
    if cont == size(X,1)
        idx = 1;
        while(idx~=num_part)
            ind = randi(N*M);
            xi = floor(ind/M) + (mod(ind,M)~=0);
            yi = mod(ind,M)*(mod(ind,M)~=0) + M*(mod(ind,M)==0);

            %fprintf('xi:%d  yi:%d\n',xi,yi)

            if mapa_real(xi,yi) == log(0)
                xi = (xi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
                yi = (yi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);        
                X(idx,:) = [xi;yi;pi_to_pi(0.5*randn(1))];
                idx = idx + 1;
            end
        end

%         for i=1:num_part
%             ind = randi(N*M);
%             xi = floor(ind/M)+1;
%             yi = mod(ind,M);
%             xi = (xi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
%             yi = (yi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
%             X(i,:) = [xi;yi;0.5*randn(1)];
%         end
    end
    % 7.5. APROXIMAMOS EL NUEVO GAUSIANO DE LAS PARTICULAS
    [mu, P] = compute_gaussian_from_samples(X);
    
    
    % 7.6. PLOTEAMOS RESULTADOS
    %   -> LAS PARTICULAS
    set(handler_particles, 'xdata',X(:,1),  'ydata',X(:,2)); 
    %   -> TRAYECTORIA DEL ROBOT
    plot(x(1), x(2), '.r')
    %   -> MEDIA DE LAS MUESTRAS
    plot(mu(1), mu(2), '.b')
    %   -> COVARIANZA DE LAS MUESTRAS
    ellipse_points = sigma_ellipse(mu(1:2), P(1:2,1:2), 2);
    set(handle_P, 'xdata', ellipse_points(1,:), 'ydata', ellipse_points(2,:))
    
    
    %plot_map(Mp,grid_dim)    
    %plot_robot(x,L,'r',Dim)
    
    %pause
end
