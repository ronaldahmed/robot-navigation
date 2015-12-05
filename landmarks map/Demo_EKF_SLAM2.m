% Demo_EKF
% Author : Ivan A. Calle Flores
% Description: Localizacion de un robot movil de tipo diferencial usando
%              el filtro extendido de Kalman
clc; clear all; close all

% VARIABLES GLOBALES
global NN       % Numero de marcas 
global done     % Flags de las marcas visitadas

%----------------------------------------------------------------------%
% 1. CONFIGURACION DEL MAPA DE REFERENCIAS "LANDMARKS"
%----------------------------------------------------------------------%
%  ESTABLECEMOS EL MAPA
mapa = ...
  [-1.0  -2.0;
    1.0  -2.0;
    3.0  -2.0;
    5.5  -2.0;
    8.5  -2.0;
    8.5   0.0;
    8.5   3.0;
    5.5   3.0;
    3.0   3.0;
    1.0   3.0;
   -1.0   3.0]';                          

%  ESTABLECEMOS LAS MARCAS 'ETIQUETAS' DE LAS REFERENCIAS - "SIGNATURES"
marcas_mapa = 1:size(mapa,2);

%  MAPA FINAL[3xNlandmarks]
mapa = [mapa;marcas_mapa];
NN = size(mapa,2);
done = zeros(1,NN);

%----------------------------------------------------------------------%
% 2. CONFIGURACION DEL MAPA DE PUNTOS DE PASO DEL ROBOT "VIA_POINTS"
%----------------------------------------------------------------------%
via_points =...
   [1.0  -1.0;
    7.0  -1.0;
    7.0   2.0;
    0.0   2.0];
via_points = via_points';   % Cada punto es una columna



%----------------------------------------------------------------------%
% 3. PLOTEAMOS EL ENTORNO DE NAVEGACION
%----------------------------------------------------------------------%
fig1 = figure;    
hold on
xlabel('x(m)')
ylabel('y(m)')
plot(mapa(1,:),mapa(2,:), 'r+','MarkerSize',10,'LineWidth',2)
axis([-2  10 -4  5])
plot(via_points(1,:),via_points(2,:), 'b+','MarkerSize',10,'LineWidth',2)


%----------------------------------------------------------------------%
% 4. PARAMETROS DEL SENSOR "RANGE-BEARING"
%----------------------------------------------------------------------%
MAX_RANGE = 4.0;                    % Rango maximo
sigmaR = 0.05;                      % Desviaciones estandar
sigmaB = (2.0*pi/180);       
sigmaS = 0.05;               
std_sensor = [sigmaR;sigmaB;sigmaS];

%  HANDLERS PARA ANIMAR LAS MEDIDAS DE LOS SENSORES
K = 10;
handle_sensor = zeros(1,K);
for f=1:K
    handle_sensor(f) = plot(0,0, 'm');
end



%----------------------------------------------------------------------%
% 5. PARAMETROS DEL ROBOT
%----------------------------------------------------------------------%
%  PARAMETROS MECANICOS
D = 0.1;        % Diametro de las ruedas
L = 0.15;       % Distancia de las ruedas al punto central del robot

%  PARAMETROS DE ERROR
alpha1 = 1e-2;  alpha2 = 1e-3;  
alpha3 = 1e-3;  alpha4 = 1e-2;
alpha5 = 1e-3;  alpha6 = 1e-3;
alpha_VELOCITY = [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6];

%  CONFIGURACION INICIAL
x = [0;-1;0*pi/8];
plot_robot(x, L, 'r')



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
% 8. LAZO CENTRAL
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
    
    
    
    % 8.2 SIMULAMOS EL MOVIMIENTO DEL ROBOT
    x = sample_motion_model_velocity(x, u, dt, alpha_VELOCITY);
    
    
    
    % 8.3. OBTENEMOS LAS REFERENCIAS QUE ESTAN AL ALCANCE DEL ROBOT
    [vlm,Nobs] = get_visible_landmarks(x,mapa,MAX_RANGE);
        
    
    
    % 8.4. FILTRO EXTENDIDO DE KALMAN
    %   a. Paso de Prediccion
    [mu, P] = EKF_Prediction_SLAM(mu, P, u, alpha_VELOCITY, dt);   
    if(DEBUG_PREDICCION)       
        fprintf('STEP %d - PASO DE PREDICCION\step', step)
        disp('mu')
        disp(mu')
        fprintf('Matriz P\step')    
        disp(P)
    end    
    
    %   b. Paso de correccion
     if(Nobs > 0)
        for i=1:Nobs
            
            % -> Simulamos la medida del sensor laser
            z = range_bearing_model(x, vlm(:,i), std_sensor);

            % -> Ploteamos la medida de los sensores
            ppx = [x(1)  x(1) + z(1)*cos(z(2) + x(3))];
            ppy = [x(2)  x(2) + z(1)*sin(z(2) + x(3))];
            set(handle_sensor(i),'xdata', ppx, 'ydata', ppy)
            
            % -> Compute the update step
            [mu, P] = EKF_Update_SLAM(mu, P, z, std_sensor);
        end
     end
     
     if(DEBUG_CORRECCION)       
        fprintf('STEP %d - PASO DE CORRECCION\step', step)
        disp('mu')
        disp(mu)
        fprintf('Matriz P\step')
        disp(P)
    end

    
    
    % 8.5. PLOTEAMOS RESULTADOS    
    %   -> TRAYECTORIA DEL ROBOT
    plot(x(1), x(2), '.r')
    %   -> MEDIA Y COVARIANZA DEL 'BELIEF' DEL ROBOT
    plot(mu(1), mu(2), '.b')
    ellipse_points = sigma_ellipse(mu(1:2), P(1:2,1:2), 2);
    set(handle_P, 'xdata', ellipse_points(1,:), 'ydata', ellipse_points(2,:))
    
    % PLOTEAMOS SLAM : elipses de las marcas
    for kk=1:NN
        start = 3*kk + 1;
        ee = start + 1;
        ellipse_points = sigma_ellipse(mu(start:ee), P(start:ee,start:ee), 2);
        set(handle_lm(kk),'xdata', ellipse_points(1,:), 'ydata', ellipse_points(2,:))
    end    
    
    
    
    % 8.6. DELTA DE TIEMPO 
    step = step+1;
    pause(0.01)
    %pause
    %clc
end