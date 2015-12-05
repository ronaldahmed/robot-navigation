 % Demo_EKF
% Author : Ivan A. Calle Flores
% Description: Localizacion de un robot movil de tipo diferencial usando
%              el filtro extendido de Kalman
clc; clear all; close all

%----------------------------------------------------------------------%
% 1. CONFIGURACION DEL MAPA DE REFERENCIAS "LANDMARKS"
%----------------------------------------------------------------------%
%  ESTABLECEMOS EL MAPA
mapa = ...
    [-1.0   -2.0;
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



%----------------------------------------------------------------------%
% 2. CONFIGURACION DE LA FIGURA
%----------------------------------------------------------------------%
fig1 = figure;    
hold on
xlabel('x(m)')
ylabel('y(m)')



%----------------------------------------------------------------------%
% 3. PLOTEAMOS EL ENTORNO
%----------------------------------------------------------------------%
plot(mapa(1,:),mapa(2,:), 'r+','MarkerSize',10,'LineWidth',2)
axis([-2  10 -4  5])



%----------------------------------------------------------------------%
% 4. PARAMETROS DEL SENSOR "RANGE-BEARING"
%----------------------------------------------------------------------%
MAX_RANGE = 5.5;                    % Maxima distancia
sigmaR = 0.1;                       % Desviaciones estandar
sigmaB = (3.0*pi/180);       
sigmaS = 0.05;               
std_sensor = [sigmaR;sigmaB;sigmaS];

%  HANDLERS PARA ANIMAR LAS MEDIDAS DE LOS SENSORES
K = 20;
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
alpha1 = 1e-2;  alpha2 = 1e-2;  
alpha3 = 1e-2;  alpha4 = 1e-2;
alpha5 = 1e-3;  alpha6 = 1e-3;
alpha_VELOCITY = [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6];

%  INTERVALO DE TIEMPO
dt = 0.5;



%----------------------------------------------------------------------%
% 6. CONFIGURACION INICIAL DEL ROBOT
%----------------------------------------------------------------------%
%  CONFIGURACION INICIAL
x = [0;-1;0*pi/4];
plot_robot(x, L, 'r')

%  DISTRIBUCION INICIAL - Initial belief
mu = x + [0.03*randn(2,1); 0.05*randn(1)];          % Mean
P = 1e-2*eye(3);                                    % Covarianze
plot_robot(mu, L, 'b')
ellipse_points = sigma_ellipse(mu(1:2), P(1:2,1:2), 2);
handle_P  = plot(ellipse_points(1,:),ellipse_points(2,:),'r'); 



%----------------------------------------------------------------------%
% 7. LAZO CENTRAL
%----------------------------------------------------------------------%
disp('FILTRO EXTENDIDO DE KALMAN')
disp('-----------------------------')
disp('Condiciones iniciales')
fprintf('  x = [%2.4f, %2.4f, %2.4f]\n', x)
fprintf('  mu = [%2.4f, %2.4f, %2.4f]\n', mu)
disp(' ')
disp(' Presionar una tecla')
pause

N = 150;     % Numero de pasos
for n=1:N 
    % 7.1 ESTABLECEMOS LOS CONTROLES
    if(n<70)        
        v = 0.2;                % Velocidad translacional
        w = 0;                  % Velocidad rotacional
    elseif(n>=70 && n<74)
        v = 0.2;
        w = 1*pi/4;
    elseif(n>=74 && n<100)
        v = 0.2;
        w = 0*pi/8;
    elseif(n>=100 && n<104)
        v = 0.2;
        w = 1*pi/4;
    else
        v = 0.2;
        w = 0;
    end
    u = [v; w];
    % 7.2 SIMULAMOS EL MOVIMIENTO DEL ROBOT
    x = sample_motion_model_velocity(x, u, dt, alpha_VELOCITY);

    
    % 7.3. OBTENEMOS LAS REFERENCIAS QUE ESTAN AL ALCANCE DEL ROBOT
    [vlm,Nobs] = get_visible_landmarks(x,mapa,MAX_RANGE);
    
    
    % 7.4. FILTRO EXTENDIDO DE KALMAN
    %   a. Paso de Prediccion
    [mu, P] = EKF_Prediction_step(mu, P, u, alpha_VELOCITY, dt);
    
    %   b. Paso de correccion
    prob_z_acum = 1;
    if(Nobs > 0)
        for i=1:Nobs
            
            % -> Simulamos la medida del sensor laser
            z = range_bearing_model(x, vlm(:,i), std_sensor);

            % -> Ploteamos la medida de los sensores
            ppx = [x(1)  x(1) + z(1)*cos(z(2) + x(3))];
            ppy = [x(2)  x(2) + z(1)*sin(z(2) + x(3))]; 
            set(handle_sensor(i),'xdata', ppx, 'ydata', ppy)
            
            % -> Compute the update step
            [mu, P,pz] = EKF_Correction_step(mu, P, z, vlm(:,i), std_sensor);
            prob_z_acum = prob_z_acum*pz;
        end
    end
    plot(x(1),x(2),'.r')
    plot(mu(1),mu(2),'.b')
    ellipse_points = sigma_ellipse(mu(1:2), P(1:2,1:2), 2);
    set(handle_P,'xdata',ellipse_points(1,:),'ydata',ellipse_points(2,:))
    fprintf('Prob de medida real : %.4f\n',prob_z_acum)

    pause(0.1)
end
plot_robot(x,0.1)
plot_robot(mu,0.1)