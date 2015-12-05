% demo 1.2: intento de SLAM
% usando S laser beams y likelihood field
clc
clear all
close all

%----------------------------------------------------------------------%
% 1. OCCUPANCY GRID MAP CONF
%----------------------------------------------------------------------%
%  MAP READING
%Map notation  ... %1: ocupado , 0 : libre

mapa_real = load('map3.csv');
mapa_real = mapa_real';
grid_dim = .04;
wall_threshold = 0.3;            % threshold to be considered as a wall
% mapa_real = make_map(grid_dim);
mapa_real = log(mapa_real);
[N,M] = size(mapa_real);

% Initial cell likelihood
pm_occ = sum(sum(mapa_real==0)) / (N*M);
lp_0 = log(pm_occ/(1 - pm_occ));
lp_occ = 0.2*lp_0;
lp_free = 2*lp_0;
Mp = ones(size(mapa_real))*lp_0;

%----------------------------------------------------------------------%
% 2. PLOT REAL MAP AND INITIAL SURROUNDINGS
%----------------------------------------------------------------------%
figure(3)
plot_map(mapa_real,grid_dim)
figure(1)
plot_map(mapa_real,grid_dim)

%----------------------------------------------------------------------%
% 3. "RANGE-FINDER" SENSOR PARAMETERS
%----------------------------------------------------------------------%
MAX_RANGE = 2;                      % maximum range
sigmaR = 0.05;                      % standard deviations
sigmaB = (2.0*pi/180);
sigmaC = 0.05;
std_sensor = [sigmaR,sigmaB,sigmaC];
betha = (2.0*pi/180)*10;            % beam angle

zhit = 0.5;
zshort = 0.1;
zmax = 500;
zrand = 1;
Z_weights = [zhit,zshort,zmax,zrand];   % average weights of error sources
S = (-pi/2:(5*pi/180):pi/2);            % angles of 18 laser beams
xsensor = [0,0]';                       % relative position of sensor in robot

%  SENSOR MEASUREMENTS ANIMATION HANDLERS
K = 100;
handle_sensor = zeros(1,K);
for f=1:K
    handle_sensor(f) = plot(0,0, 'y');
end


%----------------------------------------------------------------------%
% 4. LIKELIHOOD FIELD PREPROCESSING
%----------------------------------------------------------------------%
%nearest_wall = preprocess_nearest_wall(mapa_real,grid_dim);


%----------------------------------------------------------------------%
% 5. ROBOT PARAMETERS
%----------------------------------------------------------------------%
%  MECHANICAL PARAMETERS
D = 0.1;        % Diametro de las ruedas
L = 0.15;       % Distancia de las ruedas al punto central del robot
Wr = 0.3;       % ancho robot
Hr = 0.6;       % largo robot
Dim = [Wr Hr D];

%  ERROR PARAMETERS
alpha1 = 1e-2;  alpha2 = 1e-3;  
alpha3 = 1e-3;  alpha4 = 1e-2;
alpha5 = 1e-3;  alpha6 = 1e-3;
alpha_VELOCITY = [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6];

%  TIME INTERVAL
dt = 0.2;

%----------------------------------------------------------------------%
% 6. INITIAL SETUP FOR ROBOT
%----------------------------------------------------------------------%
%  INITIAL CONFIGURATION
%x = [0.25;1;0];
x = [0.1;0.5;0];
plot_robot(x, L, 'r',Dim)


%  PARTICLE SET
num_part = 100;            % Number of particles
W = ones(num_part,1)/num_part;    % Nomalized likelihood for weights
%  Initializing set
X = zeros(num_part,3);              
for i=1:num_part
    ind = randi(N*M);
    xi = floor(ind/M)+1;
    yi = mod(ind,M);
    xi = (xi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
    yi = (yi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
    X(i,:) = [xi;yi;0.5*randn(1)];
end
% for i=1:num_part
%     X(i,:) = [x(1:2)+0.03*randn(2,1);  x(3)+0.05*randn(1)];
% end


%  GET GAUSSIAN APROX OF PARTICLES
[mu, P] = compute_gaussian_from_samples(X);

% HANDLER TO PLOT PARTICLES
handler_particles = plot(X(:,1),X(:,2),'w.','markersize',4);
handle_P  = plot(0,0,'b', 'erasemode', 'xor'); 


%----------------------------------------------------------------------%
% 7. MAIN LOOP
%----------------------------------------------------------------------%
disp('SLAM w/ MCL and OccGridMapp')
disp('-----------------------------')
disp('Initial conditions')
fprintf('  x = [%2.4f, %2.4f, %2.4f]\n', x)
fprintf('  mu = [%2.4f, %2.4f, %2.4f]\n', mu)
disp(' ')

N = 25;     % Number of steps
for n=1:N
    % 7.1 SET CONTROLS
    %if(n<16)        
    if n<25
        v = 0.5;                % Translational velocity
        w = 0;                  % Rotacional velocity
    elseif(n>=16 && n<21)
        v = 0.2;
        w = 2*pi/4;
    elseif(n>=21 && n<60)
        v = 0.2;
        w = 0*pi/8;
    elseif(n>=60 && n<70)
        v = 0.2;
        w = 1*pi/4;
    else
        v = 0.2;
        w = 0;
    end
    u = [v; w];
    
    % 7.2 ROBOT MOVEMENT SIMULATION
    x = sample_motion_model_velocity(x,u,Hr,dt, alpha_VELOCITY);
    
    % 7.3. GET CELLS NEARBY TO ROBOT
    [cells,frontier,missed] = get_cells_range_SLAM(x,S,mapa_real,grid_dim,MAX_RANGE,wall_threshold);
    
    % 7.4. PARTICLE FILTER
    %   a. Prediction step
    for i=1:num_part
         X(i,:) = sample_motion_model_velocity(X(i,:), u,Hr, dt, alpha_VELOCITY)';
    end
    
    % -> Simulation of measurement of laser sensor 
    zt = range_finder_grid_model(x, frontier, missed,grid_dim, std_sensor,size(mapa_real,1));

    for j=1:size(cells,1)
        mi = cells(j,:);
        [lp,flag] = inverse_range_finder_model(x, mi,zt,grid_dim, betha,std_sensor,...
                                                MAX_RANGE,lp_0,lp_occ,lp_free);
        
        if flag
            Mp(mi(1),mi(2)) = min(Mp(mi(1),mi(2)) + lp - lp_0,0);
        end
    end    
    
    near_wall = nearest_wall_SLAM(Mp,wall_threshold, grid_dim);
    
    for j=1:num_part
        for k=1:size(zt,1)
            if zt(k,1) ~= Inf
                ppx = [x(1)  x(1) + zt(k,1)*cos(zt(k,2) + x(3))];
                ppy = [x(2)  x(2) + zt(k,1)*sin(zt(k,2) + x(3))];
                set(handle_sensor(k),'xdata', ppx, 'ydata', ppy)

                wm = likelihood_field_range_finder_model(X(j,:)',xsensor,...
                           zt(k,:)',near_wall, grid_dim, std_sensor,Z_weights);
                W(j) = W(j) * wm;
            end        
        end
    end
 

    % c. RESAMPLING FINAL STEP
    index = simple_resample(W);
    X = X(index,:);
    W = ones(1,num_part)/num_part;
    
    % d. Reset particles if all of them are the same (robot is lost)
    temp = X(1,:);
    cont = 1;
    for i=2:num_part
        if temp(1) == X(i,1) && temp(2) == X(i,2) && temp(3) == X(i,3)
            cont = cont +1;
        end
    end
    if cont == size(X,1)
        for i=1:num_part
            ind = randi(N*M);
            xi = floor(ind/M)+1;
            yi = mod(ind,M);
            xi = (xi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
            yi = (yi-1)*grid_dim + grid_dim/2 + 0.05*randn(1);
            X(i,:) = [xi;yi;0.5*randn(1)];
        end
    end
    % 7.5. APROXIMATE NEW GAUSSIAN OF PARTICLES
    [mu, P] = compute_gaussian_from_samples(X);
        
    % 7.6. PLOT RESULTS
    %   -> PARTICLES
    set(handler_particles, 'xdata',X(:,1),  'ydata',X(:,2)); 
    %   -> ROBOT TRAJECTORY
    plot(x(1), x(2), '.r')
    %   -> SAMPLES MEAN
    plot(mu(1), mu(2), '.b')
    %   -> SAMPLES CO-VARIANCE
    ellipse_points = sigma_ellipse(mu(1:2), P(1:2,1:2), 2);
    set(handle_P, 'xdata', ellipse_points(1,:), 'ydata', ellipse_points(2,:))
    
    plot_map(Mp,grid_dim)
%    plot_robot(x,L,'r')
end
