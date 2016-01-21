% MONTE CARLO LOCALIZATION ALGORITHM: PARTICLES FILTER
% using S laser beams and likelihood fields
clc
clear all
close all

%% 0. MAP SELECTION
map_id = 0; % 0: map1
            % 1: map2
            % 2: map3

%% 1. GRID MAP SET UP
%----------------------------------------------------------------------%
%  READING MAP
%notation  ... %1: occupied cell , 0 : free cell
mapa_real = [];
switch(map_id)
    case 0
        mapa_real = load('map1.csv');
    case 1
        mapa_real = load('map2.csv');
    case 2
        mapa_real = load('map3.csv');
end
mapa_real = fliplr(mapa_real');
grid_dim = .1;

[N,M] = size(mapa_real);


%% 2. PLOT REAL MAP
%----------------------------------------------------------------------%
figure(1)
plot_map(mapa_real,grid_dim)
hold on

%% 3. "RANGE-FINDER" SENSOR PARAMETERS
%----------------------------------------------------------------------%
Z_max = 5.6;                      % Rango maximo
% Noise modeling
std_hit = 0.0075;     % std of p_hit modeled as Gaussian N(0,s), +-30mm a 1m, +-120 a 4m Hoyuko, 7.5cm datasheet
% p_rand : uniform, 1/zmax
% p_max: point mass distro
% Noise weights
zhit = 3;
zrand = 1;
zmax = 5;

Z_weights = [zhit,zrand,zmax];          % pesos de ponderacion de las fuentes de error
Z_weights = Z_weights/ sum(Z_weights); % normalization Sum(Z)=1
angle_range = 20*pi/18; % 200° coverage
n_sample_intervals = 20;
S = (-angle_range/2:(angle_range/n_sample_intervals):angle_range/2);            % 20 beam rays
xsensor = [0,0]';                       % pos relativa del sensor en el robot


%% 5. ROBOT PARAMTERS
%----------------------------------------------------------------------%
%  MECHANICAL PARAMETERS
D = 0.2;        % Diametro de las ruedas
L = 0.15;       % Distancia de las ruedas al punto central del robot
Wr = 0.3;       % ancho robot
Hr = 0.6;       % largo robot
max_steering_angle = 34*pi/180;  % maximum steering angle
Dim = [Wr Hr D];
%  NOISE PARAMETERS
alpha1 = 1e-3;  alpha2 = 1e-4;  
alpha3 = 1e-3;  alpha4 = 1e-4;
alpha5 = 1e-4;  alpha6 = 1e-5;
alpha_VELOCITY = [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6];


%% 4. LIKELIHOOD PREPROCESSING
%----------------------------------------------------------------------%
nearest_wall = preprocess_nearest_wall(mapa_real,grid_dim);


%%  PARTICLES INITIAL SETUP
%----------------------------------------------------------------------%
num_part = 1000;            % number of particles
W = ones(num_part,1)/num_part;    % Weights -----------> uniform probability

%  Initialize set
X = zeros(num_part,3);              
for i=1:num_part
    X(i,:) = sample_particle(mapa_real,grid_dim);
end

%  Gaussian aprox of particles
[mu, P] = compute_gaussian_from_samples(X);


%%  HANDLERS TO PLOT SNESOR MEASUREMENTS
%----------------------------------------------------------------------%
K = 100;
handle_sensor_ray = zeros(1,K);
for f=1:K
    handle_sensor_ray(f) = plot([0,0],[0,0],'y');
end
handle_sensor_frontier = plot(0,0,'ro');
handle_sensor_missed = plot(0,0,'go');

%%  HANDLER TO PLOT PARTICLES
handler_particles = plot(X(:,1),X(:,2),'b.','markersize',4);
handle_P  = plot(0,0,'b', 'erasemode', 'xor');


%% Control signals
dt = 0.05;         % time interval (s)
x = [];       % Initial position state (m,m,rad)
switch(map_id)
    case 0
        x = [1;8;0]; 
    case 1
        x = [0.3;8.5;0];
    case 2
        x = [2;9;-pi/2]; 
end

vo = 1*grid_dim/dt;   % constant linear velocity
wo = vo*tan(max_steering_angle)/Hr;             % constant angular velocity

%% Action Sequence
V=[];
if map_id==0
    V = [V,get_controls(1.35,0      ,vo,wo,dt)];   % straight for 1.5m
    V = [V,get_controls(0  ,-pi/2,vo,wo,dt)];   % turn 90° right
    V = [V,get_controls(5  ,0      ,vo,wo,dt)];   % straight for 5m
    V = [V,get_controls(0  ,-pi  ,vo,wo,dt)];   % turn back
    V = [V,get_controls(5  ,0    ,vo,wo,dt)];   % straight for 5m
    V = [V,get_controls(0  ,-pi/2,vo,wo,dt)];   % turn 90° right
    V = [V,get_controls(4  ,0    ,vo,wo,dt)];   % straight for 3 m
elseif map_id==1
    V = [V,get_controls(0.8,0      ,vo,wo,dt)];   % straight for 1.5m
    V = [V,get_controls(0  ,-pi/2,vo,wo,dt)];   % turn 90° right
    V = [V,get_controls(4  ,0      ,vo,wo,dt)];   % straight for 5m
    V = [V,get_controls(0  ,pi/2,vo,wo,dt)];   % turn 90° left
    V = [V,get_controls(4.5  ,0      ,vo,wo,dt)];   % straight for 5m
    V = [V,get_controls(0  ,pi/2,vo,wo,dt)];   % turn 90° left
elseif map_id==2
    V = [V,get_controls(1  ,0      ,vo,wo,dt)];   % straight for 1.5m
    V = [V,get_controls(0  ,pi/4,vo,wo,dt)];   % turn 45° left
    V = [V,get_controls(2  ,0      ,vo,wo,dt)];   % straight for 1.5m
    V = [V,get_controls(0  ,pi/4,vo,wo,dt)];   % turn 45° left
    V = [V,get_controls(6  ,0      ,vo,wo,dt)];   % straight for 1.5m
end

n_steps = size(V,2);    % steps in simulation

%% 7. MAIN LOOP
%----------------------------------------------------------------------%
disp('MCL: PARTICLES FILTER')
disp('-----------------------------')
disp('Initial conditions')
fprintf('  x = [%2.4f, %2.4f, %2.4f]\n', x)
fprintf('  mu = [%2.4f, %2.4f, %2.4f]\n', mu)
disp(' ')

handle_textbox= annotation('textbox',[.7,0.13, .1, .1],'String','Iteration:0  ','FitBoxToText','on');

plot(x(1), x(2), '.r')

for n=1:n_steps
    % save the screenshot
    
    %% Uncomment this section to pause since iteration pause_it
    % pause_it = 60
    %if n>pause_it
    %    h=msgbox('Hurry up, take a picture!');
    %    uiwait(h)
    %end
    
    % 7.1 SET CONTROL SIGNALS
    u = V(:,n);
    
    % 7.2 ROBOT MOVEMENT SIMULATION
    x = sample_motion_model_velocity(x,u,Hr,dt, alpha_VELOCITY);
    %x = noise_free_motion_model_velocity(x,u,dt);

    % If current position is outside map or cell is occupied, error
    cx = floor(x(1)/grid_dim) + 1;
    cy = floor(x(2)/grid_dim) + 1;
    if cx<1 || cx > N || cy<1 || cy>M || mapa_real(cx,cy)==1
        disp('Robot collisioned with an obstacle or exited the map...')
        break
    end
    
    % 7.3. GET CELLS IN CURRENT ROBOT SCOPE
    [cells,frontier,missed] = get_cells_range(x,S,mapa_real,grid_dim,Z_max);
    
    % 7.4. PARTICLES FILTER
    %   a. Prediction step
    for i=1:num_part
         X(i,:) = sample_motion_model_velocity(X(i,:), u,Hr,dt, alpha_VELOCITY)';
    end
    
    % -> Simulation of range finder sensor measurement
    zt = range_finder_grid_model(x, frontier, missed,grid_dim,std_hit,Z_max);
    
    % -> Iterate over measurements and plot
    for j=1:num_part
        end_points=[];
        missed_points=[];        
        for k=1:size(zt,1)
            if zt(k,2)>0
                d = -grid_dim/2;
            else
                d = grid_dim/2;
            end
            phi = pi_to_pi(zt(k,2) + x(3));
            if zt(k,1) ~= Z_max
                ppx = [x(1),x(1) + zt(k,1)*cos(phi) + d];
                ppy = [x(2),x(2) + zt(k,1)*sin(phi) + d];
                end_points = [end_points;ppx(2),ppy(2)];
                
                wm = likelihood_field_range_finder_model(X(j,:)',xsensor,...
                           zt(k,:)',nearest_wall, grid_dim, std_hit,Z_weights,Z_max);
                W(j) = W(j) * wm;
            else
                dist = Z_max + std_hit*randn(1);
                ppx = [x(1),x(1) + dist*cos(phi) + d];
                ppy = [x(2),x(2) + dist*sin(phi) + d];
                missed_points = [missed_points;ppx(2),ppy(2)];                
            end
            %set(handle_sensor_ray(k),'XData', ppx, 'YData', ppy)
        end
    end
    
    figure(1)

    % c. RESAMPLING STEP
    index = simple_resample(W);
    X = X(index,:);
    W = ones(1,num_part)/num_part;
    
    % 7.5. GET NEW GAUSSIAN APROXIMATION OF PARTICLES
    %[mu, P] = compute_gaussian_from_samples(X);
    
    %% 7.6. PLOT RESULTS
    %   -> ROBOT TRAJECTORY
    plot(x(1), x(2), '.r')
    %   -> LAS PARTICULAS
    set(handler_particles, 'XData',X(:,1),  'YData',X(:,2)); 

    %%   -> SAMPLES MEDIAN
    %plot(mu(1), mu(2), '.b')
    %%   -> SAMPLES COVARIANCE
    %ellipse_points = sigma_ellipse(mu(1:2), P(1:2,1:2), 2);
    %set(handle_P, 'XData', ellipse_points(1,:), 'YData', ellipse_points(2,:))
    message = strcat('Iteration: ',num2str(n));
    set(handle_textbox,'String',message)
    pause(0.2)
end
