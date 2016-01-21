%% OCCUPANCY GRID MAPPING ALGORITHM
clc
clear all
close all

%-----%% 0. MAP SELECTION
map_id = 2; % 0: map1
            % 1: map2
            % 2: map3

%% 1. GRIP MAP SETUP
%----------------------------------------------------------------------%
%  MAP READING
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


%% Prior probability for every cell
pm_occ = sum(sum(mapa_real==1)) / (N*M);
lp_0 = log(pm_occ/(1 - pm_occ)); % less than 0
%lp_0 = -abs(lp_0);
%lp_occ = 0.2*lp_0;
%lp_free = 2.5*lp_0;
lp_occ = 0.5*lp_0;
lp_free = 1.5*lp_0;
Mp = ones(size(mapa_real))*lp_0;


%% 2. PLOT REAL MAP AND INITIAL STATE OF INFERED MAP
%----------------------------------------------------------------------%
%figure(3)
%plot_map(mapa_real,grid_dim)

figure(1)
%plot_map(Mp,grid_dim,0,lp_0,1)    % logform mode


%% 3. "RANGE-FINDER" SENSOR PARAMETERS
%----------------------------------------------------------------------%
Z_max = 5.6;                      % Maximum range
% Noise modeling
std_hit = 0.0075;     % std of p_hit modeled as Gaussian N(0,s), +-30mm a 1m, +-120 a 4m Hoyuko, 7.5cm datasheet
% p_rand : uniform, 1/zmax
% p_max: point mass distro
% Noise weights
zhit = 3;
zrand = 1;
zmax = 5;

Z_weights = [zhit,zrand,zmax];          % average weights for prob dist. of noises.
Z_weights = Z_weights/ sum(Z_weights); % normalization Sum(Z)=1
angle_range = 20*pi/18; % 200° coverage
n_sample_intervals = 20;
beta = angle_range/n_sample_intervals;   % beam width

S = (-angle_range/2:beta:angle_range/2);            % 20 beam rays
xsensor = [0,0]';                       % relative position of sensor in robot


%% 5. ROBOT PARAMETERS
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
alpha3 = 1e-4;  alpha4 = 1e-5;
alpha5 = 1e-4;  alpha6 = 1e-5;
alpha_VELOCITY = [alpha1 alpha2 alpha3 alpha4 alpha5 alpha6];


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
    V = [V,get_controls(1.4,0      ,vo,wo,dt)];   % straight for 1.5m
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
    V = [V,get_controls(4.3  ,0      ,vo,wo,dt)];   % straight for 5m
    V = [V,get_controls(0  ,1.1*pi/2,vo,wo,dt)];   % turn 90° left
    V = [V,get_controls(4  ,0      ,vo,wo,dt)];   % straight for 5m
elseif map_id==2
    V = [V,get_controls(1  ,0      ,vo,wo,dt)];   % straight for 1.5m
    V = [V,get_controls(0  ,pi/4,vo,wo,dt)];   % turn 45° left
    V = [V,get_controls(1.5  ,0      ,vo,wo,dt)];   % straight for 1.5m
    V = [V,get_controls(0  ,pi/4,vo,wo,dt)];   % turn 45° left
    V = [V,get_controls(5.5  ,0      ,vo,wo,dt)];   % straight for 1.5m
    V = [V,get_controls(0  ,-pi,vo,wo,dt)];   % turn 45° left
    V = [V,get_controls(7  ,0      ,vo,wo,dt)];   % straight for 1.5m
end

n_steps = size(V,2);    % steps in simulation
pos_states = [];
pos_states = [pos_states;x(1:2)'];

%% 7. LAZO CENTRAL
%----------------------------------------------------------------------%
disp('OCCUPANCY GRID MAPPING')
disp('-----------------------------')
disp(' ')

handle_textbox= annotation('textbox',[.7,0.13, .1, .1],'String','Iteration:0  ','FitBoxToText','on');
%plot(x(1), x(2), '.r')

for n=1:n_steps
    disp(strcat('Iteration: ',num2str(n)))
    %% Uncomment this section to pause since iteration pause_it
    % pause_it = 60
    %if n>pause_it
    %    plot_map(Mp,grid_dim,0,lp_0,1)
    %    plot(x(1), x(2), '.r')
    %    h=msgbox('Hurry up, take a picture!');
    %    uiwait(h)
    %end
    
    % 7.1 SET CONTROL SIGNALS
    u = V(:,n);
    
    % 7.2 SIMULATE ROBOT MOVEMENT
    x = sample_motion_model_velocity(x,u,Hr,dt, alpha_VELOCITY);
    
    % If current position is outside map or cell is occupied, error
    cx = floor(x(1)/grid_dim) + 1;
    cy = floor(x(2)/grid_dim) + 1;
    if cx<1 || cx > N || cy<1 || cy>M || mapa_real(cx,cy)==1
        disp('Robot collisioned with an obstacle or exited the map...')
        break
    end
    
    % 7.3. GET CELLS IN CURRENT ROBOT SCOPE
    [cells,frontier,missed] = get_cells_range(x,S,mapa_real,grid_dim,Z_max);
    %[cells,frontier,missed] = get_visible_frontier(x,mapa_real,grid_dim,MAX_RANGE);
    
    zt = range_finder_grid_model(x, frontier, missed,grid_dim,std_hit,Z_max);
    
    %figure(1)
    for j=1:size(cells,1)
        mi = cells(j,:);
        lp = inverse_range_finder_model(x, mi,zt,grid_dim, beta,...
                                                Z_max,lp_0,lp_occ,lp_free);
        if lp~=lp_0
            Mp(mi(1),mi(2)) = Mp(mi(1),mi(2)) + lp - lp_0;
            Mp(mi(1),mi(2)) = min(Mp(mi(1),mi(2)), lp_occ);
        end
    end
    
    pos_states = [pos_states;x(1:2)'];
    %pause(0.2)
end

plot_map(Mp,grid_dim,0,lp_0,1)
plot(pos_states(:,1),pos_states(:,2),'.r')
message = strcat('Iteration: ',num2str(n_steps));
set(handle_textbox,'String',message)