function V = get_controls(distance,angle,vo,wo,dt)
% Retrieves neccesary controls to move robot 'distance' m and modify global
% orientation 'angle' rads
% distance (m): straight distance to move
% angle (rads): change in global orientation
%    turn left: -pi/2
%    turn right: pi/2
%    turn aroud: pi (U turn from right ) | -pi
% vo,wo : constant linear/rot velocities
% dt (s): time step in simulation
% grid_size: grid resolution

% return: V = [v;w]:control signals (linear,rot vels). Size is amount of time
% steps neccesary to reach (distance,angle) target


V = [];
% move forward
if angle==0
    V = [vo * ones(1,round(distance/(vo*dt) ) );  % vo
             zeros(1,round(distance/(vo*dt) ) )]; % 0 rad/s
% turning 'angle' rads:
elseif distance==0
    V = [       vo           * ones(1,abs(round(angle/(wo*dt))) );                        % vo
         wo*((-1)^(angle<0)) * ones(1,abs(round(angle/(wo*dt))) )];      % 0 rad/s
end

