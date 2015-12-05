function xp = sample_motion_model_velocity(x,u,L,T,alpha)

% SAMPLE_MOTION_MODEL_VELOCITY Implements the algorithm that is described
%                              in the table 5.3. See Page 124
% 
% Output
%       xp: Sample pose at time "t" of the robot.
%
% Inputs
%       x: Vector that represent the pose "[x;y;theta]" at time "t-1" 
%       u: Vector of control = [v; w];
%   delta: Time interval(seg.)
%   alpha: DEFINE EL RUIDO


%----------------------------------------------------------------------%
% 1. PARAMETERS CHECKING
%----------------------------------------------------------------------%
%  1.1. Make sure the angle is in the correct range
theta = x(3);
if(theta > pi || theta <= -pi)
    error('sample_motion_model_velocity: The angle theta of the robot is not in the range.')
end

%  1.2. Check the dimensions of the vector of errors
if length(alpha) ~= 6
    error('sample_motion_model_velocity: Vector of errors must be of size 4')
end


%----------------------------------------------------------------------%
% SIMULAMOS LAS VELOCIDADES REALES(son las velocidades comandadas +
%  ruido gaussiano)
%----------------------------------------------------------------------%
%  2.1. Translational velocity
var = alpha(1)*u(1)^2 + alpha(2)*u(2)^2;        % Varianza en v
v = u(1) + sqrt(var)*randn(1);                  % Velocida real

%  2.2. Steering angle
var = alpha(3)*u(1)^2 + alpha(4)*u(2)^2;        % Variance
phi = u(2) + sqrt(var)*randn(1);                  % Actual rot. velocity

%  2.3. Additional rotational velocity
% var = alpha(5)*u(1)^2 + alpha(6)*u(2)^2;        % Variance
% gamma = sqrt(var)*randn(1);

%   -> Ubicacion
xx = x(1);%-(L/2)*cos(theta);
yy = x(2);%-(L/2)*sin(theta);

xx_p = xx + T*v*cos(theta);
yy_p = yy + T*v*sin(theta);
theta_p = theta + T*v*tan(phi)/L;
theta_p = pi_to_pi(theta_p);

%  NUEVA CONFIGURACION DEL ROBOT
xp = [xx_p; yy_p;theta_p];

end % END
