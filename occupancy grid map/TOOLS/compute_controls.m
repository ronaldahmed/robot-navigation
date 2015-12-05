function [v, w, iwp] = compute_controls(x, wp, iwp, VMAX)
% COMPUTE_CONTROLS Compute the values of the controls in order to follow 
%                  the waypoints. 
%
% Description
%   
%   This function calculates the translational and rotational velocities
%   needed in order to follow the waypoints.
%   If the distance to the current waypoint is more than "minD" we adjust
%   the steering to follow the current waypoint; otherwise, we adjust the 
%   steering to follow the next waypoint.
%
% OUTPUTS:
%     v: New translational velocity
%     w: New angular velocity
%   iwp: Index of new waypoint(Can be the current or a new one)
%
% INPUTS:
%      x: Pose of the robot(the representative of the belief)
%     wp: Matrix of waypoints[2xNwp]
%    iwp: Index to current waypoint
%   VMAX: Maximun translational velocity

%----------------------------------------------------------------------%
% 1. PARAMETERS CHECKING
%----------------------------------------------------------------------%
%  1.1. MAKE SURE THE CORRECT DIMENSIONS OF WAYPOINTS
if size(wp,1)~=2
    error('COMPUTE_CONTROLS: Matrix of waypoints must have 2 rows')
end

%  1.2. DISTANCIA MINIMA
minD = 0.1;


%----------------------------------------------------------------------%
% 2. DETERMINE IF CURRENT WAYPOINT WAS REACHED
%----------------------------------------------------------------------%
%  2.1. GET THE CURRENT WAYPOINT
cwp = wp(:,iwp);

%  2.2. COMPUTE THE DISTANCE TO THE CURRENT "wayppoint"
d2 = (cwp(1) - x(1))^2  +  (cwp(2) - x(2))^2;

%  2.3. Check the distance to the current waypoint to update the "waypoint"
if d2 < minD^2
    
    % a. The distance to the current way point is less than the threshold
    %    so switch to next waypoint
    iwp = iwp + 1;
    
    % b. If the robot have reached final waypoint, set "iwp=0" and return
    if iwp > size(wp,2)
        v = 0;
        w = 0;
        iwp = 0;
        return;
    end    
    
    % c. Get nex waypoint
    cwp = wp(:,iwp);
end



%----------------------------------------------------------------------%
% 3. SET THE APROPRIATE CONTROLS TO CONTINUE FOLLOWING THE WAY POINTS
%----------------------------------------------------------------------%
%  3.1. COMPUTE THE "angle" FROM THE LOCATION TO THE CURRENT WAYPOINT
delta = atan2(cwp(2)-x(2), cwp(1)-x(1))  -  x(3);
delta = pi_to_pi(delta);

%  3.2. SET THE VELOCITIES
if(abs(delta) >= pi/2)
    v = VMAX/128;
    w = sign(delta)*pi/4;
elseif(abs(delta) >= pi/4  && abs(delta) < pi/2)
    v = VMAX/64;
    w = sign(delta)*pi/8;
elseif(abs(delta) >= pi/8  &&  abs(delta) < pi/4)
    v = VMAX/32;
    w = sign(delta)*pi/16;
elseif(abs(delta) >= pi/16 &&  abs(delta) < pi/8)
    v = VMAX/16;
    w = sign(delta)*pi/32;
elseif(abs(delta) >= pi/32 && abs(delta) < pi/16)
    v = VMAX/8;
    w = sign(delta)*pi/64;
elseif(abs(delta) >= pi/64 && abs(delta) < pi/32)
    v = VMAX/4;
    w = sign(delta)*pi/128;
elseif(abs(delta) >= pi/128 && abs(delta) < pi/256)
    v = VMAX/2;
    w = sign(delta)*pi/256;
else
    v = VMAX;
    w = 0;
end



end % END