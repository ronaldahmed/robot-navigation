function q = likelihood_field_range_finder_model(X,x_sensor,zt,N,dim,std_hit,Zw,z_max)
% retorna probabilidad de medida range finder :)
% X col, zt col, xsen col
[n,m] = size(N);

% Robot global position and orientation
theta = X(3);

% Beam global angle
theta_sen = zt(2);
phi = pi_to_pi(theta + theta_sen);

%Tranf matrix in case sensor has relative position respecto to robot's CG
rotS = [cos(theta),-sin(theta);sin(theta),cos(theta)];

% Prob. distros parameters
sigmaR = std_hit;
zhit  = Zw(1);
zrand = Zw(2);
zmax  = Zw(3);

% Actual algo
q = 1;
if zt(1) ~= z_max
    % get global pos of end point of measument
    xz = X(1:2) + rotS*x_sensor + zt(1)*[cos(phi);
                                         sin(phi)];
    xi = floor(xz(1)/dim) + 1;
    yi = floor(xz(2)/dim) + 1;
    
    % if end point doesn't lay inside map: unknown
    if xi<1 || xi>n || yi<1 || yi>m
        q = 1.0/z_max; % all measurements equally likely, uniform in range [0-zmax]
        return
    end
    
    dist2 = N(xi,yi);
    gd = gauss_1D(0,sigmaR,dist2);
    q = zhit*gd + zrand/zmax;
end

end