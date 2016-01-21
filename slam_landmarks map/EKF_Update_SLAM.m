function [mu_up, P_up] = EKF_Update_SLAM(mu, P, z, std_sensor)
global done
global NN

n = (size(mu,1)-3)/3;
Fxj = [eye(3), zeros(3,3*n)];
mu_up = 0;
P_up = 0;
sigma = std_sensor.^2;
Q = diag(sigma);

r = z(1);
phi = z(2);
idj = round(z(3));

if ~done(idj)
    done(idj) = true;
    mu(3*idj+1:3*idj+3) = mu(1:3) + r*[cos(phi+mu(3));
                   sin(phi+mu(3));
                   0];
end
delta = mu(3*idj+1:3*idj+2) - mu(1:2);

q = delta'*delta;
q = sqrt(q);
z_med = [q;
         atan2(delta(2),delta(1)) - mu(3);
         mu(3*idj+3) ];
     
z_med(2) = pi_to_pi(z_med(2));
Fxj = zeros(6,3*NN+3);
Fxj(1:3,1:3) = eye(3);
Fxj(4:6,(3*idj + 1):(3*idj + 3)) = eye(3);

dx = delta(1);
dy = delta(2);
%Jac del mod sensor
if q==0
    c1 = [0,0,0]';
    c2 = [0,0,0]';
    c3 = [0,-1,0]';
else
    c1 = [-dx/q;
           dy / (q*q);
                0];
    c2 = [-dy / q;
          -dx / (q*q);
            0];
    c3 = [0,-1,0]';
end
H = [c1,c2,c3,-c1,-c2,[0;0;1]];
H = H*Fxj;
S = H*P*H'+Q;
K = P*H'*inv(S);

difZ = z-z_med;
difZ(2) = pi_to_pi(difZ(2));

mu_up = mu + K*difZ;
mu_up(3) = pi_to_pi(mu_up(3));
P_up = (eye(3*NN+3) - K*H)*P;

end