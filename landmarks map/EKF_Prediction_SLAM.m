function [mu_pred, P_pred] = EKF_Prediction_SLAM(mu, P, u, alpha, dt)
global NN

Fx = [eye(3), zeros(3,3*NN)];

[mu_pred,~,~,~,flag] = noise_free_motion_model_velocity(mu(1:3),u,dt);

%%
mu_pred = mu + Fx'*(mu_pred - mu(1:3));
%%
v = u(1);
w = u(2);
theta = mu(3);

if flag==0
    % traslacional
    G = [0,0,-v*dt*sin(theta);
         0,0,v*dt*cos(theta);
         0,0,0];
    V = [dt*cos(theta),0;
         dt*sin(theta),0;
         0            ,0];
else
    G = [0,0, -(v/w) * (cos(theta)-cos(theta+w*dt));
         0,0, -(v/w) * (sin(theta)-sin(theta+w*dt));
         0,0, 0];
    c1 = [(-sin(theta) + sin(theta+w*dt) )/w;
          ( cos(theta) - cos(theta+w*dt))/w;
          0];
    c2 = [v*(sin(theta) - sin(theta+w*dt))/w^2 + v*cos(theta+w*dt)*dt/w;
         -v*( cos(theta)-cos(theta+w*dt) )/w^2 + v*sin(theta+w*dt)*dt/w;
         dt];
    V = [c1 c2];
end
M = [alpha(1)*v*v + alpha(2)*w*w,0; 0,alpha(3)*v*v + alpha(4)*w*w];
R = V*M*V';
G = eye(3*NN+3) + Fx'*G*Fx;

%%
P_pred = G*P*G' + Fx'*R*Fx;
%%
end