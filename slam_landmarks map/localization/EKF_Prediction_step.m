function [mu_pred, P_pred] = EKF_Prediction_step(mu, P, u, alpha, dt)
v = u(1);
w = u(2);
theta = mu(3);

[mu_pred,~,~,~,flag] = noise_free_motion_model_velocity(mu,u,dt);
if flag==0
    % traslacional
    G = [1,0,-v*dt*sin(theta);
         0,1,v*dt*cos(theta);
         0,0,1];
    V = [dt*cos(theta),0;
         dt*sin(theta),0;
         0,0];
else
    G = [1,0, -(v/w) * (cos(theta)-cos(theta+w*dt));
         0,1, -(v/w) * (sin(theta)-sin(theta+w*dt));
         0,0, 1];
    c1 = [(-sin(theta) + sin(theta+w*dt) )/w;
          ( cos(theta) - cos(theta+w*dt))/w;
          0];
    c2 = [v*(sin(theta) - sin(theta+w*dt))/w^2 + v*cos(theta+w*dt)*dt/w;
         -v*( cos(theta)-cos(theta+w*dt) )/w^2 + v*sin(theta+w*dt)*dt/w;
         dt];
    V = [c1 c2];
end

M = [alpha(1)*v*v + alpha(2)*w*w,0; 0,alpha(3)*v*v + alpha(4)*w*w];
P_pred = G*P*G' + V*M*V';

end