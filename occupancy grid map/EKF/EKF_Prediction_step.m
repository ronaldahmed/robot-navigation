function [mu_pred, P_p] = EKF_Prediction_step(mu, P, u, alpha_VELOCITY, dt)
    theta=mu(3);
    v=u(1);
    ang=u(2); %  antes w
    L=0.2;
    % Calculamos la pred del mov traslacional del robot
    [mu_pred,~,~,~,~]=noise_free_motion_model_velocity(mu,u,dt);
    % hallamos el jacobiano respecto a los estados
    G=[1,0,-v*dt*sin(theta);...
       0,1,v*dt*cos(theta);...
       0,0,1];
    % Hallamos el jacobiano respecto a los controles
    V=[dt*cos(theta),0;
       dt*sin(theta),0;
       dt*tan(ang)/L,dt*v*sec(ang)^2/L];
    % Matriz de incertidumbre en los controles
    M=[alpha_VELOCITY(1)*v^2+alpha_VELOCITY(2)*ang^2,0;...
       0,alpha_VELOCITY(3)*v^2+alpha_VELOCITY(4)*ang^2];
    % Calculo de la nueva media
    %mup=mu+[-v/w*sin(theta)+v/w*sin(theta+w*dt);v/w*cos(theta)-v/w*cos(theta+w*dt);w*dt];
    % Matriz de covarianza de prediccion
    P_p=G*P*G'+V*M*V';
end
    