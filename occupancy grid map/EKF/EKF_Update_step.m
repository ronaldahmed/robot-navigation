function [muc, Pc] = EKF_Update_step(mup, Pp, z, mi, std_sensor,M,dim);
    %Matriz de covarianzas Q
    Q=[std_sensor(1)^2 0 0;...
        0 std_sensor(2)^2 0;...
        0 0 std_sensor(3)^2];
    % COmponentes de la celda
    mjx = dim*(mi(1)-1) + dim/2;
    mjy = dim*(mi(2)-1) + dim/2;
    mjs = (mi(1)-1)*M + mi(2);
    
    % componentes de la media
    mux=mup(1);
    muy=mup(2);
    
    % distancia cuadrada de la media a la marca
    q=(mjx-mux)^2 + (mjy-muy)^2;
    % Medida esperada "predecida" por el robot
    zpred=[sqrt(q);...
        pi_to_pi(atan2(mjy-muy,mjx-mux)-mup(3));...
        mjs];
    % Calculo del jacobiano H de la medida de los sensores
    if(q==0)
        %disp('Cuidado con R')
        H=[0 0 0;...
            0 0 -1;...
            0 0 0];
    else
        H=[-(mjx-mux)/sqrt(q), -(mjy-muy)/sqrt(q), 0;...
            (mjy-muy)/q, -(mjx-mux)/q, -1;...
            0, 0, 0];
    end
    % Calculo de la matriz S
    S=H*Pp*H'+Q;
    % calculo de la matriz K
    K=Pp*H'*S^(-1);
    % vector de innovacion error en la smedidas
    v=z-zpred;
    v(2)=pi_to_pi(v(2));
    % Calculo de la media de correcion
    muc=mup+K*(v);
    muc(3)=pi_to_pi(muc(3));
    % Calculo de la matriz de medicion corregida
    Pc=(eye(3)-K*H)*Pp;
    