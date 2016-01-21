function [xp,xc,yc,r,f] = noise_free_motion_model_velocity(x,u,dt)
%implementa el algoritmo de movimiento del robot
L=0.6;
%movimiento traslacional
%calculo de la nueva configuracion
xp=x+[dt*u(1)*cos(x(3));dt*u(1)*sin(x(3));dt*u(2)];
%seteamos las coordenadas del circulo de rotacion
xc=inf;
yc=inf;
%seteamos el radio del circulo y del flag
r=inf;
f=0;
%nos aseguramos que la orientacion este en el rango(-pi;pi)  
xp(3)=pi_to_pi(xp(3));
end 