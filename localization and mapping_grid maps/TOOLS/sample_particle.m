function x = sample_particle(map,grid_dim)
[N,M] = size(map);
sampled = 0;
while sampled==0
    xi = randi(N);
    yi = randi(M);
    if map(xi,yi) == 0
        xi = (xi-1)*grid_dim + grid_dim/2 + 0.005*randn(1);
        yi = (yi-1)*grid_dim + grid_dim/2 + 0.005*randn(1);
        x = [xi;yi;pi_to_pi(-pi + 2*pi*rand(1))];
        sampled = 1;
    end
end

end