function map = make_map(dim)
n = 95;
m = 120;
map = ones(n,m);

% ---------
map(1:70,10:40) = zeros(70,31);
map(20:35,10:15) = ones(16,6);
map(45:60,38:40) = ones(16,3);

map(70:100,10:80) = zeros(31,71);
map(20:100,60:90) = zeros(81,31);
map(70:75,50:70) = ones(6,21);
map(95:100,75:85) = ones(6,11);
map(95:100,10:30) = ones(6,21);
map(90:100,15:25) = ones(11,11);

end