function  keep = simple_resample(w)
m = length(w);
keep = zeros(1,m);
w = w/sum(w);

for i=1:m
    r = rand(1);
    ac = w(1);
    id =1;
    while ac < r
        id=id+1;
        ac = ac + w(id);
    end
    keep(i) = id;
end

end