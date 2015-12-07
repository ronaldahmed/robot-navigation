%%  toExplore()  regresa direcciones de puntos disponibles a explorar segun
%%  la estructura no-holonomica del vehiculo.
function adjacents=toExplore(current_dir,dmin)
    % variables son puntos cardinales
    e=dmin*[ 1  0]';
    w=dmin*[-1  0]';
    n=dmin*[ 0  1]';
    s=dmin*[ 0 -1]';
    ne=dmin*[1  1]';
    se=dmin*[1 -1]';
    nw=dmin*[-1 1]';
    sw=dmin*[-1 -1]';
    
    switch current_dir
        case 0
            adjacents=[ne e se];
        case 45
            adjacents=[n ne e];
        case 90
            adjacents=[nw n ne];
        case 135
            adjacents=[w nw n];
        case 180
            adjacents=[sw w nw];
        case 225
            adjacents=[s sw w];
        case 270
            adjacents=[se s sw];
        case 315
            adjacents=[e se s];
        otherwise
            adjacents=[];
    end
end