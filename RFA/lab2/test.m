list_a=[.1 1 10 100 1000 10000 100000];
list_b=[.1 1 10 100 1000 10000 100000];
list_b

function [E] = test_dataset_parameters(a,b)
    E = a+b;
endfunction
f      = @ test_dataset_parameters;   % a simulated  error function

[ Grid_a, Grid_b ] = ndgrid( list_a, list_b );

F = log( f( Grid_a, Grid_b ) );   % log error evaluated on the gridpoints

SurfObj = surf( list_a, list_b, F );
Axes = get( SurfObj, 'parent' )
set( Axes, 'xscale', 'log', 'yscale', 'log' )   % use logarithmic axes