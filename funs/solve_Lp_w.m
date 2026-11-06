function   x   =  solve_Lp_w( y, lambda, p )
J     =   4;
tau   =  (2*lambda.*(1-p)).^(1/(2-p)) + p*lambda.*(2*(1-p)*lambda).^((p-1)/(2-p));
x     =   zeros( size(y) );
i0    =   find( abs(y)>tau );
if length(i0)>=1
    y0    =   y(i0);
    t     =   abs(y0);
    lambda0 = lambda(i0);
    for  j  =  1 : J
        t    =  abs(y0) - p*lambda0.*(max(t,eps)).^(p-1);
        t    =  max(t, 0);
    end
    x(i0)   =  sign(y0).*t;
end
