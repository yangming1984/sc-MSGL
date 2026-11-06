function [E_new]= solve_L12norm(E, gamma2)
[m,n] = size(E);
E_new = zeros(m,n);
for k=1:n
    e = E(:,k);
    [tao, mu] = search_tao_mu(e, gamma2);
    e_temp = abs(e) - gamma2*tao/(1.0 + gamma2*tao)*mu;
    e_temp(find(e_temp<0)) = 0;
    E_new(:,k) = sign(e).*e_temp;
end
function [tao,mu] = search_tao_mu(e, gamma2)
d = length(e);
e_abs = abs(e);
[temp,S] = sort(e_abs,'descend');
tao = d;
mu = 1.0/d*sum(e_abs);
while (tao>1) && (e_abs(S(tao)) - gamma2*tao/(1+gamma2*tao)*mu <0)
    mu = tao/(tao-1)*mu - 1.0/(tao-1)*e_abs(S(tao));
    tao = tao-1;
end
