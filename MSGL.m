function [S, Z, info] = MSGL(data, opts)
if nargin < 2 || isempty(opts), opts = struct(); end
kernel_on = getdef(opts, 'kernel_on', 'X');
norm_mode = getdef(opts, 'norm_mode', 'feature');
d         = getdef(opts, 'd', 6);
max_iter  = getdef(opts, 'max_iter', 75);
tol       = getdef(opts, 'tol', 1e-6);
alpha     = getdef(opts, 'alpha', 100);
gamma     = getdef(opts, 'gamma', 40);
beta      = getdef(opts, 'beta', 0.1);
p_s       = getdef(opts, 'p_s', 0.9);
verbose   = getdef(opts, 'verbose', true);
sigma_scale = getdef(opts, 'sigma_scale', 1.0);
fusion_w    = getdef(opts, 'fusion_w', []);
auto_weight = getdef(opts, 'auto_weight', 'off');
w_min       = getdef(opts, 'w_min', 0.1);
knn_k       = getdef(opts, 'knn_k', 0);
knn_mode    = getdef(opts, 'knn_mode', 'union');
mst_repair  = getdef(opts, 'mst_repair', false);
init_S      = getdef(opts, 'init_S', []);
rho_mode    = getdef(opts, 'rho_mode', 'geom');
eta         = getdef(opts, 'eta', 2);
rho_mu      = getdef(opts, 'rho_mu', 10);
rho_tau     = getdef(opts, 'rho_tau', 2);
kernel_scale = getdef(opts, 'kernel_scale', []);
inv_mode    = getdef(opts, 'inv_mode', 'inv');
track_graph = getdef(opts, 'track_graph', false);
save_state  = getdef(opts, 'save_state', false);
disable_tensor = getdef(opts, 'disable_tensor', false);
disable_projection = getdef(opts, 'disable_projection', false);
V = numel(data);
N = size(data{1},1);
omega = getdef(opts, 'omega', ones(V,1));
assert(numel(omega) == V, 'omega must have one entry per view (got %d, V=%d)', numel(omega), V);
if isempty(fusion_w)
    fw = ones(V,1);
else
    fw = double(fusion_w(:));
    assert(numel(fw) == V, 'fusion_w must have one entry per view');
end
if islogical(auto_weight)
    if auto_weight, auto_weight = 'forward'; else, auto_weight = 'off'; end
end
aw_on = ~strcmpi(auto_weight, 'off');
assert(strcmpi(auto_weight,'off') || strcmpi(auto_weight,'forward') || strcmpi(auto_weight,'inverse'), ...
    'auto_weight must be ''off'', ''forward'' or ''inverse''');
assert(strcmpi(rho_mode,'geom') || strcmpi(rho_mode,'adaptive'), 'rho_mode must be geom or adaptive');
assert(strcmpi(inv_mode,'inv') || strcmpi(inv_mode,'pinv'), 'inv_mode must be inv or pinv');
for v = 1:V
    data{v} = double(data{v});
    assert(size(data{v},1) == N, 'all views must have the same number of rows (samples)');
end
X = cell(1, V);
sigma = zeros(1, V);
T = cell(1, V);
T0 = cell(1, V);
for v = 1:V
    switch lower(norm_mode)
        case 'none'
            X{v} = data{v};
        otherwise
            X{v} = NormalizeData(data{v}, norm_mode);
    end
    switch lower(kernel_on)
        case 'x'
            sigma(v) = optSigma(X{v});
        case 'data'
            sigma(v) = optSigma(data{v});
        otherwise
            error('kernel_on must be ''X'' or ''data''');
    end
    options.KernelType = 'Gaussian';
    options.t = sigma_scale * sigma(v);
    switch lower(kernel_on)
        case 'x'
            T{v} = constructKernel(X{v}, X{v}, options);
        case 'data'
            T{v} = constructKernel(data{v}, data{v}, options);
    end
end
if knn_k > 0
    for v = 1:V
        T0{v} = T{v};
        T{v} = knn_truncate(T{v}, knn_k, knn_mode);
        if mst_repair
            T{v} = mst_repair_kernel(T{v}, T0{v});
        end
    end
else
    T0 = T;
end
if ~isempty(kernel_scale)
    ks = double(kernel_scale(:));
    assert(numel(ks) == V, 'kernel_scale must have one entry per view');
    for v = 1:V, T{v} = ks(v) * T{v}; end
end
T_tensor = cat(3, T{:,:});
all_dim = 0;
for k = 1:V
    all_dim = all_dim + size(X{k},2);
    Z{k} = zeros(N,N);
    J{k} = zeros(N,N);
    C{k} = zeros(N,N);
end
Z_tensor = cat(3, Z{:,:});
J_tensor = cat(3, J{:,:});
C_tensor = cat(3, C{:,:});
if isempty(init_S)
    S_tensor = T_tensor;
else
    init_S = double(init_S);
    assert(isequal(size(init_S), [N N]), 'init_S must be N x N');
    S_tensor = repmat(init_S, [1, 1, V]);
end
sX = [N, N, V];
H = eye(N) - 1/N*ones(N);
condXTX = zeros(1,V);
for ii = 1:V
    dim = size(X{ii},2);
    A_x = X{ii}'*H*X{ii};
    condXTX(ii) = cond(1e-6*eye(dim,dim) + A_x);
    switch lower(inv_mode)
        case 'inv'
            invXTX{ii} = inv(1e-6*eye(dim,dim) + A_x);
        case 'pinv'
            invXTX{ii} = pinv(1e-6*eye(dim,dim) + A_x);
    end
end
rho = 1e-4;
rho_max = 1e12;
if aw_on, w_aw = ones(V,1)/V; end
iter = 0;
err = inf;
stop_reason = 'max_iter';
rho_traj = zeros(max_iter,1); primal_traj = zeros(max_iter,1); dual_traj = zeros(max_iter,1);
w_traj = zeros(max_iter,V);
dens_traj = zeros(max_iter,1); comp_traj = zeros(max_iter,1);
while iter < max_iter
    if verbose
        fprintf('----processing iter %d--------\n', iter+1);
    end
    J_prev = J_tensor;
    for ii = 1:V
        S_v = S_tensor(:,:,ii);
        J_v = J_tensor(:,:,ii);
        C_v = C_tensor(:,:,ii);
        T_v = (alpha*S_v + rho*J_v - C_v)./(alpha + rho);
        if aw_on
            if strcmpi(auto_weight, 'forward')
                lam_ii = 2*beta*w_aw(ii)/(alpha + rho);
            else
                lam_ii = 2*beta/((alpha + rho)*max(w_aw(ii), 1e-12));
            end
        else
            lam_ii = 2*beta/(alpha + rho);
        end
        Z{ii} = solve_L12norm(T_v, lam_ii);
    end
    Z_tensor = cat(3, Z{:,:});
    if disable_tensor
        J_tensor = Z_tensor;
    else
        z = Z_tensor(:);
        c = C_tensor(:);
        [j, ~] = wshrinkObj_weight_lp(z + 1/rho*c, omega./rho, sX, 0, 3, p_s);
        J_tensor = reshape(j, sX);
    end
    for ii = 1:V
        J{ii} = J_tensor(:,:,ii);
    end
    if aw_on
        f_aw = zeros(V,1);
        for ii = 1:V
            f_aw(ii) = norm(Z{ii} - J{ii}, 'fro')^2 + eps;
        end
        w_raw = (1./f_aw) / sum(1./f_aw);
        w_aw = max(w_raw, w_min);
        w_aw = w_aw / sum(w_aw);
    end
    for ii = 1:V
        if disable_projection
            W{ii} = eye(size(X{ii},2));
        else
            tmp_S = S_tensor(:,:,ii);
            tmp_D = diag(sum(tmp_S));
            tmp_L = tmp_D - tmp_S;
            M = invXTX{ii}*(X{ii}'*tmp_L*X{ii});
            [tmp_W,~] = eigs(M, d);
            W{ii} = tmp_W*diag(1./sqrt(diag(tmp_W'*tmp_W)));
        end
    end
    for ii = 1:V
        A = zeros(N);
        distx = L2_distance_1(W{ii}'*X{ii}', W{ii}'*X{ii}');
        tmp_Z = Z{ii};
        for jj = 1:N
            dxi = distx(jj,:);
            ad = -(dxi - alpha*tmp_Z(jj,:))/(alpha + gamma);
            A(jj,:) = EProjSimplex_new(ad);
        end
        S_tensor(:,:,ii) = A;
    end
    C_tensor = C_tensor + rho * (Z_tensor - J_tensor);
    r_now = norm(J_tensor(:) - Z_tensor(:), inf);
    s_now = rho * norm(J_tensor(:) - J_prev(:), inf);
    switch lower(rho_mode)
        case 'geom'
            rho = min(eta * rho, rho_max);
        case 'adaptive'
            if r_now > rho_mu * s_now
                rho = min(rho_tau * rho, rho_max);
            elseif s_now > rho_mu * r_now
                rho = max(rho / rho_tau, 1e-8);
            end
    end
    err = norm(J_tensor(:) - Z_tensor(:), inf);
    rec = iter + 1;
    primal_traj(rec) = r_now;
    dual_traj(rec) = s_now;
    rho_traj(rec) = rho;
    if aw_on, w_traj(rec,:) = w_aw'; end
    if track_graph
        dens = 0; comp = 0;
        for ii = 1:V
            S_v = S_tensor(:,:,ii);
            Av = double(S_v > 0); Av(1:N+1:end) = 0;
            dens = dens + nnz(Av)/(N*(N-1));
            Gv = graph(Av, 'upper');
            comp = max(comp, numel(unique(conncomp(Gv))));
        end
        dens_traj(rec) = dens/V;
        comp_traj(rec) = comp;
    end
    if err < tol
        stop_reason = 'tol';
        break;
    end
    iter = iter + 1;
end
S = zeros(N,N);
if aw_on && isempty(fusion_w)
    fw_final = w_aw;
else
    fw_final = fw;
end
for k = 1:V
    S = S + fw_final(k)*Z{k};
end
S = (S + S')/2;
n_rec = min(iter + 1, max_iter);
info.iter = iter;
info.err = err;
info.stop_reason = stop_reason;
info.rho_final = rho;
info.rho_traj = rho_traj(1:n_rec);
info.primal_traj = primal_traj(1:n_rec);
info.dual_traj = dual_traj(1:n_rec);
info.cond = condXTX;
info.sigma = sigma;
if aw_on
    info.w_traj = w_traj(1:n_rec,:);
    info.w_final = w_aw;
end
if track_graph
    info.density_traj = dens_traj(1:n_rec);
    info.components_traj = comp_traj(1:n_rec);
end
if save_state
    info.T = T;
    info.J = J;
end
end
function K = knn_truncate(K, k, mode)
N = size(K,1);
k = min(k, N);
[~, idx] = sort(K, 2, 'descend');
M = false(N);
for i = 1:N
    M(i, idx(i,1:k)) = true;
end
switch lower(mode)
    case 'union'
        K = K .* (M | M.');
    case 'mutual'
        K = K .* (M & M.');
    case 'soft'
        K = K .* (M + 0.1*(~M));
        K = max(K, K.');
    otherwise
        error('knn_mode must be ''union'', ''mutual'' or ''soft''');
end
end
function K = mst_repair_kernel(K, K0)
N = size(K,1);
A = double(K > 0); A(1:N+1:end) = 0;
bin = conncomp(graph(A, 'upper'));
if numel(unique(bin)) <= 1, return; end
Wk = -log(max(K0, 1e-300)); Wk(1:N+1:end) = 0;
Tr = minspantree(graph(Wk, 'upper'));
[ii, jj] = findedge(Tr);
for e = 1:numel(ii)
    K(ii(e),jj(e)) = max(K(ii(e),jj(e)), K0(ii(e),jj(e)));
    K(jj(e),ii(e)) = max(K(jj(e),ii(e)), K0(jj(e),ii(e)));
end
end
function val = getdef(s, name, def)
if isfield(s, name) && ~isempty(s.(name))
    val = s.(name);
else
    val = def;
end
end
