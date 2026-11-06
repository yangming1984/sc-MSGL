function run_MSGL(dataset, config)
if nargin < 1 || isempty(dataset), dataset = 'klein'; end
if nargin < 2 || isempty(config), config = 'tau03_hvg2000_pc50'; end
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, 'funs'));
addpath(fullfile(here, 'twist'));
addpath(fullfile(here, 'ClusteringMeasure'));
mat_path = fullfile(here, 'dataset', [dataset '_suite'], [config '.mat']);
if ~exist(mat_path, 'file')
    error('数据文件不存在: %s', mat_path);
end
D = load(mat_path);
if ~isfield(D, 'truth'), error('%s 缺少 truth', mat_path); end
gt = double(D.truth(:));
N = numel(gt);
K = numel(unique(gt));
if isfield(D, 'views')
    V = D.views;
elseif isfield(D, 'X')
    V = {D.X};
else
    error('MAT 文件必须含 views 或 X');
end
if ~iscell(V), error('views 必须是 cell 数组'); end
for i = 1:numel(V)
    if iscell(V{i}) && numel(V{i}) == 1, V{i} = V{i}{1}; end
    V{i} = double(V{i});
    if size(V{i}, 1) ~= N, error('视图 %d 行数 %d 与 N=%d 不符', i, size(V{i},1), N); end
    V{i} = fillmissing(V{i}, 'constant', 0);
end
Vn = numel(V);
target = sqrt(N);
data = cell(1, Vn);
for i = 1:Vn
    data{i} = V{i} * (target / max(norm(V{i}, 'fro'), eps));
end
rng(100);
opts = struct('kernel_on', 'X', 'norm_mode', 'none', 'alpha', 300, ...
    'gamma', 40, 'beta', 0.1, 'p_s', 0.9, 'd', 6, ...
    'omega', ones(Vn, 1), 'verbose', false);
[S, ~, info] = MSGL(data, opts);
C = SpectralClustering(S, K);
[~, nmi, ~] = compute_nmi(gt, C);
ACC = Accuracy(C, double(gt));
[AR, ~, ~, ~] = RandIndex(gt, C);
fprintf('%s / %s: N=%d, K=%d, V=%d | ACC=%.4f, NMI=%.4f, ARI=%.4f (iters=%d, %s)\n', ...
    dataset, config, N, K, Vn, ACC(1), nmi(1), AR(1), info.iter, info.stop_reason);
end
