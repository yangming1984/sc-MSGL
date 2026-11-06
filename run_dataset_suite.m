function run_dataset_suite(dataset, action)
if nargin < 1 || isempty(dataset), error('dataset is required'); end
if nargin < 2 || isempty(action), action = 'mainline'; end

here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here,'funs')); addpath(fullfile(here,'twist'));
addpath(fullfile(here,'ClusteringMeasure'));

dataset = char(dataset);
suite_dir = fullfile(here, 'dataset', lower(dataset) + "_suite");
out_dir = fullfile(here, 'results', dataset);
if ~exist(suite_dir,'dir'), error('Suite directory does not exist: %s', suite_dir); end
if strcmpi(action,'reset')
    error('reset is intentionally not destructive; remove the dataset result folder after verifying it, then rerun.');
end
if ~strcmpi(action,'mainline')
    error('Unknown action %s. Supported action: mainline.', action);
end
if ~exist(fullfile(here,'results'),'dir'), mkdir(fullfile(here,'results')); end
if ~exist(out_dir,'dir'), mkdir(out_dir); end

csv_path = fullfile(out_dir, [dataset '_mainline_results.csv']);
diag_path = fullfile(out_dir, [dataset '_mainline_diagnostics.csv']);
ckpt_path = fullfile(out_dir, 'checkpoint.mat');

completed = load_completed(csv_path, dataset);
if exist(ckpt_path,'file')
    C = load(ckpt_path);
    if isfield(C,'completed_keys')
        completed = union(completed, cellstr(C.completed_keys(:)));
    end
end

files = dir(fullfile(suite_dir,'tau*_hvg*_pc*.mat'));
files = files(~contains({files.name}, '_E10.mat') & ~contains({files.name}, '_E20.mat'));
if isempty(files), error('No base suite MAT files under %s', suite_dir); end
[~,ix] = sort({files.name}); files = files(ix);

write_header_if_needed(csv_path, 'dataset,config,method,rep,N,K,V,ACC,NMI,ARI,iter,err,stop_reason');
write_header_if_needed(diag_path, 'dataset,config,N,K,V,aligned_target,mean_fro_before,mean_fro_after');

N_REP = 10;
fprintf('Dataset %s: %d suite configurations; output=%s\n', dataset, numel(files), out_dir);
for fi = 1:numel(files)
    cfg = erase(files(fi).name,'.mat');
    fprintf('\n=== %s / %s ===\n', dataset, cfg);
    D = load(fullfile(files(fi).folder,files(fi).name));
    if ~isfield(D,'truth'), error('%s has no truth', files(fi).name); end
    gt = double(D.truth(:)); N = numel(gt); K = numel(unique(gt));
    V = clean_views(D, N);
    Vn = numel(V);
    [data, before_norm, after_norm] = align_views(V);
    if ~diagnostic_exists(diag_path,cfg)
        append_diag(diag_path,dataset,cfg,N,K,Vn,sqrt(N),mean(before_norm),mean(after_norm));
    end

    methods = {'MSGL_fro'};
    for mi = 1:numel(methods)
        method = methods{mi};
        for r = 1:N_REP
            key = sprintf('%s|%s|%s|%d',dataset,cfg,method,r);
            if any(strcmp(completed,key))
                fprintf('skip %s rep%d (checkpoint)\n',method,r); continue;
            end
            try
                rng(100+r);
                opts = struct('kernel_on','X','norm_mode','none','alpha',300, ...
                    'gamma',40,'beta',0.1,'p_s',0.9,'d',6, ...
                    'omega',ones(Vn,1),'verbose',false,'save_state',false);
                [S,~,info] = MSGL(data,opts);
                C = SpectralClustering(S,K);
                m = metrics_of(C,gt);
                stop_reason = get_stop_reason(info);
                append_result(csv_path,dataset,cfg,method,r,N,K,Vn,m,info,stop_reason);
                completed{end+1} = key;
                completed = unique(completed,'stable');
                completed_keys = completed;
                save(ckpt_path,'completed_keys','-v7');
                fprintf('%s rep%d: ACC=%.4f NMI=%.4f ARI=%.4f\n',method,r,m.acc,m.nmi,m.ari);
            catch ME
                fail_path = fullfile(out_dir,'failures.log');
                fid=fopen(fail_path,'a'); fprintf(fid,'[%s] %s|%s|%s|%d\n%s\n',datestr(now,31),dataset,cfg,method,r,getReport(ME)); fclose(fid);
                fprintf(2,'FAILED %s rep%d: %s\n',method,r,ME.message);
                rethrow(ME);
            end
        end
    end
end
fprintf('\n%s mainline complete. Results: %s\n',dataset,csv_path);
end

function V = clean_views(D,N)
if isfield(D,'views')
    V = D.views;
elseif isfield(D,'X')
    V = {D.X};
else
    error('MAT file must contain views or X');
end
if ~iscell(V), error('views must be a cell array'); end
for i=1:numel(V)
    if iscell(V{i}) && numel(V{i})==1, V{i}=V{i}{1}; end
    V{i}=double(V{i});
    if size(V{i},1) ~= N, error('view %d has %d rows, expected %d',i,size(V{i},1),N); end
    V{i}=fillmissing(V{i},'constant',0);
end
end

function [data,before_norm,after_norm] = align_views(V)
target=sqrt(size(V{1},1)); data=cell(size(V)); before_norm=zeros(numel(V),1); after_norm=zeros(numel(V),1);
for i=1:numel(V)
    before_norm(i)=norm(V{i},'fro');
    data{i}=V{i}*(target/max(before_norm(i),eps));
    after_norm(i)=norm(data{i},'fro');
end
end

function m=metrics_of(C,gt)
[~,nmi,~]=compute_nmi(gt,C); A=Accuracy(C,double(gt)); [AR,~,~,~]=RandIndex(gt,C);
m.acc=A(1); m.nmi=nmi(1); m.ari=AR(1);
end

function s=get_stop_reason(info)
if isfield(info,'stop_reason'), s=char(string(info.stop_reason)); else, s='unknown'; end
end

function keys=load_completed(path,dataset)
keys={}; if ~exist(path,'file'), return; end
try
    T=readtable(path,'TextType','string');
    for i=1:height(T)
        keys{end+1}=sprintf('%s|%s|%s|%d',dataset,char(T.config(i)),char(T.method(i)),T.rep(i));
    end
catch
    warning('Could not read existing results; checkpoint will still be used.');
end
keys=unique(keys,'stable');
end

function write_header_if_needed(path,header)
if ~exist(path,'file') || dir(path).bytes==0
    fid=fopen(path,'w'); fprintf(fid,'%s\n',header); fclose(fid);
end
end

function append_result(path,dataset,cfg,method,r,N,K,V,m,info,stop_reason)
fid=fopen(path,'a'); fprintf(fid,'%s,%s,%s,%d,%d,%d,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%s\n', ...
    dataset,cfg,method,r,N,K,V,m.acc,m.nmi,m.ari,getfield_default(info,'iter',NaN), ...
    getfield_default(info,'err',NaN),stop_reason); fclose(fid);
end

function append_diag(path,dataset,cfg,N,K,V,target,before,after)
fid=fopen(path,'a'); fprintf(fid,'%s,%s,%d,%d,%d,%.10g,%.10g,%.10g\n',dataset,cfg,N,K,V,target,before,after); fclose(fid);
end

function tf=diagnostic_exists(path,cfg)
tf=false; if ~exist(path,'file') || dir(path).bytes==0, return; end
try
    T=readtable(path,'TextType','string'); tf=any(strcmp(string(T.config),cfg));
catch
    tf=false;
end
end

function x=getfield_default(s,f,d)
if isfield(s,f), x=s.(f); else, x=d; end
end

function s=getReport(ME)
s=ME.message; for i=1:numel(ME.stack), s=sprintf('%s\n  %s:%d',s,ME.stack(i).name,ME.stack(i).line); end
end
