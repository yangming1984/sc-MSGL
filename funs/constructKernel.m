function [K,elapse] = constructKernel(fea_a,fea_b,options)
if (~exist('options','var'))
   options = [];
else
   if ~isstruct(options)
       error('parameter error!');
   end
end
if ~isfield(options,'KernelType')
    options.KernelType = 'Gaussian';
end
switch lower(options.KernelType)
    case {lower('Gaussian')}
        if ~isfield(options,'t')
            options.t = 1;
        end
    case {lower('Polynomial')}
        if ~isfield(options,'d')
            options.d = 2;
        end
    case {lower('PolyPlus')}
        if ~isfield(options,'d')
            options.d = 2;
        end
    case {lower('Linear')}
        ;
    otherwise
        error('KernelType does not exist!');
end
tmp_T = cputime;
switch lower(options.KernelType)
    case {lower('Gaussian')}
        if isempty(fea_b)
            D = EuDist2(fea_a,[],0);
        else
            D = EuDist2(fea_a,fea_b,0);
        end
        K = exp(-D/(2*options.t^2));
    case {lower('Polynomial')}
        if isempty(fea_b)
            D = full(fea_a * fea_a');
        else
            D = full(fea_a * fea_b');
        end
        K = D.^options.d;
    case {lower('PolyPlus')}
        if isempty(fea_b)
            D = full(fea_a * fea_a');
        else
            D = full(fea_a * fea_b');
        end
        K = (D+1).^options.d;
    case {lower('Linear')}
        if isempty(fea_b)
            K = full(fea_a * fea_a');
        else
            K = full(fea_a * fea_b');
        end
    otherwise
        error('KernelType does not exist!');
end
if isempty(fea_b)
    K = max(K,K');
end
elapse = cputime - tmp_T;
