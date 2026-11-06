function [ProcessData] = NormalizeData(X, mode)
if nargin < 2
    mode = 'feature';
end
ProcessData = zeros(size(X));
switch lower(mode)
    case 'feature'
        n = size(X,2);
        for i = 1:n
            ProcessData(:,i) = X(:,i) ./ max(1e-12, norm(X(:,i)));
        end
    case 'sample'
        n = size(X,1);
        for i = 1:n
            ProcessData(i,:) = X(i,:) ./ max(1e-12, norm(X(i,:)));
        end
    otherwise
        error('NormalizeData: unknown mode ''%s'' (use ''feature'' or ''sample'')', mode);
end
end
