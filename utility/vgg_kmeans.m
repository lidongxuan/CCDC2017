function [CX, sse] = vgg_kmeans(X, nclus, varargin)

% VGG_KMEANS    initialize K-means clustering
%               [CX, sse] = vgg_kmeans(X, nclus, optname, optval, ...)
%
%               - X: input points (one per column)
%               - nclus: number of clusters
%               - opts (defaults):
%                    maxiters (inf): maxmimum number of iterations
%                    mindelta (eps): minimum change in SSE per iteration
%                       verbose (1): 1=print progress
%
%               - CX: cluster centers
%               - sse: SSE��Ŀ�꺯��ֵ��

% Author: Mark Everingham <me@robots.ox.ac.uk>
% Date: 13 Jan 03


opts = struct('maxiters', inf, 'mindelta', eps, 'verbose', 1);

%nargin�Ǻ�����������ĸ�����Ҳ��˵����������С��2������������Ϊ��
%opts = struct('maxiters', inf,'mindelta', eps, 'verbose', 1);
if nargin > 2
    opts=vgg_argparse(opts,varargin);%����д��ṹ��
end

%�������X��������
perm=randperm(size(X,2));
%CXΪ������X�����ǰnclus��������ɵľ����൱�������nclus��������Ϊ����ĳ�ʼ������
CX=X(:,perm(1:nclus));

%��ʼ��Ŀ�꺯��ֵΪinf
sse0 = inf;
iter = 0;%��������

%����
while iter < opts.maxiters
    
    [CX, sse] = vgg_kmiter(X, CX);%����һ��
    
    if opts.verbose%��ʾ��������
        fprintf('iter %d: sse = %g (%g secs)\n', iter, sse, t)
    end
    
    k = sse0-sse;
    if k < opts.mindelta%�Ƿ�ﵽ��ֵ
        break
    end
    
    sse0=sse;
    iter=iter+1;
    
end%���ڵ�������ʱ�����

