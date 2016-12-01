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
%               - sse: SSE（目标函数值）

% Author: Mark Everingham <me@robots.ox.ac.uk>
% Date: 13 Jan 03


opts = struct('maxiters', inf, 'mindelta', eps, 'verbose', 1);

%nargin是函数输入参数的个数，也就说如果输入参数小于2，将采用配置为：
%opts = struct('maxiters', inf,'mindelta', eps, 'verbose', 1);
if nargin > 2
    opts=vgg_argparse(opts,varargin);%配置写入结构体
end

%随机打乱X的列序列
perm=randperm(size(X,2));
%CX为随机后的X矩阵的前nclus列向量组成的矩阵，相当于随机采nclus个向量作为聚类的初始的向量
CX=X(:,perm(1:nclus));

%初始化目标函数值为inf
sse0 = inf;
iter = 0;%迭代次序

%迭代
while iter < opts.maxiters
    
    [CX, sse] = vgg_kmiter(X, CX);%更新一次
    
    if opts.verbose%显示迭代详情
        fprintf('iter %d: sse = %g (%g secs)\n', iter, sse, t)
    end
    
    k = sse0-sse;
    if k < opts.mindelta%是否达到阈值
        break
    end
    
    sse0=sse;
    iter=iter+1;
    
end%大于迭代次数时则结束

