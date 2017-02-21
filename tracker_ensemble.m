%基于卷积特征的CF（Correlation filter）跟踪算法
% Input:
%   - video_path:          图片序列路径
%   - img_files:           图标序列的名称列表
%   - pos:                 目标的初始中心位置
%   - target_sz:           目标的初始大小
% 	- padding:             跟踪区域的padding参数
%   - lambda:              岭回归（ridge regression）的正则项
%   - output_sigma_factor: 高斯标记labelling的空间带宽
%   - interp_factor:       模型更新的学习率
%   - cell_size:           空间量化水平，尺度比例
%   - show_visualization:  1代表输出可视化结果，0代表不输出
% Output:
%   - positions:           预测出的目标位置
%   - time:                跟踪这一帧所耗时间

function [positions, time] = tracker_ensemble(video_path, img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, cell_size, show_visualization)

% ================================================================================
% 环境设置
% ================================================================================
% indLayers = 1; 
% nweights=1;
indLayers = [1,2]; 
nweights=[1,0.2];
numLayers = length(indLayers);

if min(target_sz)>32
    bili=32/min(target_sz);
else
    bili=1;
end

% 获取图片尺寸
%im_sz     = size(imread([video_path img_files{1}]))
im_sz = floor(size(imread([video_path img_files{1}]))*bili);
target_sz=floor(target_sz*bili);
% 确定搜索域尺寸。针对目标相对整个背景的大小选择合理的搜索域尺寸。padding结构体里提供了三个针对不同情况的比例系数
window_sz = get_search_window(target_sz, im_sz, padding);

% 计算高斯标记函数的sigma，是目标尺寸宽高乘积的平方根，乘以output_sigma_factor（0.1），再除以cell_size（4）
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size    d=bsxfun(@times,c,[1 2]);

%l1_patch_num为搜索域的尺寸除以cell_size
l1_patch_num = floor(window_sz / cell_size);

% gaussian_shaped_labels是形成搜索域/cell_size大小的二维高斯分布标签，毕竟波峰移到左上角（循环平移，因此四个角都是峰值）
% 对gaussian_shaped_labels形成的高斯分布进行快速傅里叶变换，得到yf
yf = fft2(gaussian_shaped_labels(output_sigma, l1_patch_num));
% 根据yf的尺寸构建出二维cos平面。（为了避免边界不连续）
cos_window = hann(size(yf,1)) * hann(size(yf,2))';
% 建立可视化视频界面
if(show_visualization)
    update_visualization = show_video(img_files, video_path);
end
% 初始化用于计算FPS和distance precision的参数
time      = 0;
positions = zeros(numel(img_files), 2);
nweights  = reshape(nweights,1,1,[]);
% Note: variables ending with 'f' are in the Fourier domain.
model_xf     = cell(1, numLayers);
model_alphaf = cell(1, numLayers);

im = imread([video_path img_files{1}]);
im = imResample(im,im_sz(1:2));
pos=floor(pos*bili);

% 提取不同层次的卷积特征
% 在图中抠出搜索窗大小（window_sz）的区域（如果越界，超过的值全部设为边界值），命名为patch，作为提取特征的输入
patch = get_subwindow(im, pos, window_sz);%确保了目标位置在patch的中心
%判断该图片是否为RGB三通道图片，是则转换为灰度图

Fisize=100;
patchsize = [6 6];
[kernel] = get_kernel(patchsize,patch,target_sz,Fisize);
kernel=  bsxfun(@minus,kernel,mean(kernel));
kernel_org=kernel;


% ================================================================================
% 开始跟踪
% ================================================================================
for frame = 1:numel(img_files),
    im = imread([video_path img_files{frame}]); % 载入图片
    im=imResample(im,im_sz(1:2));
    
    tic();
    
    % ================================================================================
    % 根据学习到的目标模型预测目标位置
    % ================================================================================
    if frame > 1%如果不是首帧
        % 提取不同层次的卷积特征
        % 在图中抠出搜索窗大小（window_sz）的区域（如果越界，超过的值全部设为边界值），命名为patch，作为提取特征的输入
        patch = get_subwindow(im, pos, window_sz);
        
%        [kernel_] = get_kernel(patchsize,patch,target_sz,Fisize);
%        kernel_=  bsxfun(@minus,kernel_,mean(kernel_));
%        %kernel=0.9*kernel+0.1*kernel_;
%        kernel=0.9*kernel_org+0.1*kernel_;
     
        % 提取层级卷积特征
        feat  = get_features_my(patch, cos_window, kernel,patchsize,indLayers);
        % 计算目标位置
        pos  = predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
            model_xf, model_alphaf);
    end
    
    % ================================================================================
    % 根据层级卷积特征学习correlation filters
    % ================================================================================
    % 提取层级卷积特征
    % feat  = get_features(patch, cos_window, indLayers);
    feat  = get_features_my(patch, cos_window, kernel,patchsize,indLayers);
    % 更新模型
    [model_xf, model_alphaf] = updateModel(feat, yf, interp_factor, lambda, frame, ...
        model_xf, model_alphaf);
    
    % ================================================================================
    % Save predicted position and timing
    % ================================================================================
    positions(frame,:) = pos/bili;
    %disp(1/toc());
    time = time + toc();
    % Visualization
    if show_visualization,
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        %disp(box);
        stop = update_visualization(frame, box/bili);
        if stop, break, end  %user pressed Esc, stop early
        drawnow;
        % pause(0.05)  % uncomment to run slower
    end
end

end






