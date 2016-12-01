
% Input:
%     - video:              the name of the selected video
%     - show_visualization: set to True for visualizing tracking results
%     - show_plots:         set to True for plotting quantitative results
% Output:
%     - precision:          precision thresholded at 20 pixels
%

function [precision, fps] = run_tracker(video, show_visualization, show_plots)

%数据集的路径.将数据集data放在工作目录的上级目录
base_path   = '../data';

% 目标周围的搜索区域padding参数
padding = struct('generic', 1.8, 'large', 1, 'height', 0.4);

lambda = 1e-4;              % 正则参数Regularization parameter (见 Eqn 3 )
output_sigma_factor = 0.1;  % 高斯label的空间带宽

interp_factor = 0.01;       % 模型更新的学习率 (见 Eqn 6a, 6b)
cell_size = 4;              % 空间cell尺寸，Spatial cell size

%是否使用GPU计算？
global enableGPU;
% enableGPU = 'ture';
enableGPU = 0;

%通过load_video_info函数确定目标的处置位置（pos），目标尺寸大小（target_sz），所有帧的目标位置（ground_truth）等信息
 [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);
        
% 集合所有与跟踪相关的参数，进行跟踪
 [positions, time] = tracker_ensemble(video_path, img_files, pos, target_sz, ...
            padding, lambda, output_sigma_factor, interp_factor, ...
            cell_size, show_visualization);
        
% Calculate and show precision plot, as well as frames-per-second
 precisions = precision_plot(positions, ground_truth, video, show_plots);
fps = numel(img_files) / time;
        
fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps)
        
        if nargout > 0,
                %return precisions at a 20 pixels threshold
                precision = precisions(20);
        end
end
