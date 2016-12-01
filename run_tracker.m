
% Input:
%     - video:              the name of the selected video
%     - show_visualization: set to True for visualizing tracking results
%     - show_plots:         set to True for plotting quantitative results
% Output:
%     - precision:          precision thresholded at 20 pixels
%

function [precision, fps] = run_tracker(video, show_visualization, show_plots)

%���ݼ���·��.�����ݼ�data���ڹ���Ŀ¼���ϼ�Ŀ¼
base_path   = '../data';

% Ŀ����Χ����������padding����
padding = struct('generic', 1.8, 'large', 1, 'height', 0.4);

lambda = 1e-4;              % �������Regularization parameter (�� Eqn 3 )
output_sigma_factor = 0.1;  % ��˹label�Ŀռ����

interp_factor = 0.01;       % ģ�͸��µ�ѧϰ�� (�� Eqn 6a, 6b)
cell_size = 4;              % �ռ�cell�ߴ磬Spatial cell size

%�Ƿ�ʹ��GPU���㣿
global enableGPU;
% enableGPU = 'ture';
enableGPU = 0;

%ͨ��load_video_info����ȷ��Ŀ��Ĵ���λ�ã�pos����Ŀ��ߴ��С��target_sz��������֡��Ŀ��λ�ã�ground_truth������Ϣ
 [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);
        
% ���������������صĲ��������и���
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
