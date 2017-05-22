
function [precision, fps] = run_tracker(video, show_visualization, show_plots)

%���ݼ���·��.�����ݼ�data���ڹ���Ŀ¼���ϼ�Ŀ¼
base_path   = '../data/OTB2013';

% Ŀ����Χ����������padding����
% padding = struct('generic', 1.8, 'large', 1, 'height', 0.4);
padding = struct('generic', 2.8, 'large', 2, 'height', 1.4);

lambda = 1e-4;              % ��ع飨ridge regression����������
output_sigma_factor = 0.1;  % ��˹label�Ŀռ����

interp_factor = 0.01;       % ģ�͸��µ�ѧϰ��
cell_size = 2;              % �ռ�cell�ߴ�

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
        
% ����precisions��fps
 precisions = precision_plot(positions, ground_truth, video, show_plots);
fps = numel(img_files) / time;
        
fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps)
        
        if nargout > 0,
                %��ֵ��20������
                precision = precisions(20);
        end
end
