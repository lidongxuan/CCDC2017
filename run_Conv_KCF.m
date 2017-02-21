
function results = run_Conv_KCF(seq, res_path, bSaveImage)

addpath('utility','matconvnet/matlab');
% Image file names
img_files = seq.s_frames;
% Seletected target size
target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
% Initial target position
pos       = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);

% Ŀ����Χ����������padding����
% padding = struct('generic', 1.8, 'large', 1, 'height', 0.4);
padding = struct('generic', 2.8, 'large', 2, 'height', 1.4);

lambda = 1e-4;              % �������Regularization parameter (�� Eqn 3 )
output_sigma_factor = 0.1;  % ��˹label�Ŀռ����

interp_factor = 0.01;       % ģ�͸��µ�ѧϰ�� (�� Eqn 6a, 6b)
cell_size = 2;              % �ռ�cell�ߴ磬Spatial cell size

%�Ƿ�ʹ��GPU���㣿
global enableGPU;
% enableGPU = 'ture';
enableGPU = 0;

video_path='';

show_visualization=false;
        
% ���������������صĲ��������и���
 [positions, time] = tracker_ensemble(video_path, img_files, pos, target_sz, ...
            padding, lambda, output_sigma_factor, interp_factor, ...
            cell_size, show_visualization);
        
% ================================================================================
% Return results to benchmark, in a workspace variable
% ================================================================================
rects      = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
rects(:,3) = target_sz(2);
rects(:,4) = target_sz(1);
results.type   = 'rect';
results.res    = rects;
results.fps    = numel(img_files)/time;
end
