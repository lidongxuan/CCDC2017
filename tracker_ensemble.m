%���ھ��������CF��Correlation filter�������㷨
% Input:
%   - video_path:          ͼƬ����·��
%   - img_files:           ͼ�����е������б�
%   - pos:                 Ŀ��ĳ�ʼ����λ��
%   - target_sz:           Ŀ��ĳ�ʼ��С
% 	- padding:             ���������padding����
%   - lambda:              ��ع飨ridge regression����������
%   - output_sigma_factor: ��˹���labelling�Ŀռ����
%   - interp_factor:       ģ�͸��µ�ѧϰ��
%   - cell_size:           �ռ�����ˮƽ���߶ȱ���
%   - show_visualization:  1����������ӻ������0�������
% Output:
%   - positions:           Ԥ�����Ŀ��λ��
%   - time:                ������һ֡����ʱ��

function [positions, time] = tracker_ensemble(video_path, img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, cell_size, show_visualization)

% ================================================================================
% ��������
% ================================================================================
indLayers = 1; 
nweights=1;
% indLayers = [1,2]; 
% nweights=[1,0.5];
% indLayers = [37, 28, 19];   % ѡ��VGG Net�� Conv5-4, Conv4-4, and Conv3-4 
% nweights  = [1, 0.5, 0.02]; % ����CF��Ӧ��Ȩֵ Weights for combining correlation filter responses
numLayers = length(indLayers);

% ��ȡͼƬ�ߴ�
im_sz     = size(imread([video_path img_files{1}]));
% ȷ��������ߴ硣���Ŀ��������������Ĵ�Сѡ������������ߴ硣padding�ṹ�����ṩ��������Բ�ͬ����ı���ϵ��
window_sz = get_search_window(target_sz, im_sz, padding);
% �����˹��Ǻ�����sigma����Ŀ��ߴ��߳˻���ƽ����������output_sigma_factor��0.1�����ٳ���cell_size��4��
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size    d=bsxfun(@times,c,[1 2]);

%l1_patch_numΪ������ĳߴ����cell_size
l1_patch_num = floor(window_sz / cell_size);

% gaussian_shaped_labels���γ�������/cell_size��С�Ķ�ά��˹�ֲ���ǩ���Ͼ������Ƶ����Ͻǣ�ѭ��ƽ�ƣ�����ĸ��Ƕ��Ƿ�ֵ��
% ��gaussian_shaped_labels�γɵĸ�˹�ֲ����п��ٸ���Ҷ�任���õ�yf
yf = fft2(gaussian_shaped_labels(output_sigma, l1_patch_num));
% ����yf�ĳߴ繹������άcosƽ�档��Ϊ�˱���߽粻������
cos_window = hann(size(yf,1)) * hann(size(yf,2))';
% �������ӻ���Ƶ����
if(show_visualization)
    update_visualization = show_video(img_files, video_path);
end
% ��ʼ�����ڼ���FPS��distance precision�Ĳ���
time      = 0;
positions = zeros(numel(img_files), 2);
nweights  = reshape(nweights,1,1,[]);
% Note: variables ending with 'f' are in the Fourier domain.
model_xf     = cell(1, numLayers);
model_alphaf = cell(1, numLayers);

im = imread([video_path img_files{1}]);

% ��ȡ��ͬ��εľ������
% ��ͼ�пٳ���������С��window_sz�����������Խ�磬������ֵȫ����Ϊ�߽�ֵ��������Ϊpatch����Ϊ��ȡ����������
patch = get_subwindow(im, pos, window_sz);%ȷ����Ŀ��λ����patch������
%�жϸ�ͼƬ�Ƿ�ΪRGB��ͨ��ͼƬ������ת��Ϊ�Ҷ�ͼ

Fisize=100;
patchsize = [6 6];
[kernel] = get_kernel(patchsize,patch,target_sz,Fisize);
kernel=  bsxfun(@minus,kernel,mean(kernel));
kernel_org=kernel;


% ================================================================================
% ��ʼ����
% ================================================================================
for frame = 1:numel(img_files),
    im = imread([video_path img_files{frame}]); % ����ͼƬ
    if ismatrix(im)
        im = cat(3, im, im, im);
    end  
    tic();
    
    % ================================================================================
    % ����ѧϰ����Ŀ��ģ��Ԥ��Ŀ��λ��
    % ================================================================================
    if frame > 1%���������֡
        % ��ȡ��ͬ��εľ������
        % ��ͼ�пٳ���������С��window_sz�����������Խ�磬������ֵȫ����Ϊ�߽�ֵ��������Ϊpatch����Ϊ��ȡ����������
        patch = get_subwindow(im, pos, window_sz);
        
        [kernel_] = get_kernel(patchsize,patch,target_sz,Fisize);
        kernel_=  bsxfun(@minus,kernel_,mean(kernel_));
        %kernel=0.9*kernel_org+0.1*kernel_;
        kernel=0.9*kernel+0.1*kernel_;
        
        % ��ȡ�㼶�������
        % feat  = get_features(patch, cos_window, indLayers);
        feat  = get_features_my(patch, cos_window, kernel,patchsize,indLayers);
        % ����Ŀ��λ��
        pos  = predictPosition(feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
            model_xf, model_alphaf);
        target_sz=predictSize(target_sz,pos,im);
        % angle = predictAngle(feat_last, pos,temp_pos);
    end
    
    % ================================================================================
    % ���ݲ㼶�������ѧϰcorrelation filters
    % ================================================================================
    % ��ȡ�㼶�������
    % feat  = get_features(patch, cos_window, indLayers);
    feat  = get_features_my(patch, cos_window, kernel,patchsize,indLayers);
    % ����ģ��
    [model_xf, model_alphaf] = updateModel(feat, yf, interp_factor, lambda, frame, ...
        model_xf, model_alphaf);
    
    % ================================================================================
    % Save predicted position and timing
    % ================================================================================
    positions(frame,:) = pos;
    time = time + toc();
    % Visualization
    if show_visualization,
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        disp(box);
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early
        drawnow
        % pause(0.05)  % uncomment to run slower
    end
end

end






