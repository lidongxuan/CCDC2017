function [Fio] = get_kernel(patchsize,subimage,target_sz, Fisize)

if size(subimage,3)==3
    subimage	= double(rgb2gray(subimage));
else
    subimage	= double(subimage);
end 

size_subimage=size(subimage);
%����С��Ĵ�С��6*6����ԭ����32*32Ŀ���������Բ���Ϊ1���в��������Կ��Բ��������ľ���Ϊ[32-6+1,32-6+1]
patchnum(1) = length(patchsize(1)/2 : 1: (target_sz(1)-patchsize(1)/2));
patchnum(2) = length(patchsize(2)/2 : 1: (target_sz(2)-patchsize(2)/2));

%prod��������Ԫ�ص����˻�����prod��[1:5]������120.
%�������ȳ�ʼ������6*6�У�27*27�е�0����patch
%�൱��ÿ��Сpatch��6*6�ľ���ת����������һ����27*27��������������
patch = zeros(prod(patchsize), prod(patchnum));

y = patchsize(1)/2;
x = patchsize(2)/2;

%patch_centx��patch_centyΪ��Ŀ�������ھ��Ȳ�����27*27��С�����������λ��
patch_centy = y : 1: (target_sz(1)-y);
patch_centy = patch_centy + floor((size_subimage(1)- target_sz(1))/2);
patch_centx = x : 1: (target_sz(2)-x);
patch_centx = patch_centx + floor((size_subimage(2)- target_sz(2))/2);

%��27*27������С�飨��С6*6��������ֵ����һ�������У���patch��
%ÿ��Сpatch��6*6�ľ���ת����������һ����27*27��������������
l =1;
for j = 1: patchnum(1)                   % sliding window
    for k = 1:patchnum(2)
        data = subimage(patch_centy(j)-y+1 : patch_centy(j)+y, patch_centx(k)-x+1 : patch_centx(k)+x);
        %imshow(data);
        patch(:, l) = reshape(data,numel(data),1);
        l = l+1;
    end
end
    

% patch = bsxfun(@minus,patch,mean(patch));

%��������������Ϊ10
cluster_options.maxiters = 10;
%���ò���ʾ��ϸ��Ϣ
cluster_options.verbose  = 0;
%����k��ֵ���࣬�˲��������������Ŀ��ΪFisize�����ص�FiΪ������
Fio = vgg_kmeans(double(patch), Fisize, cluster_options);

end