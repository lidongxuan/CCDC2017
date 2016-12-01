function [Fio] = get_kernel(patchsize,subimage,target_sz, Fisize)

if size(subimage,3)==3
    subimage	= double(rgb2gray(subimage));
else
    subimage	= double(subimage);
end 

size_subimage=size(subimage);
%采样小块的大小是6*6，在原来的32*32目标区域内以步长为1进行采样，所以可以采样数量的矩阵为[32-6+1,32-6+1]
patchnum(1) = length(patchsize(1)/2 : 1: (target_sz(1)-patchsize(1)/2));
patchnum(2) = length(patchsize(2)/2 : 1: (target_sz(2)-patchsize(2)/2));

%prod计算数组元素的连乘积。如prod（[1:5]）返回120.
%这里是先初始化生成6*6行，27*27列的0矩阵patch
%相当于每个小patch是6*6的矩阵，转成列向量，一共有27*27个这样的列向量
patch = zeros(prod(patchsize), prod(patchnum));

y = patchsize(1)/2;
x = patchsize(2)/2;

%patch_centx，patch_centy为在目标区域内均匀采样的27*27个小块的中心坐标位置
patch_centy = y : 1: (target_sz(1)-y);
patch_centy = patch_centy + floor((size_subimage(1)- target_sz(1))/2);
patch_centx = x : 1: (target_sz(2)-x);
patch_centx = patch_centx + floor((size_subimage(2)- target_sz(2))/2);

%将27*27个采样小块（大小6*6）的像素值放在一个矩阵中，即patch。
%每个小patch是6*6的矩阵，转成列向量，一共有27*27个这样的列向量
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

%聚类最大迭代次数为10
cluster_options.maxiters = 10;
%设置不显示详细信息
cluster_options.verbose  = 0;
%进行k均值聚类，滤波器个数（类别数目）为Fisize，返回的Fi为聚类结果
Fio = vgg_kmeans(double(patch), Fisize, cluster_options);

end