function feat = get_features_my(im, cos_window, kernel,patchsize,indLayers)

size_scale=[0.5,0.7,0.9,1,1.111,1.428,2]

if size(im,3)==3
    im	= double(rgb2gray(im));
else
    im	= double(im);
end 

feat = cell(length(indLayers), 1);
sz_window = size(cos_window);
sz_im=size(im);
sz_kernel=size(kernel);
l=1;
patch = zeros(prod(patchsize), (sz_im(1)-patchsize(1)+1)*(sz_im(2)-patchsize(2)+1));
    
for i=1:sz_im(1)-patchsize(1)+1
    for j=1:sz_im(2)-patchsize(2)+1
        data=im(i:(i+patchsize(1)-1),j:(j+patchsize(2)-1));
        patch(:, l) = reshape(data,numel(data),1);
        l=l+1;
    end
end
patch = bsxfun(@minus,patch,mean(patch));
pooling_size=[2 2];
xx=kernel'*patch;  %¾í»ý²Ù×÷

conv_1= zeros(sz_im(1)-patchsize(1)+1+rem(sz_im(1)-patchsize(1)+1,pooling_size(1)),...
                sz_im(2)-patchsize(2)+1+rem(sz_im(2)-patchsize(2)+1,pooling_size(2)),...
                sz_kernel(2));
sz_conv_1=size(conv_1);

for i=1:sz_conv_1(3)
    conv_1(1:sz_im(1)-patchsize(1)+1,...
        1:sz_im(2)-patchsize(2)+1,...
        i)=reshape(xx(i,:),sz_im(2)-patchsize(2)+1,sz_im(1)-patchsize(1)+1)';
end

conv_1 = imResample(conv_1, sz_window(1:2));
if ~isempty(cos_window),
    conv_1 = bsxfun(@times, conv_1, cos_window);
end
feat{1}=conv_1;

% pooling_1=zeros(sz_conv_1(1)/pooling_size(1),...
%                 sz_conv_1(2)/pooling_size(2),...
%                 sz_kernel(2));
% sz_pooling_1=size(pooling_1);

% for k=1:sz_pooling_1(3)
%     for i= 1:pooling_size(1):sz_pooling_1(1)
%         for j= 1:pooling_size(2):sz_pooling_1(2)
%             pooling_1(i,j,k)=max(max(abs(conv_1(i:i+pooling_size(1),j:j+pooling_size(2),k))));
%         end
%     end
% end
% pooling_1 = imResample(pooling_1, sz_window(1:2));
% if ~isempty(cos_window),
%     pooling_1 = bsxfun(@times, pooling_1, cos_window);
% end
% feat{2}=pooling_1;

end
