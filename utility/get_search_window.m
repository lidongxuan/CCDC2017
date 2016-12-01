function window_sz = get_search_window( target_sz, im_sz, padding)
% GET_SEARCH_WINDOW
%针对目标相对整个背景的大小选择合理的搜索域尺寸。padding结构体里提供了三个针对不同情况的比例系数

if(target_sz(1)/target_sz(2) > 2)
    % For objects with large height, we restrict the search window with padding.height
    window_sz = floor(target_sz.*[1+padding.height, 1+padding.generic]);
    
elseif(prod(target_sz)/prod(im_sz(1:2)) > 0.05)
    % For objects with large height and width and accounting for at least 10 percent of the whole image,
    % we only search 2x height and width
    window_sz=floor(target_sz*(1+padding.large));
    
else
    %otherwise, we use the padding configuration
    window_sz = floor(target_sz * (1 + padding.generic));
    
end


end

