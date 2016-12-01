function [model_xf, model_alphaf] = updateModel(feat, yf, interp_factor, lambda, frame, ...
        model_xf, model_alphaf)

    numLayers = length(feat);

    % ================================================================================
    % Initialization
    % ================================================================================
    xf       = cell(1, numLayers);
    alphaf   = cell(1, numLayers);

    % ================================================================================
    % Model update
    % ================================================================================
    for ii=1 : numLayers%Eq(3)
        xf{ii} = fft2(feat{ii});
        kf = sum(xf{ii} .* conj(xf{ii}), 3) / numel(xf{ii});% conj is used for calculating gonge
        alphaf{ii} = yf./ (kf+ lambda);   % Fast training
    end

    % Model initialization or update
    if frame == 1,  % First frame, train with a single image
        for ii=1:numLayers
            model_alphaf{ii} = alphaf{ii};
            model_xf{ii} = xf{ii};
        end
    else
        % Online model update using learning rate interp_factor
        for ii=1:numLayers
            model_alphaf{ii} = (1 - interp_factor) * model_alphaf{ii} + interp_factor * alphaf{ii};
            model_xf{ii}     = (1 - interp_factor) * model_xf{ii}     + interp_factor * xf{ii};
        end
    end
end