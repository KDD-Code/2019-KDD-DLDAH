function net = test_net_structure (net)
    net.layers = net.layers(1:21);
    n = numel(net.layers) ;
    for i=1:n
        if (strcmp(net.layers{i}.type, 'conv'))
            net.layers{i}.dilate = [1 1];
        end
        if isfield(net.layers{i}, 'weights')
            if ~isempty(net.layers{i}.weights)
                net.layers{i}.weights{1} = gpuArray(net.layers{i}.weights{1}) ;
                net.layers{i}.weights{2} = gpuArray(net.layers{i}.weights{2}) ;
            end
        end
    end
  
end
