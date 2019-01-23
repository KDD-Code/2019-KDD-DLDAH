function net = net_structure (net, codelens, classes)
    net.layers = net.layers(1:19);
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
    net.layers{20}.pad = [0,0,0,0];
    net.layers{20}.stride = [1,1];
    net.layers{20}.type = 'conv';
    net.layers{20}.name = 'fc8';
    net.layers{20}.opts = {};
    net.layers{20}.dilate = [1 1];
    net.layers{20}.weights{1} = gpuArray(0.01*randn(1,1,4096,codelens,'single'));
    net.layers{20}.weights{2} = gpuArray(0.01*randn(1,codelens,'single'));
    
    net.layers{21}.type = 'tanh'; % can be replaced by Adaptive tanh
    net.layers{21}.name = 'hash_layer';
    net.layers{21}.opts = {};
    net.layers{21}.weights = {};
    
    net.layers{22}.pad = [0,0,0,0];
    net.layers{22}.stride = [1,1];
    net.layers{22}.type = 'conv';
    net.layers{22}.name = 'fc9';
    net.layers{22}.opts = {};
    net.layers{22}.dilate = [1 1];
    net.layers{22}.weights{1} = gpuArray(0.01*randn(1,1,codelens,classes,'single'));
    net.layers{22}.weights{2} = gpuArray(0.01*randn(1,classes,'single'));
        
end
