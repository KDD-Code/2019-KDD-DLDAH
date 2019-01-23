function gpu_net = update (gpu_net, res_back, lr, N)
    weight_decay = 5*1e-4 ;
    n_layers = 22 ;
    batch_size = 128 ;
    for ii = 1:n_layers
            if isfield(gpu_net.layers{ii},'weights')
                if (ii ~= 22)
                    if strcmp(gpu_net.layers{ii}.type, 'conv')
                        gpu_net.layers{ii}.weights{1} = gpu_net.layers{ii}.weights{1}+...
                            lr*(res_back(ii).dzdw{1}/(batch_size*N) - weight_decay*gpu_net.layers{ii}.weights{1});
                        gpu_net.layers{ii}.weights{2} = gpu_net.layers{ii}.weights{2}+...
                            lr*(res_back(ii).dzdw{2}/(batch_size*N) - weight_decay*gpu_net.layers{ii}.weights{2});
                    end
                else
                    if strcmp(gpu_net.layers{ii}.type, 'conv')
                        gpu_net.layers{ii}.weights{1} = gpu_net.layers{ii}.weights{1}+...
                            lr*(res_back(ii).dzdw{1}/(batch_size*N) - weight_decay*gpu_net.layers{ii}.weights{1});
                        gpu_net.layers{ii}.weights{2} = gpu_net.layers{ii}.weights{2}+...
                            lr*(res_back(ii).dzdw{2}/(batch_size*N) - weight_decay*gpu_net.layers{ii}.weights{2});
                
                    end
                end
            end
    end
end
