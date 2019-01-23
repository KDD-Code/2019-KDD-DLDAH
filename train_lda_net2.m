function [net, U] = train_lda_net2 (X1, L1, U, net, iter , lr, eta, margin, fid)
    N = size(X1,4);
    batchsize = 128;

    index = randperm(N);
    for j = 0:ceil(N/batchsize)-1
        batch_time=tic;

        %% random select a minibatch and sample triplets
        ix = index((1+j*batchsize):min((j+1)*batchsize,N));

        %% load and preprocess an image
        im = X1(:,:,:,ix);
        im_ = single(im); % note: 0-255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
        im_ = im_ - repmat(net.meta.normalization.averageImage,1,1,1,size(im_,4));
        im_ = gpuArray(im_);

        %% load the label
        batch_label = L1(ix);
        batch_label = to_categorical(batch_label+1, 10);
        
        %% run the CNN
        res = vl_simplenn(net, im_);
        U0 = squeeze(gather(res(end).x))';
        U(ix,:) = U0; 

        %% compute the loss and gradient
        delta = U0-batch_label;
        loss = sum(delta(:).^2) / numel(ix);
        
        dJdU =  2 * delta;
        dJdU = single(dJdU);
        
        dJdoutput = gpuArray(reshape(dJdU',[1,1,size(dJdU',1),size(dJdU',2)]));
        

        res = vl_simplenn( net, im_, dJdoutput);

        %% update the parameters of CNN
        net = update(net , res, lr, N);
        batch_time = toc(batch_time);

        fprintf(' iter %d  batch %d/%d (%.1f images/s) ,lr is %d, mse loss: %f \n', iter, j+1,ceil(size(X1,4)/batchsize), batchsize/ batch_time,lr, loss);
        fprintf(fid, ' iter %d  batch %d/%d (%.1f images/s) ,lr is %d, mse loss: %f \n', iter, j+1,ceil(size(X1,4)/batchsize), batchsize/ batch_time,lr, loss);
    end
end
