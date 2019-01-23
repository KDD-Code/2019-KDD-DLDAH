function  [] = DLDAH(codelens, classes)

	lr = 5 * 1e-4;
	stepsize = 20;
	maxIter = 100;
	lrDecay = 2 / 3;
	eta = 100;

    savefolder = [num2str(codelens),'_'];
    if ~exist(savefolder, 'dir')
        mkdir(savefolder);
    end
    matfolder = [savefolder, '/mat/'];
    logfolder = [savefolder,'/log/' ];
    if ~exist(matfolder, 'dir')
        mkdir(matfolder);
    end
    if ~exist(logfolder, 'dir')
        mkdir(logfolder);
    end
    matfilename = [matfolder, sprintf('lr%f_sp%d_mi%d_decay%f_eta%d.mat', lr, stepsize, maxIter, lrDecay, eta)];
    logfilename = [logfolder, sprintf('lr%f_sp%d_mi%d_decay%f_eta%d.log', lr, stepsize, maxIter, lrDecay, eta)];


    %% prepare the dataset
    data_prepare;
    %% load the pre-trained CNN
    net = load('imagenet-vgg-fq.mat') ;
    %% load the Dataset
    %load('cifar_data_img.mat') ;
    load('cifar-10.mat');
    
    %% initialization
    net = net_structure(net, codelens, classes);
    U = zeros(size(train_data,4),classes);     %% changed, the final output

    %% training
    fid = fopen(logfilename, 'w');
    for iter = 1: maxIter
        [net,U] = train_lda_net2(train_data, train_L, U, net, iter, lr, eta, codelens / 2, fid) ;

        %% learning rate changes
        if mod(iter,stepsize)==0
            lr = lr*lrDecay;
        end
    end

    %% test
    [map,B_dataset,B_test] = test(net, dataset_L, test_L,data_set, test_data);
    fprintf('codelens = %d, map = %f\n', codelens, map);
    fprintf(fid, 'codelens = %d, map = %f\n', codelens, map);
    fclose(fid);

    save(matfilename);
end