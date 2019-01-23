function output = to_categorical(input, classes)
    num = length(input);
    output = zeros([num, classes]);
    for i = 1:num
        output(i,input(i))=1;
    end
end

