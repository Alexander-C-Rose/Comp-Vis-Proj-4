clear;

im = imread("baboon.png");
baboon = double(im); % RGB version
baboon_gray = double(im2gray(im)); % grayscale version
%% Gabor stuff

% Gabor 
minWaveLength = 3;
mult = 2;
sigmaOnf = 0.65;
dThetaOnSigma = 1.5;
nscale = 4;
norient = 4;

% The grayscale version will be processed similar to the mosaic images
A = gaborconvolve(baboon_gray, nscale, norient, minWaveLength, mult, ...
        sigmaOnf, dThetaOnSigma);

[r, s] = size(baboon_gray);

% Smooth images
for i = 1:norient
    for j = 1:nscale
        A{i,j} = imgaussfilt(abs(A{i,j}), 2, 'FilterSize',3, 'Padding', 'symmetric');
    end
end

B = imgaussfilt(baboon, 2, 'FilterSize',3, 'Padding', 'symmetric'); % RGB

% Create feature Vectors for each image pixel
XA = [];
for i = 1:r
    for j = 1:s
        tempA = [i;j]; % start with only the x-y coordinates
        for k = 1:nscale
            for l = 1:norient
                MagA = A{k,l};
                tempA = [tempA; MagA(i,j)]; % create rest of feature vector
            end
        end
        if ((i == j) && (j == 1))
                XA = tempA;
        else 
            XA = [XA, tempA];
        end
    end
end


% Do a feature vector for only color and xy
XB = [];
for i = 1:r
    for j = 1:s
        loc = [i;j]; % start with only the x-y coordinates
        tempB = [loc; B(i,j,1); B(i,j,2); B(i,j,3)];
        if ((i == j) && (j == 1))
            XB = tempB;
        else 
            XB = [XB, tempB];
        end
    end
end
D = sprintf('Color_Gabor.mat'); % String representing the filename
save(D, "XA", "XB");
clear;
%% Normalize
load("Color_Gabor.mat");
% the max function returns the maximum value in a column of a matrix. Since
% I need the maximum of each row, I transpose A, then take the max,
% resulting in a row vector with maximums for normalizing. A similar
% approach is used for minimum function.

max_val = max(transpose(XA));
min_val = min(transpose(XA));
[rA, sA] = size(XA);

xA = [];
for i = 1:rA
    for j = 1:sA
        xA(i,j) = val_norm(XA(i,j), min_val(i), max_val(i));
    end
end

max_val = max(transpose(XB));
min_val = min(transpose(XB));
[rB, sB] = size(XB);

xB = [];
for i = 1:rB
    for j = 1:sB
        xB(i,j) = val_norm(XB(i,j), min_val(i), max_val(i));
    end
end

% Save normalized feature vectors to a file
D = sprintf('Color_Normalized.mat'); % String representing the filename
save(D, "xA", "xB");


%% K-means

clear;
load("Color_Normalized.mat");
cluster_num = 4; % Try using 4 regions
for i = 1:15 % run kmeans 15 times and record the relevant data
    [idx, CA{i}, sumd] = kmeans(transpose(xA), cluster_num);
    kA{i} = idx;
    sumdA(:,i) = sumd;
    
    [idx, CB{i}, sumd] = kmeans(transpose(xB), cluster_num);
    kB{i} = idx;
    sumdB(:,i) = sumd;

    KA{i} = transpose(reshape(kA{i}, [968,966]));
    KB{i} = transpose(reshape(kB{i}, [968,966]));
end

% Add the cluster distances together for a given k-means. This is the
% divergence calculation where column i corresponds to k{i}.
sumdA = sum(sumdA);
sumdB = sum(sumdB);

% save data
save("Color_kmeans.mat", "KA", "KB", "CA", "CB" ,"sumdA", "sumdB");

%% display kmeans results
clear;
load("Color_kmeans.mat");

% By labeling the best and worst, I can choose them to initialize EM and
% compare.
[best, best_I] = min(sumdA);
[worst, worst_I] = max(sumdA);
[bestB, best_IB] = min(sumdB);
[worstB, worst_IB] = max(sumdB);
% Display A results
figure(Color="White");
for i = 1:15
    subplot(3,5,i);
    imshow(mat2gray(KA{i}));
    if (i == best_I)
        title("Best");
    elseif(i == worst_I)
        title("Worst");
    else
        title(sprintf("Iteration %i", i));
    end
end

% display B results
figure(Color="White");
for i = 1:15
    subplot(3,5,i);
    imshow(mat2gray(KB{i}));
    if (i == best_IB)
        title("Best");
    elseif(i == worst_IB)
        title("Worst");
    else
        title(sprintf("Iteration %i", i));
    end
end
