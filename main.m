clear;

mosA = double(imread("mosaicA.bmp"));
mosB = double(imread("mosaicB.bmp"));

%% K-means clustering stuff

% Gabor 
minWaveLength = 3;
mult = 2;
sigmaOnf = 0.65;
dThetaOnSigma = 1.5;
nscale = 4;
norient = 4;

A = gaborconvolve(mosA, nscale, norient, minWaveLength, mult, ...
        sigmaOnf, dThetaOnSigma);
B = gaborconvolve(mosB, nscale, norient, minWaveLength, mult, ...
        sigmaOnf, dThetaOnSigma);

[rA, sA] = size(mosA);
[rB, sB] = size(mosB);

% Smooth images
for i = 1:norient
    for j = 1:nscale
        A{i,j} = imgaussfilt(abs(A{i,j}), 2, 'FilterSize',3, 'Padding', 'symmetric');
        B{i,j} = imgaussfilt(abs(B{i,j}), 2, 'FilterSize',3, 'Padding', 'symmetric');
    end
end

% Create feature Vectors for each image pixel
XA = [];

for i = 1:rA
    for j = 1:sA
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

XB = [];
for i = 1:rB
    for j = 1:sB
        tempB = [i;j]; % start with only the x-y coordinates
        for k = 1:nscale
            for l = 1:norient
                MagB = B{k,l};
                tempB = [tempB; MagB(i,j)];
            end
        end
        if ((i == j) && (j == 1))
                XB = tempB;
        else 
            XB = [XB, tempB];
        end
    end
end

% Save feature vectors to a file
D = sprintf('Gabor_%i_%i.mat', nscale, norient); % String representing the filename
save(D, "XA", "XB");

%% Normalize

clear;
load("Gabor_4_4.mat");
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
D = sprintf('Normalized.mat'); % String representing the filename
save(D, "xA", "xB");


%% K-means

clear;
load("Normalized.mat");

for i = 1:15 % run kmeans 15 times and record the relevant data
    [idx, CA, sumd] = kmeans(transpose(xA), 4); % 4 textures in mosaic A
    kA{i} = idx;
    sumdA(:,i) = sumd;
    
    [idx, CB, sumd] = kmeans(transpose(xB), 3); % 3 textures in mosaic B
    kB{i} = idx;
    sumdB(:,i) = sumd;

    KA{i} = transpose(reshape(kA{i}, [256,256]));
    KB{i} = transpose(reshape(kB{i}, [256,256]));
end

% Add the cluster distances together for a given k-means. This is the
% divergence calculation where column i corresponds to k{i}.
sumdA = sum(sumdA);
sumdB = sum(sumdB);

% By labeling the best and worst, I can choose them to initialize EM and
% compare.

[best, best_I] = min(sumdA);
[worst, worst_I] = max(sumdA);
[bestB, best_IB] = min(sumdB);
[worstB, worst_IB] = max(sumdB);

% Accuracy calculations
truA = double(imread("mapA.bmp"));
truB = double(imread("mapB.bmp"));
perA_best = accuracy(truA, KA{best_I});
perA_worst = accuracy(truA, KA{worst_I});
perB_best = accuracy(truB, KB{best_IB});
perB_worst = accuracy(truB, KB{worst_IB});

for i = 1:15
    perA(i) = accuracy(truA, KA{i});
    perB(i) = accuracy(truB, KB{i});
end

% save data
save("kmeans.mat", "KA", "KB", "CA", "CB" ,"perA_worst", "perA_best", "perB_worst", ...
    "perB_best", "best_I", "best_IB", "worst_I", "worst_IB", "perA", "perB");

%% display kmeans results
clear;
load("kmeans.mat");
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
        title(sprintf("per(%i) = %g", i, round(perA(i),4)));
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
        title(sprintf("per(%i) = %g", i, round(perB(i),4)));
    end
end
