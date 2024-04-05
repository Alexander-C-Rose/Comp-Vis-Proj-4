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

% Create feature Vectors for each image pixel
XA = [];

for i = 1:rA
    for j = 1:sA
        tempA = [];
        for k = 1:nscale
            for l = 1:norient
                MagA = abs(A{k,l});
                tempA = [tempA; MagA(i,j)];
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
        tempB = [];
        for k = 1:nscale
            for l = 1:norient
                MagB = abs(B{k,l});
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


