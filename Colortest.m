% Color image testing and so forth. 
load("Color_kmeans.mat");
load("Color_Normalized.mat");
cluster_num = 4;
% Choosing to run EM on the Gabor data with the best divergence (number 1)
[to_display_A, log_out_A] = EM_function(xA, KA{1}, CA{1}, cluster_num, 1, 100);

vidObj = VideoWriter('Baboon Gabor Best.avi'); % Prepare the new file.
vidObj.FrameRate = 5;
open(vidObj); % Create an animation.
[~, iterations] = size(log_out_A);
% figure for video writing
toVid = figure(Color="White", Position=get(0,'ScreenSize'));
for i = 1:iterations
    %plot the output of EM
    subplot(1, 2, 1);
    imshow(mat2gray(to_display_A(:,:,i)));
    title("Output from EM");
    % Plot log-likelihood
    subplot(1,2,2);
    plot(1:i, log_out_A(1:i));
    title("Log-likelihood vs iteration");
    % Plot accuracy
    drawnow;
    pause(.1);

    % Draw the videoframe
    currFrame = getframe(toVid);
    writeVideo(vidObj, currFrame);
end
close(vidObj);

%% Next video

% Choosing to run EM on the Gabor data with the worst results (number 10)
[to_display_A, log_out_A] = EM_function(xA, KA{10}, CA{10}, cluster_num, 1, 100);

vidObj = VideoWriter('Baboon Gabor worst.avi'); % Prepare the new file.
vidObj.FrameRate = 5;
open(vidObj); % Create an animation.
[~, iterations] = size(log_out_A);
% figure for video writing
toVid = figure(Color="White", Position=get(0,'ScreenSize'));
for i = 1:iterations
    %plot the output of EM
    subplot(1, 2, 1);
    imshow(mat2gray(to_display_A(:,:,i)));
    title("Output from EM");
    % Plot log-likelihood
    subplot(1,2,2);
    plot(1:i, log_out_A(1:i));
    title("Log-likelihood vs iteration");
    % Plot accuracy
    drawnow;
    pause(.1);

    % Draw the videoframe
    currFrame = getframe(toVid);
    writeVideo(vidObj, currFrame);
end
close(vidObj);

%% Next video

% Choosing to run EM on the RGB data with the best divergence
[to_display_A, log_out_A] = EM_function(xB, KB{11}, CB{11}, cluster_num, 1, 100);

vidObj = VideoWriter('Baboon XYRGB Best.avi'); % Prepare the new file.
vidObj.FrameRate = 5;
open(vidObj); % Create an animation.
[~, iterations] = size(log_out_A);
% figure for video writing
toVid = figure(Color="White", Position=get(0,'ScreenSize'));
for i = 1:iterations
    %plot the output of EM
    subplot(1, 2, 1);
    imshow(mat2gray(to_display_A(:,:,i)));
    title("Output from EM");
    % Plot log-likelihood
    subplot(1,2,2);
    plot(1:i, log_out_A(1:i));
    title("Log-likelihood vs iteration");
    % Plot accuracy
    drawnow;
    pause(.1);

    % Draw the videoframe
    currFrame = getframe(toVid);
    writeVideo(vidObj, currFrame);
end
close(vidObj);

%% Next Video
clear;
load("Color_kmeans.mat");
load("Color_Normalized.mat");
cluster_num = 4;
% Choosing to run EM on the RGB data with worst results
[to_display_A, log_out_A] = EM_function(xB, KB{2}, CB{2}, cluster_num, 1, 100);

vidObj = VideoWriter('Baboon XYRGB worst.avi'); % Prepare the new file.
vidObj.FrameRate = 5;
open(vidObj); % Create an animation.
[~, iterations] = size(log_out_A);
% figure for video writing
toVid = figure(Color="White", Position=get(0,'ScreenSize'));
for i = 1:iterations
    %plot the output of EM
    subplot(1, 2, 1);
    imshow(mat2gray(to_display_A(:,:,i)));
    title("Output from EM");
    % Plot log-likelihood
    subplot(1,2,2);
    plot(1:i, log_out_A(1:i));
    title("Log-likelihood vs iteration");
    % Plot accuracy
    drawnow;
    pause(.1);

    % Draw the videoframe
    currFrame = getframe(toVid);
    writeVideo(vidObj, currFrame);
end
close(vidObj);

%% Final Video
% In this section, I try adding RGB components to the Gabor feature vectors
% and see what happens. For this, I reshape some of the vectors and re-run
% kmeans. 

clear;
load("Color_Normalized.mat");

% Since the XY data is already present in the Gabor, remove it from the RGB
% That is, remove rows 1 and 2 from the data
xB = xB(3:5,:);

% Concatenate to the bottom of the gabor data
xA = [xA; xB];
clear xB;

cluster_num = 4; % Try using 4 regions
for i = 1:15 % run kmeans 15 times and record the relevant data
    [idx, CA, sumd] = kmeans(transpose(xA), cluster_num);
    kA{i} = idx;
    sumdA(:,i) = sumd;

    KA{i} = transpose(reshape(kA{i}, [968,966]));
end

% Add the cluster distances together for a given k-means. This is the
% divergence calculation where column i corresponds to k{i}.
sumdA = sum(sumdA);

% Display results of k-means
% By labeling the best and worst, I can choose them to initialize EM and
% compare.
[~, best_I] = min(sumdA);
[~, worst_I] = max(sumdA);

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


[to_display_A, log_out_A] = EM_function(xA, KA{best_I}, CA, cluster_num, 1, 100);

vidObj = VideoWriter('Baboon GaborRGB combined.avi'); % Prepare the new file.
vidObj.FrameRate = 5;
open(vidObj); % Create an animation.
[~, iterations] = size(log_out_A);
% figure for video writing
toVid = figure(Color="White", Position=get(0,'ScreenSize'));
for i = 1:iterations
    %plot the output of EM
    subplot(1, 2, 1);
    imshow(mat2gray(to_display_A(:,:,i)));
    title("Output from EM");
    % Plot log-likelihood
    subplot(1,2,2);
    plot(1:i, log_out_A(1:i));
    title("Log-likelihood vs iteration");
    % Plot accuracy
    drawnow;
    pause(.1);

    % Draw the videoframe
    currFrame = getframe(toVid);
    writeVideo(vidObj, currFrame);
end
close(vidObj);