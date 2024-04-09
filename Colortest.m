% Color image testing and so forth. 
load("Color_kmeans.mat");
load("Color_Normalized.mat");
cluster_num = 4;
% Choosing to run EM on the Gabor data with the best divergence
[to_display_A, log_out_A] = EM_function(xA, KA{14}, CA, cluster_num, 1, 100);

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

% Choosing to run EM on the Gabor data I think looks best
[to_display_A, log_out_A] = EM_function(xA, KA{12}, CA, cluster_num, 1, 100);

vidObj = VideoWriter('Baboon Gabor personal.avi'); % Prepare the new file.
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
[to_display_A, log_out_A] = EM_function(xB, KB{2}, CB, cluster_num, 1, 100);

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

% Choosing to run EM on the RGB data I think looks best
[to_display_A, log_out_A] = EM_function(xB, KB{8}, CB, cluster_num, 1, 100);

vidObj = VideoWriter('Baboon XYRGB personal.avi'); % Prepare the new file.
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