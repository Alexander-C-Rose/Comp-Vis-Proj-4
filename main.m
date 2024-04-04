% K-means clustering stuff

mosA = double(imread("mosaicA.bmp"));
mosB = double(imread("mosaicB.bmp"));

kA = kmeans(mosA, 4);    % there are 4 regions in this image
kB = kmeans(mosB, 3);    % there are 3 regions in this image

figure(1);
imshow(kA);

figure(2);
imshow(kB);

