% x1 = imread('262A2812.tif');
% x2 = imread('262A2811.tif');
% x3 = imread('262A2810.tif'); %people talking
% 
% x1 = imread('262A2868.tif');
% x2 = imread('262A2867.tif');
% x3 = imread('262A2866.tif'); %people standing

% x1 = imread('262A2631.tif');
% x2 = imread('262A2630.tif');
% x3 = imread('262A2629.tif'); %man standing
%  of pic
% x1 = imread('262A2945.tif');
% x2 = imread('262A2944.tif');
% x3 = imread('262A2943.tif'); %bbq

% x1 = imread('n3.tif');
% x2 = imread('n15.tif');
% x3 = imread('15.tif'); %bbq
% x4 = imread('original.tif');

x1 = imread('262A3225.tif');
x2 = imread('262A3226.tif');
x3 = imread('262A3227.tif'); %008

r1 = x1(:,:,1); g1 = x1(:,:,2); b1 = x1(:,:,3);
r2 = x2(:,:,1); g2 = x2(:,:,2); b2 = x2(:,:,3);
r3 = x3(:,:,1); g3 = x3(:,:,2); b3 = x3(:,:,3);
%changing everything to rgb arrays

% apply to all 18 colors
[h1,v1,d1,a1] = test(r1);
[h2,v2,d2,a2] = test(g1);
[h3,v3,d3,a3] = test(b1);
[h4,v4,d4,a4] = test(r2);
[h5,v5,d5,a5] = test(g2);
[h6,v6,d6,a6] = test(b2);
[h7,v7,d7,a7] = test(r3);
[h8,v8,d8,a8] = test(g3);
[h9,v9,d9,a9] = test(b3);
%a: approximation, h: horizontal, v:vertical, d:diagonal (Sub-bands)

%highest or highest abs?
[length,width] = size(r1);
rnewh = ones(length/2,width/2);
rnewv = ones(length/2,width/2);
rnewd = ones(length/2,width/2);
gnewh = ones(length/2,width/2);
gnewv = ones(length/2,width/2);
gnewd = ones(length/2,width/2);
bnewh = ones(length/2,width/2);
bnewv = ones(length/2,width/2);
bnewd = ones(length/2,width/2);
%allocating arrays for rgb

% add and average all of subbands
rnewa = (2*a1+a4+a7)/4;
gnewa = (2*a2+a5+a8)/4;
bnewa = (2*a3+a6+a9)/4;

% find highest subband coefficient for the rest and create one copy
for i= 1:length/2
    for j = 1:length/2
        rnewh(i,j) = max([h1(i,j),h4(i,j),h7(i,j)]);
        rnewv(i,j) = max([v1(i,j),v4(i,j),v7(i,j)]);
        rnewd(i,j) = max([d1(i,j),d4(i,j),d7(i,j)]);

        gnewh(i,j) = max([h2(i,j),h5(i,j),h8(i,j)]);
        gnewv(i,j) = max([v2(i,j),v5(i,j),v8(i,j)]);
        gnewd(i,j) = max([d2(i,j),d5(i,j),d8(i,j)]);

        bnewh(i,j) = max([h3(i,j),h6(i,j),h9(i,j)]);
        bnewv(i,j) = max([v3(i,j),v6(i,j),v9(i,j)]);
        bnewd(i,j) = max([d3(i,j),d6(i,j),d9(i,j)]);
    end
end

% for i= 1:length/2
%     for j = 1:length/2
%         temp1 = [h1(i,j),h4(i,j),h7(i,j)];
%         temp2 = [v1(i,j),v4(i,j),v7(i,j)];
%         temp3 = [d1(i,j),d4(i,j),d7(i,j)];
%         [~,index1] = max(abs(temp1));
%         [~,index2] = max(abs(temp2));
%         [~,index3] = max(abs(temp3));
%         rnewh(i,j) = temp1(index1);
%         rnewv(i,j) = temp2(index2);
%         rnewd(i,j) = temp3(index3);
%         %finding highest abs value
% 
%         temp4 = [h2(i,j),h5(i,j),h8(i,j)];
%         temp5 = [v2(i,j),v5(i,j),v8(i,j)];
%         temp6 = [d2(i,j),d5(i,j),d8(i,j)];
%         [~,index4] = max(abs(temp4));
%         [~,index5] = max(abs(temp5));
%         [~,index6] = max(abs(temp6));
%         gnewh(i,j) = temp4(index4);
%         gnewv(i,j) = temp5(index4);
%         gnewd(i,j) = temp6(index6);
% 
%         temp7 = [h3(i,j),h6(i,j),h9(i,j)];
%         temp8 = [v3(i,j),v6(i,j),v9(i,j)];
%         temp9 = [d3(i,j),d6(i,j),d9(i,j)];
%         [~,index7] = max(abs(temp7));
%         [~,index8] = max(abs(temp8));
%         [~,index9] = max(abs(temp9));
%         bnewh(i,j) = temp4(index7);
%         bnewv(i,j) = temp5(index8);
%         bnewd(i,j) = temp6(index9);
%     end
% end
% inverse transform
finalred = ihaart2(rnewa,rnewh,rnewv,rnewd);
finalgreen = ihaart2(gnewa,gnewh,gnewv,gnewd);
finalblue = ihaart2(bnewa,bnewh,bnewv,bnewd);
%% 
 
finalcolor(:,:,1) = finalred;
finalcolor(:,:,2) = finalgreen;
finalcolor(:,:,3) = finalblue;
finalcolor = uint16(finalcolor);
%%
imshow(finalcolor)
%%
figure(1)
subplot(2,2,1)
imshow(finalcolor)
title('output image')
subplot(2,2,2)
imshow(x1)
title('input image')
subplot(2,2,3)
imshow(x2)
title('input image')
subplot(2,2,4)
imshow(x3)
title('input image')
%%
figure(1)
subplot(1,3,1)
imshow(x3)
title('input image')
subplot(1,3,2)
imshow(x1)
title('input image')
subplot(1,3,3)
imshow(x2)
title('input image')

%%


max1 = max(finalcolor(:,:,1));
max11 = max(max1);
max2 = max(finalcolor(:,:,2));
max21 = max(max2);
max3 = max(finalcolor(:,:,3));
max31 = max(max3);
% find max of each value in matlab
maximum = [max11 max21 max31];

max4 = max(x4(:,:,1));
max41 = max(max4);
max5 = max(x4(:,:,2));
max51 = max(max5);
max6 = max(x4(:,:,3));
max61 = max(max6);
% find max of each value in matlab
maximum1 = [max41 max51 max61];
%%
maximum1 = double(maximum1);
maximum = double(maximum);
finalcolor = double(finalcolor);
x4 = double(x4);
for i=1:3
    for j=1:length
        for k = 1:width
            finalcolor(j,k,i) = finalcolor(j,k,i)/(maximum(i));
            x4(j,k,i) = x4(j,k,i)/(maximum1(i));
        end
    end
end
peakpsnr = psnr(finalcolor,x4);

%% writing result to a .tif file
% t = Tiff('myimage008.tif','w');  
% tagstruct.ImageLength = size(finalcolor,1); 
% tagstruct.ImageWidth = size(finalcolor,2);
% tagstruct.Photometric = Tiff.Photometric.RGB;
% tagstruct.BitsPerSample = 16;
% tagstruct.SamplesPerPixel = 3;
% tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
% tagstruct.Software = 'MATLAB'; 
% setTag(t,tagstruct)
% write(t,finalcolor);
% close(t);
% imshow('myimage008.tif')
%% normalize to find psnr
% hdr_image = hdrread('HDRImg.hdr');
% 
% max1 = max(finalcolor(:,:,1));
% max11 = max(max1);
% max2 = max(finalcolor(:,:,2));
% max21 = max(max2);
% max3 = max(finalcolor(:,:,3));
% max31 = max(max3);
% % find max of each value in matlab
% max11 = single(max11);
% 
% finalcolor = single(finalcolor);
% for i=1:3
%     for j=1:length
%         for k = 1:width
%             finalcolor(j,k,i) = finalcolor(j,k,i)/max11;
%         end
%     end
% end
% peakpsnr = psnr(finalcolor,hdr_image);

%%
max4 = max(rnewa);
max41 = max(max4);
max5 = max(v1);
max51 = max(max5);
max6 = max(h1);
max61 = max(max6);
max7 = max(d1);
max71 = max(max7);

% for j=1:length
%     for k = 1:width
%         rnewa(j,k) = rnewa(j,k)/max41;
% %         v1(j,k) = v1(j,k)/max51;
% %         d1(j,k) = d1(j,k)/max61;
% %         h1(j,k) = h1(j,k)/max71;
%     end
% end

rnewa = uint8(rnewa);
figure(1)
subplot(2,2,1)
imshow(rnewa)
title('Approximation sub-band')
subplot(2,2,2)
imshow(v1)
title('Vertical sub-band')
subplot(2,2,3)
imshow(h1)
title('Horizontal sub-band')
subplot(2,2,4)
imshow(d1)
title('Diagonal sub-band')