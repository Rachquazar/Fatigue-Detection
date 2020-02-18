clear; close all;
addpath(genpath('.'));

% % Load Models
fitting_model = 'models/Chehra_f1.0.mat';
load(fitting_model); 

t = 0:1:10;
earV = [];
i = 0;

vfr = VideoReader('<video_name>.avi');
vfr.CurrentTime = 18;
   
% % Test Path
%   image_path = 'EAR/';
%   img_list = dir([image_path,'*.jpg ']);

while hasFrame(vfr)
    img = readFrame(vfr);

% % Perform Fitting

    % % Load Image
    test_image = img; 
    % imshow(test_image); hold on;
    
    disp('Detecting Face...');
    faceDetector = vision.CascadeObjectDetector();
    bbox = step(faceDetector, img);            
    test_init_shape = InitShape(bbox, refShape);
    test_init_shape = reshape(test_init_shape, 49, 2);
    % plot(test_init_shape(:, 1),test_init_shape(:, 2), 'ro');
    
    if size(test_image,3) == 3
        test_input_image = im2double(rgb2gray(test_image));
    else
        test_input_image = im2double((test_image));
    end
    
    disp('Fitting...');    
    % % Maximum Number of Iterations 
    % % 3 < MaxIter < 7
    MaxIter = 6;
    test_points = Fitting(test_input_image, test_init_shape, RegMat, MaxIter);
    
    % plot(test_points(:,1), test_points(: ,2), 'g*', 'MarkerSize', 6); hold off;
    % legend('Initialization', 'Final Fitting');
    
    % set(gcf,'units','normalized','outerposition',[0 0 1 1]);
    % pause;
    % close all;

    RGB = insertMarker(img, test_points);
    imshow(RGB);

P1  = [test_points(20,1) test_points(20,2)];
P2  = [test_points(21,1) test_points(21,2)];
P3  = [test_points(22,1) test_points(22,2)];
P4  = [test_points(23,1) test_points(23,2)];
P5  = [test_points(24,1) test_points(24,2)];
P6  = [test_points(25,1) test_points(25,2)];
P7  = [test_points(26,1) test_points(26,2)];
P8  = [test_points(27,1) test_points(27,2)];
P9  = [test_points(28,1) test_points(28,2)];
P10 = [test_points(29,1) test_points(29,2)];
P11 = [test_points(30,1) test_points(30,2)];
P12 = [test_points(31,1) test_points(31,2)];

EAR1 = (norm((P2-P6),2) + norm((P3-P5),2))/ (2*(norm((P1-P4),2)));
EAR2 = (norm((P8-P12),2) + norm((P9-P11),2))/ (2*(norm((P7-P10),2)));

EAR = (EAR1+EAR2)/2;
i = i+1;
earV(i) = EAR; 

%xlFilename = 'E:\Face detectors\chehra\Chehra_v0.1_MatlabFit\EAR\earvalues.xlsx';
%xlRange = 'A1';
%xlswrite(xlFilename,earV,'earvalues',xlRange);
%A = dataset('XLSFile',earvalues,'EAR',true);

display(EAR)
end

plot(earV)