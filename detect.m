% run('../vlfeat-0.9.20/toolbox/vl_setup')
imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);
load('mysvm.mat');

bboxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

cellSize = 6;
dim = 36;
bboxes_final = [];
confidences_final = [];
image_names_final = [];
r_0 = 0;
%  scales = [0.75  1 1.25 1.5 1.75 2 2.25 2.5 2.75 3 4];
scales = [0.5 1 1.5];
% scales = 1
scales = fliplr(scales)';
num_scales = numel(scales);

for i=1:nImages
    number_of_boxes = [];
    temp_boxes = [];
    temp_confs = [];
    temp_names = [];
    feat_dims = zeros(num_scales,2);
    

    % load and show the image
    im = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
%     im = (im-mean(im(:)))/std(im(:));
      % im = im2single(rgb2gray(imread('class.jpg')));
    imshow(im);
    hold on;
    
    [im_h, im_w, im_c] = size(im);
    multiscale_confs = -Inf .* ones(((ceil(im_h/cellSize))-5)*((ceil(im_w/cellSize))-5),num_scales);
    [multiscale_confs,feat_dims] = msConfs(im,scales,cellSize,w,b);

    % generate a grid of features across the entire image. you may want to 
    % try generating features more densely (i.e., not in a grid)

    
    % get the most confident predictions for this scale
    [~,inds] = sort(multiscale_confs(:,:),'descend');
    num_inds = size(multiscale_confs,1);
%     break;
    
    if num_inds > 20,
        inds = inds(1:20,:); % (use a bigger number for better recall)
        num_inds = 20;
    end
    
    number_of_boxes = [number_of_boxes; num_inds];
    %For each scale, take top 20 boxes, scale tdfgdfgdfghem appropriately, add them
    %to the bounding box list
    i_scales = [];
    for j=1:num_scales
        scale = scales(j);
        for n=1:num_inds
            [row,col] = ind2sub([feat_dims(j,1) feat_dims(j,2)],inds(n,j));
            bbox = [ (col*cellSize)./scale...
                (row*cellSize)./scale ...
                ((col+cellSize-1)*cellSize)./scale ...
                ((row+cellSize-1)*cellSize)./scale];
            confs = reshape(multiscale_confs(1:feat_dims(j,1)*feat_dims(j,2),j),feat_dims(j,:));
            
            conf = confs(row,col);
            image_name = {imageList(i).name};
            
            plot_rectangle = [bbox(1), bbox(2); ...
                bbox(1), bbox(4); ...
                bbox(3), bbox(4); ...
                bbox(3), bbox(2); ...
                bbox(1), bbox(2)];

            temp_boxes = [temp_boxes; bbox];
            temp_confs = [temp_confs; conf];
            temp_names = [temp_names; image_name];
            bboxes = [bboxes; bbox];
            confidences = [confidences; conf];
            image_names = [image_names; image_name];
            i_scales = [i_scales; scale];
        end
    end
   
    (bboxes(:,4)-bboxes(:,2)).*(bboxes(:,3)-bboxes(:,1));
    
    rb = temp_boxes;
    rc = temp_confs;
    rs = i_scales;
    
    r_0 = r_0+1;
    
    bboxes_temp = [];
    confidences_temp = [];
    image_names_temp = [];
    ks = [];
    %X = arrayHelper(rb,rc,rs);    
    [X,Y,Z] = nonMaxSupp(rb,rc,rs);
    
    
    for index=1:size(X,1),
        plot_rectangle = [X(index,1), X(index,2); ...
            X(index,1), X(index,4); ...
            X(index,3), X(index,4); ...
            X(index,3), X(index,2); ...
            X(index,1), X(index,2)];
        plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
        % pause;
    end
      pause;
      clf;



bboxes_final = [bboxes_final ; bboxes_temp];
confidences_final = [confidences_final ; confidences_temp];
image_names_final = [image_names_final ; image_names_temp];

fprintf('got preds for image %d/%d\n', i,nImages);
end


label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);


function X = arrayHelper(boxes,confidences,scales),
    if isempty(boxes),
        X = [];
        return;
    end

    %Take first box
    box_i = boxes(1,:);
    boxes(1,:) = [];
    confidences_temp = confidences(1);
    confidences(1) = [];
    scales_temp = scales(1);
    scales(1) = [];
    indices = 1;
    
    bboxes_temp = box_i;
    
    
    bi=[max(box_i(1),boxes(:,1))  max(box_i(2),boxes(:,2))  min(box_i(3),boxes(:,3))  min(box_i(4),boxes(:,4))];
    iw = bi(:,3) - bi(:,1)+1;
    ih = bi(:,4) - bi(:,2)+1;
    
    
    
    box_i_area = (box_i(3) - box_i(1))*(box_i(4)-box_i(2));
    i_area = (iw .* ih);
    
    boxes_area = (boxes(:,3)-boxes(:,1)).*(boxes(:,4)-boxes(:,2));
    u_area = box_i_area + boxes_area - i_area;
%     u_area
%     i_area
        
    b_inds = find((i_area./u_area) > 0.01);
    bboxes_temp = [bboxes_temp;boxes(b_inds,:)];
    boxes(b_inds,:) = [];
    confidences_temp = [confidences_temp;confidences(b_inds)];
    confidences(b_inds) = [];
    scales_temp = [scales_temp;scales(b_inds)];
    scales(b_inds) = [];
        
    
    largest_index = find(confidences_temp == max(confidences_temp));
%     X = [bboxes_temp(largest_index,:)./ scales_temp(largest_index); arrayHelper(boxes,confidences,scales)];
    X = [bboxes_temp(largest_index,:)  ; arrayHelper(boxes,confidences,scales)];

        
end

function [X,Y,Z] = nonMaxSupp(boxes,confidences,scales)
    if isempty(boxes),
        X = [];
        Y=[];
        Z = [];
        return;
    end
    nBoxes = size(boxes,1);
    finalBoxes = [];
    finalConfs = [];
    finalScales = [];
    tempBoxes = [];
    tempConfs = [];
    tempScales = [];
    for i=1:nBoxes,
        if isempty(boxes),
            break;
        end
        
        box_a = boxes(1,:);
        boxes(1,:) = [];
        tempBoxes = box_a;
        ba_area = (box_a(3)-box_a(1)+1)*(box_a(4)-box_a(2)+1);
        
        conf_a = confidences(1);
        confidences(1) = [];
        tempConfs = conf_a;
        scale_a = scales(1);
        scales(1) = [];
        tempScales = scale_a;
        
        %Check if first box intersects with any others
        bi = [max(box_a(1),boxes(:,1)) max(box_a(2),boxes(:,2)) min(box_a(3),boxes(:,3)) min(box_a(4),boxes(:,4))];
        iw = bi(:,3) - bi(:,1) + 1;
        ih = bi(:,4) - bi(:,2) + 1;
        i_area = iw .* ih;
        u_area = ba_area + ((boxes(:,3)-boxes(:,1)+1).*(boxes(:,4)-boxes(:,2) + 1)) - i_area;
        
        %Find the boxes that intersect, remove them from the box array, put
        %them into an array with the original box for non_max suppression.
        i_indices = find((i_area./u_area)>0.05);
        i_boxes = [box_a; boxes(i_indices,:)];
        boxes(i_indices,:) = [];
        
        i_confs = [conf_a; confidences(i_indices)];
        confidences(i_indices) = [];
        i_scales = [scale_a; scales(i_indices)];
        scales(i_indices) = [];
        
        max_ind = find(i_confs == max(i_confs(:)));
        addedBox = i_boxes(max_ind,:);
        addedConf = i_confs(max_ind);
        addedScale = i_scales(max_ind);
        
        finalBoxes = [finalBoxes; addedBox];
        finalConfs = [finalConfs; addedConf];
        finalScales = [finalScales; addedScale];

    
    end
    X = finalBoxes;
    Y = finalConfs;
    Z = finalScales;
end


    

function [MSC,D] = msConfs(im,scales,cellSize,w,b),
    nScales = numel(scales);
%     im = (im - mean(im(:)))/std(im(:));
    [im_h im_w im_c] = size(im);
    D = zeros(nScales,2);
    
    MSC = -Inf .* ones(ceil(im_h/cellSize)*ceil(im_w/cellSize),nScales);
    
    for j=1:nScales,
        scale = scales(j);
        im2 = imresize(im,[im_h * scale im_w * scale]);
        im2 = (im2 - mean(im2(:)))/std(im2(:));
        feats = vl_hog(im2,cellSize);

        [rows,cols,~] = size(feats);
        confs = zeros(rows,cols);
        D(j,:) = [rows cols];
        for r=1:rows-5
            for c=1:cols-5
                % create feature vector for the current window and classify it using the SVM model,
                im_wind = feats(r:r+5,c:c+5,:);
                im_wind = im_wind(:);
                score = w'*im_wind + b;
                confs(r,c) = score;
            end
        end
        MSC(1:size(confs,1)*size(confs,2),j)=confs(:);
    end
end