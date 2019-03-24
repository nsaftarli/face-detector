close all
clear
% run('../vlfeat-0.9.20/toolbox/vl_setup')

pos_imageDir = 'cropped_training_images_faces';
pos_imageList = dir(sprintf('%s/*.jpg',pos_imageDir));
pos_nImages = length(pos_imageList);

neg_imageDir = 'cropped_training_images_notfaces';
neg_imageList = dir(sprintf('%s/*.jpg',neg_imageDir));
neg_nImages = length(neg_imageList);

cellSize = 6;
featSize = 31*cellSize^2;

augFeatures(featSize,cellSize);
pos_feats = zeros(pos_nImages,featSize);
% for i=1:pos_nImages
%     im = im2single(imread(sprintf('%s/%s',pos_imageDir,pos_imageList(i).name)));
%     feat = vl_hog(im,cellSize);
%     pos_feats(i,:) = feat(:);
%     fprintf('got feat for pos image %d/%d\n',i,pos_nImages);
%     % pause;
% %     imhog = vl_hog('render', feat);
% %     subplot(1,2,1);
% %     imshow(im);
% %     subplot(1,2,2);
% %     imshow(imhog)
% %     pause;
% end

% neg_feats = zeros(neg_nImages,featSize);
% for i=1:neg_nImages
%     im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageList(i).name)));
%     feat = vl_hog(im,cellSize);
%     neg_feats(i,:) = feat(:);
%     fprintf('got feat for neg image %d/%d\n',i,neg_nImages);
% %     imhog = vl_hog('render', feat);
% %     subplot(1,2,1);
% %     imshow(im);
% %     subplot(1,2,2);
% %     imshow(imhog)
% %     pause;
% end

% save('pos_neg_feats.mat','pos_feats','neg_feats','pos_nImages','neg_nImages')


function X = augFeatures(featSize,cellSize)
    pos_dir = 'augmented_faces/';
    neg_dir = 'augmented_notfaces/';
    orig = 'original';
    lr = 'lrflip';
    ud = 'udflip';
    noise = 'noise';
    rot = 'rot';
    pos_orig_imageDir = [pos_dir orig];
    pos_orig_imageList =dir(sprintf('%s/*.jpg',pos_orig_imageDir));
    pos_lr_imageDir = [pos_dir lr];
    pos_lr_imageList =dir(sprintf('%s/*.jpg',pos_lr_imageDir));
%      pos_ud_imageDir = [pos_dir ud];
%      pos_ud_imageList =dir(sprintf('%s/*.jpg',pos_ud_imageDir));
     pos_noise_imageDir = [pos_dir noise];
     pos_noise_imageList =dir(sprintf('%s/*.jpg',pos_noise_imageDir));
     pos_rot_imageDir = [pos_dir rot];
     pos_rot_imageList = dir(sprintf('%s/*.jpg',pos_rot_imageDir));
     
     

    pos_imageList = [pos_orig_imageList; pos_lr_imageList; pos_noise_imageList; pos_rot_imageList];


    neg_orig_imageDir = [neg_dir orig];
    neg_orig_imageList =dir(sprintf('%s/*.jpg',neg_orig_imageDir));
    neg_lr_imageDir = [neg_dir lr];
    neg_lr_imageList =dir(sprintf('%s/*.jpg',neg_lr_imageDir));

     neg_noise_imageDir = [neg_dir noise];
     neg_noise_imageList =dir(sprintf('%s/*.jpg',neg_noise_imageDir));
     neg_rot_imageDir = [neg_dir rot];
     neg_rot_imageList = dir(sprintf('%s/*.jpg',neg_rot_imageDir));

    neg_imageList = [neg_orig_imageList; neg_lr_imageList; neg_noise_imageList; neg_rot_imageList];

    pos_nImages = length(pos_imageList);
    neg_nImages = length(neg_imageList);


    cellSize=6;
    featSize = 31 * 6.^2;
    pos_feats = zeros(pos_nImages,featSize);
    neg_feats = zeros(neg_nImages,featSize);

    
        
    for i=1:pos_nImages,
        im = im2double(imread(sprintf('%s/%s',pos_imageList(i).folder, pos_imageList(i).name)));
%         pos_imageList(i).name
        im = (im - mean(im(:)))/std(im(:));
        im = im2single(im);
        feat = vl_hog(im,cellSize);
        pos_feats(i,:) = feat(:);
%              fprintf('got feat for pos image %d/%d\n',i,pos_nImages);
    end

    for i=1:neg_nImages,
        im = im2double(imread(sprintf('%s/%s',neg_imageList(i).folder, neg_imageList(i).name)));
        im = (im - mean(im(:)))/std(im(:));
        im = im2single(im);

        feat = vl_hog(im,cellSize);
        neg_feats(i,:) = feat(:);
        % fprintf('got feat for neg image %d/%d\n',i,neg_nImages);
    end
    filename = ['aug_pos_neg_feats.mat'];
    save(filename,'pos_feats','neg_feats','pos_nImages','neg_nImages','-v7.3','-nocompression');
    
end













