% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
aug_imageDir_neg = 'augmented_notfaces';
mkdir(new_imageDir);
mkdir(aug_imageDir_neg);

new_valDir = 'cropped_validation_images_notfaces';

mkdir(new_valDir)

dim = 36;

num = 1;
x = 0;
while n_have < n_want

    %Read in an image, split into 3 channels
    im = imread([imageList(num).folder '/' imageList(num).name]);
    num = num+1;
    im_R = im(:,:,1);
    im_G = im(:,:,2);
    im_B = im(:,:,3);

    %Get dimensions of image as well as windows per image
    [im_h, im_w, im_c] = size(im);
    num_y = floor(im_h / 36);
    num_x = floor(im_w / 36);
    num_patches = min(num_x,num_y);


    for i=1:num_patches,
        %Take 36x36 patches of each channel
        for j=1:num_patches,
            im_patch = im((i-1)*36+1:i*36,(j-1)*36+1:j*36,:);

            im_patch = rgb2gray(im_patch);
            imwrite(im_patch,[aug_imageDir_neg '/original/' int2str(n_have) '_' int2str(i+j) '.jpg']);

             leftRightFlip(im_patch,n_have,i);
%             upDownFlip(im_patch,n_have,i);
             addNoise(im_patch,n_have,i);
             rotIm(im_patch,n_have,i);

            n_have = n_have + 0.5;
            x = x+3;

            if n_have >= n_want,
                break;
            end 
        end
        if n_have >= n_want,
            break;
        end
    end


    % n_have = n_have + 3*num_patches;
    if n_have >= n_want,
        break;
    end
end

function X = leftRightFlip(image_patch,n_have,num)
    aug_dir = './augmented_notfaces/lrflip';

    im = fliplr(image_patch);
    
    imwrite(im,[aug_dir '/' int2str(n_have) '_' int2str(num) '.jpg']);


%     imwrite(lr(:,:,1),[aug_dir '/' int2str(n_have) '_' int2str(num) 'r.jpg']);
%     imwrite(lr(:,:,2),[aug_dir '/' int2str(n_have) '_' int2str(num) 'g.jpg']);
%     imwrite(lr(:,:,3),[aug_dir '/' int2str(n_have) '_' int2str(num) 'b.jpg']);
%     return;
end

function X = upDownFlip(image_patch,n_have,num)
    aug_dir = './augmented_notfaces/udflip';

    im = flipud(image_patch);
    imwrite(im,[aug_dir '/' int2str(n_have) '_' int2str(num) '.jpg']);

end

function X = addNoise(image_patch,n_have,num)
    aug_dir = './augmented_notfaces/noise';

    [h, w, c] = size(image_patch);

    x = randi([1 20],[h w c]);

%     im =  rgb2gray(uint8(image_patch + uint8(x)));
    imwrite(image_patch,[aug_dir '/' int2str(n_have) '_' int2str(num) '.jpg']);


%     imwrite(noisy(:,:,1),[aug_dir '/' int2str(n_have) '_' int2str(num) 'r.jpg']);
%     imwrite(noisy(:,:,2),[aug_dir '/' int2str(n_have) '_' int2str(num) 'g.jpg']);
%     imwrite(noisy(:,:,3),[aug_dir '/' int2str(n_have) '_' int2str(num) 'b.jpg']);
    return;
end

function X = rotIm(image_patch,n_have,num)
    aug_dir = './augmented_notfaces/rot';
    
    [h,w] = size(image_patch);
    
    x = randi([-15 15]);
    
%     im = rgb2gray(imrotate(image_patch,x));
    im = imresize(image_patch, [36 36]);
%     imwrite(uint8(im(:,:,1)),[aug_dir '/' int2str(n_have) '_' int2str(num) 'r.jpg']);
%     imwrite(uint8(im(:,:,2)),[aug_dir '/' int2str(n_have) '_' int2str(num) 'g.jpg']);
%     imwrite(uint8(im(:,:,3)),[aug_dir '/' int2str(n_have) '_' int2str(num) 'b.jpg']);
    imwrite(im,[aug_dir '/' int2str(n_have) '_' int2str(num) '.jpg']);
end






