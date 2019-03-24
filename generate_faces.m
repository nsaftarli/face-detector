% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'cropped_training_images_faces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

aug_imageDir_pos = 'augmented_faces';
mkdir(aug_imageDir_pos); 

dim = 36;
num = 1;
x = 0;
while n_have < n_want

    %Read in an image, split into 3 channels
    im = imread([imageList(num).folder '/' imageList(num).name]);
    num = num+1;


    %Get dimensions of image as well as windows per image
    [im_h, im_w] = size(im);

     
    imwrite(im,[aug_imageDir_pos '/original/' int2str(n_have) '_' int2str(i) '.jpg']);


        leftRightFlip(im,n_have,i);
        upDownFlip(im,n_have,i);
        addNoise(im,n_have,i);
        rotIm(im,n_have,i);

        n_have = n_have + 1;
        x = x+1;

        if n_have >= n_want,
            break;
        end 
end





function X = leftRightFlip(image_patch,n_have,num)
    aug_dir = './augmented_faces/lrflip';

    lr = fliplr(image_patch);

    imwrite(lr,[aug_dir '/' int2str(n_have) '_' int2str(num) '.jpg']);
    return;
end

function X = upDownFlip(image_patch,n_have,num)
    aug_dir = './augmented_faces/udflip';

    ud = flipud(image_patch);

    imwrite(ud,[aug_dir '/' int2str(n_have) '_' int2str(num) '.jpg']);
    return;
end

function X = addNoise(image_patch,n_have,num)
    aug_dir = './augmented_faces/noise';

    [h, w] = size(image_patch);




    x = randi([1 20],[h w]);

    noisy =  image_patch + uint8(x);

    imwrite(noisy,[aug_dir '/' int2str(n_have) '_' int2str(num) '.jpg']);

    return;
end

function X = rotIm(image_patch,n_have,num)
    aug_dir = './augmented_faces/rot';
    
    [h,w] = size(image_patch);
    
%    pause;
    
    x = randi([-15 15]);
    
    im = imrotate(image_patch,x);
    im = imresize(im, [36 36]);
    imwrite(im,[aug_dir '/' int2str(n_have) '_' int2str(num) '.jpg']);
end






