% getStimulusMovies.m
%
% script to create a stimulus movie, either white noise or series of natural images
%
% "argments":  variables expected by this script:
%   option.stimulus
%   nFrames - e.g. 375
%   imgSiz - e.g. 16
%   nPixels - e.g. 16*16
%   iMovie <= 20 or 25

% "returns":  new stuff created by this script:
%   stimMovie = nPixels x nFrames array (double)

if strcmp(option.stimulus,'white')  % build it here
    stimMovie  = 2*(rand(nPixels,nFrames) - 0.5);  % white noise, range -1 to +1
elseif strcmp(option.stimulus,'McGill_clips')   % these are 32^2 x 480
    % read file
    files.thisFileName = ['McGill_clips_0' int2str(iMovie) '.mat'];
    fprintf(1,'loading stimulus file: %s\n',files.thisFileName);
    load(files.thisFileName);  % -> thisNewMovie (int8, 32x32x480)
 
    mSiz = size(thisNewMovie,1);    % pixel size of movie
    if (imgSiz>mSiz)
        error('requested imgSiz is too large for this movie-set');
    elseif (imgSiz<mSiz)  % movie has too many pixels:  need to crop
        begX = round((mSiz-imgSiz)/2);
        endX = begX + imgSiz - 1;
        xMovie = thisNewMovie(begX:endX,begX:endX,:);
    else   % exactly right number of pixels
        xMovie = thisNewMovie;
    end
    
    % create stimMovie, range -1 to +1, imgSiz^2 x 480
    stimMovie = double(reshape(xMovie,nPixels,nFrames));
    stimMovie = stimMovie/128;  % range -1 to +1 
else
    error('invalid option.stimulus');
end
    





