clear all;  fprintf(1,'\n\n\n\n\n\n');   close all;  

rng('default');  % "standard" random number seed -> reproducible simulations 

addpath('netlab3_3');

%  choose stimulus   %%%%%%%%%%%%%%%
commandwindow
stimType = input('stimulus:  white noise (1) or natural images (2):  ');
if stimType==1
    option.stimulus = 'white';
else
    option.stimulus = 'McGill_clips';
end
clear stimType;

alphaRange = logspace(-1, 4, 50); % 50 log-spaced values between 0.1 and 10000
algoType = input('stimulus:  cross-correlation (1) or scg-regression (2):  ');
if algoType==1
    option.algorithm = 'crossCorr';
else
    option.algorithm = 'scg';
end
clear algoType;

% partition data: training, validation, and test 
nMovies.train  = 3;
nMovies.valid = 1; 
nMovies.test = 1; % New test partition
nMovies.total = nMovies.train + nMovies.valid + nMovies.test; 
if (nMovies.total > 6)
    error('sorry, we only have 6 movie response datasets');
end

imgSiz  = 32;     % width/height of stimulus/filter/rfMap  
nPixels = imgSiz^2; 

durSec = 5;   refreshHz = 75;  % simulate 5 seconds at 75 hz frame rate
nFrames = durSec*refreshHz;  % e.g. 5 sec at 75 hz  (="ndata" in netlab)
if nFrames>375
    error('too many frames for these movie files !');
end

% specify model receptive field (Gabor function followed by half-power law)
model.lambda = 8;
model.phase  = 0;
model.ori    = 0;  
model.pwrExp = 2; % input('power law exponent:   ');

%  graph specs
fig.stim.pos          = [400 200 300 300];     %[xOff yOff xSize ySize];
fig.stim.handle       = figure('position',fig.stim.pos,'toolbar','none','menubar','none','name','stimulus');
fig.model.pos         = [50 600 300 400];  
fig.model.handle      = figure('position',fig.model.pos,'toolbar','none','menubar','none','name','model');

%  create model filter, and plot in Figure 1 
rfModel = makeModelRF(model,imgSiz);    % creates model filter (Gabor function)
rfModelVec  = reshape(rfModel,1,nPixels);      % make a 1d version, for later use            

%  partition full dataset into 3 subsets: training, validation, and test 
stimMovie      = zeros(nPixels,nFrames);
stimMovieTrain   = [];
stimMovieValid  = [];
stimMovieTest = []; % New
respTrain      = [];
respValid     = [];
respTest = []; % New

for iMovie=1:nMovies.total
    getStimulusMovies;            % -> stimMovie = nPixels x nFrames, range -1 to +1
    output = rfModelVec*stimMovie;    % linear filter response to the stimulus    
    output = hwr(output);  % half-wave rectify (set negative values to zero)
    output = output.^model.pwrExp;  % power-low for positive values
    
    % accumulate results in dataset partitions: 
    if iMovie <= nMovies.train
        stimMovieTrain = [stimMovieTrain stimMovie]; 
        respTrain    = [respTrain    output];        
    elseif iMovie <= nMovies.train + nMovies.valid
        stimMovieValid  = [stimMovieValid stimMovie];  
        respValid     = [respValid    output];          
    else
        stimMovieTest = [stimMovieTest stimMovie];  % New
        respTest = [respTest output];  % New
    end                      
end  % end of iMovie-loop

if strcmp(option.algorithm,'scg')
    % initialize options for optimization
    nin  = imgSiz^2; % number of inputs
    nout = 1;        % number of outputs:  one neuron
    netOptions     = zeros (1,18); 
    netOptions(1)  = 0;
    netOptions(2)  = .0001;      % termination criterion: distance moved
    netOptions(3)  = netOptions(2); % for scg, use VERY small value, eg 10^-9
    netOptions(14) = 200;    % max no of iterations - should be >= no of dim.s ?

    % Store VAF values and RF maps for different alphas
    vafValues = zeros(size(alphaRange));
    rfMaps = zeros(imgSiz, imgSiz, length(alphaRange));

    for aIdx = 1:length(alphaRange)
        alpha = alphaRange(aIdx);
    
        % estimate rfMap, for this alpha
        net = glm(nin, nout,'linear',alpha);       % initialize structure 

        net.w1 = 0*net.w1;    net.b1 = 0*net.b1;   % sparse prior
        [net, netOptions] = netopt(net,netOptions,stimMovieTrain',respTrain','scg');
        rfMap2d   = reshape(net.w1,imgSiz,imgSiz);  % reshape to 2d 
        rfMapVec    = reshape(rfMap2d,nPixels,1);    % make a 1-d version

        % Compute VAF for training dataset
        predRespTrain = rfMapVec'*stimMovieTrain;
        residTrain = respTrain - predRespTrain;
        vafTrain = 1 - var(residTrain)/var(respTrain);
        vafValues(aIdx) = vafTrain;

        % Store the RF map for this alpha
        rfMaps(:,:,aIdx) = rfMap2d;
    end
    figure;
    semilogx(alphaRange, vafValues, 'b*-');
    xlabel('Alpha');
    ylabel('VAF');
    title('VAF vs. Alpha on Training Dataset');
    grid on;

    [~, optimalIdx] = max(vafValues);
    alphasForSubplots = [1, optimalIdx, length(alphaRange)];

    figure;
    subplot(2,2,1);
    imagesc(rfModel); axis image; axis off; title('Actual RF');
    colormap('gray');
    
    for i = 1:3
        subplot(2,2,i+1);
        imagesc(rfMaps(:,:,alphasForSubplots(i))); axis image; axis off;
        title(sprintf('Estimated RF, alpha=%.2f', alphaRange(alphasForSubplots(i))));
    end


elseif strcmp(option.algorithm,'crossCorr')
    nLags=1; maxLag=0;  % (settings to make xcorr.m give us what we want)
    crossCorrAll = zeros(nPixels,nMovies.train*nFrames);
    for iPix=1:nPixels
        crossCorrAll(iPix,:) = xcorr(respTrain,stimMovieTrain(iPix,:),maxLag,'unbiased');  % cross-correlation
    end
    rfMap = crossCorrAll(:,end-nLags+1:end);  % only take positive lagged values
    rfMap2d = reshape(rfMap,imgSiz,imgSiz);  % reshape into 2d
    rfMapVec  = reshape(rfMap2d,nPixels,1);    % make a 1-d version
    clear iPix crossCorrAll rfMap nLags maxLag;
else
    error('unrecognized algorithm');
end

% show some example stimulus images
figure(fig.stim.handle);
for ix=1:4
    stimImgVec = stimMovie(:,ix+20);
    stimImg = reshape(stimImgVec,imgSiz,imgSiz);
    subplot(2,2,ix);
    imagescZadj(stimImg); axis image; axis off; colormap('gray');
end

% graph estimated rfMap, below "actual" (model) receptive field:
figure(fig.model.handle); 
subplot(2,1,1);
imagescZadj(rfModel);  hold on; axis image; axis off; colorbar; title('RF model filter');
subplot(2,1,2);  
imagescZadj(rfMap2d); 
if strcmp(option.algorithm,'scg')
    title(sprintf('scg RF estimate for alpha = %3.1f',alpha));  
else
    title(sprintf('RF estimate for cross-correlation'));      
end
axis image;  axis off;  colorbar;  

% use rfMap estimate to generate prediction of the training and validation responses)
predRespTrain = rfMapVec'*stimMovieTrain; 
predRespValid = rfMapVec'*stimMovieValid; 

residValidNew = respValid - predRespValid;  % residual - error in prediction of validation response

% calculate VAF for validation dataset
vaf.R_matrix = corrcoef(respValid,predRespValid);  % -> 2x2 matrix, ones on diagonal
vaf.offDiag = vaf.R_matrix(1,2);
vaf.vaf = vaf.offDiag^2.;
fprintf(1,'\nVAF for validation dataset = %5.1f percent\n', 100*vaf.vaf);   
