% assign_1d_RF_sysIdent_overfit.m 
%
% demonstrate overfitting, for simple regression estimation of 1-d (spatial) receptive field
% use hold-back dataset, compare learning curves for training vs validation sets
%   use to illustrate how "early stopping" can work

% note: error = mean-squares not sum-squares

% for early stopping:  save RF estimate when prediction of validation set starts to get worse

clear all;  close all;  fprintf(1,'\n\n\n\n\n\n');

rng('default');   % "standard" random number seed -> reproducible simulations

nRFpts = 32;    % number of points in receptive field (== number of parameters to be estimated)
nMeasTrain = 60;    % number of measurements to use for receptive field estimation
nMeasValid = 40;    % additional measurements to use for validation

noiseAmp = 0.4; % amplitude of noise

eta = input('learning rate:   ');  % learning rate

num_iterations = input('number of batch-mode iterations:   '); 

% define a model receptive field (Gabor function), and plot it
xPtsK = 1:1:nRFpts;
mu = nRFpts/2;   lambda = nRFpts/5;   sig = lambda*0.5;
env = exp(-(xPtsK-mu).^2/(2*sig^2));  % Gaussian envelope
receptiveField = env.*sin(2*pi*xPtsK/lambda);
figure(1);    
plot(xPtsK,receptiveField,'b-');      grid;

% create 2 input signals (stimulus sets):   white noise, range from -1 to +1
stimTrain = (rand(nRFpts,nMeasTrain) - 0.5);   % nMeasTrain measurements, for nRFpts pixels

% simulate response of the model system (receptive field) to input signal:
respTrain = receptiveField*stimTrain + noiseAmp*randn(1,nMeasTrain);  % (with some added noise) 
% stim   = nRFpts x nMeas
% resp   = 1 x nMeas           % note stim and resp are ~ zero-mean (as they need to be)
% w      = 1 x nRFpts

% w = randn(1,nRFpts);  % random initial weights (conventional practice) 
w = zeros(1,nRFpts);  % initialize weights (receptive field estimate) - "sparse prior"

errTrain = zeros(num_iterations,1);     % initialize histories

for iteration = 1:num_iterations    % loop over iterations

   respCalc = w*stimTrain;     % predicted response for estimation dataset

   % gradient descent
   dw = (respCalc - respTrain)*stimTrain'; %  gradient
   w = w - eta*dw;   % learning rule:  update weights   
   
   errTrain(iteration) = mean((respTrain - respCalc).^2);   % record error-squared for history

   plot(xPtsK,receptiveField,'b-',xPtsK,w,'r-');   grid;

   xMin = min(xPtsK);    xMax = max(xPtsK);   % set axis limits, to keep things stable
   yMin = 1.5*min(receptiveField);  yMax = 1.5*max(receptiveField);
   axis ([xMin xMax  yMin yMax]);
   legend('actual receptive field','estimated receptive field');
   drawnow  
end

figure(2); 
plot(1:1:num_iterations,errTrain,'b-');  %,1:1:num_iterations,errValid,'r-');
han.legend = legend('errTrain','Location','NorthEast');  % legend in upper right
set(han.legend,'FontSize',10);
set(han.legend,'FontWeight','bold');
set(han.legend,'EdgeColor','black');
grid on;  xlabel('iterations');  ylabel('MSE');
drawnow;


