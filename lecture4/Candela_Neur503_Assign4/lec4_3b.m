clear all; close all; fprintf(1,'\n\n\n\n\n\n');

rng('default'); % "standard" random number seed -> reproducible simulations

nRFpts = 32; % number of points in receptive field (== number of parameters to be estimated)
nMeasTrain = 60; % number of measurements to use for receptive field estimation
nMeasValid = 40; % additional measurements to use for validation

noiseAmp = 0.4; % amplitude of noise

eta = input('learning rate: '); % learning rate
num_iterations = input('number of batch-mode iterations: ');

% define a model receptive field (Gabor function), and plot it
xPtsK = 1:1:nRFpts;
mu = nRFpts/2; lambda = nRFpts/5; sig = lambda*0.5;
env = exp(-(xPtsK-mu).^2/(2*sig^2)); % Gaussian envelope
receptiveField = env.*sin(2*pi*xPtsK/lambda);
figure(1);
plot(xPtsK,receptiveField,'b-'); grid;
title('Actual Receptive Field');
xlabel('Position'); ylabel('Response');

% create input signals (stimulus sets) for training and validation: white noise, range from -1 to +1
stimTrain = (rand(nRFpts,nMeasTrain) - 0.5); % nMeasTrain measurements for training
stimValid = (rand(nRFpts,nMeasValid) - 0.5); % nMeasValid measurements for validation

% simulate response of the model system (receptive field) to input signal for both datasets
respTrain = receptiveField * stimTrain + noiseAmp * randn(1, nMeasTrain); % (with some added noise)
respValid = receptiveField * stimValid + noiseAmp * randn(1, nMeasValid);

% Initialize weights (receptive field estimate) - "sparse prior"
w = zeros(1, nRFpts);

errTrain = zeros(num_iterations, 1); % initialize error histories
errValid = zeros(num_iterations, 1); % for validation

bestErr = Inf;
bestIteration = 0;
bestW = w;

for iteration = 1:num_iterations % loop over iterations

    % Training
    respCalcTrain = w * stimTrain;
    dw = (respCalcTrain - respTrain) * stimTrain'; % gradient
    w = w - eta * dw; % update weights
    errTrain(iteration) = mean((respTrain - respCalcTrain).^2); % record MSE for training

    % Validation
    respCalcValid = w * stimValid;
    errValid(iteration) = mean((respValid - respCalcValid).^2); % record MSE for validation

    % Early Stopping Check
    if errValid(iteration) < bestErr
        bestErr = errValid(iteration);
        bestIteration = iteration;
        bestW = w;
    end

    % Plotting during training
    if mod(iteration, 10) == 0 % Update plot every 10 iterations
        figure(1);
        plot(xPtsK, receptiveField, 'b-', xPtsK, w, 'r-'); grid;
        axis([min(xPtsK), max(xPtsK), 1.5*min(receptiveField), 1.5*max(receptiveField)]);
        legend('Actual Receptive Field', 'Estimated Receptive Field');
        title(['Receptive Fields at Iteration ', num2str(iteration)]);
        xlabel('Position'); ylabel('Response');
        drawnow
    end
end

% Plot the best estimation
figure(1);
plot(xPtsK, receptiveField, 'b-', xPtsK, bestW, 'r-'); grid;
legend('Actual Receptive Field', 'Estimated Receptive Field');
title('Best Estimation');
xlabel('Position'); ylabel('Response');

% Plotting Learning Curves
figure(3);
plot(1:num_iterations, errTrain, 'b-', 1:num_iterations, errValid, 'r-');
legend('Training Error', 'Validation Error', 'Location', 'NorthEast');
xlabel('Iterations'); ylabel('MSE');
title('Learning Curves');
hold on;
plot(bestIteration, bestErr, 'go', 'DisplayName', 'Best Iteration'); % Indicate best iteration
hold off;
grid on; 

% Reporting
fprintf('Final Results:\n');
fprintf('Learning Rate: %f\n', eta);
fprintf('Number of Iterations: %d\n', num_iterations);
fprintf('Best Iteration: %d\n', bestIteration);
