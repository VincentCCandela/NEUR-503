function f1 = makeModelRF(model,imgSiz)
%
% create a filter (Gabor) for receptive field model of a V1 simple cell
%
% arguments:  imgSiz, carrLambda, carrOri
% returns:  f1
   
carrLambda = model.lambda;
carrOri    = model.ori;

nPts = imgSiz;       
carrSF     = 1 / carrLambda;
envSigX     = carrLambda/3;  
envSigY     = envSigX;
carrOriRad = (pi/180)*carrOri;
carrPh      = model.phase;    carrPhRad  = (pi/180)*carrPh;  % 0=sine,90=cosine
envOri      = 0;    envOriRad  = (pi/180)*envOri;
for x=1:nPts
   for y=1:nPts     % (faster than meshgrid, for small array sizes)
	  xx = (x-nPts/2)*cos(envOriRad) - (y-nPts/2)*sin(envOriRad);
	  yy = (x-nPts/2)*sin(envOriRad) + (y-nPts/2)*cos(envOriRad);
      env = exp( -(xx).^2/(2*envSigX*envSigX) - (yy).^2/(2*envSigY*envSigY) );
	  carr = sin(2*pi*carrSF*(x*cos(carrOriRad)+y*sin(carrOriRad)) + carrPhRad);	  
      f1(y,x) = env * carr;
   end
end
