function imagescZadj(rfMap)
%
%imagescZadj - function to make imagesc-plot with z-scale adjusted so midpoint==0
%   usage:   imagescZadj(array)
%
%   if array is 1-d, reshapes as square image


yMax = max(max(rfMap));  yMin = min(min(rfMap));
yMx = max([abs(yMax) abs(yMin)]);

if min(size(rfMap))==1         % handle images reshaped as 1-d
    siz = sqrt(length(rfMap));
    if ~(siz==round(siz))
        error('not a square image array !');
    end
    rfMap = reshape(rfMap,siz,siz);
end

imagesc(rfMap,[-yMx yMx]);

return

