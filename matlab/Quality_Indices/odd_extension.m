%%%%%%%%%%%%%% Q2n aux. function
function ext = odd_extension(A,winx,winy)

if errargn(mfilename,nargin,3,nargout,1), error('*'), end
if errargt(mfilename,winx,'in0'), error('*'), end
if errargt(mfilename,winy,'in0'), error('*'), end
if (rem(winx,2) ~= 1 || rem(winy,2) ~= 1)
   disp(' ')
   disp('Errore: le dimensioni delle finestre devono essere entrambe dispari');
   disp(' ')
   return
end

[dimy,dimx] = size(A);
wx = (winx-1)/2;
wy = (winy-1)/2;
ext = zeros(dimy+winy-1,dimx+winx-1);
ext(wy+1:wy+dimy,wx+1:wx+dimx) = A;

for k = 1:wy
   ext(wy-k+1,wx+1:wx+dimx) = A(k+1,:);
   ext(wy+dimy+k,wx+1:wx+dimx) = A(dimy-k,:);
end
for k = 1:wx
   ext(:,wx-k+1) = ext(:,wx+k+1);
   ext(:,wx+dimx+k) = ext(:,dimx+wx-k);
end
   
end

