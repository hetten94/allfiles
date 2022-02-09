clc;
clear;
fid = fopen('NoF1.txt');
%fid = fopen('Samp.txt');
T = textscan(fid, '%f%f');
fclose(fid);
X = T{1};
%XX =T{1};
Y = T{2};
%YY = T{2};
X1 = -X';
Y1 = Y';

m = length(X');
m1 = floor(m/48)+1;
Xnew = zeros(1,m1);
k=0;
for i = 1:48:(m-48)
   max = X1(48*floor(i/49)+1);
    for j = i:1:i+47
         if X1(j+1)>max
             max = X1(j+1);
             k = j+1;
        end
        %maxm = max - (X1(k+17)+X1(k+18)+X1(k+19)+X1(k+20)+X1(k+21)+X1(k+22)+X1(k+23)+X1(k+24)+X1(k+25)+X1(k+26))/10;
        maxm = max - (X1(k+7)+X1(k+8)+X1(k+9))/3;
        if maxm < 0
            maxm = 0;
        end
    end
   Xnew(floor(i/48)+1) = maxm;
end
Xnew1 = Xnew';
%formatSpec = '%f\n';
%fileID = fopen('outX8_5A26.11.20.txt','wt');
%fprintf(fileID,formatSpec,Xnew1);
%fclose(fileID);
n = length(Y');
n1 = floor(n/48)+1;
Ynew = zeros(1,n1);
k=0;
for i = 1:48:(n-48)
   max = Y1(48*floor(i/49)+1);
    for j = i:1:i+47
         if Y1(j+1)>max
             max = Y1(j+1);
             k = j+1;
         end
        %maxm = max - (Y1(k+17)+Y1(k+18)+Y1(k+19)+Y1(k+20)+Y1(k+21)+Y1(k+22)+Y1(k+23)+Y1(k+24)+Y1(k+25)+Y1(k+26))/10;
        maxm = max - (Y1(k+7)+Y1(k+8)+Y1(k+9))/3;
        if maxm < 0
            maxm = 0;
        end
    end
   Ynew(floor(i/48)+1) = maxm;
end

Ynew1 = Ynew';

% N = length(Ynew1);
% G = zeros(N,1);
% dG2 = zeros(N,1);
% k=0;
% for j=1:1:N
% k = k+1;
% s = 0;
% s2 = 0;
% s3 = 0;
% s4 = 0;
% s5 = 0;
% s6 = 0;
%     for i = 1:1:j
%         s = s + Xnew1(i)*Ynew1(i);
%         s2 = s2 + Xnew1(i);
%         s3 = s3 + Ynew1(i);
%         s4 = s4 + Xnew1(i)*Ynew1(i)*Xnew1(i)*Ynew1(i);
%         s5 = s5 + Xnew1(i)*Xnew1(i);
%         s6 = s6 + Ynew1(i)*Ynew1(i);
%     end
%     G(k) = s/(s2*s3)*j;
%     dG2(k) = sqrt(((s4/j-s*s/(j*j))/(s*s) + (s5/j-s2*s2/(j*j))/(s2*s2) + (s6/j-s3*s3/(j*j))/(s3*s3))/(j-1))*s/(s2*s3)*j*j;
%     if j==1
%         dG2(1) = 0;
%     end
% end
% G = G';
% dG2 = dG2';
formatSpec = '%f\n';
fileID = fopen('outSNoF1.txt','wt');
fileID1 = fopen('outINoF1.txt','wt');
fprintf(fileID,formatSpec,Xnew1);
fprintf(fileID1,formatSpec,Ynew1);
fclose(fileID);
fclose(fileID1);
% fileID2 = fopen('outGPMT8.5.txt','wt');
% fileID3 = fopen('outdG2PMT8.5.txt','wt');
% fprintf(fileID2,formatSpec,G);
% fprintf(fileID3,formatSpec,dG2);
% fclose(fileID2);
% fclose(fileID3);




