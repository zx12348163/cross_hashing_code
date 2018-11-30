function A = LoadMatrixBin(strFile, type)

fid = fopen(strFile, 'rb');
h = fread(fid, 1, 'int32');
w = fread(fid, 1, 'int32');
m = fread(fid, h*w, type);
m = reshape(m, w, h);
A = m';
fclose(fid);