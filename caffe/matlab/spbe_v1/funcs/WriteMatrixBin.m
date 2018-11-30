function WriteMatrixBin(A, strFile, type)
[h, w] = size(A);
fid = fopen(strFile, 'wb');
fwrite(fid, h, 'int32');
fwrite(fid, w, 'int32');
fwrite(fid, A', type);
fclose(fid);