function ITQTest

bTraining = 1;
bCoding = 1;
iIter = 0;
   
 
strGistDir = 'E:\yanxia\dataset\SPQ\GistRemoveMean\';
strTrainFile = [strGistDir 'matTrain.bin'];
strQueryFile = [strGistDir 'matQuery.bin'];
strDatabaseFile = [strGistDir 'matBase.bin'];

iCode = 64;

if(bTraining)
    matTrain = LoadMatrixBin(strTrainFile, 'double');
    
    %PCA
    matCov = matTrain' * matTrain;
    [V, D] = eig(matCov);
    V = V';
    V = V(end:-1:1, :);
    
    %ITQ
    matProPca = V(1:iCode, :); %c*D
    matTrainPca = matTrain * matProPca'; %N*c
    [B, R] = ITQ(matTrainPca, iIter);

    WriteMatrixBin(matProPca, 'matProPca.bin', 'double');
    WriteMatrixBin(R, 'R.bin', 'double');
    
    matPro = R' * matProPca;
    WriteMatrixBin(matPro, 'matPro.bin', 'double');
else
    matPro = LoadMatrixBin('matPro.bin', 'double');
end
   
   
if(bCoding)
    matQuery = LoadMatrixBin(strQueryFile, 'double');
    matBase = LoadMatrixBin(strDatabaseFile, 'double');
    
    matQueryCode = ITQCoding(matQuery, matPro);
    WriteMatrixBin(matQueryCode, 'matCodeQuery.bin', 'uint');
    
    matBaseCode = ITQCoding(matBase, matPro);
    WriteMatrixBin(matBaseCode, 'matCodeDatabase.bin', 'uint');    
end

end

function matCode = ITQCoding(matData, matPro)
     matDataPro = matData * matPro';
     matCodeBin = int32(zeros(size(matDataPro)));
     matCodeBin(matDataPro >=0.0) = int32(1);
     matCode = compactbit(matCodeBin);
end

