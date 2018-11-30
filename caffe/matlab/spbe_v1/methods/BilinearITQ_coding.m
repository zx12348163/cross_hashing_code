function matCodeBin = BilinearITQ_coding(matData, matR1, matR2)

    dim1 = size(matR1, 1);
    dim2 = size(matR2, 1);
    b1 = size(matR1, 2);
    b2 = size(matR2, 2);
    matData = matData';
    nNum = size(matData, 2);
    X = reshape(matData, dim1, dim2, nNum);

    BB = zeros(b1, b2, nNum,'single');
    for j=1:nNum
        matProj = matR1'*X(:,:,j)*matR2;
        BB(:,:,j) = (matProj>=0);
    end

    matCodeBin = reshape(BB, b1*b2, nNum);
    matCodeBin = matCodeBin';
end