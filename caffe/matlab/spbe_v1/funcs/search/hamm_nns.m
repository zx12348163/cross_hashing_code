% Running k-nearest neighbor search according to Hamming distance
% codes from https://github.com/norouzi/ckmeans
% time&memory-consuming

function [ind dist] = hamm_nns(B, C, knn)

% Inputs:
%       B: (m/8)*n -- n m-bit codes represented in compressed format,
%          created by compactbit. Every 8 bits is represented by a unit8.
%       C: (m/8)*n -- n m-bit query codes represented in compressed format
%          that should be compared with the binary codes in B using Hamming
%          distance.
%       knn: scaler -- number of nearest neighbors to be found.

% Outputs:
%       ind: indices of nearest elements from B (one-based)
%       dist: hamming distance of corresponding ind
%       counter: a count vector that counts how many of the k nearest
%                items have a hamming distance of 0, 1, ..., nbits


assert(size(C, 1) == size(B, 1), ['number of rows of B and C should' ...
                                  ' be equal.']);
n = size(B, 2);
m = size(B, 1) * 8;

[ind, dist] = linscan_hamm_knn_mex(B, C, n, m, knn);
