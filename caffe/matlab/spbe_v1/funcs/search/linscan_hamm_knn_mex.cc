// A mex wrapper around linscan_hamm_query to call Hamming k-nearest
// neighbor search from within matlab.
//
// NOTE: the output indices are one-based.

#include <algorithm>
#include "linscan.h"
#include <math.h>
#include <mex.h>
#include <stdio.h>
#include "types.h"

// Inputs --------------------

#define mxcodes         prhs[0]  // A matrix of data points against
                                 // which nearest neighbors search is
                                 // performed.
#define mxqueries       prhs[1]  // A matrix of query points for the
                                 // search.
#define mxN             prhs[2]  // Number of data points to use in
                                 // search.
#define mxB             prhs[3]  // Number of bits per code.
#define mxK             prhs[4]  // Number of kNN results to return.

// Outputs --------------------

#define mxres           plhs[0]
#define mxdist          plhs[1]


void myAssert(int a, const char *b) {
  if (!a)
    mexErrMsgTxt(b);
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )    
{
  if (nrhs != 5)
    mexErrMsgTxt("Wrong number of inputs\n");
  if (nlhs != 2)
    mexErrMsgTxt("Wrong number of outputs\n");
	
  UINT32 N = (UINT32) *(mxGetPr(mxN));
  int B = (int) *(mxGetPr(mxB));
  int K = (int) *(mxGetPr(mxK));
	
  UINT8 *codes = (UINT8*) mxGetPr(mxcodes);
  UINT8 *queries = (UINT8*) mxGetPr(mxqueries);
	
  int NQ = mxGetN(mxqueries);
  int dim1codes = mxGetM(mxcodes);
  int dim1queries = mxGetM(mxqueries);

  myAssert(mxGetN(mxcodes) >= N, "codes < N");
  myAssert(dim1codes >= B / 8, "dim1codes < B/8");
  myAssert(dim1queries >= B / 8, "dim1queries < B/8");
  myAssert(B % 8 == 0, "mod(B, 8) != 0");

  mxres = mxCreateNumericMatrix(K, NQ, mxUINT32_CLASS, mxREAL);
  UINT32 *res = (UINT32 *) mxGetPr(mxres);
  mxdist = mxCreateNumericMatrix(K, NQ, mxUINT32_CLASS, mxREAL);  
  UINT32 *dist = (UINT32 *) mxGetPr(mxdist);

  linscan_hamm_query(res, dist, codes, queries, N, NQ, B, K,
                     dim1codes, dim1queries);
}
