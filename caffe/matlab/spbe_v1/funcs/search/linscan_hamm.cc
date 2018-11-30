#include "linscan.h"
#include <string.h>
#include "bitops.h"
#include "types.h"
#include "StructForCompare.h"
#include <vector>


/**
 * Performs kNN search by linear scan in Hamming distance between
 * binary codes and queries
 *
 * Inputs:
 *
 *   N: number of codes in the db
 *   NQ: number of queries to be answered
 *   B: number of bits in the db/query codes that should be taken into
 *      account in Hamming distance
 *   K: number of results to be returned for each query ie, k in kNN
 *   codes: an array of UINT8 storing the db codes
 *   queries: an array of UINT8 storing the query codes
 *   dim1codes: number of words in the database codes -- most likely
 *              dim1codes = B/8
 *   dim2codes: number of words in the query codes -- most likely
 *              dim2codes = B/8
 *
 * Outputs:
 *
 *   counter: int[(B+1)*N], stores the number of db items with
 *            different Hamming distances (ranging from 0 to B) from
 *            each query.
 *   res: int[K*N], stores the ids of K nearest neighbors for each
 *        query (zero_based)
 */
void linscan_hamm_query(UINT32 *res, UINT32* dist, UINT8 *codes,
                        UINT8 *queries, int N, UINT32 NQ, int B,
                        unsigned int K, int dim1codes,
                        int dim1queries) {
  int B_over_8 = B / 8;
  UINT8 * pqueries = queries;
  UINT32 * pres = res;
  UINT32 * pdist = dist;

  int i;
#ifndef SINGLE_CORE
#pragma omp parallel shared(i) private(pqueries, pres, pdist)
#endif
  {
#ifndef SINGLE_CORE
#pragma omp for
#endif
    for (i=0; i<NQ; i++) 
	{
#ifndef SINGLE_CORE
      pqueries = queries + (UINT64)i*(UINT64)dim1queries;
      pres = res + (UINT64)i*(UINT64)K;
	  pdist = dist + (UINT64)i*(UINT64)K;
#endif

	  std::vector<PairForRank<int>> vecDist(N);
      UINT8 * pcodes = codes;
      for (int j = 0; j < N; j++, pcodes += dim1codes) 
	  {
		  vecDist[j].id = j;
		  vecDist[j].value = match(pcodes, pqueries, B_over_8);
      }
	  
	  std::sort(vecDist.begin(), vecDist.end(), PairForRank<int>::CompByValueSmallFirst);
      
      for(int j = 0; j < K; ++j)
	  {
		  pres[j] = vecDist[j].id;
		  pdist[j] = vecDist[j].value;
	  }      
	    
#ifdef SINGLE_CORE
      pres += K;
	  pdist += K;
      pqueries += dim1queries;
#endif
    }
  }
}
