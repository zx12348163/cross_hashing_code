#include "types.h"

#ifndef LINSCAN_H__
#define LINSCAN_H__

void linscan_hamm_query(UINT32 *res, UINT32* dist, 
						UINT8 *codes, UINT8 *queries, int N, UINT32 NQ, 
						int B, unsigned int K, int dim1codes, int dim1queries);

#endif  // LINSCAN_H__
