#pragma once
#include <Windows.h>
#include <algorithm>

template<typename TYPE>
class PairForRank
{
public:
	PairForRank()
	{
	}

	PairForRank(TYPE inValue, int inID):value(inValue), id(inID)
	{
	}

	TYPE value;
	int id;

	static bool CompByValueSmallFirst(const PairForRank<TYPE> &lhs, const PairForRank<TYPE> &rhs)
	{
		return lhs.value < rhs.value;
	}
				
	static bool CompByValueLargeFirst(const PairForRank<TYPE> &lhs, const PairForRank<TYPE> &rhs)
	{
		return lhs.value > rhs.value;
	}
};
