#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
void updateLossyCompElement_Double(unsigned char* curBytes, unsigned char* preBytes, 
		int reqBytesLength, int resiBitsLength,  LossyCompressionElement *lce)
{
	int resiIndex, intMidBytes_Length = 0;
	int leadingNum = compIdenticalLeadingBytesCount_double(preBytes, curBytes); //in fact, float is enough for both single-precision and double-precisiond ata.
	int fromByteIndex = leadingNum;
	int toByteIndex = reqBytesLength; //later on: should use "< toByteIndex" to tarverse....
	if(fromByteIndex < toByteIndex)
	{
		intMidBytes_Length = reqBytesLength - leadingNum;
		memcpy(lce->integerMidBytes, &(curBytes[fromByteIndex]), intMidBytes_Length);
	}
	int resiBits = 0;
	if(resiBitsLength!=0)
	{
		resiIndex = reqBytesLength;
		if(resiIndex < 8)
			resiBits = (curBytes[resiIndex] & 0xFF) >> (8-resiBitsLength);
	}
	lce->leadingZeroBytes = leadingNum;
	lce->integerMidBytes_Length = intMidBytes_Length;
	lce->resMidBitsLength = resiBitsLength;
	lce->residualMidBits = resiBits;
}

inline void longToBytes_bigEndian(unsigned char *b, unsigned long num) 
{
	b[0] = (unsigned char)(num>>56);
	b[1] = (unsigned char)(num>>48);
	b[2] = (unsigned char)(num>>40);
	b[3] = (unsigned char)(num>>32);
	b[4] = (unsigned char)(num>>24);
	b[5] = (unsigned char)(num>>16);
	b[6] = (unsigned char)(num>>8);
	b[7] = (unsigned char)(num);
//	if(dataEndianType==LITTLE_ENDIAN_DATA)
//		symTransform_8bytes(*b);
}

void compressSingleDoubleValue(DoubleValueCompressElement *vce, double tgtValue, double precision, double medianValue,
		int reqLength, int reqBytesLength, int resiBitsLength)
{
	double normValue = tgtValue - medianValue;

	ldouble lfBuf;
	lfBuf.value = normValue;

	int ignBytesLength = 64 - reqLength;
	if(ignBytesLength<0)
		ignBytesLength = 0;

	long tmp_long = lfBuf.lvalue;
	longToBytes_bigEndian(vce->curBytes, tmp_long);

	lfBuf.lvalue = (lfBuf.lvalue >> ignBytesLength)<<ignBytesLength;

	//double tmpValue = lfBuf.value;

	vce->data = lfBuf.value+medianValue;
	vce->curValue = tmp_long;
	vce->reqBytesLength = reqBytesLength;
	vce->resiBitsLength = resiBitsLength;
}

inline void intToBytes_bigEndian(unsigned char *b, unsigned int num)
{
	b[0] = (unsigned char)(num >> 24);	
	b[1] = (unsigned char)(num >> 16);	
	b[2] = (unsigned char)(num >> 8);	
	b[3] = (unsigned char)(num);	
	
	//note: num >> xxx already considered endian_type...
//if(dataEndianType==LITTLE_ENDIAN_DATA)
//		symTransform_4bytes(*b); //change to BIG_ENDIAN_DATA
}

inline short computeReqLength_double_MSST19(double realPrecision)
{
	short reqExpo = getPrecisionReqLength_double(realPrecision);
	return 12-reqExpo;
}


unsigned int optimize_intervals_double_1D_opt_MSST19(double *oriData, size_t dataLength, double realPrecision)
{
	size_t i = 0, radiusIndex;
	double pred_value = 0;
	double pred_err;
	size_t *intervals = (size_t*)malloc(confparams_cpr->maxRangeRadius*sizeof(size_t));
	memset(intervals, 0, confparams_cpr->maxRangeRadius*sizeof(size_t));
	size_t totalSampleSize = 0;//dataLength/confparams_cpr->sampleDistance;

	double * data_pos = oriData + 2;
	double divider = log2(1+realPrecision)*2;
	int tempIndex = 0;
	while(data_pos - oriData < dataLength){
		if(*data_pos == 0){
        		data_pos += confparams_cpr->sampleDistance;
        		continue;
		}
		tempIndex++;
		totalSampleSize++;
		pred_value = data_pos[-1];
		pred_err = fabs((double)*data_pos / pred_value);
		radiusIndex = (unsigned long)fabs(log2(pred_err)/divider+0.5);
		if(radiusIndex>=confparams_cpr->maxRangeRadius)
			radiusIndex = confparams_cpr->maxRangeRadius - 1;
		intervals[radiusIndex]++;

		data_pos += confparams_cpr->sampleDistance;
	}
	//compute the appropriate number
	size_t targetCount = totalSampleSize*confparams_cpr->predThreshold;
	size_t sum = 0;
	for(i=0;i<confparams_cpr->maxRangeRadius;i++)
	{
		sum += intervals[i];
		if(sum>targetCount)
			break;
	}
	if(i>=confparams_cpr->maxRangeRadius)
		i = confparams_cpr->maxRangeRadius-1;

	unsigned int accIntervals = 2*(i+1);
	unsigned int powerOf2 = roundUpToPowerOf2(accIntervals);

	if(powerOf2<64)
		powerOf2 = 64;

	free(intervals);
	return powerOf2;
}


TightDataPointStorageD* SZ_compress_double_1D_MDQ_MSST19(double *oriData,
size_t dataLength, double realPrecision, double valueRangeSize, double medianValue_f)
{
#ifdef HAVE_TIMECMPR
	double* decData = NULL;
	if(confparams_cpr->szMode == SZ_TEMPORAL_COMPRESSION)
		decData = (double*)(multisteps->hist_data);
#endif

	//struct ClockPoint clockPointBuild;
	//TimeDurationStart("build", &clockPointBuild);
	unsigned int quantization_intervals;
	if(exe_params->optQuantMode==1)
		quantization_intervals = optimize_intervals_double_1D_opt_MSST19(oriData, dataLength, realPrecision);
	else
		quantization_intervals = exe_params->intvCapacity;
	//updateQuantizationInfo(quantization_intervals);
	int intvRadius = quantization_intervals/2;

	double* precisionTable = (double*)malloc(sizeof(double) * quantization_intervals);
	double inv = 2.0-pow(2, -(confparams_cpr->plus_bits));
    for(int i=0; i<quantization_intervals; i++){
        double test = pow((1+realPrecision), inv*(i - intvRadius));
        precisionTable[i] = test;
    }

	struct TopLevelTableWideInterval levelTable;
    MultiLevelCacheTableWideIntervalBuild(&levelTable, precisionTable, quantization_intervals, realPrecision, confparams_cpr->plus_bits);

	size_t i;
	int reqLength;
	double medianValue = medianValue_f;
	//double medianInverse = 1 / medianValue_f;
	//short radExpo = getExponent_double(realPrecision);

	reqLength = computeReqLength_double_MSST19(realPrecision);

	int* type = (int*) malloc(dataLength*sizeof(int));

	double* spaceFillingValue = oriData; //

	DynamicIntArray *exactLeadNumArray;
	new_DIA(&exactLeadNumArray, dataLength/2/8);

	DynamicByteArray *exactMidByteArray;
	new_DBA(&exactMidByteArray, dataLength/2);

	DynamicIntArray *resiBitArray;
	new_DIA(&resiBitArray, DynArrayInitLen);

	unsigned char preDataBytes[8];
	intToBytes_bigEndian(preDataBytes, 0);

	int reqBytesLength = reqLength/8;
	int resiBitsLength = reqLength%8;
	double last3CmprsData[3] = {0};

	//size_t miss=0, hit=0;

	DoubleValueCompressElement *vce = (DoubleValueCompressElement*)malloc(sizeof(DoubleValueCompressElement));
	LossyCompressionElement *lce = (LossyCompressionElement*)malloc(sizeof(LossyCompressionElement));

	//add the first data
	type[0] = 0;
	compressSingleDoubleValue_MSST19(vce, spaceFillingValue[0], realPrecision, reqLength, reqBytesLength, resiBitsLength);
	updateLossyCompElement_Double(vce->curBytes, preDataBytes, reqBytesLength, resiBitsLength, lce);
	memcpy(preDataBytes,vce->curBytes,8);
	addExactData(exactMidByteArray, exactLeadNumArray, resiBitArray, lce);
	listAdd_double(last3CmprsData, vce->data);
	//miss++;
#ifdef HAVE_TIMECMPR
	if(confparams_cpr->szMode == SZ_TEMPORAL_COMPRESSION)
		decData[0] = vce->data;
#endif

	//add the second data
	type[1] = 0;
	compressSingleDoubleValue_MSST19(vce, spaceFillingValue[1], realPrecision, reqLength, reqBytesLength, resiBitsLength);
	updateLossyCompElement_Double(vce->curBytes, preDataBytes, reqBytesLength, resiBitsLength, lce);
	memcpy(preDataBytes,vce->curBytes,8);
	addExactData(exactMidByteArray, exactLeadNumArray, resiBitArray, lce);
	listAdd_double(last3CmprsData, vce->data);
	//miss++;
#ifdef HAVE_TIMECMPR
	if(confparams_cpr->szMode == SZ_TEMPORAL_COMPRESSION)
		decData[1] = vce->data;
#endif
	int state;
	//double checkRadius;
	double curData;
	double pred = vce->data;

    double predRelErrRatio;

	const uint64_t top = levelTable.topIndex, base = levelTable.baseIndex;
	const uint64_t range = top - base;
	const int bits = levelTable.bits;
	uint64_t* const buffer = (uint64_t*)&predRelErrRatio;
	const int shift = 52-bits;
	uint64_t expoIndex, mantiIndex;
	uint16_t* tables[range+1];
	for(int i=0; i<=range; i++){
		tables[i] = levelTable.subTables[i].table;
	}

	for(i=2;i<dataLength;i++)
	{
		curData = spaceFillingValue[i];
		predRelErrRatio = curData / pred;

		expoIndex = ((*buffer & 0x7fffffffffffffff) >> 52) - base;
		if(expoIndex <= range){
			mantiIndex = (*buffer & 0x000fffffffffffff) >> shift;
			state = tables[expoIndex][mantiIndex];
		}else{
			state = 0;
		}

		if(state)
		{
			type[i] = state;
			pred *= precisionTable[state];
			//hit++;
			continue;
		}

		//unpredictable data processing
		type[i] = 0;
		compressSingleDoubleValue_MSST19(vce, curData, realPrecision, reqLength, reqBytesLength, resiBitsLength);
		updateLossyCompElement_Double(vce->curBytes, preDataBytes, reqBytesLength, resiBitsLength, lce);
		memcpy(preDataBytes,vce->curBytes,8);
		addExactData(exactMidByteArray, exactLeadNumArray, resiBitArray, lce);
		pred =  vce->data;
		//miss++;
#ifdef HAVE_TIMECMPR
		if(confparams_cpr->szMode == SZ_TEMPORAL_COMPRESSION)
			decData[i] = vce->data;
#endif

	}//end of for

//	printf("miss:%d, hit:%d\n", miss, hit);

	size_t exactDataNum = exactLeadNumArray->size;

	TightDataPointStorageD* tdps;

	new_TightDataPointStorageD(&tdps, dataLength, exactDataNum,
			type, exactMidByteArray->array, exactMidByteArray->size,
			exactLeadNumArray->array,
			resiBitArray->array, resiBitArray->size,
			resiBitsLength,
			realPrecision, medianValue, (char)reqLength, quantization_intervals, NULL, 0, 0);
    tdps->plus_bits = confparams_cpr->plus_bits;

	//free memory
	free_DIA(exactLeadNumArray);
	free_DIA(resiBitArray);
	free(type);
	free(vce);
	free(lce);
	free(exactMidByteArray); //exactMidByteArray->array has been released in free_TightDataPointStorageF(tdps);
	free(precisionTable);
	freeTopLevelTableWideInterval(&levelTable);
	return tdps;
}


void SZ_compress_args_double_NoCkRngeNoGzip_1D_pwr_pre_log_MSST19(unsigned char** newByteData, double *oriData, double pwrErrRatio, size_t dataLength, size_t *outSize, double valueRangeSize, double medianValue_f,
																unsigned char* signs, bool* positive, double min, double max, double nearZero){
	double multiplier = pow((1+pwrErrRatio), -3.0001);
	for(int i=0; i<dataLength; i++){
		if(oriData[i] == 0){
			oriData[i] = nearZero * multiplier;
		}
	}

	double median_log = sqrt(fabs(nearZero * max));

	TightDataPointStorageD* tdps = SZ_compress_double_1D_MDQ_MSST19(oriData, dataLength, pwrErrRatio, valueRangeSize, median_log);

	tdps->minLogValue = nearZero / ((1+pwrErrRatio)*(1+pwrErrRatio));
	if(!(*positive)){
		unsigned char * comp_signs;
		// compress signs
		unsigned long signSize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, signs, dataLength, &comp_signs);
		tdps->pwrErrBoundBytes = comp_signs;
		tdps->pwrErrBoundBytes_size = signSize;
	}
	else{
		tdps->pwrErrBoundBytes = NULL;
		tdps->pwrErrBoundBytes_size = 0;
	}
	free(signs);

	convertTDPStoFlatBytes_double(tdps, newByteData, outSize);
	if(*outSize>3 + MetaDataByteLength + exe_params->SZ_SIZE_TYPE + 1 + sizeof(double)*dataLength)
		SZ_compress_args_double_StoreOriData(oriData, dataLength, newByteData, outSize);

	free_TightDataPointStorageD(tdps);
}

double computeRangeSize_double_MSST19(double* oriData, size_t size, double* valueRangeSize, double* medianValue, unsigned char * signs, bool* positive, double* nearZero)
{
    size_t i = 0;
    double min = oriData[0];
    double max = min;
    *nearZero = min;

    for(i=1;i<size;i++)
    {
        double data = oriData[i];
        if(data <0){
            signs[i] = 1;
            *positive = false;
        }
        if(oriData[i] != 0 && fabs(oriData[i]) < fabs(*nearZero)){
            *nearZero = oriData[i];
        }
        if(min>data)
            min = data;
        else if(max<data)
            max = data;
    }

    *valueRangeSize = max - min;
    *medianValue = min + *valueRangeSize/2;
    return min;
}