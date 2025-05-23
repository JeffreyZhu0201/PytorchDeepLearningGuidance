import numpy

def matrixmul(a:list[list[int|float]],
			  b:list[list[int|float]])-> list[list[int|float]]:
	a = numpy.array(a)
	b = numpy.array(b)
	
	if a.shape[1] != b.shape[0]:
		return -1

	return a@b
