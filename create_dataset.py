
import extract_features as ef
import numpy as np

with open('./input.csv','r',encoding='utf-8') as input_file:
	with open('./dataset.csv','w',encoding='utf-8') as dataset:
		for line in input_file:
			r       = line.split(',')
			x       = r[0].strip()
			y       = r[1].strip()

			example = ef.extractFeatures(x)
			result  = '{0},{1}\n'.format(
				np.array2string(example, separator=','),
				y
			)
			result = result.replace('[','')
			result = result.replace(']','')
			result = result.replace(' ','')
			dataset.write(result)
