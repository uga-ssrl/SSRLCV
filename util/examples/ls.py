from copy import deepcopy as cp

# Note all matrices of the form  [[row1],...,[rown]] 

# Matrix Scaling
def s(a,A):
	matrix = cp(A)
	for row in range(len(matrix)):
		for col in range(len(matrix[0])):
			matrix[row][col] = float(a)*matrix[row][col]
	return matrix

# Matrix Addition
def add(A,B):
	matrix = cp(A)
	if len(A) != len(B) or len(A[0]) != len(B[0]):
		return "Dimensions not equal"
	for row in range(len(matrix)):
		for col in range(len(matrix[0])):
			matrix[row][col] = A[row][col]+B[row][col]
	return matrix

# Matrix Multiplication
def mult(A,B):
	mat1 = cp(A)
	mat2 = cp(B)

	if len(mat1[0]) != len(mat2):
		return "Dimensions not equal"

	final = []
	for row in range(len(mat1)):
		temp = []
		for col in range(len(mat2[0])):
			summ = 0
			for mult in range(len(mat2)):
				summ += mat1[row][mult] * mat2[mult][col]
			temp.append(summ)
		final.append(temp)
	return final

# Transpose
def t(A):
	matrix = cp(A)
	final = []
	for col in range(len(matrix[0])):
		temp = []
		for row in range(len(matrix)):
			temp.append(matrix[row][col])
		final.append(temp)
	return final

def selectCol(A,col,row_start):
	matrix = cp(A)
	final = []
	
	for row in range(row_start,len(matrix)):
		final.append([matrix[row][col]])
	return final

def eDist(vect):
 	mat = cp(vect)
	summ = 0
	for a in mat:
		summ += a[0]**2
	return summ**.5

def I(size):
	final = []
	for row in range(size):
		temp = []
		for col in range(size):
			if row == col:
				temp.append(1.0)
			else : 
				temp.append(0.0)
		final.append(temp)
	return final

def resize(A,num_rows,num_cols):
	matrix = cp(A)
	nrows = len(matrix)
	ncols = len(matrix[0])
	final = []

	for row in range(num_rows):
		temp = []
		for col in range(num_cols):
			row_i = row - (num_rows-nrows)
			col_i = col - (num_cols-ncols)

			if row_i >= 0 and col_i >= 0:
				temp.append(matrix[row_i][col_i])
			elif row == col:
				temp.append(1)
			else :
				temp.append(0)
		final.append(temp)
	return final

def QR(A):
	mat = cp(A)
	U = []

	for k in range(len(mat[0])):
		a1 = selectCol(mat,k,k)
		beta = -1*(a1[0][0]/abs(a1[0][0]))*eDist(a1)
		e1 = [[1]]
		for k1 in range(1,len(a1)):
			e1.append([0])
		y = add(a1,s(-1*beta,e1))
		alpha = (2**.5)/eDist(y)
		v = s(alpha,y)
		u = add(I(len(v)),s(-1,mult(v,t(v))))
		u = resize(u,len(mat),len(mat))
		U.append(resize(u,len(mat),len(mat)))
		mat = mult(u,mat)

	# Everything has been calculated
	R = cp(mat)
	Qs = cp(U[len(U)-1])
	for i in range(1,len(U)):
		Qs = mult(Qs,U[len(U)-1-i])

	Q = t(Qs)

	return [Q,R]

def upperTriangleSolve(T,B):
	solutions = [] 
	for i in range(len(T[0])):
		solutions.append(0)

	num_solutions = len(T[0])
	for row in range(num_solutions):
		current_row = len(T[0])-1-row
		summ = 0
		for col in range(row+1):
			current_col = len(T[0])-1-col
			if col == row:
				sol = (B[current_row][0]-summ)/T[current_row][current_col]
				solutions[current_row] = [sol]
			else:
				summ += T[current_row][current_col]*solutions[current_col][0]
	return solutions

def solve(A,B):
	if len(A) != len(B):
		return "incorrect data"
	R,Q = QR(A)
	# Solving Ax = QRx = B => Rx = QsB =y => Rx = y 
	y = mult(t(Q),B)
	
	Solution = upperTriangleSolve(R,y)
	return Solution
