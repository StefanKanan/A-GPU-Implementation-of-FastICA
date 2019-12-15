import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
#####
# from numba import cuda
import pyculib
#####
import pycuda.gpuarray as gpuarray
from pycuda.cumath import tanh
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel
#####
import skcuda.linalg as gpuLinalg
from skcuda.misc import sum as gpuSum
from skcuda.linalg import inv as invGPU
from skcuda.linalg import diag
from skcuda.misc import min as gpuMin
from skcuda.misc import max as gpuMax
gpuLinalg.init()

#####################################
#Part I - Set tiles
#####################################

Tile = 32
Tile2 = 1024
TileMat = 32

####################################
#Part II - Functions
####################################

mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void MatMulABT(float *A, float *B, float *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
  {
    
    __shared__ float SA[TILE][TILE];
    __shared__ float SB[TILE][TILE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int Row = blockIdx.y*TILE + ty; //which block + TILE for index + threadId
    int Col = blockIdx.x*TILE + tx;
    
    float sum = 0;
    
    for (int i=0; i < (TILE + ACols - 1)/TILE; i++) {
        if (i*TILE + tx < ACols && Row < ARows)
            SA[ty][tx] = A[Row*ACols + i*TILE + tx];
        else
            SA[ty][tx] = 0.0;
          
        if (i*TILE + ty < BCols && Col < BRows)
            SB[ty][tx] = B[Col*BCols + i*TILE + ty];
        else
            SB[ty][tx] = 0.0;
        
        __syncthreads();
        
        for (int k=0; k < TILE; k++)
            sum += SA[ty][k]*SB[k][tx];
        
        __syncthreads();
    }
    
    int R = (blockIdx.y*blockDim.y + threadIdx.y)*CCols;
    int Co = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (Row < CRows && Col < CCols)
        C[Co + R] = sum;
  }
  """ % {'TILE_SIZE': TileMat})
MatMulABT = mod.get_function("MatMulABT")


mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void MatMulATB(float *A, float *B, float *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
  {
    
    __shared__ float SA[TILE][TILE];
    __shared__ float SB[TILE][TILE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int ystep = blockDim.y;
    int ylim = ACols/TILE; //Check how threads are managed
    
    int Col = blockIdx.x*TILE + tx;
        
    for (int j=0; j < ylim; j+=ystep) {
    
    int Row = (blockIdx.y + j)*TILE + ty; //which block + TILE for index + threadId

    float sum = 0;

    for (int i=0; i < (TILE + ARows - 1)/TILE; i++) {
        if (i*TILE + tx < ARows && Row < ACols)
            SA[ty][tx] = A[(i*TILE + tx)*ACols + Row]; //tx is used for rows, check how you manage threads //Too many columns for Row -> ty
        else
            SA[ty][tx] = 0.0;

        if (i*TILE + ty < BRows && Col < BCols)
            SB[ty][tx] = B[(i*TILE + ty)*BCols + Col]; //Col should use ty
        else
            SB[ty][tx] = 0.0;

        __syncthreads();

        for (int k=0; k < TILE; k++)
            sum += SA[ty][k]*SB[k][tx];

        __syncthreads();
    }
    
    int R = (blockIdx.y*blockDim.y + threadIdx.y)*CCols;
    int Co = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (Row < CRows && Col < CCols)
        C[Co + R] = sum;
    
    }
  }
  """ % {'TILE_SIZE': TileMat})
MatMulATB = mod.get_function("MatMulATB")

mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void MatMulATB(float *A, float *B, float *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
  {
    
    __shared__ float SA[TILE][TILE];
    __shared__ float SB[TILE][TILE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int Row = blockIdx.y*TILE + ty; //which block + TILE for index + threadId
    int Col = blockIdx.x*TILE + tx;
    
    float sum = 0;
    
    for (int i=0; i < (TILE + ARows - 1)/TILE; i++) {
        if (i*TILE + tx < ARows && Row < ACols)
            SA[ty][tx] = A[(i*TILE + tx)*ACols + Row];
        else
            SA[ty][tx] = 0.0;
          
        if (i*TILE + ty < BRows && Col < BCols)
            SB[ty][tx] = B[(i*TILE + ty)*BCols + Col];
        else
            SB[ty][tx] = 0.0;
        
        __syncthreads();
        
        for (int k=0; k < TILE; k++)
            sum += SA[ty][k]*SB[k][tx];
        
        __syncthreads();
    }
    
    int R = (blockIdx.y*blockDim.y + threadIdx.y)*CCols;
    int Co = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (Row < CRows && Col < CCols)
        C[Co + R] = sum;
  }
  """ % {'TILE_SIZE': TileMat})
MatMulATB = mod.get_function("MatMulATB")

mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void MatMulATB(float *A, float *B, float *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
  {
    
    __shared__ float SA[TILE][TILE];
    __shared__ float SB[TILE][TILE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int Row = blockIdx.x*TILE + tx; //which block + TILE for index + threadId
    int Col = blockIdx.y*TILE + ty;
    
    float sum = 0;
    
    for (int i=0; i < (TILE + ARows - 1)/TILE; i++) {
        if (i*TILE + ty < ARows && Row < ACols)
            SA[tx][ty] = A[(i*TILE + ty)*ACols + Row];
        else
            SA[tx][ty] = 0.0;
          
        if (i*TILE + tx < BRows && Col < BCols)
            SB[tx][ty] = B[(i*TILE + tx)*BCols + Col];
        else
            SB[tx][ty] = 0.0;
        
        __syncthreads();
        
        for (int k=0; k < TILE; k++)
            sum += SA[tx][k]*SB[k][ty];
        
        __syncthreads();
    }
    
    int R = (blockIdx.x*blockDim.x + threadIdx.x)*CCols;
    int Co = blockIdx.y*blockDim.y + threadIdx.y;
    
    if (Row < CRows && Col < CCols)
        C[Co + R] = sum;
  }
  """ % {'TILE_SIZE': TileMat})
MatMulATB = mod.get_function("MatMulATB")

mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void MatMul(float *A, float *B, float *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols)
  {
    
    __shared__ float SA[TILE][TILE];
    __shared__ float SB[TILE][TILE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int Row = blockIdx.y*TILE + ty; //which block + TILE for index + threadId
    int Col = blockIdx.x*TILE + tx;
    
    float sum = 0;
    
    for (int i=0; i < (TILE + ACols - 1)/TILE; i++) {
        if (i*TILE + tx < ACols && Row < ARows)
            SA[ty][tx] = A[Row*ACols + i*TILE + tx];
        else
            SA[ty][tx] = 0.0;
          
        if (i*TILE + ty < BRows && Col < BCols)
            SB[ty][tx] = B[(i*TILE + ty)*BCols + Col];
        else
            SB[ty][tx] = 0.0;
        
        __syncthreads();
        
        for (int k=0; k < TILE; k++)
            sum += SA[ty][k]*SB[k][tx];
        
        __syncthreads();
    }
    
    int R = (blockIdx.y*blockDim.y + threadIdx.y)*CCols;
    int Co = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (Row < CRows && Col < CCols)
        C[Co + R] = sum;
  }
  """ % {'TILE_SIZE': TileMat})
MatMul = mod.get_function("MatMul")


mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void sumCol(float *x, float *y, int rows, int cols)
  {
    
    __shared__ float SA[TILE][TILE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int Row = blockIdx.y*blockDim.y + ty;
    int Col = blockIdx.x*blockDim.x + tx;
    
    if (Row < rows && Col < cols)
        SA[ty][tx] = y[Row*cols + Col];
    else
        SA[ty][tx] = 0.0;
    __syncthreads();
    
    if (Row < rows && Col < cols) {
        for (unsigned int i=1; i<blockDim.y; i*=2) {
            if (ty %% (2*i) == 0)
                SA[ty][tx] += SA[ty + i][tx];
        }
        
        if (ty == 0)
            x[Col + blockIdx.y*cols] = SA[0][tx];
    }
  }
  """ % {'TILE_SIZE': Tile})
sumCol = mod.get_function("sumCol")

#Newer version
mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void sumCol(float *x, float *y, int rows, int cols)
  {
    
    __shared__ float SA[TILE][TILE];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int Row = blockIdx.x*blockDim.x + tx;
    int Col = blockIdx.y*blockDim.y + ty;
    
    if (Row < rows && Col < cols)
        SA[tx][ty] = y[Row*cols + Col];
    else
        SA[tx][ty] = 0.0;
    __syncthreads();
    
    if (Row < rows && Col < cols) {
        for (unsigned int i=1; i<blockDim.x; i*=2) {
            if (tx %% (2*i) == 0)
                SA[tx][ty] += SA[tx + i][ty];
        }
        
        if (tx == 0)
            x[Col + blockIdx.x*cols] = SA[0][ty];
    }
  }
  """ % {'TILE_SIZE': Tile})
sumCol = mod.get_function("sumCol")

mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void Max(float *x, float *o_data, int n)
  {
    __shared__ float sdata[TILE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
        sdata[tid] = x[i];
    else
        sdata[tid] = 0.0;
    __syncthreads();
    
    for (unsigned int s=1; s < blockDim.x; s*=2) {
        if (tid %% (2*s) == 0)
            sdata[tid] = max(sdata[tid], sdata[tid + s] );
        __syncthreads();
    }
    
    if (tid == 0)
        o_data[blockIdx.x] = sdata[0];
  }
  """ % {'TILE_SIZE': Tile2})
gpuMaxc = mod.get_function("Max")

mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void Min(float *x, float *o_data, int n)
  {
    __shared__ float sdata[TILE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i < n)
        sdata[tid] = x[i];
    else
        sdata[tid] = 0.0;
    __syncthreads();
    
    for (unsigned int s=1; s < blockDim.x; s*=2) {
        if (tid %% (2*s) == 0)
            sdata[tid] = max(sdata[tid], sdata[tid + s] );
        __syncthreads();
    }
    
    if (tid == 0)
        o_data[blockIdx.x] = sdata[0];
  }
  """ % {'TILE_SIZE': Tile2})
gpuMinc = mod.get_function("Min")

mod = SourceModule("""
  
  __global__ void absSub(float *x, float *y, float *c, int n)
  {
    
    for (int i = threadIdx.y + blockIdx.y*blockDim.y; i < n; i += blockDim.y * gridDim.y) {
        for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < n; j += blockDim.x * gridDim.x) {
            x[j + n*i] = abs( abs(y[j + n*i]) - abs(c[j + n*i]) );
        }
    }
  }
  """)
absSub = mod.get_function("absSub")

mod = SourceModule("""
  
  __global__ void diag(float *x, float *y, int rows, int cols)
  {
    for (int j = threadIdx.x + blockIdx.x*blockDim.x; j + j*cols < rows*cols; j += blockDim.x * gridDim.x) {
            x[j] = y[j + j*cols];
    }
  }
  """)
gpuDiag = mod.get_function("diag")

#################################################
#Part III - Set tile
#################################################

Tile3 = 1024


#################################################
#Part IV - Functions
#################################################

"""1 - hypTan*hypTan"""
#Only uses X dim
mod = SourceModule("""
  __global__ void elementWise(float *x, int n)
  {
    float store = 0;
    for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        store = x[i];
        x[i] = 1 - store*store;//x[i]*x[i];
    }
  }
  """)
elementWise = mod.get_function("elementWise")

#IMPORTANT: SET TILE SIZE
"""ONES(DIM, DIM)*RowSum * C"""
mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void MatVecMulRow(float *x, float *y, int n)
  {
    __shared__ float Vec[TILE];
    
    for (int i = threadIdx.y + blockIdx.y*blockDim.y; i < n; i += blockDim.y * gridDim.y) {
        Vec[threadIdx.y] = y[i];
        __syncthreads();
        for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < n; j += blockDim.x * gridDim.x)
            x[j + n*i] = x[j + n*i]*Vec[threadIdx.y];
    }
  }
  """ % {'TILE_SIZE': Tile3})

mod = SourceModule("""

  #define TILE %(TILE_SIZE)d
  
  __global__ void MatVecMul(float *x, float *y, int n)
  {
    
    //__shared__ float Vec[TILE];
    
    //int ty = threadIdx.y;
    //int tx = threadIdx.x;
    
    for (int i = threadIdx.y + blockIdx.y*blockDim.y; i < n; i += blockDim.y * gridDim.y) {
        for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < n; j += blockDim.x * gridDim.x) {
            //Vec[tx] = y[j]
            x[j + n*i] = x[j + n*i]*y[j];
        }
    }
  }
  """ % {'TILE_SIZE': Tile3})
MatVecMul = mod.get_function("MatVecMul")

mod = SourceModule("""
  
  __global__ void Sub(float *x, float *y, float *c, int n)
  {
    
    for (int i = threadIdx.y + blockIdx.y*blockDim.y; i < n; i += blockDim.y * gridDim.y) {
        for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < n; j += blockDim.x * gridDim.x) {
            x[j + n*i] = y[j + n*i] - c[j + n*i];
        }
    }
  }
  """)
Sub = mod.get_function("Sub")

mod = SourceModule("""
  
  __global__ void Div(float *x, int y, int n)
  {
    
    for (int i = threadIdx.y + blockIdx.y*blockDim.y; i < n; i += blockDim.y * gridDim.y) {
        for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < n; j += blockDim.x * gridDim.x) {
            x[j + n*i] = x[j + n*i]/y;
        }
    }
  }
  """)
Div = mod.get_function("Div")

mod = SourceModule("""
  
  __global__ void Copy(float *x, float *y, int n)
  {
    
    for (int i = threadIdx.y + blockIdx.y*blockDim.y; i < n; i += blockDim.y * gridDim.y) {
        for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < n; j += blockDim.x * gridDim.x) {
            x[j + n*i] = y[j + n*i];
        }
    }
  }
  """)
Copy = mod.get_function("Copy")


mod = SourceModule("""
  
  __global__ void Mul(float *x, float *y, float z, int n)
  {
    
    for (int i = threadIdx.y + blockIdx.y*blockDim.y; i < n; i += blockDim.y * gridDim.y) {
        for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < n; j += blockDim.x * gridDim.x) {
            x[j + n*i] = y[j + n*i]*z;
        }
    }
  }
  """)
Mul = mod.get_function("Mul")

mod = SourceModule("""
  
  __global__ void GPUtanh(float *x, int cols, int rows)
  {
    
    for (int i = threadIdx.y + blockIdx.y*blockDim.y; i < rows; i += blockDim.y * gridDim.y) {
        for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < cols; j += blockDim.x * gridDim.x) {
            x[j + cols*i] = tanh(x[j + cols*i]);
        }
    }
  }
  """)
gpuTanh = mod.get_function("GPUtanh")

mod = SourceModule("""
  
  __global__ void CopyRow(float *x, float *y, int n)
  {
    
    for (int j = threadIdx.x + blockIdx.x*blockDim.x; j < n; j += blockDim.x * gridDim.x) {
        x[j] = y[j];
    }
  }
  """)
CopyRow = mod.get_function("CopyRow")

compareGpu = ReductionKernel(np.float32, neutral="0",
        reduce_expr="max(a, b)", map_expr="abs( abs(x[i]) - abs(y[i]) )",
        arguments="float *x, float *y")

def gpuMatMul(A, B, C, transa='n', transb='n', block=(32, 32, 1)):
    block = (TileMat, TileMat, 1)
    
    transa = transa.lower()
    transb = transb.lower()
    bx, by, bz = block
    Arow, Acol = A.shape
    Brow, Bcol = B.shape
    if bx > Tile:
        bx = Tile
    if by > Tile:
        by = Tile
    block = (bx, by, bz)
    
    if transa == 'n' and transb == 'n':
        if Acol != Brow:
            raise Exception('Not aligned, column in A: {} with row in B: {}'.format(Acol, Brow))
        MatMul(A, B, C, np.int32(A.shape[0]), np.int32(A.shape[1]),
              np.int32(B.shape[0]), np.int32(B.shape[1]), np.int32(C.shape[0]), np.int32(C.shape[1]),
              block=block, grid=(int(np.ceil(B.shape[1]/block[0])), int(np.ceil(A.shape[0]/block[1])), 1))
    elif transa == 't' and transb == 'n':
        if Arow != Brow:
            raise Exception('Not aligned, column in A: {} with row in B: {}'.format(Arow, Brow))
        MatMulATB(A, B, C, 
                  np.int32(A.shape[0]), np.int32(A.shape[1]), #ARow, ACol
                  np.int32(B.shape[0]), np.int32(B.shape[1]), np.int32(C.shape[0]), np.int32(C.shape[1]),
                  block=block,
                  grid=
#                       (int(np.ceil(B.shape[1]/block[0])), int(np.ceil(A.shape[1]/block[1])), 1)) #grid
                      (int(np.ceil(A.shape[1]/block[0])), int(np.ceil(B.shape[1]/block[1])), 1)) #grid
    elif transa == 'n' and transb == 't':
        if Acol != Bcol:
            raise Exception('Not aligned, column in A: {} with row in B: {}'.format(Acol, Bcol))
        MatMulABT(A, B, C, np.int32(A.shape[0]), np.int32(A.shape[1]),
              np.int32(B.shape[0]), np.int32(B.shape[1]), np.int32(C.shape[0]), np.int32(C.shape[1]),
              block=block, grid=(int(np.ceil(B.shape[0]/block[0])), int(np.ceil(A.shape[0]/block[1])), 1))
        
def gpuSumCol(A, B, block, C):
    block = (Tile, Tile, 1)
    
    row, col = A.shape
    x, y, z = block
    bx = int(np.ceil(col/x))
    by = int(np.ceil(row/y)) #Huge amount

    bRow, bCol = B.shape
    if bRow < by:
        by = bRow
        print('Not enough rows in B. Fewer rows will be summed.')
    if bCol < bx:
        bx = bCol
        print('Not enough columns in B. Fewer columns will be summed.')
    grid = (bx, by, 1)
    
    sumCol(B, A, np.int32(row), np.int32(col), block=block, grid=grid ) #Error
    
    while by > 1:
        row = by
        by = int(np.ceil(by/y))
        grid = (bx, by, 1)
        
        sumCol(B, B, np.int32(row), np.int32(col), block=block, grid=grid )
    
    block = (x*y, 1, 1)
    bx = int(np.ceil(col/(x*y)))
    grid = (bx, 1, 1)
    
    CopyRow(C, B, np.int32(col), block=block, grid=grid)

#Newer version
def gpuSumCol(A, B, block, C):
    block = (Tile, Tile, 1)
    
    row, col = A.shape
    x, y, z = block
    bx = int(np.ceil(col/y))
    by = int(np.ceil(row/x)) #Huge amount

    bRow, bCol = B.shape
    if bRow < by:
        by = bRow
        print('Not enough rows in B. Fewer rows will be summed.')
    if bCol < bx:
        bx = bCol
        print('Not enough columns in B. Fewer columns will be summed.')
    grid = (by, bx, 1)
    
    sumCol(B, A, np.int32(row), np.int32(col), block=block, grid=grid ) #Error
    
    while by > 1:
        row = by
        by = int(np.ceil(by/x))
        grid = (by, bx, 1)
        
        sumCol(B, B, np.int32(row), np.int32(col), block=block, grid=grid )
    
    block = (x*y, 1, 1)
    bx = int(np.ceil(col/(x*y)))
    grid = (bx, 1, 1)
    
    CopyRow(C, B, np.int32(col), block=block, grid=grid)

#todo
def findMax(A, block):
    block = (Tile2, 1, 1)
    
    if len(A.shape) > 1:
        row, col = A.shape
    else:
        row = A.shape[0]
        col = 1
    x, y, z = block
    bx = int(np.ceil(col*row/x))

    grid = (bx, 1, 1)
    B = gpuarray.zeros(bx, np.float32)

    gpuMaxc(A, B, np.int32(row*col), block=block, grid=grid )

    while bx > 1:
        row = bx
        bx = int(np.ceil(bx/x))
        grid = (bx, 1, 1)
        
        gpuMaxc(B, B, np.int32(row), block=block, grid=grid )
    
    block = (1, 1, 1)
    grid = (1, 1, 1)
    
    C = gpuarray.zeros(1, np.float32)
    CopyRow(C, B, np.int32(1), block=block, grid=grid)
    
    return C

def findMin(A, block):
    block = (Tile2, 1, 1)
    
    if len(A.shape) > 1:
        row, col = A.shape
    else:
        row = A.shape[0]
        col = 1
    x, y, z = block
    bx = int(np.ceil(col*row/x))

    grid = (bx, 1, 1)
    B = gpuarray.zeros(bx, np.float32)

    gpuMinc(A, B, np.int32(row*col), block=block, grid=grid )

    while bx > 1:
        row = bx
        bx = int(np.ceil(bx/x))
        grid = (bx, 1, 1)
        
        gpuMinc(B, B, np.int32(row), block=block, grid=grid )
    
    block = (1, 1, 1)
    grid = (1, 1, 1)
    
    C = gpuarray.zeros(1, np.float32)
    CopyRow(C, B, np.int32(1), block=block, grid=grid)
    
    return C

def compareGpuC(A, B, block):
    row, col = A.shape
    C = gpuarray.zeros((row, col), np.float32)
    
    x, y, z = block
    bx = int(np.ceil(col/x))
    by = int(np.ceil(row/y))

    grid = (bx, by, 1)
    
    absSub(C, A, B, block=block, grid=grid)
    
    C = findMax(C, block)
    
    return C

def findDiag(A, diag, block=(32, 1, 1)):
    row, col = A.shape
    row = np.int32(row)
    col = np.int32(col)
    x, y, z = block
    
    bx = int(np.ceil(row/x))
    gpuDiag(diag, A, row, col, block=(32, 1, 1), grid=(bx, 1, 1))
    
    return diag

################################################
#Part V - FastICA
################################################



def PCA(data, firstEig=None, lastEig=None):
    if lastEig is None:
        lastEig = len(data) - 1
    if firstEig is None:
        firstEig = 0

    OldDim = len(data) #Amount of components

    COV = np.cov(data) # Dim x Dim
    D, E = np.linalg.eig(COV)

    rankTolerance = 1e-7
    maxLastEig = np.sum(D > rankTolerance)
    if maxLastEig == 0:
        raise Exception('Eigenvalues of the covariance matrix are all smaller than tolerance \
        Please make sure that your data matrix contains nonzero values. \
        \nIf the values are very small, try rescaling the data matrix.\n')

    maxLastEig = maxLastEig - 1 #for index
    eigenvalues = np.sort(D)[::-1]

    if lastEig > maxLastEig:
        lastEig = maxLastEig

    if lastEig < OldDim-1: #if lastEig essentially changes
        lowerLimitValue = (eigenvalues[lastEig] + eigenvalues[lastEig + 1]) / 2
    else:
        lowerLimitValue = eigenvalues[OldDim - 1] - 1 #It isn't inclusive

    lowerColumns = D > lowerLimitValue

    if firstEig > 0:
        higherLimitValue = (eigenvalues[firstEig - 1] + eigenvalues[firstEig]) / 2
    else:
        higherLimitValue = eigenvalues[0] + 1 #It isn't inclusive

    higherColumns = D < higherLimitValue

    selectedColumns = lowerColumns & higherColumns
    
    E = E[:, selectedColumns]
    D = np.diag(D[selectedColumns]) #Eigenvalues

    return E, D

"""This version of FastICA, copies B computes the square root of the matrix and copies it back to the GPU"""
def FastICASymmSwap(X, whitening, dewhitening, maxIterations, threshold):
    Dim, NumOfSampl = X.shape
    Dim = np.int32(Dim)
#     Dim = len(X)
#     NumOfSampl = len(X[0])

    B = linalg.orth(np.random.random((Dim, Dim))).astype(np.float32) #linalg.orth makes the array non contiguous
    #B.flags['C_CONTIGUOUS']
    
    print(B)
    B_gpu = gpuarray.to_gpu(np.ascontiguousarray(B, np.float32))
    #Bold
    Bold_gpu = gpuarray.zeros((Dim, Dim), np.float32)
    Bold = np.zeros((Dim, Dim))
    #W
    A = np.zeros((Dim, Dim)) #maybe dtype
    #CTC
    CTC_gpu = gpuarray.zeros((Dim, Dim), np.float32)
    #X
    X_gpu = gpuarray.to_gpu(X.astype(np.float32))
    #hypTan
    hypTan_gpu = gpuarray.zeros((NumOfSampl, Dim), np.float32)
    #rowSum
    rowSum_gpu = gpuarray.zeros(Dim, np.float32)
    #minAbsCos
    minAbsCos_gpu = gpuarray.zeros((Dim, Dim), np.float32)
    
    #helpers
    sqrt = linalg.sqrtm  # sqrt on a matrix
    inv = np.linalg.inv
    eig = linalg.eig

    for i in range(0, maxIterations + 1):
        print(i, maxIterations)
        if i == maxIterations:
            print('Component {} did not converge after {} iterations'.format(i, maxIterations))
            B = B_gpu.get()
            if B.size != 0: #not empty

                B = B @ np.real(inv(sqrt(B.T @ B)))
                W = B.T @ whitening
                A = dewhitening @ B

                print('A:\n', A)
                print('W:\n', W)
                
                return A, W
        gpuLinalg.dot(B_gpu, B_gpu, transa='T', out=CTC_gpu) #WORKS! #B@B.T
        CTC = CTC_gpu.get()
        
        D, E = eig(CTC) #CHECK IF IMAGINARY

        D = np.sqrt(abs(D)) #We abs all the negative eigenvalues
        D_gpu = gpuarray.to_gpu(np.ascontiguousarray(np.diag(D)))
        E_gpu = gpuarray.to_gpu(np.ascontiguousarray(E))
        Einv_gpu =invGPU(E_gpu)
        
        gpuLinalg.dot(E_gpu, D_gpu, out=CTC_gpu) #WORKS!
        gpuLinalg.dot(CTC_gpu, Einv_gpu, out=CTC_gpu) #WORKS!
        
#Manual CPU
#         CTC = E@np.diag(D)@np.linalg.inv(E)
#         CTC_gpu.set(CTC.astype(np.float32))

#Direct
#         CTC = sqrt(CTC)
#         CTC_gpu.set(CTC)

        invGPU(CTC_gpu, overwrite=True)
        CTC_gpu = CTC_gpu.real
        
        gpuLinalg.dot(B_gpu, CTC_gpu, out=B_gpu) #WORKS! #B@(B@B.T)^-1
        
        gpuLinalg.dot(B_gpu, Bold_gpu, transa='T', out=minAbsCos_gpu) #This sometimes gives different values then C.T @ Bold
        minAbsCos2 = gpuMin(abs(gpuMin(minAbsCos_gpu))).get()
        
#         minAbsCos = min(abs(np.diag(C.T @ Bold)))
        minAbsCos = minAbsCos2[0, 0]
#         print(minAbsCos2)
#         print(1-minAbsCos < threshold)
#         print('IS IT THE SAME:', np.allclose(minAbsCos_gpu.get(), C.T@Bold))
#         print('IS IT THE SAME:', 1 - minAbsCos2[0, 0] < threshold)
        if 1 - minAbsCos < threshold:
            print('here')
            
            C = B_gpu.get()
        
            A = dewhitening @ C
            W = C.T @ whitening
            
            print('A:\n', A)
            print('W:\n', W)
            return A, W
        
        ######################### READ ABOUT DYNAMIC PARALLELISM AND UNIFIED MEMORY
        Copy(Bold_gpu, B_gpu, Dim, block=(4, 4, 1), grid=(1, 1, 1)) #Bold = B
        
        gpuLinalg.dot(X_gpu, B_gpu, transa='T', out=hypTan_gpu) #WORKS! #X.T@B
        tanh(hypTan_gpu, out=hypTan_gpu) #WORKS! #Elementwise
        gpuLinalg.dot(X_gpu, hypTan_gpu, out=CTC_gpu) #WORKS! #X@hypTan

        n = Dim*NumOfSampl
        elementWise(hypTan_gpu, np.int32(n), block=(128, 1, 1), grid=(1,1,1)) #1 - hypTan*hypTan
        
        gpuSum(hypTan_gpu, axis=0, out=rowSum_gpu)
        MatVecMul(B_gpu, rowSum_gpu, Dim, block=(4, 4, 1), grid=(1,1,1))
        
        Sub(B_gpu, CTC_gpu, B_gpu, Dim, block=(4, 4, 1), grid=(1,1,1)) #C = left - right
        Div(B_gpu, np.int32(NumOfSampl), Dim, block=(4, 4, 1), grid=(1,1,1)) #Division by scalar

"""This version of FastICA, uses approximation for B"""
def FastICASymmApro(X, whitening, dewhitening, maxIterations, threshold):
    Threads = 32
    ThreadBlock = (Threads, Threads, 1)
    
    Dim, NumOfSampl = X.shape
    Dim = np.int32(Dim)
    B = linalg.orth(np.random.random((Dim, Dim))).astype(np.float32) #linalg.orth makes the array non contiguous
    #B.flags['C_CONTIGUOUS']
    
    print(B)
    B_gpu = gpuarray.to_gpu(np.ascontiguousarray(B, np.float32))
    #Bold
    Bold_gpu = gpuarray.zeros((Dim, Dim), np.float32)
    Bold = np.zeros((Dim, Dim))
    #W
    A = np.zeros((Dim, Dim)) #maybe dtype
    #CTC
    CTC_gpu = gpuarray.zeros((Dim, Dim), np.float32)
    #hypTan
    hypTan_gpu = gpuarray.zeros((NumOfSampl, Dim), np.float32)
    #rowSum
    row = int(np.ceil(NumOfSampl/Threads))
    Sum_gpu = gpuarray.zeros((row, Dim), np.float32)
    rowSum_gpu = gpuarray.zeros(Dim, np.float32)
    #minAbsCos
    minAbsCos_gpu = gpuarray.zeros((Dim, Dim), np.float32)
    #left, right
    left_gpu = gpuarray.zeros((Dim, Dim), np.float32)
    right_gpu = gpuarray.zeros((Dim, Dim), np.float32)
    #Identity
    I_gpu = gpuarray.to_gpu(np.eye(Dim).astype(np.float32))
    Check_gpu = gpuarray.zeros((Dim, Dim), np.float32)
    #diag
    diag_gpu = gpuarray.zeros(Dim, np.float32)
    
    # start = cuda.Event()
    # end = cuda.Event()
    
    # start.record()
    
    #X
    X_gpu = gpuarray.to_gpu(X.astype(np.float32))
    
    for i in range(0, maxIterations + 1):
#         print(i, maxIterations)
        if i == maxIterations:
            print('Component {} did not converge after {} iterations'.format(i, maxIterations))
            B = B_gpu.get()
            if B.size != 0: #not empty

                B = B @ np.real(inv(sqrt(B.T @ B)))
                W = B.T @ whitening
                A = dewhitening @ B

                print('A:\n', A)
                print('W:\n', W)
                
                return A, W
            return None, None #TODO
        
        f = True
        j = 0
        
        gpuMatMul(B_gpu, B_gpu, CTC_gpu, transb='T')
        gpuSumCol(CTC_gpu, Sum_gpu, ThreadBlock, rowSum_gpu)
        
#         print(np.allclose(rowSum_gpu.get(), gpuSum(CTC_gpu, axis=0).get()))
        norm = findMax(rowSum_gpu, (31, 1, 1))
#         norm = gpuMax(rowSum_gpu)
    
        Div(B_gpu, norm, Dim, block=ThreadBlock, grid=(1,1,1)) #Division by scalar
        #maybe check every 5 iterations
        while f:
            Mul(left_gpu, B_gpu, np.float32(3/2), Dim, block=ThreadBlock, grid=(1,1,1)) #Division by scalar
            
            gpuMatMul(B_gpu, B_gpu, right_gpu, transb='T')
            gpuMatMul(right_gpu, B_gpu, right_gpu)
            Mul(right_gpu, right_gpu, np.float32(1/2), Dim, block=ThreadBlock, grid=(1,1,1)) #Division by scalar
            
            Sub(B_gpu, left_gpu, right_gpu, Dim, block=ThreadBlock, grid=(1,1,1)) #C = left - right
            gpuMatMul(B_gpu, B_gpu, Check_gpu, transb='T')
            
            if j >= 20:
                f = compareGpuC(Check_gpu, I_gpu, ThreadBlock).get()
                f = not f <= threshold
                j = 0
            j += 1
#             j+=1

        gpuMatMul(B_gpu, Bold_gpu, minAbsCos_gpu, transa='T')

#         minAbsCos2 = findMin(abs(findDiag(minAbsCos_gpu, diag_gpu)), (128, 1, 1)).get()
        minAbsCos2 = gpuMin(abs(diag(minAbsCos_gpu))).get()
#         print( abs( diag(minAbsCos_gpu) ) )

        minAbsCos = minAbsCos2[0]
    
        if 1 - minAbsCos < threshold:
            print('Converged!') #TODO
            
                
            # end.record()
            # end.synchronize()

#             secs = start.time_till(end)*1e-3
#             return secs
#             print('Seconds: ', secs)
            
            # C = B_gpu.get()
        
            # A = dewhitening @ C
            # W = C.T @ whitening
            
            # print('A:\n', A)
            # print('W:\n', W)
            # return A, W
        
        Copy(Bold_gpu, B_gpu, Dim, block=ThreadBlock, grid=(1, 1, 1)) #Bold = B
        
        gpuMatMul(X_gpu, B_gpu, hypTan_gpu, transa='T')
        
        n = int( np.ceil(hypTan_gpu.shape[0]/Threads) )
        if n > 65536:
            n = 65535
        
#         n=1
        gpuTanh(hypTan_gpu, np.int32(hypTan_gpu.shape[1]), np.int32(hypTan_gpu.shape[0]),
                block=ThreadBlock, grid=(1, n, 1))
        gpuMatMul(X_gpu, hypTan_gpu, CTC_gpu)
        
        n = Dim*NumOfSampl
        
        m = int( np.ceil(hypTan_gpu.shape[0]/(Threads*Threads) ) )
        if m > 65536:
            m = 65535
#         m = 1
    
        elementWise(hypTan_gpu, np.int32(n), block=(Threads*Threads, 1, 1), grid=(m,1,1)) #1 - hypTan*hypTan
        
        gpuSumCol(hypTan_gpu, Sum_gpu, ThreadBlock, rowSum_gpu)
        MatVecMul(B_gpu, rowSum_gpu, Dim, block=ThreadBlock, grid=(1,1,1))
        
        Sub(B_gpu, CTC_gpu, B_gpu, Dim, block=ThreadBlock, grid=(1,1,1)) #C = left - right
        Div(B_gpu, np.int32(NumOfSampl), Dim, block=ThreadBlock, grid=(1,1,1)) #Division by scalar


#Using g(y) = tanh(y)
def FastICA(mixedsig, approach='Swap', maxIterations=10, threshold=0.0001):
    Dim = len(mixedsig)

    #Center X
    m = np.mean(mixedsig, 1)
    m = m.reshape((Dim, 1))

    mixedmean = mixedsig.copy() - m

    #PCA and whitening
    #E - the eigenvectors, an orthogonal matrix, Dim x Dim-k
    #D - the eigenvalues
    E, D = PCA(mixedsig) #maybe check if D contains negative values
    #todo read theory about mixedmin

    WhiteningMatrix = np.linalg.inv(np.sqrt(D)) @ E.T
    DewhiteningMatrix = E @ np.sqrt(D)

    Whitesig = WhiteningMatrix @ mixedsig
    print(mixedsig.shape)
    print(Whitesig.shape)
    if not np.all(np.isreal(Whitesig)):
        raise Exception('Whitened matrix has imaginary values!') #should this be a warning

    #A - mixing matrix
    #W - demixing matrix
    if approach == 'Swap':
        A, W = FastICASymmSwap(Whitesig, WhiteningMatrix, DewhiteningMatrix, maxIterations, threshold)
    if approach == 'Apro':
        A, W = FastICASymmApro(Whitesig, WhiteningMatrix, DewhiteningMatrix, maxIterations, threshold)
#         time = FastICASymmApro(Whitesig, WhiteningMatrix, DewhiteningMatrix, maxIterations, threshold)
#         %prun FastICASymmApro(Whitesig, WhiteningMatrix, DewhiteningMatrix, maxIterations, threshold)

    return W@(mixedmean + m), A, W
#     return time

######################################################
from scipy.io.wavfile import read, write
import numpy as np

rate, data = read('signals30.wav')
data = data.T.copy()
# data = data.astype(np.float32)
A = np.load('A.npy')

frames = 0

mixedsig = A[:,:]@data#[:, :]

I, AB, W = FastICA(mixedsig[:, :], 'Apro', 100, 0.001)