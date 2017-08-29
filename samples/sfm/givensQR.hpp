/*
# Copyright (c) 2014-2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __NVX_GIVENS_QR_HPP__
#define __NVX_GIVENS_QR_HPP__

#include "matrix.hpp"
#include <stdexcept>
#include <math.h>
#include <algorithm>

namespace mathalgo
{
template<typename T>
class Givens
{
public:
    Givens() : m_oQ(1,1), m_oR(1,1), m_oJ(2,2)
    {
    }

    //Calculate the inverse of a matrix using the QR decomposition.
    // param: A - matrix to inverse
    matrix<T> Inverse( const matrix<T>& oMatrix )
    {
        if ( oMatrix.cols() != oMatrix.rows() )
        {
            throw std::domain_error( "matrix has to be square" );
        }
        matrix<T> oIdentity = matrix<T>::identity( oMatrix.rows() );
        Decompose( oMatrix );
        return Solve( oIdentity );
    }

    //Performs QR factorization using Givens rotations.
    void Decompose( const matrix<T>& oMatrix )
    {
        int nRows = oMatrix.rows();
        int nCols = oMatrix.cols();

        if ( nRows == nCols )
        {
            nCols--;
        }
        else if ( nRows < nCols )
        {
            nCols = nRows - 1;
        }

        m_oQ = matrix<T>::identity(nRows);
        m_oR = oMatrix;

        for ( int j = 0; j < nCols; j++ )
        {
            for ( int i = j + 1; i < nRows; i++ )
            {
                GivensRotation( m_oR(j,j), m_oR(i,j) );
                PreMultiplyGivens( m_oR, j, i );
                PreMultiplyGivens( m_oQ, j, i );
            }
        }

        m_oQ = m_oQ.transpose();
    }

    //Find the solution for a matrix.
    //http://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems
    matrix<T> Solve( const matrix<T>& oMatrix )
    {
        matrix<T> oQtM( m_oQ.transpose() * oMatrix );
        int nCols = m_oR.cols();
        matrix<T> oS( 1, nCols );
        for (int i = nCols-1; i >= 0; i-- )
        {
            oS(0,i) = oQtM(i, 0);
            for ( int j = i + 1; j < nCols; j++ )
            {
                oS(0,i) -= oS(0,j) * m_oR(i, j);
            }
            oS(0,i) /= m_oR(i, i);
        }

        return oS;
    }

    const matrix<T>& GetQ()
    {
        return m_oQ;
    }

    const matrix<T>& GetR()
    {
        return m_oR;
    }

private:

    //Givens rotation is a rotation in the plane spanned by two coordinates axes.
    //http://en.wikipedia.org/wiki/Givens_rotation
    void GivensRotation( T a, T b )
    {
        T t,s,c;
        if (b == 0)
        {
            c = (a >= 0) ? T(1) : T(-1);
            s = 0;
        }
        else if (a == 0)
        {
            c = 0;
            s = (b >= 0) ? T(-1) : T(1);
        }
        else if (std::abs(b) > std::abs(a))
        {
            t = a/b;
            s = -1/sqrt(1+t*t);
            c = -s*t;
        }
        else
        {
            t = b/a;
            c = 1/sqrt(1+t*t);
            s = -c*t;
        }
        m_oJ(0,0) = c; m_oJ(0,1) = -s;
        m_oJ(1,0) = s; m_oJ(1,1) = c;
    }

    //Get the premultiplication of a given matrix
    //by the Givens rotation.
    void PreMultiplyGivens( matrix<T>& oMatrix, int i, int j )
    {
        int nRowSize = oMatrix.cols();

        for ( int nRow = 0; nRow < nRowSize; nRow++ )
        {
            T nTemp = oMatrix(i,nRow) * m_oJ(0,0) + oMatrix(j,nRow) * m_oJ(0,1);
            oMatrix(j,nRow) = oMatrix(i,nRow) * m_oJ(1,0) + oMatrix(j,nRow) * m_oJ(1,1);
            oMatrix(i,nRow) = nTemp;
        }
    }

private:
    matrix<T> m_oQ, m_oR, m_oJ;
};

}

#endif
