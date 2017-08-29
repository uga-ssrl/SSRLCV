/*
# Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __NVX_SFM_MATRIX_HPP__
#define __NVX_SFM_MATRIX_HPP__

#include <vector>
#include <tuple>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace mathalgo
{

template<class T>
class matrix
{
public:
    matrix(unsigned int nRows, unsigned int nCols) :
        m_nRows( nRows ),
        m_nCols( nCols ),
        m_oData( nRows*nCols, 0 )
    {
        if ( !nRows || !nCols )
        {
            throw std::range_error( "invalid matrix size" );
        }
    }

    static matrix identity( unsigned int nSize )
    {
        matrix oResult( nSize, nSize );

        int nCount = 0;
        std::generate( oResult.m_oData.begin(), oResult.m_oData.end(),
                       [&nCount, nSize]() { return !(nCount++ % (nSize + 1)); } );

        return oResult;
    }

    inline T operator()(unsigned int nRow, unsigned int nCol) const
    {
        if ( nRow >= m_nRows || nCol >= m_nCols )
        {
            throw std::out_of_range( "position out of range" );
        }

        return m_oData[nCol+m_nCols*nRow];
    }

    inline T& operator()(unsigned int nRow, unsigned int nCol)
    {
        if ( nRow >= m_nRows || nCol >= m_nCols )
        {
            throw std::out_of_range( "position out of range" );
        }

        return m_oData[nCol+m_nCols*nRow];
    }

    inline matrix operator*( const matrix& other) const
    {
        if ( m_nCols != other.m_nRows )
        {
            throw std::domain_error( "matrix dimensions are not multiplicable" );
        }

        matrix oResult( m_nRows, other.m_nCols );
        for ( unsigned int r = 0; r < m_nRows; ++r )
        {
            for ( unsigned int ocol = 0; ocol < other.m_nCols; ++ocol )
            {
                for ( unsigned int c = 0; c < m_nCols; ++c )
                {
                    oResult(r,ocol) += (*this)(r,c) * other(c,ocol);
                }
            }
        }

        return oResult;
    }

    inline matrix transpose() const
    {
        matrix oResult( m_nCols, m_nRows );
        for ( unsigned int r = 0; r < m_nRows; ++r )
        {
            for ( unsigned int c = 0; c < m_nCols; ++c )
            {
                oResult(c,r) += (*this)(r,c);
            }
        }
        return oResult;
    }

    inline unsigned int rows() const
    {
        return m_nRows;
    }

    inline unsigned int cols() const
    {
        return m_nCols;
    }

    inline std::vector<T> data() const
    {
        return m_oData;
    }

    void print() const
    {
        for ( unsigned int r = 0; r < m_nRows; r++ )
        {
            for ( unsigned int c = 0; c < m_nCols; c++ )
            {
                std::cout << (*this)(r,c) << "\t";
            }
            std::cout << std::endl;
        }
    }

private:
    unsigned int m_nRows;
    unsigned int m_nCols;
    std::vector<T> m_oData;
};

}

#endif
