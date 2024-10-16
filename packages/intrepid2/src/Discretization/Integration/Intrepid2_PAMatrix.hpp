// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_PAMatrix.hpp
    \brief  Header file for the Intrepid2::PAMatrix class; provides support matrix partial assembly.
    \author Created by Nathan V. Roberts.
*/

#ifndef __INTREPID2_PAMATRIX_HPP__
#define __INTREPID2_PAMATRIX_HPP__

#include "Intrepid2_ConfigDefs.hpp"

#include "Intrepid2_Types.hpp"

#include "Intrepid2_Data.hpp"
#include "Intrepid2_TransformedBasisValues.hpp"

#include "Kokkos_Core.hpp"

namespace Intrepid2 {

  /** \class Intrepid2::PAMatrix
      \brief Provides support for structure-aware integration.
  */
  template<typename DeviceType = Kokkos::DefaultExecutionSpace::device_type,
           typename Scalar = double>
  class PAMatrix {
  public:
    Data<DeviceType,Scalar> _composedTransform;
    TensorData<Scalar,DeviceType> _cellMeasures;
    TransformedBasisValues<Scalar,DeviceType> _vectorDataLeft, _vectorDataRight;
    
    /** \brief   Constructs a <b>PAMatrix</b>  representing the contraction of \a <b>vectorDataLeft</b> against \a <b>vectorDataRight</b> containers on
                 point and space dimensions, weighting each point according to <b>cellMeasures</b>.

        \param  vectorDataLeft        [in] - Left input container, with logical shape (C,F1,P,D)
        \param  cellMeasures             [in] - Point weight container, with logical shape (C,P)
        \param  vectorDataRight      [in] - Right input container with logical shape (C,F2,P,D)
        \param  sumInto                        [in] - If TRUE, sum into given output array, otherwise overwrite it. Default: FALSE.

        On construction, computes (if it will be needed) a <b>composedTransform</b> object with logical shape (C,P), (C,P,D), or (C,P,D,D) that stores (det J) M^T_L M_R, where det J represents <b>cellMeasures</b> and M_L and M_R represent the basis transformations for the left and right basis, respectively.
    */
    PAMatrix(const TransformedBasisValues<Scalar,DeviceType> vectorDataLeft,
             const TensorData<Scalar,DeviceType> cellMeasures,
             const TransformedBasisValues<Scalar,DeviceType> vectorDataRight);
    
    /** \brief   Constructs a <b>PAMatrix</b>  representing the contraction of \a <b>vectorData</b> against itself in
                 point and space dimensions, weighting each point according to <b>cellMeasures</b>.

        \param  vectorData                 [in] - Transformed basis values input container, with logical shape (C,F,P,D)
        \param  cellMeasures             [in] - Point weight container, with logical shape (C,P)
        
        On construction, computes (if it will be needed) a <b>composedTransform</b> object with logical shape (C,P), (C,P,D), or (C,P,D,D) that stores (det J) M^T M, where det J represents <b>cellMeasures</b> and M represents the basis transformations for the reference-space basis.
    */
    PAMatrix(const TransformedBasisValues<Scalar,DeviceType> vectorData,
             const TensorData<Scalar,DeviceType> cellMeasures);
    
    /** \brief   Allocates storage for a fully-assembled matrix.
        \return <b>integrals</b>, a container with logical shape (C,F1,F2), suitable for passing to assemble().
    */
    Data<Scalar,DeviceType> allocateMatrixStorage();
  
    /** \brief   Allocates and returns a container with shape (C,F1).
        \return  a container with logical shape (C,F1), suitable for passing to extractColumn().
    */
    Data<DataScalar,DeviceType> allocateColumnStorage();
    
    /** \brief   Allocates and returns a container with shape (C,F), where F=min(F1,F2).
        \return  a container with logical shape (C,F), suitable for passing to extractDiagonal().
    */
    Data<DataScalar,DeviceType> allocateDiagonalStorage();
    
    /** \brief   Allocates and returns a view with shape (C,F2).
        \return  a container with logical shape (C,F2), suitable for passing to extractRow().
    */
    Data<DataScalar,DeviceType> allocateRowStorage();
    
    /** \brief   Allocates and returns a multi-vector with shape (C,F2).
        \return  a container with logical shape (C,F2), suitable for passing to apply() as input.
    */
    ScalarView<DataScalar,DeviceType> allocateInputVector();
    
    /** \brief   Allocates and returns a multi-vector with shape (C,F2,N), where N is the number of input vectors.
        \return  a container with logical shape (C,F2,N), suitable for passing to apply() as input.
    */
    ScalarView<DataScalar,DeviceType> allocateInputMultiVector(const ordinal_type &n);
    
    /** \brief   Allocates and returns a multi-vector with shape (C,F1).
        \return  a container with logical shape (C,F1), suitable for passing to apply() as output.
    */
    ScalarView<DataScalar,DeviceType> allocateOutputVector();
    
    /** \brief   Allocates and returns a multi-vector with shape (C,F1,N), where N is the number of output vectors.
        \return  a container with logical shape (C,F1,N), suitable for passing to apply() as output.
    */
    ScalarView<DataScalar,DeviceType> allocateOutputMultiVector(const ordinal_type &n);
    
    /** \brief  Applies the matrix to <b>inputVector</b>, placing the result in <b>outputVector</b>, without explicit assembly and storage of the matrix itself.

        \param  outputVector             [out] - the result of applying the matrix to the input vector
        \param  inputVector               [in] - the vector to which the matrix will applied
        
        <b>outputVector</b> and <b>inputVector</b> may have shapes (C,F1) and (C,F2), representing single vectors, or shapes (C,F1,N) and (C,F2,N), representing multi-vectors.
    */
    void apply(const ScalarView<DataScalar,DeviceType> &outputVector,
               const ScalarView<DataScalar,DeviceType> & inputVector);
    
    /** \brief   Fully assembles the matrix.
        \param   integrals          [out] - Output matrix, with logical shape (C,F,F).  See allocateMatrixStorage().
    */
    void assemble(Data<Scalar,DeviceType> &integrals);
    
    /** \brief   Extracts the <b>j</b>th column of the matrix, placing it in <b>column</b>.
        \param   row          [out] - Output container with logical shape (C,F1).  See allocateColumnStorage().
    */
    void extractColumn(const Data<Scalar,DeviceType> &column, const ordinal_type &j);
    
    /** \brief   Extracts the diagonal of the matrix, placing it in <b>diagonal</b>.  If the matrix is not square, returns the portion of the matrix for which i==j.
        \param   diagonal          [out] - Output container with logical shape (C,F), F=min(F1,F2).  See allocateDiagonalStorage().
    */
    void extractDiagonal(const Data<Scalar,DeviceType> &diagonal);
    
    /** \brief   Extracts the <b>i</b>th row of the matrix, placing it in <b>row</b>.
        \param   row          [out] - Output view with logical shape (C,F2).  See allocateRowStorage().
    */
    void extractRow(const Data<Scalar,DeviceType> &row, const ordinal_type &i);
    
    /** \brief   Computes and returns the matrix entry at (i,j).
        \param   i          [in] - row index.
        \param   j          [in] - column index.
        \return  The matrix value at (i,j).
    */
    Scalar extractEntry(const ordinal_type &i, const ordinal_type &j);
  }; // end PAMatrix class

} // end namespace Intrepid2

// include templated definitions
#include <Intrepid2_PAMatrixDef.hpp>

#endif
