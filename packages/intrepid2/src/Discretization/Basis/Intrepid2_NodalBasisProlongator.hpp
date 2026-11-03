// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_NodalBasisProlongator.hpp
    \brief  Header file for the Intrepid2 NodalBasisProlongator class.
    \author Created by N. V. Roberts.
 */

#ifndef Intrepid2_NodalBasisProlongator_h
#define Intrepid2_NodalBasisProlongator_h

#include "Intrepid2_ConfigDefs.hpp"
#include "Intrepid2_Types.hpp"
#include "Intrepid2_Utils.hpp"

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_DirectSumBasis.hpp"
#include "Intrepid2_TensorBasis.hpp"

#include "Teuchos_LAPACK.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"

#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace Intrepid2 {

namespace Impl {

  /** \brief Presents a coefficient container of rank 1, 2, or 3 as a rank-2 container indexed by (flattened cell, field ordinal).

      For rank 1 the cell index is ignored; for rank 3, cell = c0 * C1 + c1.
      Not intended for use outside of NodalBasisProlongator.
   */
  template<class ViewType, int rank>
  struct ProlongatorCellAccessor
  {
    ViewType view_;
    ordinal_type C1_; // extent of second cell dimension; only used for rank 3

    KOKKOS_INLINE_FUNCTION
    decltype(auto) operator()(const ordinal_type &cellOrdinal, const ordinal_type &fieldOrdinal) const
    {
      if constexpr (rank == 1)
      {
        return view_(fieldOrdinal);
      }
      else if constexpr (rank == 2)
      {
        return view_(cellOrdinal, fieldOrdinal);
      }
      else
      {
        return view_(cellOrdinal / C1_, cellOrdinal % C1_, fieldOrdinal);
      }
    }
  };

  /** \brief Applies one factor of a (possibly one-factor) Kronecker product to a tensor of coefficients.

      The input coefficients for one block, at offset inOffset_, are viewed as a tensor with shape (left_, inMid_, right) in
      "first index fastest" order; the output, at offset outOffset_, has shape (left_, outMid_, right).  The functor computes
        out(l, m, r) = sum_j A(m, j) * in(l, j, r),
      where A = M_ (an outMid_ x inMid_ matrix) if transpose_ is false, and A = M_^T (M_ is inMid_ x outMid_) if it is true.

      Not intended for use outside of NodalBasisProlongator.
   */
  template<class Scalar, class InAccessor, class OutAccessor, class MatrixViewType>
  struct ProlongatorStageFunctor
  {
    InAccessor     in_;
    OutAccessor    out_;
    MatrixViewType M_;
    ordinal_type   left_;
    ordinal_type   inMid_;
    ordinal_type   outMid_;
    ordinal_type   inOffset_;
    ordinal_type   outOffset_;
    bool           transpose_;
    bool           sumInto_;

    KOKKOS_INLINE_FUNCTION
    void operator()(const ordinal_type &cellOrdinal, const ordinal_type &outOrdinal) const
    {
      const ordinal_type l   = outOrdinal % left_;
      const ordinal_type t   = outOrdinal / left_;
      const ordinal_type m   = t % outMid_;
      const ordinal_type r   = t / outMid_;
      const ordinal_type inBase = inOffset_ + l + left_ * inMid_ * r;

      Scalar sum = 0.0;
      if (transpose_)
      {
        for (ordinal_type j=0; j<inMid_; j++)
        {
          sum += M_(j,m) * in_(cellOrdinal, inBase + left_ * j);
        }
      }
      else
      {
        for (ordinal_type j=0; j<inMid_; j++)
        {
          sum += M_(m,j) * in_(cellOrdinal, inBase + left_ * j);
        }
      }
      if (sumInto_)
      {
        out_(cellOrdinal, outOffset_ + outOrdinal) += sum;
      }
      else
      {
        out_(cellOrdinal, outOffset_ + outOrdinal) = sum;
      }
    }
  };
} // namespace Impl

/** \class Intrepid2::NodalBasisProlongator
    \brief Prolongation (and its transpose) from a coarse nodal basis to a finer nodal basis spanning a superset of the coarse space.

    If the fine basis \f$\{\psi_i\}\f$ is dual to the functionals \f$\ell_i\f$ defined by its dof coordinates and coefficients
    (\f$\ell_i(u) = u(x_i)\f$ for scalar-valued bases, \f$\ell_i(u) = u(x_i) \cdot c_i\f$ for vector-valued bases), then a coarse
    basis function \f$\phi_j\f$ in the span of the fine basis satisfies \f$\phi_j = \sum_i P_{ij} \psi_i\f$ with
    \f$P_{ij} = \ell_i(\phi_j)\f$.  We compute \f$P = V_f^{-1} V_c\f$, with \f$(V_f)_{ik} = \ell_i(\psi_k)\f$ and \f$(V_c)_{ij} = \ell_i(\phi_j)\f$;
    \f$V_f\f$ is the identity up to roundoff, but solving with it keeps the computation correct when that holds only approximately.
    On construction, we also verify that the coarse space is contained in the fine space, throwing an exception otherwise.

    Bases are decomposed structurally before any matrices are computed:
    - Direct-sum bases (Intrepid2::Basis_DirectSumBasis, e.g. the H(curl) and H(div) bases in Intrepid2::DerivedNodalBasisFamily)
      whose coarse and fine summands have matching structure are split into their summands.
    - Tensor-product bases (Intrepid2::Basis_TensorBasis) whose coarse and fine instances have the same number of tensorial components
      are prolongated in terms of those components: the prolongation operator is the Kronecker product of the component prolongation
      operators, and it is applied by sum factorization, one component at a time.  For a hexahedral basis of degree p this costs
      O(p^4) per cell, rather than the O(p^6) of a dense matrix-vector product, and the stored matrices are O(p^2) rather than O(p^6).
    - Any other basis is treated as a single block with a dense prolongation matrix.

    Prolongation matrices are computed on the host once per (coarse, fine) basis pair and stored in a static container, in the same
    way as the orientation matrices stored by Intrepid2::OrientationTools; the container is cleared at Kokkos::finalize(), or on demand
    via clearProlongationMatrixData().  Because component bases are shared across many tensor bases (e.g. the line bases used for
    H(grad), H(curl), H(div), and H(vol) on the quadrilateral and hexahedron), the component matrices are reused across function spaces.

    Coefficient containers may have shape (F), (C,F), or (C0,C1,F); the final dimension indexes basis functions.  Matrices are stored
    as doubles; the basis output and point scalar types must therefore be arithmetic types.  Coefficient containers may have any scalar
    type for which double * Scalar is defined (including Sacado types).
 */
template<typename DeviceType,
         typename OutputScalar = double,
         typename PointScalar  = double>
class NodalBasisProlongator {
public:
  static_assert(std::is_arithmetic<OutputScalar>::value && std::is_arithmetic<PointScalar>::value,
                "NodalBasisProlongator requires arithmetic OutputScalar and PointScalar types; coefficient Views passed to apply() may still have non-arithmetic scalar types.");

  using BasisBase       = Basis<DeviceType,OutputScalar,PointScalar>;
  using BPtr            = BasisPtr<DeviceType,OutputScalar,PointScalar>;
  using HostDeviceType  = typename Kokkos::HostSpace::device_type;
  using HostBasisType   = Basis<HostDeviceType,OutputScalar,PointScalar>;
  using TensorBasisType    = Basis_TensorBasis<BasisBase>;
  using DirectSumBasisType = Basis_DirectSumBasis<BasisBase>;

  //! (fine field ordinal, coarse field ordinal)
  using MatrixViewType     = Kokkos::View<double**,DeviceType>;
  using HostMatrixViewType = Kokkos::View<double**,HostDeviceType>;

  /** \brief Key :: coarse basis name, coarse degree, fine basis name, fine degree, dof data (cardinalities, dof coordinates, and dof coefficients of both bases).

      Basis names do not distinguish point types (e.g. equispaced and warp-blend nodes), so the dof data is included in the key.
   */
  using KeyType        = std::tuple<std::string,ordinal_type,std::string,ordinal_type,std::vector<double> >;
  using MatrixDataType = std::map<KeyType,MatrixViewType>;

  //! Prolongation matrices, computed once per (coarse, fine) basis pair; cleared at Kokkos::finalize().
  static MatrixDataType prolongationMatrixData;

private:
  //! A contiguous range of coarse and fine field ordinals, related by the Kronecker product of the matrices in components.
  struct Block
  {
    ordinal_type coarseOffset;
    ordinal_type fineOffset;
    ordinal_type coarseCardinality;
    ordinal_type fineCardinality;
    std::vector<MatrixViewType> components; // first component varies fastest in the field ordinal
    std::vector<ordinal_type>   coarseDims; // coarse cardinality of each component
    std::vector<ordinal_type>   fineDims;   // fine cardinality of each component
  };

  std::vector<Block> blocks_;
  ordinal_type coarseCardinality_;
  ordinal_type fineCardinality_;
  ordinal_type maxProlongationWorkSize_ = 0; // largest intermediate tensor in a multi-stage apply()
  ordinal_type maxTransposeWorkSize_    = 0; // largest intermediate tensor in a multi-stage applyTranspose()
  ordinal_type maxComponentCount_       = 1;

  //! Appends the dof data used to distinguish bases that share a name and degree.
  static void appendDofData(const HostBasisType &basis, std::vector<double> &data)
  {
    const ordinal_type cardinality = basis.getCardinality();
    const ordinal_type spaceDim    = basis.getDomainDimension();
    data.push_back(cardinality);
    data.push_back(spaceDim);

    Kokkos::DynRankView<double,HostDeviceType> dofCoords("dofCoords", cardinality, spaceDim);
    basis.getDofCoords(dofCoords);
    for (ordinal_type i=0; i<cardinality; i++)
    {
      for (ordinal_type d=0; d<spaceDim; d++)
      {
        data.push_back(dofCoords(i,d));
      }
    }

    if (isVectorValued(basis))
    {
      Kokkos::DynRankView<double,HostDeviceType> dofCoeffs("dofCoeffs", cardinality, spaceDim);
      basis.getDofCoeffs(dofCoeffs);
      for (ordinal_type i=0; i<cardinality; i++)
      {
        for (ordinal_type d=0; d<spaceDim; d++)
        {
          data.push_back(dofCoeffs(i,d));
        }
      }
    }
  }

  static bool isVectorValued(const HostBasisType &basis)
  {
    auto values = basis.allocateOutputView(1, OPERATOR_VALUE);
    return values.rank() == 3;
  }

  /** \brief Evaluates the dof functionals of functionalBasis applied to the members of basis.
      \return a column-major matrix V with V(i,j) = l_i(phi_j), where l_i is the i-th dof functional of functionalBasis and phi_j the j-th member of basis.
   */
  static Teuchos::SerialDenseMatrix<int,double> evaluateFunctionals(const HostBasisType &functionalBasis, const HostBasisType &basis)
  {
    const ordinal_type numFunctionals = functionalBasis.getCardinality();
    const ordinal_type numFields      = basis.getCardinality();
    const ordinal_type spaceDim       = functionalBasis.getDomainDimension();
    const bool vectorValued           = isVectorValued(functionalBasis);

    Kokkos::DynRankView<double,HostDeviceType> dofCoords("dofCoords", numFunctionals, spaceDim);
    functionalBasis.getDofCoords(dofCoords);

    Kokkos::DynRankView<double,HostDeviceType> dofCoeffs;
    if (vectorValued)
    {
      dofCoeffs = Kokkos::DynRankView<double,HostDeviceType>("dofCoeffs", numFunctionals, spaceDim);
      functionalBasis.getDofCoeffs(dofCoeffs);
    }

    auto values = basis.allocateOutputView(numFunctionals, OPERATOR_VALUE);
    basis.getValues(values, dofCoords, OPERATOR_VALUE);

    Teuchos::SerialDenseMatrix<int,double> V(numFunctionals, numFields);
    for (ordinal_type i=0; i<numFunctionals; i++)
    {
      for (ordinal_type j=0; j<numFields; j++)
      {
        if (vectorValued)
        {
          double value = 0.0;
          for (ordinal_type d=0; d<spaceDim; d++)
          {
            value += dofCoeffs(i,d) * values(j,i,d);
          }
          V(i,j) = value;
        }
        else
        {
          V(i,j) = values(j,i);
        }
      }
    }
    return V;
  }

  /** \brief Computes the prolongation matrix P (fine cardinality x coarse cardinality) on the host, and verifies that coarse = fine * P at the coarse basis's dof coordinates.
   */
  static HostMatrixViewType computeProlongationMatrix(const HostBasisType &coarseBasis, const HostBasisType &fineBasis)
  {
    INTREPID2_TEST_FOR_EXCEPTION(coarseBasis.getBaseCellTopology().getKey() != fineBasis.getBaseCellTopology().getKey(), std::invalid_argument,
                                 "coarseBasis and fineBasis must be defined on the same cell topology");
    INTREPID2_TEST_FOR_EXCEPTION(coarseBasis.getNumTensorialExtrusions() != fineBasis.getNumTensorialExtrusions(), std::invalid_argument,
                                 "coarseBasis and fineBasis must be defined on the same cell topology");
    INTREPID2_TEST_FOR_EXCEPTION(isVectorValued(coarseBasis) != isVectorValued(fineBasis), std::invalid_argument,
                                 "coarseBasis and fineBasis must agree on whether they are vector-valued");

    const ordinal_type coarseCardinality = coarseBasis.getCardinality();
    const ordinal_type fineCardinality   = fineBasis.getCardinality();

    INTREPID2_TEST_FOR_EXCEPTION(coarseCardinality > fineCardinality, std::invalid_argument,
                                 "coarseBasis cardinality may not exceed that of fineBasis");

    auto Vf = evaluateFunctionals(fineBasis, fineBasis);
    auto P  = evaluateFunctionals(fineBasis, coarseBasis); // will be overwritten with Vf^{-1} Vc

    Teuchos::LAPACK<int,double> lapack;
    std::vector<int> pivots(fineCardinality);
    int info = 0;
    lapack.GESV(fineCardinality, coarseCardinality, Vf.values(), Vf.stride(), pivots.data(), P.values(), P.stride(), &info);
    INTREPID2_TEST_FOR_EXCEPTION(info != 0, std::runtime_error,
                                 "Fine-basis dof functionals are singular on the fine basis; is fineBasis a nodal basis?");

    HostMatrixViewType hostMatrix("NodalBasisProlongator matrix", fineCardinality, coarseCardinality);
    for (ordinal_type i=0; i<fineCardinality; i++)
    {
      for (ordinal_type j=0; j<coarseCardinality; j++)
      {
        hostMatrix(i,j) = P(i,j);
      }
    }

    // Verify that the coarse space lies in the span of the fine basis by checking coarse_j = sum_i P(i,j) fine_i at the coarse dof coordinates
    // (which are generally distinct from the fine dof coordinates, at which the identity holds by construction).
    {
      const ordinal_type spaceDim  = coarseBasis.getDomainDimension();
      Kokkos::DynRankView<double,HostDeviceType> points("points", coarseCardinality, spaceDim);
      coarseBasis.getDofCoords(points);

      auto coarseValues = coarseBasis.allocateOutputView(coarseCardinality, OPERATOR_VALUE);
      auto fineValues   = fineBasis  .allocateOutputView(coarseCardinality, OPERATOR_VALUE);
      coarseBasis.getValues(coarseValues, points, OPERATOR_VALUE);
      fineBasis  .getValues(fineValues,   points, OPERATOR_VALUE);

      const bool vectorValued = (coarseValues.rank() == 3);
      const ordinal_type numComponents = vectorValued ? spaceDim : 1;
      double maxError = 0.0, maxValue = 0.0;
      for (ordinal_type j=0; j<coarseCardinality; j++)
      {
        for (ordinal_type pointOrdinal=0; pointOrdinal<coarseCardinality; pointOrdinal++)
        {
          for (ordinal_type d=0; d<numComponents; d++)
          {
            const double coarseValue = vectorValued ? coarseValues(j,pointOrdinal,d) : coarseValues(j,pointOrdinal);
            double fineValue = 0.0;
            for (ordinal_type i=0; i<fineCardinality; i++)
            {
              fineValue += hostMatrix(i,j) * (vectorValued ? fineValues(i,pointOrdinal,d) : fineValues(i,pointOrdinal));
            }
            maxError = std::max(maxError, std::abs(fineValue - coarseValue));
            maxValue = std::max(maxValue, std::abs(coarseValue));
          }
        }
      }
      const double tol = 1e-10 * std::max(1.0, maxValue);
      INTREPID2_TEST_FOR_EXCEPTION(maxError > tol, std::invalid_argument,
                                   "coarseBasis does not lie in the span of fineBasis; prolongation is undefined");
    }

    return hostMatrix;
  }

  //! Returns true if coarse and fine have the same direct-sum/tensor structure down to the components, so that blocks can be built from that structure.
  static bool directSumStructuresMatch(const BasisBase *coarseBasis, const BasisBase *fineBasis)
  {
    auto coarseDirectSum = dynamic_cast<const DirectSumBasisType*>(coarseBasis);
    auto fineDirectSum   = dynamic_cast<const DirectSumBasisType*>(fineBasis);
    if ((coarseDirectSum == nullptr) != (fineDirectSum == nullptr)) return false;
    if (coarseDirectSum == nullptr) return true; // neither is a direct sum; leaves are handled by addBlocks()
    return directSumStructuresMatch(coarseDirectSum->getBasis1().get(), fineDirectSum->getBasis1().get())
        && directSumStructuresMatch(coarseDirectSum->getBasis2().get(), fineDirectSum->getBasis2().get());
  }

  void addBlocks(const BasisBase *coarseBasis, const BasisBase *fineBasis, const ordinal_type coarseOffset, const ordinal_type fineOffset)
  {
    auto coarseDirectSum = dynamic_cast<const DirectSumBasisType*>(coarseBasis);
    auto fineDirectSum   = dynamic_cast<const DirectSumBasisType*>(fineBasis);
    if ((coarseDirectSum != nullptr) && (fineDirectSum != nullptr))
    {
      auto coarse1 = coarseDirectSum->getBasis1();
      auto fine1   = fineDirectSum  ->getBasis1();
      addBlocks(coarse1.get(), fine1.get(), coarseOffset, fineOffset);
      addBlocks(coarseDirectSum->getBasis2().get(), fineDirectSum->getBasis2().get(),
                coarseOffset + coarse1->getCardinality(), fineOffset + fine1->getCardinality());
      return;
    }

    Block block;
    block.coarseOffset      = coarseOffset;
    block.fineOffset        = fineOffset;
    block.coarseCardinality = coarseBasis->getCardinality();
    block.fineCardinality   = fineBasis  ->getCardinality();

    auto coarseTensor = dynamic_cast<const TensorBasisType*>(coarseBasis);
    auto fineTensor   = dynamic_cast<const TensorBasisType*>(fineBasis);
    bool useTensorComponents = (coarseTensor != nullptr) && (fineTensor != nullptr);
    std::vector<BPtr> coarseComponents, fineComponents;
    if (useTensorComponents)
    {
      coarseComponents = coarseTensor->getTensorBasisComponents();
      fineComponents   = fineTensor  ->getTensorBasisComponents();
      useTensorComponents = (coarseComponents.size() == fineComponents.size());
    }
    if (useTensorComponents)
    {
      ordinal_type coarseProduct = 1, fineProduct = 1;
      for (unsigned k=0; k<coarseComponents.size(); k++)
      {
        coarseProduct *= coarseComponents[k]->getCardinality();
        fineProduct   *= fineComponents  [k]->getCardinality();
      }
      useTensorComponents = (coarseProduct == block.coarseCardinality) && (fineProduct == block.fineCardinality);
    }

    if (useTensorComponents)
    {
      for (unsigned k=0; k<coarseComponents.size(); k++)
      {
        block.components.push_back(getProlongationMatrix(coarseComponents[k].get(), fineComponents[k].get()));
        block.coarseDims.push_back(coarseComponents[k]->getCardinality());
        block.fineDims  .push_back(fineComponents  [k]->getCardinality());
      }
    }
    else
    {
      block.components.push_back(getProlongationMatrix(coarseBasis, fineBasis));
      block.coarseDims.push_back(block.coarseCardinality);
      block.fineDims  .push_back(block.fineCardinality);
    }

    // record the sizes of the intermediate tensors in the sum-factorized apply() and applyTranspose()
    const ordinal_type numComponents = block.components.size();
    maxComponentCount_ = std::max(maxComponentCount_, numComponents);
    for (ordinal_type s=0; s<numComponents-1; s++)
    {
      // after stage s of apply(), components 0..s are fine and components s+1.. are coarse; vice versa for applyTranspose().
      ordinal_type prolongationSize = 1, transposeSize = 1;
      for (ordinal_type k=0; k<numComponents; k++)
      {
        prolongationSize *= (k <= s) ? block.fineDims[k]   : block.coarseDims[k];
        transposeSize    *= (k <= s) ? block.coarseDims[k] : block.fineDims[k];
      }
      maxProlongationWorkSize_ = std::max(maxProlongationWorkSize_, prolongationSize);
      maxTransposeWorkSize_    = std::max(maxTransposeWorkSize_,    transposeSize);
    }

    blocks_.push_back(block);
  }

  template<class Scalar, class InAccessor, class OutAccessor>
  void applyStage(const InAccessor &in, const OutAccessor &out, const MatrixViewType &M, const ordinal_type &numCells,
                  const ordinal_type &left, const ordinal_type &inMid, const ordinal_type &outMid, const ordinal_type &right,
                  const ordinal_type &inOffset, const ordinal_type &outOffset, const bool transpose, const bool sumInto) const
  {
    using ExecutionSpace = typename DeviceType::execution_space;
    using Functor = Impl::ProlongatorStageFunctor<Scalar, InAccessor, OutAccessor, MatrixViewType>;
    const ordinal_type outSize = left * outMid * right;
    if ((numCells == 0) || (outSize == 0)) return;
    Functor functor{in, out, M, left, inMid, outMid, inOffset, outOffset, transpose, sumInto};
    Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{numCells,outSize});
    Kokkos::parallel_for(transpose ? "NodalBasisProlongator::applyTranspose stage" : "NodalBasisProlongator::apply stage", policy, functor);
  }

  template<int rank, class OutViewType, class InViewType>
  void applyInternal(const OutViewType &outView, const InViewType &inView, const bool transpose, const bool sumInto) const
  {
    using Scalar = typename OutViewType::non_const_value_type;

    ordinal_type numCells = 1, C1 = 1;
    if constexpr (rank == 2)
    {
      numCells = outView.extent_int(0);
    }
    else if constexpr (rank == 3)
    {
      C1       = outView.extent_int(1);
      numCells = outView.extent_int(0) * C1;
    }

    using InAccessor  = Impl::ProlongatorCellAccessor<InViewType,  rank>;
    using OutAccessor = Impl::ProlongatorCellAccessor<OutViewType, rank>;
    InAccessor  in {inView,  C1};
    OutAccessor out{outView, C1};

    // work space for multi-stage (sum-factorized) application; ping-pong between two buffers when there are three or more stages
    const ordinal_type workSize = transpose ? maxTransposeWorkSize_ : maxProlongationWorkSize_;
    using WorkViewType = decltype(Impl::createMatchingDynRankView(outView, "work", numCells, workSize));
    using WorkAccessor = Impl::ProlongatorCellAccessor<WorkViewType, 2>;
    WorkViewType work[2];
    if (maxComponentCount_ >= 2) work[0] = Impl::createMatchingDynRankView(outView, "NodalBasisProlongator work 0", numCells, workSize);
    if (maxComponentCount_ >= 3) work[1] = Impl::createMatchingDynRankView(outView, "NodalBasisProlongator work 1", numCells, workSize);

    for (const auto &block : blocks_)
    {
      const ordinal_type numStages = block.components.size();
      const ordinal_type inOffset  = transpose ? block.fineOffset   : block.coarseOffset;
      const ordinal_type outOffset = transpose ? block.coarseOffset : block.fineOffset;
      std::vector<ordinal_type> dims          = transpose ? block.fineDims   : block.coarseDims; // shape of current tensor
      const std::vector<ordinal_type> outDims = transpose ? block.coarseDims : block.fineDims;

      for (ordinal_type s=0; s<numStages; s++)
      {
        ordinal_type left = 1, right = 1;
        for (ordinal_type k=0;   k<s;         k++) left  *= dims[k];
        for (ordinal_type k=s+1; k<numStages; k++) right *= dims[k];
        const ordinal_type inMid  = dims[s];
        const ordinal_type outMid = outDims[s];
        const auto &M = block.components[s];

        const bool firstStage = (s == 0);
        const bool lastStage  = (s == numStages - 1);
        if (firstStage && lastStage)
        {
          applyStage<Scalar>(in, out, M, numCells, left, inMid, outMid, right, inOffset, outOffset, transpose, sumInto);
        }
        else if (firstStage)
        {
          WorkAccessor workOut{work[s%2], 1};
          applyStage<Scalar>(in, workOut, M, numCells, left, inMid, outMid, right, inOffset, 0, transpose, false);
        }
        else if (lastStage)
        {
          WorkAccessor workIn{work[(s-1)%2], 1};
          applyStage<Scalar>(workIn, out, M, numCells, left, inMid, outMid, right, 0, outOffset, transpose, sumInto);
        }
        else
        {
          WorkAccessor workIn {work[(s-1)%2], 1};
          WorkAccessor workOut{work[s%2],     1};
          applyStage<Scalar>(workIn, workOut, M, numCells, left, inMid, outMid, right, 0, 0, transpose, false);
        }
        dims[s] = outMid;
      }
    }
  }

  template<class OutViewType, class InViewType>
  bool fineBasisExtentMismatch(const OutViewType &outView, const InViewType &inView, const ordinal_type finalDim, const bool transpose) const
  {
    const ordinal_type fineExtent = transpose ? inView.extent_int(finalDim) : outView.extent_int(finalDim);
    return fineExtent != fineCardinality_;
  }

  template<class OutViewType, class InViewType>
  bool coarseBasisExtentMismatch(const OutViewType &outView, const InViewType &inView, const ordinal_type finalDim, const bool transpose) const
  {
    const ordinal_type coarseExtent = transpose ? outView.extent_int(finalDim) : inView.extent_int(finalDim);
    return coarseExtent != coarseCardinality_;
  }

  template<class OutViewType, class InViewType>
  void applyDispatch(const OutViewType &outView, const InViewType &inView, const bool transpose, const bool sumInto) const
  {

    INTREPID2_TEST_FOR_EXCEPTION(outView.rank() != inView.rank(), std::invalid_argument,
                                 "coarseBasisView and fineBasisView must agree on rank");
    const ordinal_type rank = outView.rank();
    INTREPID2_TEST_FOR_EXCEPTION((rank < 1) || (rank > 3), std::invalid_argument, "ranks of Views must both be 1, 2, or 3");

    const ordinal_type finalDim = rank - 1;
    INTREPID2_TEST_FOR_EXCEPTION(fineBasisExtentMismatch(outView, inView, finalDim, transpose), std::invalid_argument,
                                 "Final dimension of fineBasisView must have extent equal to fine basis cardinality");
    INTREPID2_TEST_FOR_EXCEPTION(coarseBasisExtentMismatch(outView, inView, finalDim, transpose), std::invalid_argument,
                                 "Final dimension of coarseBasisView must have extent equal to coarse basis cardinality");
    for (ordinal_type d=0; d<finalDim; d++)
    {
      INTREPID2_TEST_FOR_EXCEPTION(outView.extent_int(d) != inView.extent_int(d), std::invalid_argument,
                                   "fineBasisView and coarseBasisView must agree in the cell dimension(s)");
    }

    // guard with constexpr for fixed-rank view cases, for which the kernels otherwise would not compile
    if (rank == 1)
    {
      if constexpr (supports_rank_1<OutViewType>::value && supports_rank_1<InViewType>::value)
      {
        applyInternal<1>(outView, inView, transpose, sumInto);
      }
    }
    else if (rank == 2)
    {
      if constexpr (supports_rank_2<OutViewType>::value && supports_rank_2<InViewType>::value)
      {
        applyInternal<2>(outView, inView, transpose, sumInto);
      }
    }
    else
    {
      if constexpr (supports_rank_3<OutViewType>::value && supports_rank_3<InViewType>::value)
      {
        applyInternal<3>(outView, inView, transpose, sumInto);
      }
    }
  }

public:
  /** \brief Constructor.
      \param [in] coarseBasis - the coarse basis; must lie in the span of fineBasis
      \param [in] fineBasis   - the fine basis; must be a nodal basis (one supporting getDofCoords(), and getDofCoeffs() if vector-valued)
   */
  NodalBasisProlongator(BPtr coarseBasis, BPtr fineBasis)
  {
    INTREPID2_TEST_FOR_EXCEPTION(coarseBasis->getBasisType() == BASIS_FEM_HIERARCHICAL, std::invalid_argument,
                                 "coarseBasis is hierarchical; use HierarchicalBasisProlongator");
    INTREPID2_TEST_FOR_EXCEPTION(fineBasis->getBasisType() == BASIS_FEM_HIERARCHICAL, std::invalid_argument,
                                 "fineBasis is hierarchical; use HierarchicalBasisProlongator");
    INTREPID2_TEST_FOR_EXCEPTION(coarseBasis->getFunctionSpace() != fineBasis->getFunctionSpace(), std::invalid_argument,
                                 "coarseBasis and fineBasis must agree on function space");

    coarseCardinality_ = coarseBasis->getCardinality();
    fineCardinality_   = fineBasis  ->getCardinality();

    if (directSumStructuresMatch(coarseBasis.get(), fineBasis.get()))
    {
      addBlocks(coarseBasis.get(), fineBasis.get(), 0, 0);
    }
    else
    {
      // the bases do not share a direct-sum structure; treat the whole basis pair as a single dense block
      Block block;
      block.coarseOffset      = 0;
      block.fineOffset        = 0;
      block.coarseCardinality = coarseCardinality_;
      block.fineCardinality   = fineCardinality_;
      block.components.push_back(getProlongationMatrix(coarseBasis.get(), fineBasis.get()));
      block.coarseDims.push_back(coarseCardinality_);
      block.fineDims  .push_back(fineCardinality_);
      blocks_.push_back(block);
    }
  }

  /** \brief Returns the prolongation matrix P (fine cardinality x coarse cardinality) relating coarseBasis to fineBasis, so that coarse_j = sum_i P(i,j) fine_i.

      The matrix is computed once per basis pair and stored in prolongationMatrixData; subsequent calls return the stored matrix.
   */
  static MatrixViewType getProlongationMatrix(const BasisBase *coarseBasis, const BasisBase *fineBasis)
  {
    static bool hookRegistered = false;
    if (!hookRegistered)
    {
      Kokkos::push_finalize_hook( [=] {
        prolongationMatrixData.clear();
      });
      hookRegistered = true;
    }

    auto coarseHostBasis = coarseBasis->getHostBasis();
    auto fineHostBasis   = fineBasis  ->getHostBasis();

    std::vector<double> dofData;
    appendDofData(*coarseHostBasis, dofData);
    appendDofData(*fineHostBasis,   dofData);

    const KeyType key(coarseBasis->getName(), coarseBasis->getDegree(), fineBasis->getName(), fineBasis->getDegree(), dofData);
    const auto found = prolongationMatrixData.find(key);

    MatrixViewType matrix;
    if (found == prolongationMatrixData.end())
    {
      auto hostMatrix = computeProlongationMatrix(*coarseHostBasis, *fineHostBasis);
      matrix = MatrixViewType("NodalBasisProlongator matrix", hostMatrix.extent(0), hostMatrix.extent(1));
      Kokkos::deep_copy(matrix, hostMatrix);
      prolongationMatrixData.insert(std::make_pair(key, matrix));
    }
    else
    {
      matrix = found->second;
    }
    return matrix;
  }

  //! Clears the stored prolongation matrices.  Existing NodalBasisProlongator instances retain the matrices they use.
  static void clearProlongationMatrixData()
  {
    prolongationMatrixData.clear();
  }

  //! Returns the number of tensorial components in each block; a block with one component uses a dense matrix.
  std::vector<ordinal_type> getBlockComponentCounts() const
  {
    std::vector<ordinal_type> counts;
    for (const auto &block : blocks_) counts.push_back(block.components.size());
    return counts;
  }

  /** \brief Prolongate: fineBasisView = P * coarseBasisView (or += if sumInto is true).
      \param [out] fineBasisView   - coefficients in the fine basis; shape (F), (C,F), or (C0,C1,F)
      \param [in]  coarseBasisView - coefficients in the coarse basis; same rank and cell extents as fineBasisView
      \param [in]  sumInto         - if true, add into fineBasisView rather than overwriting it
   */
  template<class OutViewType, class InViewType>
  void apply(const OutViewType &fineBasisView, const InViewType &coarseBasisView, const bool sumInto = false) const
  {
    applyDispatch(fineBasisView, coarseBasisView, false, sumInto);
  }

  /** \brief Apply the transpose of prolongation: coarseBasisView = P^T * fineBasisView (or += if sumInto is true).
      \param [out] coarseBasisView - coefficients in the coarse basis; shape (F), (C,F), or (C0,C1,F)
      \param [in]  fineBasisView   - coefficients in the fine basis; same rank and cell extents as coarseBasisView
      \param [in]  sumInto         - if true, add into coarseBasisView rather than overwriting it
   */
  template<class OutViewType, class InViewType>
  void applyTranspose(const OutViewType &coarseBasisView, const InViewType &fineBasisView, const bool sumInto = false) const
  {
    applyDispatch(coarseBasisView, fineBasisView, true, sumInto);
  }
};

//Definition of static members
template<typename DeviceType, typename OutputScalar, typename PointScalar>
typename NodalBasisProlongator<DeviceType,OutputScalar,PointScalar>::MatrixDataType NodalBasisProlongator<DeviceType,OutputScalar,PointScalar>::prolongationMatrixData;

} // namespace Intrepid2

#endif /* Intrepid2_NodalBasisProlongator_h */
