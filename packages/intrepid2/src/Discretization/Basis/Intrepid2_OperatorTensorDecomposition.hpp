//
//  Intrepid2_OperatorTensorDecomposition.hpp
//  Trilinos
//
//  Created by Roberts, Nathan V on 7/30/22.
//

#ifndef Intrepid2_OperatorTensorDecomposition_h
#define Intrepid2_OperatorTensorDecomposition_h

#include "Intrepid2_Types.hpp"

#include <vector>

namespace Intrepid2
{

template<typename DeviceType, typename OutputType, typename PointType>
class Basis;

/** \struct  Intrepid2::OperatorTensorDecomposition
    \brief   For a multi-component tensor basis, specifies the operators to be applied to the components to produce the composite operator on the tensor basis.
*/
struct OperatorTensorDecomposition
{
  // if we want to make this usable on device, we could switch to Kokkos::Array instead of std::vector.  But this is not our immediate use case.
  std::vector< std::vector<EOperator> > ops; // outer index: vector entry ordinal; inner index: basis component ordinal. (scalar-valued operators have a single entry in outer vector)
  std::vector<double> weights; // weights for each vector entry
  ordinal_type numBasisComponents_;
  
  OperatorTensorDecomposition(const std::vector<EOperator> &opsBasis1, const std::vector<EOperator> &opsBasis2, const std::vector<double> vectorComponentWeights)
  :
  weights(vectorComponentWeights),
  numBasisComponents_(2)
  {
    const ordinal_type size = opsBasis1.size();
    const ordinal_type opsBasis2Size = opsBasis2.size();
    const ordinal_type weightsSize = weights.size();
    INTREPID2_TEST_FOR_EXCEPTION(size != opsBasis2Size, std::invalid_argument, "opsBasis1.size() != opsBasis2.size()");
    INTREPID2_TEST_FOR_EXCEPTION(size != weightsSize,   std::invalid_argument, "opsBasis1.size() != weights.size()");
    
    for (ordinal_type i=0; i<size; i++)
    {
      ops.push_back(std::vector<EOperator>{opsBasis1[i],opsBasis2[i]});
    }
  }
  
  OperatorTensorDecomposition(const std::vector< std::vector<EOperator> > &vectorEntryOps, const std::vector<double> &vectorComponentWeights)
  :
  ops(vectorEntryOps),
  weights(vectorComponentWeights)
  {
    const ordinal_type numVectorComponents = ops.size();
    const ordinal_type weightsSize = weights.size();
    INTREPID2_TEST_FOR_EXCEPTION(numVectorComponents != weightsSize,   std::invalid_argument, "opsBasis1.size() != weights.size()");
    
    INTREPID2_TEST_FOR_EXCEPTION(numVectorComponents == 0,   std::invalid_argument, "must have at least one entry!");
    
    ordinal_type numBases = 0;
    for (ordinal_type i=0; i<numVectorComponents; i++)
    {
      if (numBases == 0)
      {
        numBases = ops[i].size();
      }
      else if (ops[i].size() != 0)
      {
        const ordinal_type opsiSize = ops[i].size();
        INTREPID2_TEST_FOR_EXCEPTION(numBases != opsiSize, std::invalid_argument, "must have one operator for each basis in each nontrivial entry in vectorEntryOps");
      }
    }
    INTREPID2_TEST_FOR_EXCEPTION(numBases == 0, std::invalid_argument, "at least one vectorEntryOps entry must be non-trivial");
    numBasisComponents_ = numBases;
  }
  
  OperatorTensorDecomposition(const std::vector<EOperator> &basisOps, const double weight = 1.0)
  :
  ops({basisOps}),
  weights({weight}),
  numBasisComponents_(basisOps.size())
  {}
  
  OperatorTensorDecomposition(const EOperator &opBasis1, const EOperator &opBasis2, double weight = 1.0)
  :
  ops({ std::vector<EOperator>{opBasis1, opBasis2} }),
  weights({weight}),
  numBasisComponents_(2)
  {}
  
  OperatorTensorDecomposition(const EOperator &opBasis1, const EOperator &opBasis2, const EOperator &opBasis3, double weight = 1.0)
  :
  ops({ std::vector<EOperator>{opBasis1, opBasis2, opBasis3} }),
  weights({weight}),
  numBasisComponents_(3)
  {}
  
  ordinal_type numVectorComponents() const
  {
    return ops.size(); // will match weights.size()
  }
  
  ordinal_type numBasisComponents() const
  {
    return numBasisComponents_;
  }
  
  double weight(const ordinal_type &vectorComponentOrdinal) const
  {
    return weights[vectorComponentOrdinal];
  }
  
  bool identicallyZeroComponent(const ordinal_type &vectorComponentOrdinal) const
  {
    INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(vectorComponentOrdinal < 0,                      std::invalid_argument, "vectorComponentOrdinal is out of bounds");
    INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(vectorComponentOrdinal >= numVectorComponents(), std::invalid_argument, "vectorComponentOrdinal is out of bounds");
    return ops[vectorComponentOrdinal].size() == 0;
  }
  
  EOperator op(const ordinal_type &vectorComponentOrdinal, const ordinal_type &basisOrdinal) const
  {
    INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(vectorComponentOrdinal < 0,                      std::invalid_argument, "vectorComponentOrdinal is out of bounds");
    INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(vectorComponentOrdinal >= numVectorComponents(), std::invalid_argument, "vectorComponentOrdinal is out of bounds");
    if (identicallyZeroComponent(vectorComponentOrdinal))
    {
      return OPERATOR_MAX; // by convention: zero in this component
    }
    else
    {
      INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(basisOrdinal < 0,                    std::invalid_argument, "basisOrdinal is out of bounds");
      INTREPID2_TEST_FOR_EXCEPTION_DEVICE_SAFE(basisOrdinal >= numBasisComponents_, std::invalid_argument, "basisOrdinal is out of bounds");
      return ops[vectorComponentOrdinal][basisOrdinal];
    }
  }
  
  //! takes as argument bases that are components in this decomposition, and decomposes them further if they are tensor bases.  Returns a fully expanded decomposition.
  template<typename DeviceType, typename OutputValueType, class PointValueType>
  OperatorTensorDecomposition expandedDecomposition(std::vector< Teuchos::RCP<Basis<DeviceType,OutputValueType,PointValueType> > > &bases)
  {
    const ordinal_type basesSize = bases.size();
    INTREPID2_TEST_FOR_EXCEPTION(basesSize != numBasisComponents_, std::invalid_argument, "The number of bases provided must match the number of basis components in this decomposition");
    
    ordinal_type numExpandedBasisComponents = 0;
    using BasisBase   = Basis<DeviceType,OutputValueType,PointValueType>;
    std::vector<bool> componentIsTensor(bases.size());
    for (ordinal_type basisComponentOrdinal=0; basisComponentOrdinal<numBasisComponents_; basisComponentOrdinal++)
    {
      const ordinal_type numComponents = bases[basisComponentOrdinal]->getTensorBasisComponents().size();
      numExpandedBasisComponents += numComponents;
      componentIsTensor[basisComponentOrdinal] = numComponents > 1;
    }
    
    std::vector< std::vector<EOperator> > expandedOps; // outer index: vector entry ordinal; inner index: basis component ordinal.
    std::vector<double> expandedWeights;
    const ordinal_type opsSize = ops.size();
    for (ordinal_type simpleVectorEntryOrdinal=0; simpleVectorEntryOrdinal<opsSize; simpleVectorEntryOrdinal++)
    {
      if (identicallyZeroComponent(simpleVectorEntryOrdinal))
      {
        expandedOps.push_back(std::vector<EOperator>{});
        expandedWeights.push_back(0.0);
        continue;
      }
      
      std::vector< std::vector<EOperator> > expandedBasisOpsForSimpleVectorEntry(1); // start out with one outer entry; expands if a component is vector-valued
      
      // this lambda appends an op to each of the vector components
      auto addExpandedOp = [&expandedBasisOpsForSimpleVectorEntry](const EOperator &op)
      {
        const ordinal_type size = expandedBasisOpsForSimpleVectorEntry.size();
        for (ordinal_type i=0; i<size; i++)
        {
          expandedBasisOpsForSimpleVectorEntry[i].push_back(op);
        }
      };
      
      // this lambda takes a scalar-valued (single outer entry) expandedBasisOps and expands it
      // according to the number of vector entries coming from the vector-valued component basis
      auto vectorizeExpandedOps = [&expandedBasisOpsForSimpleVectorEntry](const int &numSubVectors)
      {
        // we require that this only gets called once per simpleVectorEntryOrdinal -- i.e., only one basis component gets to be vector-valued.
        INTREPID2_TEST_FOR_EXCEPTION(expandedBasisOpsForSimpleVectorEntry.size() != 1, std::invalid_argument, "multiple basis components may not be vector-valued!");
        for (ordinal_type i=1; i<numSubVectors; i++)
        {
          expandedBasisOpsForSimpleVectorEntry.push_back(expandedBasisOpsForSimpleVectorEntry[0]);
        }
      };
      
      std::vector<EOperator> subVectorOps;     // only used if one of the components is vector-valued
      std::vector<double> subVectorWeights {weights[simpleVectorEntryOrdinal]};
      for (ordinal_type basisComponentOrdinal=0; basisComponentOrdinal<numBasisComponents_; basisComponentOrdinal++)
      {
        const auto &op = ops[simpleVectorEntryOrdinal][basisComponentOrdinal];
        
        if (! componentIsTensor[basisComponentOrdinal])
        {
          addExpandedOp(op);
        }
        else
        {
          OperatorTensorDecomposition basisOpDecomposition = bases[basisComponentOrdinal]->getOperatorDecomposition(op);
          if (basisOpDecomposition.numVectorComponents() > 1)
          {
            // We don't currently support a use case where we have multiple component bases that are vector-valued:
            INTREPID2_TEST_FOR_EXCEPTION(subVectorWeights.size() > 1, std::invalid_argument, "Unhandled case: multiple component bases are vector-valued");
            // We do support a single vector-valued case, though; this splits the current simpleVectorEntryOrdinal into an appropriate number of components that come in order in the expanded vector
            ordinal_type numSubVectors = basisOpDecomposition.numVectorComponents();
            vectorizeExpandedOps(numSubVectors);
            
            double weightSoFar = subVectorWeights[0];
            for (ordinal_type subVectorEntryOrdinal=1; subVectorEntryOrdinal<numSubVectors; subVectorEntryOrdinal++)
            {
              subVectorWeights.push_back(weightSoFar * basisOpDecomposition.weight(subVectorEntryOrdinal));
            }
            subVectorWeights[0] *= basisOpDecomposition.weight(0);
            for (ordinal_type subVectorEntryOrdinal=0; subVectorEntryOrdinal<numSubVectors; subVectorEntryOrdinal++)
            {
              for (ordinal_type subComponentBasis=0; subComponentBasis<basisOpDecomposition.numBasisComponents(); subComponentBasis++)
              {
                const auto &basisOp = basisOpDecomposition.op(subVectorEntryOrdinal, subComponentBasis);
                expandedBasisOpsForSimpleVectorEntry[subVectorEntryOrdinal].push_back(basisOp);
              }
            }
          }
          else
          {
            double componentWeight = basisOpDecomposition.weight(0);
            const ordinal_type size = subVectorWeights.size();
            for (ordinal_type i=0; i<size; i++)
            {
              subVectorWeights[i] *= componentWeight;
            }
            ordinal_type subVectorEntryOrdinal = 0;
            const ordinal_type numBasisComponents = basisOpDecomposition.numBasisComponents();
            for (ordinal_type subComponentBasis=0; subComponentBasis<numBasisComponents; subComponentBasis++)
            {
              const auto &basisOp = basisOpDecomposition.op(subVectorEntryOrdinal, basisComponentOrdinal);
              addExpandedOp( basisOp );
            }
          }
        }
      }
      
      // sanity check on the new expandedOps entries:
      for (ordinal_type i=0; i<static_cast<ordinal_type>(expandedBasisOpsForSimpleVectorEntry.size()); i++)
      {
        const ordinal_type size = expandedBasisOpsForSimpleVectorEntry[i].size();
        INTREPID2_TEST_FOR_EXCEPTION(size != numExpandedBasisComponents, std::logic_error, "each vector in expandedBasisOpsForSimpleVectorEntry should have as many entries as there are expanded basis components");
      }
      
      expandedOps.insert(expandedOps.end(), expandedBasisOpsForSimpleVectorEntry.begin(), expandedBasisOpsForSimpleVectorEntry.end());
      expandedWeights.insert(expandedWeights.end(), subVectorWeights.begin(), subVectorWeights.end());
    }
    // check that vector lengths agree:
    INTREPID2_TEST_FOR_EXCEPTION(expandedOps.size() != expandedWeights.size(), std::logic_error, "expandedWeights and expandedOps do not agree on the number of vector components");
    
    return OperatorTensorDecomposition(expandedOps, expandedWeights);
  }
};

}// namespace Intrepid2

#endif /* Intrepid2_OperatorTensorDecomposition_h */
