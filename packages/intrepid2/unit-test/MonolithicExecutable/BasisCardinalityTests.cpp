#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_NodalBasisFamily.hpp"
#include "Intrepid2_Types.hpp"

namespace
{
  using namespace Intrepid2;

  using LineBasis = Basis_HGRAD_LINE_Cn_FEM<typename Kokkos::DefaultExecutionSpace::device_type,double,double>;
  using BasisBase = typename LineBasis::BasisBase;
  using BasisPtr = Teuchos::RCP<BasisBase>;

  static BasisPtr getHypercubeBasis_HGRAD2(int polyOrder)
  {
    using Teuchos::rcp;

    BasisPtr lineBasis = Teuchos::rcp(new LineBasis(polyOrder) );
    BasisPtr tensorBasis = lineBasis;
    tensorBasis = Teuchos::rcp(new Basis_TensorBasis<BasisBase>(tensorBasis, lineBasis, FUNCTION_SPACE_HGRAD));

    /*for (int d=1; d<spaceDim; d++)
    {
      tensorBasis = Teuchos::rcp(new Basis_TensorBasis<BasisBase>(tensorBasis, lineBasis, FUNCTION_SPACE_HGRAD));
    }*/

    return tensorBasis;
  }

  TEUCHOS_UNIT_TEST( BasisCardinality, Hypercube )
  {
   const int polyDegree   = 1;
   auto nodalBasis        = getHypercubeBasis_HGRAD2(polyDegree);
   // the following succeeds:
   //auto nodalBasis = getLineBasis<NodalBasisFamily>(FUNCTION_SPACE_HGRAD, polyDegree);
   int expectedExtrusionCount = 1;
   TEST_EQUALITY(expectedExtrusionCount, nodalBasis->getNumTensorialExtrusions());
  }

} // namespace

