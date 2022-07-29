#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_NodalBasisFamily.hpp"
#include "Intrepid2_Types.hpp"

namespace
{
  using namespace Intrepid2;

  using LineBasis = Basis_HGRAD_LINE_Cn_FEM<typename Kokkos::DefaultExecutionSpace::device_type,double,double>;
  using BasisBase = typename LineBasis::BasisBase;
  using BasisPtr = Teuchos::RCP<BasisBase>;

  TEUCHOS_UNIT_TEST( BasisCardinality, Hypercube )
  {
   const int polyDegree   = 1;
   BasisPtr lineBasis = Teuchos::rcp(new LineBasis(polyDegree) );
   BasisPtr tensorBasis = lineBasis;
   tensorBasis = Teuchos::rcp(new Basis_TensorBasis<BasisBase>(tensorBasis, lineBasis, FUNCTION_SPACE_HGRAD));
   int expectedExtrusionCount = 1;
   TEST_EQUALITY(expectedExtrusionCount, tensorBasis->getNumTensorialExtrusions());
  }

} // namespace
