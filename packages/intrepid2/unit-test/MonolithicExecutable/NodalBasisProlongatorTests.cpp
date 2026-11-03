// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   NodalBasisProlongatorTests.cpp
    \brief  Tests against Intrepid2::NodalBasisProlongator.
    \author Created by N.V. Roberts.
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "Intrepid2_TestUtils.hpp"

#include "Intrepid2_NodalBasisProlongator.hpp"

#include "Intrepid2_DerivedBasisFamily.hpp"
#include "Intrepid2_NodalBasisFamily.hpp"

#include <random>

namespace
{
  using namespace Intrepid2;

  using DeviceType     = DefaultTestDeviceType;
  using HostDeviceType = typename Kokkos::HostSpace::device_type;
  using Prolongator    = NodalBasisProlongator<DeviceType,double,double>;
  using HostFlatView   = Kokkos::View<double**,HostDeviceType>; // (flattened cell, field)

  shards::CellTopology lineTopo()  { return shards::CellTopology(shards::getCellTopologyData<shards::Line<>         >()); }
  shards::CellTopology quadTopo()  { return shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<> >()); }
  shards::CellTopology triTopo()   { return shards::CellTopology(shards::getCellTopologyData<shards::Triangle<>     >()); }
  shards::CellTopology hexTopo()   { return shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<>   >()); }
  shards::CellTopology tetTopo()   { return shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<>  >()); }
  shards::CellTopology wedgeTopo() { return shards::CellTopology(shards::getCellTopologyData<shards::Wedge<>        >()); }

  //! getBasis(), extended to wedges.
  template<class BasisFamily>
  typename BasisFamily::BasisPtr makeBasis(const shards::CellTopology &cellTopo, const EFunctionSpace fs, const int polyOrder,
                                           const EPointType pointType = POINTTYPE_DEFAULT)
  {
    if constexpr (requires { typename BasisFamily::HGRAD_WEDGE; })
    {
      if (cellTopo.getBaseKey() == shards::Wedge<>::key)
      {
        return getWedgeBasis<BasisFamily>(fs, polyOrder, pointType);
      }
    }
    return getBasis<BasisFamily>(cellTopo, fs, polyOrder, pointType);
  }

  HostFlatView randomFlatView(const int numCells, const int cardinality, std::mt19937 &generator)
  {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    HostFlatView view("random coefficients", numCells, cardinality);
    for (int c=0; c<numCells; c++)
      for (int i=0; i<cardinality; i++)
        view(c,i) = distribution(generator);
    return view;
  }

  //! numCells is the flattened cell count; it must be 1 for rank 1, and a multiple of 2 for rank 3 (which uses (2, numCells/2, F)).
  template<class ViewType, int rank>
  ViewType allocateCoefficients(const std::string &label, const int numCells, const int cardinality)
  {
    if constexpr (rank == 1) return ViewType(label, cardinality);
    else if constexpr (rank == 2) return ViewType(label, numCells, cardinality);
    else return ViewType(label, 2, numCells / 2, cardinality);
  }

  template<int rank, class ViewType>
  void copyFlatToView(const HostFlatView &flat, const ViewType &view)
  {
    auto hostView = Kokkos::create_mirror_view(view);
    const int numCells = flat.extent_int(0), cardinality = flat.extent_int(1);
    for (int c=0; c<numCells; c++)
      for (int i=0; i<cardinality; i++)
      {
        if constexpr (rank == 1) hostView(i) = flat(c,i);
        else if constexpr (rank == 2) hostView(c,i) = flat(c,i);
        else hostView(c / view.extent_int(1), c % view.extent_int(1), i) = flat(c,i);
      }
    Kokkos::deep_copy(view, hostView);
  }

  template<int rank, class ViewType>
  HostFlatView copyViewToFlat(const ViewType &view, const int numCells)
  {
    auto hostView = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    const int cardinality = view.extent_int(rank-1);
    HostFlatView flat("flat", numCells, cardinality);
    for (int c=0; c<numCells; c++)
      for (int i=0; i<cardinality; i++)
      {
        if constexpr (rank == 1) flat(c,i) = hostView(i);
        else if constexpr (rank == 2) flat(c,i) = hostView(c,i);
        else flat(c,i) = hostView(c / view.extent_int(1), c % view.extent_int(1), i);
      }
    return flat;
  }

  //! Basis values on the host, shaped (F,P,D) -- D = 1 for scalar-valued bases.
  Kokkos::View<double***,HostDeviceType> hostValues(const Teuchos::RCP<Basis<DeviceType,double,double>> &basis,
                                                     const ViewType<double,DeviceType> &points)
  {
    auto values = basis->allocateOutputView(points.extent_int(0), OPERATOR_VALUE);
    basis->getValues(values, points, OPERATOR_VALUE);
    auto valuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    const bool vectorValued = (valuesHost.rank() == 3);
    const int F = valuesHost.extent_int(0), P = valuesHost.extent_int(1), D = vectorValued ? valuesHost.extent_int(2) : 1;
    Kokkos::View<double***,HostDeviceType> result("values", F, P, D);
    for (int i=0; i<F; i++)
      for (int p=0; p<P; p++)
        for (int d=0; d<D; d++)
          result(i,p,d) = vectorValued ? valuesHost(i,p,d) : valuesHost(i,p);
    return result;
  }

  //! Checks that sum_i fineCoeffs(c,i) fine_i == sum_j coarseCoeffs(c,j) coarse_j at the test points, for every cell c.
  void checkExpansionsMatch(const HostFlatView &coarseCoeffs, const Kokkos::View<double***,HostDeviceType> &coarseValues,
                            const HostFlatView &fineCoeffs,   const Kokkos::View<double***,HostDeviceType> &fineValues,
                            const double tol, const std::string &label, Teuchos::FancyOStream &out, bool &success)
  {
    const int numCells = coarseCoeffs.extent_int(0);
    const int P = coarseValues.extent_int(1), D = coarseValues.extent_int(2);
    double maxError = 0.0;
    for (int c=0; c<numCells; c++)
      for (int p=0; p<P; p++)
        for (int d=0; d<D; d++)
        {
          double coarseValue = 0.0, fineValue = 0.0;
          for (int j=0; j<coarseValues.extent_int(0); j++) coarseValue += coarseCoeffs(c,j) * coarseValues(j,p,d);
          for (int i=0; i<fineValues.extent_int(0);   i++) fineValue   += fineCoeffs(c,i)   * fineValues(i,p,d);
          maxError = std::max(maxError, std::abs(coarseValue - fineValue));
        }
    if (maxError > tol)
    {
      success = false;
      out << "FAILURE: " << label << ": coarse and prolongated fine expansions differ by " << maxError << std::endl;
    }
  }

  void checkFlatViewsMatch(const HostFlatView &expected, const HostFlatView &actual, const double tol,
                           const std::string &label, Teuchos::FancyOStream &out, bool &success)
  {
    double maxError = 0.0;
    for (int c=0; c<expected.extent_int(0); c++)
      for (int i=0; i<expected.extent_int(1); i++)
        maxError = std::max(maxError, std::abs(expected(c,i) - actual(c,i)));
    if (maxError > tol)
    {
      success = false;
      out << "FAILURE: " << label << ": max difference " << maxError << std::endl;
    }
  }

  /** Exercises apply() and applyTranspose(), with and without sumInto, for coefficient containers of the specified type and rank:
       - apply() is checked by comparing the coarse and fine expansions at test points;
       - applyTranspose() is checked using the adjoint identity <f, P c> = <P^T f, c>, cell by cell.
   */
  template<class ViewType, int rank>
  void testApply(const Prolongator &prolongator, const Teuchos::RCP<Basis<DeviceType,double,double>> &coarseBasis,
                 const Teuchos::RCP<Basis<DeviceType,double,double>> &fineBasis,
                 const Kokkos::View<double***,HostDeviceType> &coarseValues, const Kokkos::View<double***,HostDeviceType> &fineValues,
                 const std::string &caseLabel, std::mt19937 &generator, Teuchos::FancyOStream &out, bool &success)
  {
    const int numCells = (rank == 1) ? 1 : 4;
    const int coarseCardinality = coarseBasis->getCardinality();
    const int fineCardinality   = fineBasis  ->getCardinality();
    const std::string label = caseLabel + " (rank " + std::to_string(rank) + ")";

    // scale tolerances by the size of the values involved
    double maxValue = 1.0;
    for (int i=0; i<fineValues.extent_int(0); i++)
      for (int p=0; p<fineValues.extent_int(1); p++)
        for (int d=0; d<fineValues.extent_int(2); d++)
          maxValue = std::max(maxValue, std::abs(fineValues(i,p,d)));
    const double tol = 1e-11 * maxValue * fineCardinality;

    auto coarseFlat = randomFlatView(numCells, coarseCardinality, generator);

    auto coarseCoeffs = allocateCoefficients<ViewType,rank>("coarse", numCells, coarseCardinality);
    auto fineCoeffs   = allocateCoefficients<ViewType,rank>("fine",   numCells, fineCardinality);
    copyFlatToView<rank>(coarseFlat, coarseCoeffs);
    Kokkos::deep_copy(fineCoeffs, 1.0); // apply() should overwrite these

    prolongator.apply(fineCoeffs, coarseCoeffs);
    auto fineFlat = copyViewToFlat<rank>(fineCoeffs, numCells);
    checkExpansionsMatch(coarseFlat, coarseValues, fineFlat, fineValues, tol, label + " apply", out, success);

    prolongator.apply(fineCoeffs, coarseCoeffs, true); // sumInto: should double fineCoeffs
    auto fineFlatDoubled = copyViewToFlat<rank>(fineCoeffs, numCells);
    HostFlatView fineFlatTimesTwo("fine x 2", numCells, fineCardinality);
    for (int c=0; c<numCells; c++)
      for (int i=0; i<fineCardinality; i++)
        fineFlatTimesTwo(c,i) = 2.0 * fineFlat(c,i);
    checkFlatViewsMatch(fineFlatTimesTwo, fineFlatDoubled, tol, label + " apply (sumInto)", out, success);

    // transpose: <f, P c> = <P^T f, c>
    auto fineTestFlat = randomFlatView(numCells, fineCardinality, generator);
    auto fineTest     = allocateCoefficients<ViewType,rank>("fine test", numCells, fineCardinality);
    auto coarseResult = allocateCoefficients<ViewType,rank>("coarse result", numCells, coarseCardinality);
    copyFlatToView<rank>(fineTestFlat, fineTest);
    Kokkos::deep_copy(coarseResult, 1.0); // applyTranspose() should overwrite these

    prolongator.applyTranspose(coarseResult, fineTest);
    auto coarseResultFlat = copyViewToFlat<rank>(coarseResult, numCells);
    for (int c=0; c<numCells; c++)
    {
      double lhs = 0.0, rhs = 0.0;
      for (int i=0; i<fineCardinality;   i++) lhs += fineTestFlat(c,i)     * fineFlat(c,i);
      for (int j=0; j<coarseCardinality; j++) rhs += coarseResultFlat(c,j) * coarseFlat(c,j);
      if (std::abs(lhs - rhs) > tol * std::max(1.0, std::abs(lhs)))
      {
        success = false;
        out << "FAILURE: " << label << " applyTranspose: <f, P c> = " << lhs << " but <P^T f, c> = " << rhs << std::endl;
      }
    }

    prolongator.applyTranspose(coarseResult, fineTest, true); // sumInto: should double coarseResult
    auto coarseResultDoubled = copyViewToFlat<rank>(coarseResult, numCells);
    HostFlatView coarseTimesTwo("coarse x 2", numCells, coarseCardinality);
    for (int c=0; c<numCells; c++)
      for (int j=0; j<coarseCardinality; j++)
        coarseTimesTwo(c,j) = 2.0 * coarseResultFlat(c,j);
    checkFlatViewsMatch(coarseTimesTwo, coarseResultDoubled, tol, label + " applyTranspose (sumInto)", out, success);
  }

  template<class BasisFamily>
  void testProlongation(shards::CellTopology cellTopo, const EFunctionSpace fs, const int pCoarse, const int pFine,
                        const EPointType pointType, const std::string &familyName, Teuchos::FancyOStream &out, bool &success)
  {
    auto coarseBasis = makeBasis<BasisFamily>(cellTopo, fs, pCoarse, pointType);
    auto fineBasis   = makeBasis<BasisFamily>(cellTopo, fs, pFine,   pointType);

    std::ostringstream caseLabel;
    caseLabel << familyName << " " << cellTopo.getName() << " " << EFunctionSpaceToString(fs) << " p=" << pCoarse << "->" << pFine
              << " (" << EPointTypeToString(pointType) << ")";

    Prolongator prolongator(coarseBasis, fineBasis);

    auto points = getInputPointsView<double,DeviceType>(cellTopo, pFine + 2);
    auto coarseValues = hostValues(coarseBasis, points);
    auto fineValues   = hostValues(fineBasis,   points);

    std::mt19937 generator(pCoarse * 31 + pFine);

    testApply<Kokkos::View<double*,  DeviceType>, 1>(prolongator, coarseBasis, fineBasis, coarseValues, fineValues, caseLabel.str(), generator, out, success);
    testApply<Kokkos::View<double**, DeviceType>, 2>(prolongator, coarseBasis, fineBasis, coarseValues, fineValues, caseLabel.str(), generator, out, success);
    testApply<Kokkos::View<double***,DeviceType>, 3>(prolongator, coarseBasis, fineBasis, coarseValues, fineValues, caseLabel.str(), generator, out, success);

    using DynRankView = Kokkos::DynRankView<double,DeviceType>;
    testApply<DynRankView, 1>(prolongator, coarseBasis, fineBasis, coarseValues, fineValues, caseLabel.str(), generator, out, success);
    testApply<DynRankView, 2>(prolongator, coarseBasis, fineBasis, coarseValues, fineValues, caseLabel.str(), generator, out, success);
    testApply<DynRankView, 3>(prolongator, coarseBasis, fineBasis, coarseValues, fineValues, caseLabel.str(), generator, out, success);
  }

  template<class BasisFamily>
  void testProlongationForAllDegrees(shards::CellTopology cellTopo, const std::vector<EFunctionSpace> &functionSpaces, const int maxDegree,
                                     const EPointType pointType, const std::string &familyName, Teuchos::FancyOStream &out, bool &success)
  {
    for (auto fs : functionSpaces)
    {
      for (int pCoarse=1; pCoarse<=maxDegree; pCoarse++)
      {
        for (int pFine=pCoarse; pFine<=maxDegree; pFine++)
        {
          testProlongation<BasisFamily>(cellTopo, fs, pCoarse, pFine, pointType, familyName, out, success);
        }
      }
    }
  }

  const std::vector<EFunctionSpace> allSpaces    {FUNCTION_SPACE_HGRAD, FUNCTION_SPACE_HCURL, FUNCTION_SPACE_HDIV, FUNCTION_SPACE_HVOL};
  const std::vector<EFunctionSpace> scalarSpaces {FUNCTION_SPACE_HGRAD, FUNCTION_SPACE_HVOL};

  using DerivedNodalFamily  = DerivedNodalBasisFamily<DeviceType,double,double>;
  using StandardNodalFamily = NodalBasisFamily<DeviceType,double,double>;

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, DerivedNodal_Line )
  {
    testProlongationForAllDegrees<DerivedNodalFamily>(lineTopo(), scalarSpaces, 6, POINTTYPE_EQUISPACED, "DerivedNodal", out, success);
    testProlongationForAllDegrees<DerivedNodalFamily>(lineTopo(), scalarSpaces, 6, POINTTYPE_WARPBLEND,  "DerivedNodal", out, success);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, DerivedNodal_Quad )
  {
    testProlongationForAllDegrees<DerivedNodalFamily>(quadTopo(), allSpaces, 4, POINTTYPE_EQUISPACED, "DerivedNodal", out, success);
    testProlongationForAllDegrees<DerivedNodalFamily>(quadTopo(), allSpaces, 4, POINTTYPE_WARPBLEND,  "DerivedNodal", out, success);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, DerivedNodal_Hex )
  {
    testProlongationForAllDegrees<DerivedNodalFamily>(hexTopo(), allSpaces, 3, POINTTYPE_EQUISPACED, "DerivedNodal", out, success);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, DerivedNodal_Wedge )
  {
    testProlongationForAllDegrees<DerivedNodalFamily>(wedgeTopo(), allSpaces, 3, POINTTYPE_EQUISPACED, "DerivedNodal", out, success);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, DerivedNodal_Tri )
  {
    testProlongationForAllDegrees<DerivedNodalFamily>(triTopo(), allSpaces, 4, POINTTYPE_EQUISPACED, "DerivedNodal", out, success);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, DerivedNodal_Tet )
  {
    testProlongationForAllDegrees<DerivedNodalFamily>(tetTopo(), allSpaces, 3, POINTTYPE_EQUISPACED, "DerivedNodal", out, success);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, StandardNodal_Quad )
  {
    testProlongationForAllDegrees<StandardNodalFamily>(quadTopo(), allSpaces, 4, POINTTYPE_EQUISPACED, "StandardNodal", out, success);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, StandardNodal_Hex )
  {
    testProlongationForAllDegrees<StandardNodalFamily>(hexTopo(), allSpaces, 3, POINTTYPE_EQUISPACED, "StandardNodal", out, success);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, StandardNodal_Tri )
  {
    testProlongationForAllDegrees<StandardNodalFamily>(triTopo(), allSpaces, 4, POINTTYPE_EQUISPACED, "StandardNodal", out, success);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, StandardNodal_Tet )
  {
    testProlongationForAllDegrees<StandardNodalFamily>(tetTopo(), allSpaces, 3, POINTTYPE_EQUISPACED, "StandardNodal", out, success);
  }

  // Tensor and direct-sum bases should be prolongated in terms of their components; other bases with a single dense block.
  TEUCHOS_UNIT_TEST( NodalBasisProlongator, BlockStructure )
  {
    auto componentCounts = [](shards::CellTopology cellTopo, EFunctionSpace fs, bool derived)
    {
      const auto pt = POINTTYPE_EQUISPACED;
      auto coarse = derived ? makeBasis<DerivedNodalFamily>(cellTopo, fs, 1, pt) : makeBasis<StandardNodalFamily>(cellTopo, fs, 1, pt);
      auto fine   = derived ? makeBasis<DerivedNodalFamily>(cellTopo, fs, 2, pt) : makeBasis<StandardNodalFamily>(cellTopo, fs, 2, pt);
      return Prolongator(coarse, fine).getBlockComponentCounts();
    };
    using V = std::vector<ordinal_type>;
    TEST_COMPARE_ARRAYS(componentCounts(lineTopo(),  FUNCTION_SPACE_HGRAD, true), V({1}));
    TEST_COMPARE_ARRAYS(componentCounts(quadTopo(),  FUNCTION_SPACE_HGRAD, true), V({2}));
    TEST_COMPARE_ARRAYS(componentCounts(quadTopo(),  FUNCTION_SPACE_HCURL, true), V({2,2}));
    TEST_COMPARE_ARRAYS(componentCounts(quadTopo(),  FUNCTION_SPACE_HDIV,  true), V({2,2}));
    TEST_COMPARE_ARRAYS(componentCounts(quadTopo(),  FUNCTION_SPACE_HVOL,  true), V({2}));
    TEST_COMPARE_ARRAYS(componentCounts(hexTopo(),   FUNCTION_SPACE_HGRAD, true), V({3}));
    TEST_COMPARE_ARRAYS(componentCounts(hexTopo(),   FUNCTION_SPACE_HCURL, true), V({3,3,3}));
    TEST_COMPARE_ARRAYS(componentCounts(hexTopo(),   FUNCTION_SPACE_HDIV,  true), V({3,3,3}));
    TEST_COMPARE_ARRAYS(componentCounts(hexTopo(),   FUNCTION_SPACE_HVOL,  true), V({3}));
    TEST_COMPARE_ARRAYS(componentCounts(wedgeTopo(), FUNCTION_SPACE_HGRAD, true), V({2}));
    TEST_COMPARE_ARRAYS(componentCounts(triTopo(),   FUNCTION_SPACE_HGRAD, true), V({1}));
    TEST_COMPARE_ARRAYS(componentCounts(quadTopo(),  FUNCTION_SPACE_HGRAD, false), V({1}));
    TEST_COMPARE_ARRAYS(componentCounts(hexTopo(),   FUNCTION_SPACE_HCURL, false), V({1}));
  }

  // Matrices should be computed once per basis pair and reused, including across the components of a tensor basis.
  TEUCHOS_UNIT_TEST( NodalBasisProlongator, MatrixStorage )
  {
    Prolongator::clearProlongationMatrixData();
    TEST_EQUALITY(Prolongator::prolongationMatrixData.size(), size_t(0));

    auto hexCoarse = getBasis<DerivedNodalFamily>(hexTopo(), FUNCTION_SPACE_HGRAD, 1, POINTTYPE_EQUISPACED);
    auto hexFine   = getBasis<DerivedNodalFamily>(hexTopo(), FUNCTION_SPACE_HGRAD, 3, POINTTYPE_EQUISPACED);
    Prolongator hexProlongator(hexCoarse, hexFine);
    // all three components share the same line prolongation matrix
    TEST_EQUALITY(Prolongator::prolongationMatrixData.size(), size_t(1));

    Prolongator hexProlongatorAgain(hexCoarse, hexFine);
    TEST_EQUALITY(Prolongator::prolongationMatrixData.size(), size_t(1));

    // the H(grad) quad basis reuses the same line matrix
    auto quadCoarse = getBasis<DerivedNodalFamily>(quadTopo(), FUNCTION_SPACE_HGRAD, 1, POINTTYPE_EQUISPACED);
    auto quadFine   = getBasis<DerivedNodalFamily>(quadTopo(), FUNCTION_SPACE_HGRAD, 3, POINTTYPE_EQUISPACED);
    Prolongator quadProlongator(quadCoarse, quadFine);
    TEST_EQUALITY(Prolongator::prolongationMatrixData.size(), size_t(1));

    // warp-blend and equispaced nodes differ at degree 3, so these bases share a name and degree but need a distinct matrix
    auto lineCoarse              = getBasis<DerivedNodalFamily>(lineTopo(), FUNCTION_SPACE_HGRAD, 1, POINTTYPE_EQUISPACED);
    auto lineFineWarpBlend       = getBasis<DerivedNodalFamily>(lineTopo(), FUNCTION_SPACE_HGRAD, 3, POINTTYPE_WARPBLEND);
    auto lineFineEquispaced      = getBasis<DerivedNodalFamily>(lineTopo(), FUNCTION_SPACE_HGRAD, 3, POINTTYPE_EQUISPACED);
    auto warpBlendMatrix  = Prolongator::getProlongationMatrix(lineCoarse.get(), lineFineWarpBlend.get());
    TEST_EQUALITY(Prolongator::prolongationMatrixData.size(), size_t(2));
    auto equispacedMatrix = Prolongator::getProlongationMatrix(lineCoarse.get(), lineFineEquispaced.get());
    TEST_EQUALITY(Prolongator::prolongationMatrixData.size(), size_t(2));

    auto warpBlendHost  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), warpBlendMatrix);
    auto equispacedHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), equispacedMatrix);
    double maxDifference = 0.0;
    for (unsigned i=0; i<warpBlendHost.extent(0); i++)
      for (unsigned j=0; j<warpBlendHost.extent(1); j++)
        maxDifference = std::max(maxDifference, std::abs(warpBlendHost(i,j) - equispacedHost(i,j)));
    TEST_ASSERT(maxDifference > 1e-3);

    // prolongators keep working after the storage is cleared
    Prolongator::clearProlongationMatrixData();
    TEST_EQUALITY(Prolongator::prolongationMatrixData.size(), size_t(0));
    Kokkos::View<double**,DeviceType> coarseCoefficients("coarse", 2, hexCoarse->getCardinality());
    Kokkos::View<double**,DeviceType> fineCoefficients  ("fine",   2, hexFine  ->getCardinality());
    Kokkos::deep_copy(coarseCoefficients, 1.0);
    hexProlongator.apply(fineCoefficients, coarseCoefficients);
    // the constant function is reproduced exactly by nodal bases: all fine coefficients should be 1
    auto fineHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fineCoefficients);
    double maxError = 0.0;
    for (unsigned c=0; c<fineHost.extent(0); c++)
      for (unsigned i=0; i<fineHost.extent(1); i++)
        maxError = std::max(maxError, std::abs(fineHost(c,i) - 1.0));
    TEST_ASSERT(maxError < 1e-13);
  }

  TEUCHOS_UNIT_TEST( NodalBasisProlongator, InvalidArguments )
  {
    auto p2 = getBasis<DerivedNodalFamily>(quadTopo(), FUNCTION_SPACE_HGRAD, 2);
    auto p3 = getBasis<DerivedNodalFamily>(quadTopo(), FUNCTION_SPACE_HGRAD, 3);
    auto hvol = getBasis<DerivedNodalFamily>(quadTopo(), FUNCTION_SPACE_HVOL, 3);
    TEST_THROW(Prolongator(p3, p2),   std::invalid_argument); // coarse not contained in fine
    TEST_THROW(Prolongator(p2, hvol), std::invalid_argument); // function spaces differ

    Prolongator prolongator(p2, p3);
    Kokkos::View<double**,DeviceType> coarse("coarse", 2, p2->getCardinality());
    Kokkos::View<double**,DeviceType> wrongFine("wrong fine", 2, p2->getCardinality());
    Kokkos::View<double**,DeviceType> wrongCells("wrong cells", 3, p3->getCardinality());
    TEST_THROW(prolongator.apply(wrongFine,  coarse), std::invalid_argument);
    TEST_THROW(prolongator.apply(wrongCells, coarse), std::invalid_argument);
  }
} // namespace
