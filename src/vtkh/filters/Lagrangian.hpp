#ifndef VTK_H_LAGRANGIAN_HPP
#define VTK_H_LAGRANGIAN_HPP

#include <vtkh/vtkh.hpp>
#include <vtkh/filters/Filter.hpp>
#include <vtkh/DataSet.hpp>
#include <vtkm/cont/DataSet.h>

namespace vtkh
{

class Lagrangian : public Filter
{
public:
  Lagrangian(); 
  virtual ~Lagrangian(); 
  std::string GetName() const override;
	void SetField(const std::string &field_name);
  void SetStepSize(const double &step_size);
  void SetWriteFrequency(const int &write_frequency);
	void SetCustomSeedResolution(const int &cust_res);
	void SetSeedResolutionInX(const int &x_res);
	void SetSeedResolutionInY(const int &y_res);
	void SetSeedResolutionInZ(const int &z_res);

private:
  bool AllMessagesReceived(bool *a, int num_ranks);
  bool BoundsCheck(float x, float y, float z, double *BB);
  bool BoundsCheck(float x, float y, float z, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);
  void UpdateSeedResolution(vtkm::cont::DataSet dom);
  void InitializeUniformSeeds(vtkm::cont::DataSet dom, vtkm::Id rank);	
  void ExchangeSeedParticles(vtkm::Id rank, vtkm::Id num_ranks); 
  void ExchangeParticleInformation(vtkm::Id rank, vtkm::Id num_ranks);
  void WriteBasisFlowInformation(vtkm::Id rank, vtkm::Id num_ranks);
  void ResetSeedParticles(vtkm::cont::DataSet dom, vtkm::Id rank);

protected:
  void PreExecute() override;
  void PostExecute() override;
  void DoExecute() override;

  std::string m_field_name;
	double m_step_size;
	int m_write_frequency;
	int m_cust_res;
	int m_x_res, m_y_res, m_z_res;
};

} //namespace vtkh
#endif
