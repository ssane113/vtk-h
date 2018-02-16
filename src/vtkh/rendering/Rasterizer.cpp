#include "Rasterizer.hpp"

#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/rendering/MapperRasterizer.h>
#include <memory>

namespace vtkh {
  
Rasterizer::Rasterizer()
{
  typedef vtkm::rendering::MapperRasterizer TracerType;
  auto mapper = std::make_shared<TracerType>();
  mapper->SetCompositeBackground(false);
  this->m_mapper = mapper;
}

Rasterizer::~Rasterizer()
{
}

Renderer::vtkmCanvasPtr 
Rasterizer::GetNewCanvas(int width, int height)
{
  return std::make_shared<vtkm::rendering::CanvasRayTracer>(width, height);
}

} // namespace vtkh
