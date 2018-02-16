#ifndef VTK_H_RENDERER_RASTERIZER_HPP
#define VTK_H_RENDERER_RASTERIZER_HPP

#include <vtkh/rendering/Renderer.hpp>

namespace vtkh {

class Rasterizer : public Renderer
{
public:
  Rasterizer();
  virtual ~Rasterizer();
  static Renderer::vtkmCanvasPtr GetNewCanvas(int width = 1024, int height = 1024);
};

} // namespace vtkh
#endif
