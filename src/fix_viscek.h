#ifdef FIX_CLASS

FixStyle(vicsek, FixVicsek)

#else

#ifndef LMP_FIX_VICSEK_H
#define LMP_FIX_VICSEK_H

#include "fix.h"

namespace LAMMPS_NS {

class FixVicsek : public Fix {
 public:
  FixVicsek(class LAMMPS *, int, char **);
  virtual ~FixVicsek();
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);

 private:
  double dt, sqrtdt;           // Time step and square root of time step
  double noise_strength;       // Strength of the noise
  double v_active;             // Active velocity
  double neighbor_radius;      // Radius to consider neighbors
  int seed;                    // Random seed

  class RanMars *random;       // Random number generator
};

} // namespace LAMMPS_NS

#endif // LMP_FIX_VICSEK_H
#endif
