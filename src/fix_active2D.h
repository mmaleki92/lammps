#ifdef FIX_CLASS

FixStyle(active2d,Fixactive2D)

#else

#ifndef LMP_FIX_active2D_H
#define LMP_FIX_active2D_H

#include "fix.h"

namespace LAMMPS_NS {

class Fixactive2D : public Fix {
 public:
  Fixactive2D(class LAMMPS *, int, char **);
  virtual ~Fixactive2D();
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);


 private: 
 double dt, sqrtdt;

 protected:
  class RanMars *random;
  int seed;
  double t_start,t_stop,t_period,t_target,tsqrt;
  double gamma1,gamma2,gamma3;
  double D_t, D_r,cosphi,sinphi;
  double v_active;
  char *id_temp;
  void compute_target();

};

}

#endif
#endif