#ifdef FIX_CLASS

FixStyle(active2du,Fixactive2DU)

#else

#ifndef LMP_FIX_active2Du_H
#define LMP_FIX_active2Du_H

#include "fix.h"

namespace LAMMPS_NS {

class Fixactive2DU : public Fix {
 public:
  Fixactive2DU(class LAMMPS *, int, char **);
  virtual ~Fixactive2DU();
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);


 private: 
 double dt, sqrtdt;

 protected:
  class RanMars *random;
  int seed;
  double eta, mass, t_start,t_stop,t_period,t_target,tsqrt, zeta, Fa;
  double gamma1,gamma2,gamma3;
  double D_t, D_r,cosphi,sinphi;
  double v_active;
  char *id_temp;
  void compute_target();

};

}

#endif
#endif