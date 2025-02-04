#ifdef FIX_CLASS

FixStyle(active2dsecondorder2,Fixactive2DSecondOrder2)

#else

#ifndef LMP_FIX_active2D_second_order2_H
#define LMP_FIX_active2D_second_order2_H

#include "fix.h"


namespace LAMMPS_NS {

class Fixactive2DSecondOrder2 : public Fix {
 public:
  Fixactive2DSecondOrder2(class LAMMPS *, int, char **);
  virtual ~Fixactive2DSecondOrder2();
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  // void grow_arrays(int nmax);

 private: 
  double dt, sqrtdt;
  double **eta;  // Add eta declaration

 protected:
  class RanMars *random;
  int seed;
  double mass, t_start, t_stop, t_period, t_target, tsqrt, zeta, Fa, T;
  double gamma1, gamma2, gamma3;
  double D_t, D_r, cosphi, sinphi;  // D_r is declared here
  double v_active;
  char *id_temp;
  void compute_target();
};

}

#endif
#endif