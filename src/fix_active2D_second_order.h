#ifdef FIX_CLASS

FixStyle(active2dsecondorder,Fixactive2DSecondOrder)

#else

#ifndef LMP_FIX_active2D_second_order_H
#define LMP_FIX_active2D_second_order_H

#include "fix.h"

namespace LAMMPS_NS {

class Fixactive2DSecondOrder : public Fix {
 public:
  Fixactive2DSecondOrder(class LAMMPS *, int, char **);
  virtual ~Fixactive2DSecondOrder();
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
  double zeta;  // Friction coefficient
  double v_active;
  char *id_temp;
  void compute_target();

};

}

#endif
#endif