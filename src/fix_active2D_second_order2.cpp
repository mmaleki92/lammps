#include "fix_active2D_second_order2.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "math_extra.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

void crossProduct2D2ndOrder(double v_A[], double v_B[], double c_P[]) {
  c_P[0] = v_A[1] * v_B[2] - v_A[2] * v_B[1];
  c_P[1] = -(v_A[0] * v_B[2] - v_A[2] * v_B[0]);
  c_P[2] = v_A[0] * v_B[1] - v_A[1] * v_B[0];
}

/* ---------------------------------------------------------------------- */

Fixactive2DSecondOrder2::Fixactive2DSecondOrder2(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
  if (narg != 10) error->all(FLERR, "Illegal fix active_2d command");

  t_start = utils::numeric(FLERR, arg[3], false, lmp);
  t_target = t_start;
  t_stop = utils::numeric(FLERR, arg[4], false, lmp);
  D_t = utils::numeric(FLERR, arg[5], false, lmp);  // Diffusion coefficient
  if (D_t <= 0.0) error->all(FLERR, "Fix bd diffusion coefficient must be > 0.0");
  D_r = utils::numeric(FLERR, arg[6], false, lmp);  // Rotational Diffusion coefficient
  v_active = utils::numeric(FLERR, arg[7], false, lmp);  // Active velocity
  seed = utils::numeric(FLERR, arg[8], false, lmp);  // Seed for random number generator
  if (seed <= 0) error->all(FLERR, "Illegal fix active_2d command");
  zeta = utils::numeric(FLERR, arg[9], false, lmp);  // Zeta value
  if (zeta <= 0.0) error->all(FLERR, "Zeta must be > 0.0");

  random = new RanMars(lmp, seed + comm->me);
}


/* ---------------------------------------------------------------------- */

Fixactive2DSecondOrder2::~Fixactive2DSecondOrder2() {
  delete random;
}

/* ---------------------------------------------------------------------- */

int Fixactive2DSecondOrder2::setmask() {
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void Fixactive2DSecondOrder2::init() {
  compute_target();
  gamma1 = D_t;  // Translational diffusion coefficient
  gamma2 = sqrt(2 * D_t);  // Scaling for random translational force
  gamma3 = sqrt(2 * 3 * D_t);  // Scaling for random rotational force
}

/* ----------------------------------------------------------------------
   Compute target temperature for simulation
------------------------------------------------------------------------- */

void Fixactive2DSecondOrder2::compute_target() {
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;
  t_target = t_start + delta * (t_stop - t_start);
}

/* ----------------------------------------------------------------------
   Integration step for updating particle positions and orientations
------------------------------------------------------------------------- */

void Fixactive2DSecondOrder2::initial_integrate(int vflag) {
  double **x = atom->x; 
  double **v = atom->v;
  double **f = atom->f;
  double **mu = atom->mu;  // Orientation vectors
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int step = update->ntimestep;
  double dt = update->dt; // Time step
  double sqrtdt = sqrt(dt); // Square root of time step
  double gamma_t = 1.0 / D_t; // Translational friction (gamma = 1/D_t)
  double gamma_r = 1.0 / D_r; // Rotational friction
  double mass = 1.0;               // Particle mass (assuming m = 1 for simplicity)
  // double sqrt_2gamma_kBT = sqrt(2.0 * gamma_t * t_target);  // Noise term for Langevin thermostat
  // double zeta = 10.0;
  double KT = 1.0;

  double F_a = zeta * v_active;
  double m = 1.0;
  double sigma = 1.0;
  double epsilon = 1.0;
  double tau = sqrt(mass * sigma * sigma / epsilon);
  // Compute non-dimensional parameters
  double D_t_nd = D_t * tau / (sigma * sigma);
  double D_r_nd = D_r * tau;
  double F_a_nd = F_a * sigma / epsilon;
  double T_nd = KT / epsilon;
  double zeta_nd = zeta * tau / m;
  double dt_nd = dt / tau;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;


  // Initialize orientations at the start of the simulation
  if (step  <= 1) {
    for (int i = 0; i < nlocal; i++) {
      double angle = 2 * M_PI * random->uniform();
      mu[i][0] = cos(angle);  // x-component of orientation
      mu[i][1] = sin(angle);  // y-component of orientation
      mu[i][2] = 0.0;
      v[i][0] = 0.0;  // x-component of orientation
      v[i][1] = 0.0;  // y-component of orientation
      v[i][2] = 0.0; 
    }
  }

  // Integration step for particle motion
  for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
          // Update orientation
          double dtheta = sqrt(2.0 * D_r_nd * dt_nd) * random->gaussian();
          double cos_dtheta = cos(dtheta);
          double sin_dtheta = sin(dtheta);
          double mu_x = mu[i][0];
          double mu_y = mu[i][1];
          mu[i][0] = mu_x * cos_dtheta - mu_y * sin_dtheta;
          mu[i][1] = mu_x * sin_dtheta + mu_y * cos_dtheta;

          // Compute forces and noise
          double noise_x = sqrt(2.0 * zeta_nd * T_nd * dt_nd) * random->gaussian();
          double noise_y = sqrt(2.0 * zeta_nd * T_nd * dt_nd) * random->gaussian();

          // Update velocity
          v[i][0] += dt_nd * ((f[i][0] / epsilon - zeta_nd * v[i][0] + F_a_nd * mu[i][0])) + noise_x;
          v[i][1] += dt_nd * ((f[i][1] / epsilon - zeta_nd * v[i][1] + F_a_nd * mu[i][1])) + noise_y;
          v[i][2] = 0;        // Velocity in z direction (2D simulation)

          // Update position
          x[i][0] += v[i][0] * dt_nd * sigma;
          x[i][1] += v[i][1] * dt_nd * sigma;
          x[i][2] = 0;  // Assuming 2D
      }
  }
}