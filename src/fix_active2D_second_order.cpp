/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Fix for second-order active Brownian particles (2D)
------------------------------------------------------------------------- */

#include "fix_active2D_second_order.h"
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

/* ----------------------------------------------------------------------
   Cross product function for 2D vectors, keeping z=0
------------------------------------------------------------------------- */

void crossProduct2D2ndOrder(double v_A[], double v_B[], double c_P[]) {
  c_P[0] = v_A[1] * v_B[2] - v_A[2] * v_B[1];
  c_P[1] = -(v_A[0] * v_B[2] - v_A[2] * v_B[0]);
  c_P[2] = v_A[0] * v_B[1] - v_A[1] * v_B[0];
}

/* ----------------------------------------------------------------------
   Constructor for the fix, reading parameters and setting initial values
------------------------------------------------------------------------- */

Fixactive2DSecondOrder::Fixactive2DSecondOrder(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg != 10) error->all(FLERR, "Illegal fix active_2d command");

  t_start = utils::numeric(FLERR, arg[3], false, lmp);
  t_target = t_start;
  t_stop = utils::numeric(FLERR, arg[4], false, lmp);

  D_t = utils::numeric(FLERR, arg[5], false, lmp);    // Translational diffusion
  if (D_t <= 0.0) error->all(FLERR, "Fix bd, D_t must be > 0.0");

  D_r = utils::numeric(FLERR, arg[6], false, lmp);    // Rotational diffusion
  v_active = utils::numeric(FLERR, arg[7], false, lmp);  // Active velocity
  seed = utils::numeric(FLERR, arg[8], false, lmp);      // Random seed
  if (seed <= 0) error->all(FLERR, "Illegal fix active_2d command");
  
  zeta = utils::numeric(FLERR, arg[9], false, lmp);      // Zeta (drag)
  if (zeta <= 0.0) error->all(FLERR, "Zeta must be > 0.0");

  // Create a random number generator for each MPI rank
  random = new RanMars(lmp, seed + comm->me);
}

/* ----------------------------------------------------------------------
   Destructor
------------------------------------------------------------------------- */

Fixactive2DSecondOrder::~Fixactive2DSecondOrder() {
  delete random;
}

/* ----------------------------------------------------------------------
   Setmask: determines which parts of the integration process this fix covers
------------------------------------------------------------------------- */

int Fixactive2DSecondOrder::setmask() {
  int mask = 0;
  // Use both INITIAL_INTEGRATE (for the half-step velocity update + position update)
  // and FINAL_INTEGRATE (for the final half-step velocity update), if desired:
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ----------------------------------------------------------------------
   Init function: compute the target temperature and precompute constants
------------------------------------------------------------------------- */

void Fixactive2DSecondOrder::init() {
  compute_target();
  
  // Example usage of gamma1, gamma2, gamma3, etc., if needed
  gamma1 = D_t;              
  gamma2 = sqrt(2 * D_t);    
  gamma3 = sqrt(2 * 3 * D_t);
}

/* ----------------------------------------------------------------------
   Update target temperature T(t) if needed
------------------------------------------------------------------------- */

void Fixactive2DSecondOrder::compute_target() {
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;
  t_target = t_start + delta * (t_stop - t_start);
}

/* ----------------------------------------------------------------------
   INITIAL_INTEGRATE stage:
   1) Velocity half-step update
   2) Position full-step update
   3) Orientation update (if you handle it here)
------------------------------------------------------------------------- */

void Fixactive2DSecondOrder::initial_integrate(int /*vflag*/) {
  double **x = atom->x; 
  double **v = atom->v;
  double **f = atom->f;
  double **mu = atom->mu;   // Orientation vectors
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // Basic simulation parameters
  int step = update->ntimestep;
  double dt = update->dt;            // Timestep
  double half_dt = 0.5 * dt;         // Half-step
  double zeta_inv = 1.0 / zeta;      // Inverse drag (if used)
  
  // Non-dimensionalization references (example):
  double mass = 1.0;        // mass for each particle (example assumption)
  double sigma = 1.0;       // length scale
  double epsilon = 1.0;     // energy scale
  double tau = sqrt(mass * sigma * sigma / epsilon);  
  double dt_nd = dt / tau;  
  double F_a = zeta * v_active;      // typical active force
  double F_a_nd = F_a * sigma / epsilon; 
  double T_nd = 1.0 / epsilon;       // example if KT=1.0
  double zeta_nd = zeta * tau / mass;
  double D_r_nd = D_r * tau;

  // Optionally limit integration to the "firstgroup" subset
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // Initialize orientations if at the beginning of the run
  if (step <= 1) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        double angle = 2.0 * M_PI * random->uniform();
        mu[i][0] = cos(angle);
        mu[i][1] = sin(angle);
        mu[i][2] = 0.0;
        v[i][0] = 0.0;
        v[i][1] = 0.0;
        v[i][2] = 0.0;
      }
    }
  }

  // Main integration loop
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      // --------------------------------
      // 1) Velocity half-step update
      //    Here we incorporate active force, damping, random force, etc.
      // --------------------------------
      double noise_x = sqrt(2.0 * zeta_nd * T_nd * dt_nd) * random->gaussian();
      double noise_y = sqrt(2.0 * zeta_nd * T_nd * dt_nd) * random->gaussian();

      // Convert LAMMPS' force f[i] to nondimensional if necessary
      // Example: f_nd = f[i][0]/epsilon for x-component, etc.

      double fx_nd = (f[i][0] / epsilon) - (zeta_nd * v[i][0]) + (F_a_nd * mu[i][0]);
      double fy_nd = (f[i][1] / epsilon) - (zeta_nd * v[i][1]) + (F_a_nd * mu[i][1]);

      // Velocity half-step
      v[i][0] += half_dt * fx_nd + noise_x * 0.5; 
      v[i][1] += half_dt * fy_nd + noise_y * 0.5; 
      v[i][2] = 0.0;  // Remain 2D

      // --------------------------------
      // 2) Position full-step update
      //    Multiply by sigma to convert back to real length units
      // --------------------------------
      x[i][0] += dt * v[i][0] * sigma;
      x[i][1] += dt * v[i][1] * sigma;
      x[i][2]  = 0.0;

      // --------------------------------
      // 3) Orientation update
      //    For 2D, we track a single angle or directly update mu[].
      //    Rotational diffusion: dtheta = sqrt(2 D_r_nd dt_nd) ...
      // --------------------------------
      double dtheta = sqrt(2.0 * D_r_nd * dt_nd) * random->gaussian();
      double cos_dtheta = cos(dtheta);
      double sin_dtheta = sin(dtheta);
      double mu_old_x = mu[i][0];
      double mu_old_y = mu[i][1];

      mu[i][0] = mu_old_x * cos_dtheta - mu_old_y * sin_dtheta;
      mu[i][1] = mu_old_x * sin_dtheta + mu_old_y * cos_dtheta;
      mu[i][2] = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   FINAL_INTEGRATE stage:
   4) Velocity final half-step update
------------------------------------------------------------------------- */

void Fixactive2DSecondOrder::final_integrate() {
  double **v = atom->v;
  double **f = atom->f;
  double **mu = atom->mu;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double dt = update->dt;
  double half_dt = 0.5 * dt;

  // Non-dimensional references (match initial_integrate)
  double mass = 1.0;
  double sigma = 1.0;
  double epsilon = 1.0;
  double tau = sqrt(mass * sigma * sigma / epsilon);
  double dt_nd = dt / tau;
  double F_a = zeta * v_active;
  double F_a_nd = F_a * sigma / epsilon;
  double T_nd = 1.0 / epsilon;
  double zeta_nd = zeta * tau / mass;

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // Final half-step velocity update
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      double noise_x = sqrt(2.0 * zeta_nd * T_nd * dt_nd) * random->gaussian();
      double noise_y = sqrt(2.0 * zeta_nd * T_nd * dt_nd) * random->gaussian();

      double fx_nd = (f[i][0] / epsilon) - (zeta_nd * v[i][0]) + (F_a_nd * mu[i][0]);
      double fy_nd = (f[i][1] / epsilon) - (zeta_nd * v[i][1]) + (F_a_nd * mu[i][1]);

      // Final half-step velocity update
      v[i][0] += half_dt * fx_nd + noise_x * 0.5; 
      v[i][1] += half_dt * fy_nd + noise_y * 0.5;
      v[i][2] = 0.0; 
    }
  }
}