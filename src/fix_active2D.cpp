#include "fix_active2D.h"
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

void crossProduct2D(double v_A[], double v_B[], double c_P[]) {
    c_P[0] = v_A[1] * v_B[2] - v_A[2] * v_B[1];
    c_P[1] = -(v_A[0] * v_B[2] - v_A[2] * v_B[0]);
    c_P[2] = v_A[0] * v_B[1] - v_A[1] * v_B[0];
}

/* ---------------------------------------------------------------------- */

Fixactive2D::Fixactive2D(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
    if (narg != 9) error->all(FLERR,"Illegal fix active_2d command");

    t_start = utils::numeric(FLERR, arg[3], false, lmp);
    t_target = t_start;
    t_stop = utils::numeric(FLERR, arg[4], false, lmp);
    D_t = utils::numeric(FLERR, arg[5], false, lmp);  // Diffusion coefficient
    if (D_t <= 0.0) error->all(FLERR, "Fix bd diffusion coefficient must be > 0.0");
    D_r = utils::numeric(FLERR, arg[6], false, lmp);  // Rotational Diffusion coefficient
    v_active = utils::numeric(FLERR, arg[7], false, lmp);  // Active velocity
    seed = utils::numeric(FLERR, arg[8], false, lmp);  // Seed for random number generator
    if (seed <= 0) error->all(FLERR, "Illegal fix active 2d command");

    random = new RanMars(lmp, seed + comm->me);
}

/* ---------------------------------------------------------------------- */

Fixactive2D::~Fixactive2D() {
    delete random;
}

/* ---------------------------------------------------------------------- */

int Fixactive2D::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    return mask;
}

/* ---------------------------------------------------------------------- */

void Fixactive2D::init() {
    compute_target();
    gamma1 = D_t;  // Translational diffusion coefficient
    gamma2 = sqrt(2 * D_t);  // Scaling for random translational force
    gamma3 = sqrt(2 * 3 * D_t);  // Scaling for random rotational force
}

/* ----------------------------------------------------------------------
   Compute target temperature for simulation
------------------------------------------------------------------------- */

void Fixactive2D::compute_target() {
    double delta = update->ntimestep - update->beginstep;
    if (delta != 0.0) delta /= update->endstep - update->beginstep;
    t_target = t_start + delta * (t_stop - t_start);
}

/* ----------------------------------------------------------------------
   Integration step for updating particle positions and orientations
------------------------------------------------------------------------- */

void Fixactive2D::initial_integrate(int vflag) {
    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double **mu = atom->mu;  // Orientation vectors
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    int step = update->ntimestep;

    if (igroup == atom->firstgroup) nlocal = atom->nfirst;

    double dt = update->dt;  // Time step
    double sqrtdt = sqrt(dt);
    double sigma = 1.0;
    double Pe = v_active / (sigma * D_r);  // v_active is the same as v_0
    double gamma = 1 / D_t; // friction
    double epsilon = 1; // LJ epsilon
    double Gamma = epsilon / ((sigma * sigma) * gamma * D_r);
    compute_target();  // Update target temperature or other parameters

    // Initialize orientation at the beginning of the simulation
    if (step <= 1) {
        for (int i = 0; i < nlocal; i++) {
            double angle = 2 * M_PI * random->uniform();
            mu[i][0] = cos(angle);
            mu[i][1] = sin(angle);
        }
    }

    // Integration step for particle motion
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            // Rotational diffusion
            double dtheta = sqrt(2 * D_r * dt) * random->gaussian();
            double old_mu_x = mu[i][0];
            double old_mu_y = mu[i][1];

            double old_x = x[i][0];
            double old_y = x[i][1];
            double old_z = x[i][2];


            mu[i][0] = cos(dtheta) * old_mu_x - sin(dtheta) * old_mu_y;
            mu[i][1] = sin(dtheta) * old_mu_x + cos(dtheta) * old_mu_y;

            // Translational diffusion and active motion
            double dx = Pe * mu[i][0] * dt + dt * Gamma * f[i][0] + sqrt(2 * D_t * dt) * random->gaussian();
            double dy = Pe * mu[i][1] * dt + dt * Gamma * f[i][1] + sqrt(2 * D_t * dt) * random->gaussian();

            // Update positions
            x[i][0] += dx;
            x[i][1] += dy;
            x[i][2] = 0;  // Assuming 2D

            // Calculate and update velocities
            v[i][0] = dx / dt;  // Velocity in x direction
            v[i][1] = dy / dt;  // Velocity in y direction
            v[i][2] = 0;        // Velocity in z direction (2D simulation)
        }
    }
}

/* ---------------------------------------------------------------------- */
