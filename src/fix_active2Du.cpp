#include "fix_active2Du.h"
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
// u for underdamped
Fixactive2DU::Fixactive2DU(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
    if (narg != 7) error->all(FLERR,"Illegal fix active_2d command: fix ID group active2D zeta T Dr Fa seed");

    zeta = utils::numeric(FLERR, arg[3], false, lmp);  // Friction coefficient
    T = utils::numeric(FLERR, arg[4], false, lmp);     // Temperature
    Dr = utils::numeric(FLERR, arg[5], false, lmp);    // Rotational diffusion
    Fa = utils::numeric(FLERR, arg[6], false, lmp);    // Active force magnitude
    seed = utils::numeric(FLERR, arg[7], false, lmp);  // Random seed
    if (seed <= 0) error->all(FLERR, "Illegal fix active_2d command");

    random = new RanMars(lmp, seed + comm->me);
    eta = nullptr;
    grow_arrays(atom->nmax);
    atom->add_callback(Atom::GROW);

    mass = -1.0; // Initialize mass to be determined later
}

/* ---------------------------------------------------------------------- */

Fixactive2DU::~Fixactive2DU() {
    delete random;
    memory->destroy(eta);
}

/* ---------------------------------------------------------------------- */

int Fixactive2DU::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    mask |= FINAL_INTEGRATE;
    return mask;
}

/* ---------------------------------------------------------------------- */

void Fixactive2DU::init() {
    // Verify all particles have the same mass
    int *type = atom->type;
    int *mask = atom->mask;
    // double *mass_vec = atom->mass;
    int nlocal = atom->nlocal;
    double local_mass = -1.0;

    // for (int i = 0; i < nlocal; i++) {
    //     if (mask[i] & groupbit) {
    //         if (local_mass < 0.0) local_mass = mass_vec[type[i]];
    //         else if (mass_vec[type[i]] != local_mass)
    //             error->one(FLERR, "All particles in fix active_2d must have the same mass");
    //     }
    // }
    double mass = 1.0;

    // Determine global mass
    // double all_mass;
    // MPI_Allreduce(&local_mass, &all_mass, 1, MPI_DOUBLE, MPI_MAX, world);
    // if (all_mass <= 0.0) error->all(FLERR, "Invalid particle mass for fix active_2d");
    // mass = all_mass;
}

/* ---------------------------------------------------------------------- */

void Fixactive2DU::grow_arrays(int nmax) {
    memory->grow(eta, nmax, 2, "fix_active2Du:eta");
}

/* ---------------------------------------------------------------------- */

void Fixactive2DU::initial_integrate(int vflag) {
    double **x = atom->x;
    double **v = atom->v;
    double **mu = atom->mu;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double dt = update->dt;

    const double sqrt_dt = sqrt(dt);
    const double noise_scale = sqrt(2.0 * zeta * T / mass) * sqrt_dt;
    const double dt_half_mass = 0.5 * dt / mass;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            // Compute active force
            const double F_active[2] = {Fa * mu[i][0], Fa * mu[i][1]};

            // Generate translational noise
            eta[i][0] = noise_scale * random->gaussian();
            eta[i][1] = noise_scale * random->gaussian();

            // First half-step velocity update
            v[i][0] += dt_half_mass * (F_active[0] - zeta * v[i][0] + eta[i][0]);
            v[i][1] += dt_half_mass * (F_active[1] - zeta * v[i][1] + eta[i][1]);

            // Update position
            x[i][0] += v[i][0] * dt;
            x[i][1] += v[i][1] * dt;
            x[i][2] = 0.0;  // Maintain 2D simulation
        }
    }
}

/* ---------------------------------------------------------------------- */

void Fixactive2DU::final_integrate() {
    double **v = atom->v;
    double **f = atom->f;
    double **mu = atom->mu;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double dt = update->dt;

    const double sqrt_dt = sqrt(dt);
    const double dt_half_mass = 0.5 * dt / mass;
    const double rot_noise_scale = sqrt(2.0 * Dr) * sqrt_dt;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            // Compute active force again
            const double F_active[2] = {Fa * mu[i][0], Fa * mu[i][1]};

            // Second half-step velocity update with forces
            v[i][0] += dt_half_mass * (F_active[0] - zeta * v[i][0] + eta[i][0] + f[i][0]);
            v[i][1] += dt_half_mass * (F_active[1] - zeta * v[i][1] + eta[i][1] + f[i][1]);

            // Apply rotational diffusion
            const double dtheta = rot_noise_scale * random->gaussian();
            const double old_mu_x = mu[i][0];
            const double old_mu_y = mu[i][1];

            mu[i][0] = cos(dtheta) * old_mu_x - sin(dtheta) * old_mu_y;
            mu[i][1] = sin(dtheta) * old_mu_x + cos(dtheta) * old_mu_y;

            // Normalize orientation
            const double norm = sqrt(mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1]);
            if (norm > 0.0) {
                mu[i][0] /= norm;
                mu[i][1] /= norm;
            }
        }
    }
}