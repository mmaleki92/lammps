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
Fixactive2DSecondOrder2::Fixactive2DSecondOrder2(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
    if (narg != 8) error->all(FLERR,"Illegal fix active_2d command: fix ID group active2D zeta T Dr Fa seed");

    zeta = utils::numeric(FLERR, arg[3], false, lmp);
    T = utils::numeric(FLERR, arg[4], false, lmp);
    D_r = utils::numeric(FLERR, arg[5], false, lmp);
    Fa = utils::numeric(FLERR, arg[6], false, lmp);
    seed = utils::numeric(FLERR, arg[7], false, lmp);
    if (seed <= 0) error->all(FLERR, "Illegal fix active_2d command");

    random = new RanMars(lmp, seed + comm->me);
    atom->add_callback(Atom::GROW);

    // Initialize mass to be determined in init()
    mass = 0.0;
}

/* ---------------------------------------------------------------------- */

Fixactive2DSecondOrder2::~Fixactive2DSecondOrder2() {
    delete random;
}

/* ---------------------------------------------------------------------- */

int Fixactive2DSecondOrder2::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    mask |= FINAL_INTEGRATE;
    return mask;
}

/* ---------------------------------------------------------------------- */

void Fixactive2DSecondOrder2::init() {
    int *mask = atom->mask;
    int *type = atom->type;
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int nlocal = atom->nlocal;

    // Determine particle mass
    if (rmass) {
        double mass_val = -1.0;
        for (int i = 0; i < nlocal; ++i) {
            if (mask[i] & groupbit) {
                if (mass_val < 0.0) mass_val = rmass[i];
                else if (rmass[i] != mass_val)
                    error->all(FLERR, "All particles must have the same mass for fix active2D");
            }
        }
        double mass_all;
        MPI_Allreduce(&mass_val, &mass_all, 1, MPI_DOUBLE, MPI_MIN, world);
        if (mass_val != mass_all)
            error->all(FLERR, "All particles must have the same mass for fix active2D");
        this->mass = mass_val;
    } else {
        int type_val = -1;
        for (int i = 0; i < nlocal; ++i) {
            if (mask[i] & groupbit) {
                if (type_val < 0) type_val = type[i];
                else if (type[i] != type_val)
                    error->all(FLERR, "All particles must be of the same type for fix active2D");
            }
        }
        int type_all;
        MPI_Allreduce(&type_val, &type_all, 1, MPI_INT, MPI_MIN, world);
        if (type_val != type_all)
            error->all(FLERR, "All particles must be of the same type for fix active2D");
        this->mass = atom->mass[type_all];
    }

    if (this->mass <= 0.0)
        error->all(FLERR, "Particle mass must be positive for fix active2D");
}

/* ---------------------------------------------------------------------- */

void Fixactive2DSecondOrder2::initial_integrate(int vflag) {
    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double **mu = atom->mu;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double dt = update->dt;
    int step = update->ntimestep;

    const double noise_scale = sqrt(zeta * T * dt / mass);
    const double dt_half_mass = 0.5 * dt / mass;


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

    for (int i = 0; i < nlocal; ++i) {
        if (mask[i] & groupbit) {
            // Generate translational noise for initial half-step
            double eta_x = noise_scale * random->gaussian();
            double eta_y = noise_scale * random->gaussian();

            // Compute active force
            double F_active_x = Fa * mu[i][0];
            double F_active_y = Fa * mu[i][1];

            // Total force: pair + active - friction + noise
            double total_force_x = f[i][0] + F_active_x - zeta * v[i][0] + eta_x;
            double total_force_y = f[i][1] + F_active_y - zeta * v[i][1] + eta_y;

            // Velocity update (first half)
            v[i][0] += dt_half_mass * total_force_x;
            v[i][1] += dt_half_mass * total_force_y;

            // Position update
            x[i][0] += v[i][0] * dt;
            x[i][1] += v[i][1] * dt;
            x[i][2] = 0.0;  // Maintain 2D
        }
    }
}

/* ---------------------------------------------------------------------- */

void Fixactive2DSecondOrder2::final_integrate() {
    double **v = atom->v;
    double **f = atom->f;
    double **mu = atom->mu;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double dt = update->dt;

    const double noise_scale = sqrt(zeta * T * dt / mass);
    const double dt_half_mass = 0.5 * dt / mass;
    const double rot_noise_scale = sqrt(2.0 * D_r * dt);

    for (int i = 0; i < nlocal; ++i) {
        if (mask[i] & groupbit) {
            // Generate translational noise for final half-step
            double eta_x = noise_scale * random->gaussian();
            double eta_y = noise_scale * random->gaussian();

            // Compute active force with current mu
            double F_active_x = Fa * mu[i][0];
            double F_active_y = Fa * mu[i][1];

            // Total force: new pair + active - friction + noise
            double total_force_x = f[i][0] + F_active_x - zeta * v[i][0] + eta_x;
            double total_force_y = f[i][1] + F_active_y - zeta * v[i][1] + eta_y;

            // Velocity update (second half)
            v[i][0] += dt_half_mass * total_force_x;
            v[i][1] += dt_half_mass * total_force_y;

            // Apply rotational diffusion
            double dtheta = rot_noise_scale * random->gaussian();
            double old_mu_x = mu[i][0];
            double old_mu_y = mu[i][1];

            mu[i][0] = cos(dtheta) * old_mu_x - sin(dtheta) * old_mu_y;
            mu[i][1] = sin(dtheta) * old_mu_x + cos(dtheta) * old_mu_y;

            // Normalize orientation
            double norm = sqrt(mu[i][0]*mu[i][0] + mu[i][1]*mu[i][1]);
            if (norm > 0.0) {
                mu[i][0] /= norm;
                mu[i][1] /= norm;
            }
        }
    }
}

