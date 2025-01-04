#include "fix_vicsek.h"
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
#include <omp.h>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixVicsek::FixVicsek(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
    if (narg != 7) error->all(FLERR, "Illegal fix vicsek command");

    noise_strength = utils::numeric(FLERR, arg[3], false, lmp);
    v_active = utils::numeric(FLERR, arg[4], false, lmp);
    neighbor_radius = utils::numeric(FLERR, arg[5], false, lmp);
    seed = utils::numeric(FLERR, arg[6], false, lmp);
    if (seed <= 0) error->all(FLERR, "Seed must be > 0");

    random = new RanMars(lmp, seed + comm->me);
}

/* ---------------------------------------------------------------------- */

FixVicsek::~FixVicsek() {
    delete random;
}

/* ---------------------------------------------------------------------- */

int FixVicsek::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixVicsek::init() {
    // Initialize orientations if needed
}

/* ---------------------------------------------------------------------- */


void FixVicsek::initial_integrate(int vflag) {
    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double **mu = atom->mu;  // Orientation vectors
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    double dt = update->dt;
    double sqrtdt = sqrt(dt);

    // Temporary storage for new directions
    std::vector<double> avg_orient_x(nlocal, 0.0);
    std::vector<double> avg_orient_y(nlocal, 0.0);

    // Parallelize the outer loop
    #pragma omp parallel for
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            double sum_x = 0.0;
            double sum_y = 0.0;
            int neighbor_count = 0;

            // Calculate average orientation of neighbors
            for (int j = 0; j < nlocal; j++) {
                if (mask[j] & groupbit) {
                    double dx = x[j][0] - x[i][0];
                    double dy = x[j][1] - x[i][1];
                    double r = sqrt(dx * dx + dy * dy);

                    if (r < neighbor_radius) {
                        sum_x += mu[j][0];
                        sum_y += mu[j][1];
                        neighbor_count++;
                    }
                }
            }

            if (neighbor_count > 0) {
                avg_orient_x[i] = sum_x / neighbor_count;
                avg_orient_y[i] = sum_y / neighbor_count;
            }
        }
    }

    // Loop for updating orientation and velocity
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            double norm = sqrt(avg_orient_x[i] * avg_orient_x[i] + avg_orient_y[i] * avg_orient_y[i]);
            if (norm > 0) {
                double noise_angle = noise_strength * (random->uniform() - 0.5) * 2 * M_PI;
                mu[i][0] = avg_orient_x[i] / norm * cos(noise_angle) - avg_orient_y[i] / norm * sin(noise_angle);
                mu[i][1] = avg_orient_x[i] / norm * sin(noise_angle) + avg_orient_y[i] / norm * cos(noise_angle);
            } else {
                // Handle case where norm == 0 (e.g., assign random orientation)
                double angle = 2 * M_PI * random->uniform();
                mu[i][0] = cos(angle);
                mu[i][1] = sin(angle);
            }

            // Calculate velocity based on active velocity and orientation
            v[i][0] = v_active * mu[i][0];
            v[i][1] = v_active * mu[i][1];
            v[i][2] = 0.0;  // Assuming 2D dynamics, no velocity in z-direction

            x[i][0] += v_active * mu[i][0] * dt;
            x[i][1] += v_active * mu[i][1] * dt;
        }
    }
}
