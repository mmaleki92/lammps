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
    double **mu = atom->mu; // Orientation vectors
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    double dt = update->dt;

    // Temporary storage for new directions
    std::vector<double> avg_orient_x(nlocal, 0.0);
    std::vector<double> avg_orient_y(nlocal, 0.0);
    std::vector<int> neighbor_counts(nlocal, 0);

    // Access the neighbor list
    int **neighbors = neighbor->firstneigh;
    int *num_neighbors = neighbor->numneigh;

    // Loop through each particle
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            double sum_x = 0.0;
            double sum_y = 0.0;
            int count = 0;

            // Loop over neighbors of particle i
            for (int jj = 0; jj < num_neighbors[i]; jj++) {
                int j = neighbors[i][jj];
                if (mask[j] & groupbit) {
                    sum_x += mu[j][0];
                    sum_y += mu[j][1];
                    count++;
                }
            }

            if (count > 0) {
                avg_orient_x[i] = sum_x / count;
                avg_orient_y[i] = sum_y / count;
                neighbor_counts[i] = count;
            }
        }
    }

    // Update positions and orientations
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            double norm = sqrt(avg_orient_x[i] * avg_orient_x[i] + avg_orient_y[i] * avg_orient_y[i]);
            if (norm > 0) {
                // Add noise to the orientation
                double noise_angle = noise_strength * (random->uniform() - 0.5) * 2 * M_PI;
                mu[i][0] = avg_orient_x[i] / norm * cos(noise_angle) - avg_orient_y[i] / norm * sin(noise_angle);
                mu[i][1] = avg_orient_x[i] / norm * sin(noise_angle) + avg_orient_y[i] / norm * cos(noise_angle);
            } else {
                // Assign a random orientation if norm is zero
                double angle = 2 * M_PI * random->uniform();
                mu[i][0] = cos(angle);
                mu[i][1] = sin(angle);
            }

            // Update position based on velocity and orientation
            x[i][0] += v_active * mu[i][0] * dt;
            x[i][1] += v_active * mu[i][1] * dt;
        }
    }

    // Apply periodic boundary conditions
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            domain->remap(x[i]);
        }
    }
}
