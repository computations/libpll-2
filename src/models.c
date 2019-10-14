/*
    Copyright (C) 2015 Tomas Flouri, Diego Darriba, Alexandros Stamatakis

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact: Tomas Flouri <Tomas.Flouri@h-its.org>,
    Exelixis Lab, Heidelberg Instutute for Theoretical Studies
    Schloss-Wolfsbrunnenweg 35, D-69118 Heidelberg, Germany
*/

#include "pll.h"
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>

#define PLL_MEMORY_ALLOC_CHECK(ptr) {\
  if(ptr==NULL){\
    pll_errno = PLL_ERROR_MEM_ALLOC;\
    snprintf(pll_errmsg, 200, "Could not allocate memory for %s", #ptr ); \
    return PLL_FAILURE;\
  } \
}

#define PLL_MEMORY_ALLOC_CHECK_MSG(ptr, msg...) {\
  if(ptr==NULL){\
    pll_errno = PLL_ERROR_MEM_ALLOC;\
    snprintf(pll_errmsg, 200, msg); \
    return PLL_FAILURE;\
  } \
}

static int mytqli(double *d, double *e, const unsigned int n, double **z)
{
  unsigned int     m, l, iter, i, k;
  double  s, r, p, g, f, dd, c, b;

  for (i = 2; i <= n; i++)
    e[i - 2] = e[i - 1];

  e[n - 1] = 0.0;

  for (l = 1; l <= n; l++)
    {
      iter = 0;
      do
        {
          for (m = l; m <= n - 1; m++)
            {
              dd = fabs(d[m - 1]) + fabs(d[m]);
              if (fabs(e[m - 1]) + dd == dd)
                break;
            }
          if (m != l)
           {
             assert(iter < 30);

             g = (d[l] - d[l - 1]) / (2.0 * e[l - 1]);
             r = sqrt((g * g) + 1.0);
             g = d[m - 1] - d[l - 1] + e[l - 1] / (g + ((g < 0)?-fabs(r):fabs(r)));/*MYSIGN(r, g));*/
             s = c = 1.0;
             p = 0.0;

             for (i = m - 1; i >= l; i--)
               {
                 f = s * e[i - 1];
                 b = c * e[i - 1];
                 if (fabs(f) >= fabs(g))
                   {
                     c = g / f;
                     r = sqrt((c * c) + 1.0);
                     e[i] = f * r;
                     c *= (s = 1.0 / r);
                   }
                 else
                   {
                     s = f / g;
                     r = sqrt((s * s) + 1.0);
                     e[i] = g * r;
                     s *= (c = 1.0 / r);
                   }
                 g = d[i] - p;
                 r = (d[i - 1] - g) * s + 2.0 * c * b;
                 p = s * r;
                 d[i] = g + p;
                 g = c * r - b;
                 for (k = 1; k <= n; k++)
                   {
                     f = z[i][k-1];
                     z[i][k-1] = s * z[i - 1][k - 1] + c * f;
                     z[i - 1][k - 1] = c * z[i - 1][k - 1] - s * f;
                   }
               }

             d[l - 1] = d[l - 1] - p;
             e[l - 1] = g;
             e[m - 1] = 0.0;
           }
        }
      while (m != l);
    }



    return (1);
 }


static void mytred2(double **a, const unsigned int n, double *d, double *e)
{
  unsigned int     l, k, j, i;
  double  scale, hh, h, g, f;

  for (i = n; i > 1; i--)
    {
      l = i - 1;
      h = 0.0;
      scale = 0.0;

      if (l > 1)
        {
          for (k = 1; k <= l; k++)
            scale += fabs(a[k - 1][i - 1]);
          if (scale == 0.0)
            e[i - 1] = a[l - 1][i - 1];
          else
            {
              for (k = 1; k <= l; k++)
                {
                  a[k - 1][i - 1] /= scale;
                  h += a[k - 1][i - 1] * a[k - 1][i - 1];
                }
              f = a[l - 1][i - 1];
              g = ((f > 0) ? -sqrt(h) : sqrt(h)); /* diff */
              e[i - 1] = scale * g;
              h -= f * g;
              a[l - 1][i - 1] = f - g;
              f = 0.0;
              for (j = 1; j <= l; j++)
                {
                  a[i - 1][j - 1] = a[j - 1][i - 1] / h;
                  g = 0.0;
                  for (k = 1; k <= j; k++)
                    g += a[k - 1][j - 1] * a[k - 1][i - 1];
                  for (k = j + 1; k <= l; k++)
                    g += a[j - 1][k - 1] * a[k - 1][i - 1];
                  e[j - 1] = g / h;
                  f += e[j - 1] * a[j - 1][i - 1];
                }
              hh = f / (h + h);
              for (j = 1; j <= l; j++)
                {
                  f = a[j - 1][i - 1];
                  g = e[j - 1] - hh * f;
                  e[j - 1] = g;
                  for (k = 1; k <= j; k++)
                    a[k - 1][j - 1] -= (f * e[k - 1] + g * a[k - 1][i - 1]);
                }
            }
        }
      else
        e[i - 1] = a[l - 1][i - 1];
      d[i - 1] = h;
    }
  d[0] = 0.0;
  e[0] = 0.0;

  for (i = 1; i <= n; i++)
    {
      l = i - 1;
      if (d[i - 1] != 0.0)
        {
          for (j = 1; j <= l; j++)
            {
                g = 0.0;
                for (k = 1; k <= l; k++)
                  g += a[k - 1][i - 1] * a[j - 1][k - 1];
                for(k = 1; k <= l; k++)
                  a[j - 1][k - 1] -= g * a[i - 1][k - 1];
            }
        }
      d[i - 1] = a[i - 1][i - 1];
      a[i - 1][i - 1] = 1.0;
      for (j = 1; j <= l; j++)
        a[i - 1][j - 1] = a[j - 1][i - 1] = 0.0;
    }
}

int pll_nonsym_eigen(double** A,
        size_t n,
        size_t n_padded,
        double* eigenvalues_real,
        double* eigenvalues_imag,
        double* eigenvectors_real,
        double* eigenvectors_imag,
        double* inv_eigenvectors_real,
        double* inv_eigenvectors_imag,
        int* status)
{
    int signum;
    size_t i, j;
    gsl_eigen_nonsymmv_workspace *ws;
    gsl_vector_complex *eigenvalues;
    gsl_matrix *M;
    gsl_matrix_complex *eigenvectors, *inv_eigenvectors, *tmp_eigenvectors;
    gsl_permutation *lu_perm;


    ws = gsl_eigen_nonsymmv_alloc(n);
    PLL_MEMORY_ALLOC_CHECK_MSG(ws, "Could not allocate memory for the GSL workspace");
    M = gsl_matrix_calloc(n,n);
    PLL_MEMORY_ALLOC_CHECK_MSG(M, "Could not allocate memory for a matrix");

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            gsl_matrix_set(M, i, j, A[i][j]);
        }
    }

    eigenvectors = gsl_matrix_complex_calloc(n,n);
    PLL_MEMORY_ALLOC_CHECK(eigenvectors);
    eigenvalues = gsl_vector_complex_calloc(n);
    PLL_MEMORY_ALLOC_CHECK(eigenvalues);

    gsl_eigen_nonsymmv(M, eigenvalues, eigenvectors, ws);

    signum = 1;
    inv_eigenvectors = gsl_matrix_complex_calloc(n,n);
    PLL_MEMORY_ALLOC_CHECK(inv_eigenvectors);
    tmp_eigenvectors = gsl_matrix_complex_calloc(n,n);
    gsl_matrix_complex_memcpy(tmp_eigenvectors, eigenvectors);
    PLL_MEMORY_ALLOC_CHECK(tmp_eigenvectors);
    lu_perm = gsl_permutation_alloc(n);
    PLL_MEMORY_ALLOC_CHECK(lu_perm);

    gsl_complex eigenvec_det = gsl_linalg_complex_LU_det(tmp_eigenvectors, signum);
    if(gsl_complex_abs(eigenvec_det) < PLL_NONREV_EIGEN_DET_THRESHOLD){
      gsl_eigen_nonsymmv_free(ws);
      gsl_matrix_free(M);
      gsl_matrix_complex_free(eigenvectors);
      gsl_matrix_complex_free(inv_eigenvectors);
      gsl_matrix_complex_free(tmp_eigenvectors);
      gsl_vector_complex_free(eigenvalues);
      gsl_permutation_free(lu_perm);
      *status = PLL_NONREV_EIGEN_NONINVERTABLE;
      return PLL_SUCCESS;
    }

    gsl_linalg_complex_LU_decomp(tmp_eigenvectors, lu_perm, &signum);
    gsl_linalg_complex_LU_invert(tmp_eigenvectors, lu_perm, inv_eigenvectors);

    /*
     * TODO: Currently, to access elements I use the range checked matrix_get
     * functions. We can speed these up by defining GSL_RANGE_CHECK_OFF and
     * recompiling. It might be easier to read memory directly from the struct
     * instead. The only problem is that the way that complex numbers are
     * stored is not well documented.
     */
    for (i = 0; i < n; ++i) {
        eigenvalues_real[i] = GSL_REAL(gsl_vector_complex_get(eigenvalues, i));
        eigenvalues_imag[i] = GSL_IMAG(gsl_vector_complex_get(eigenvalues, i));
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            eigenvectors_real[j*n_padded + i] =
                GSL_REAL(gsl_matrix_complex_get(eigenvectors, j, i));
            eigenvectors_imag[j*n_padded + i] =
                GSL_IMAG(gsl_matrix_complex_get(eigenvectors, j, i));
            inv_eigenvectors_real[j*n_padded + i] =
                GSL_REAL(gsl_matrix_complex_get(inv_eigenvectors,j,i));
            inv_eigenvectors_imag[j*n_padded + i] =
                GSL_IMAG(gsl_matrix_complex_get(inv_eigenvectors,j,i));
        }
    }

    gsl_eigen_nonsymmv_free(ws);
    gsl_matrix_free(M);
    gsl_matrix_complex_free(eigenvectors);
    gsl_matrix_complex_free(inv_eigenvectors);
    gsl_matrix_complex_free(tmp_eigenvectors);
    gsl_vector_complex_free(eigenvalues);
    gsl_permutation_free(lu_perm);
    *status = PLL_NONREV_EIGEN_SUCCESS;
    return PLL_SUCCESS;
}

/* TODO: Add code for SSE/AVX. Perhaps allocate qmatrix in one chunk to avoid the
complex checking when to dealloc */
static double ** create_ratematrix(double * params,
                                   double * frequencies,
                                   unsigned int states,
                                   int unsymm)
{
  unsigned int i,j,k,success;

  double ** qmatrix;

  unsigned int params_count = 0;
  if (unsymm)
  {
    params_count = states * states - states;
  }
  else
  {
    params_count = (states * states-states) / 2;
  }
  /* normalize substitution parameters */
  double * params_normalized = (double *)malloc(sizeof(double) * params_count);
  if (!params_normalized)
    return NULL;

  memcpy(params_normalized,params,params_count*sizeof(double));

  if (params_normalized[params_count - 1] > 0.0)
  {
    for (i = 0; i < params_count; ++i)
      params_normalized[i] /= params_normalized[params_count - 1];
  }

  /* allocate qmatrix */
  qmatrix = (double **)malloc(states*sizeof(double *));
  if (!qmatrix)
  {
    free(params_normalized);
    return NULL;
  }

  success = 1;
  for (i = 0; i < states; ++i)
    if (!(qmatrix[i] = (double *)malloc(states*sizeof(double)))) success=0;

  if (!success)
  {
    for(i = 0; i < states; ++i) free(qmatrix[i]);
    free(qmatrix);
    free(params_normalized);
    return NULL;
  }


  if (unsymm)
  {
    k = 0;
    for (i = 0; i < states; ++i)
    {
      double row_sum = 0;
      for (j = 0; j < states; ++j)
      {
        if (i == j)
        {
          continue;
        }
        double factor = params_normalized[k++];
        qmatrix[i][j] = factor * sqrt(frequencies[i] * frequencies[j]);
        row_sum += qmatrix[i][j];
      }
      qmatrix[i][i] = -1 * row_sum;
    }
  }
  else
  {
    /* construct a matrix equal to sqrt(pi) * Q sqrt(pi)^-1 in order to ensure
       it is symmetric */
    /*
     * The above comment is incorrect. This computes
     *      Q .* sqrt(pi*pi'),
     * I.E. the elementwise product of Q with the outer product of pi with
     * itself. This creates a symmetric matrix, if Q was symmetric.
     */
    for (i = 0; i < states; ++i) qmatrix[i][i] = 0;

    k = 0;
    for (i = 0; i < states; ++i)
    {
      for (j = i+1; j < states; ++j)
      {
      double factor = (frequencies[i] <= PLL_EIGEN_MINFREQ ||
                       frequencies[j] <= PLL_EIGEN_MINFREQ) ? 0 : params_normalized[k];
      k++;
      qmatrix[i][j] = qmatrix[j][i] = factor * sqrt(frequencies[i] * frequencies[j]);
      qmatrix[i][i] -= factor * frequencies[j];
      qmatrix[j][j] -= factor * frequencies[i];
      }
    }
  }


  double mean = 0;
  for (i = 0; i < states; ++i)
    mean += frequencies[i] * (-qmatrix[i][i]);
  for (i = 0; i < states; ++i)
  {
    for (j = 0; j < states; ++j)
      qmatrix[i][j] /= mean;
  }

  free(params_normalized);

  return qmatrix;
}

static unsigned int eliminate_zero_states(double **mat, double *forg,
                                          unsigned int states, double *new_forg)
{
  unsigned int i, j, inew, jnew;
  unsigned int new_states = 0;
  for (i = 0; i < states; i++)
  {
    if (forg[i] > PLL_EIGEN_MINFREQ)
      new_forg[new_states++] = forg[i];
  }

  assert(new_states <= states);

  if (new_states < states)
  {
    for (i = 0, inew = 0; i < states; i++)
    {
      if (forg[i] > PLL_EIGEN_MINFREQ)
      {
        for (j = 0, jnew = 0; j < states; j++)
        {
          if (forg[j] > PLL_EIGEN_MINFREQ)
          {
            mat[inew][jnew] = mat[i][j];
            jnew++;
          }
        }
        inew++;
      }
    }
  }

  return new_states;
}

PLL_EXPORT int pll_update_eigen(pll_partition_t * partition,
                                unsigned int params_index)
{
  unsigned int i, j;
  int result_no;
  double *e = NULL, *d = NULL;
  double **a;

  double *eigenvecs = partition->eigenvecs[params_index];
  double *inv_eigenvecs = partition->inv_eigenvecs[params_index];
  double *eigenvals = partition->eigenvals[params_index];
  double *freqs = partition->frequencies[params_index];
  double *subst_params = partition->subst_params[params_index];

  unsigned int states = partition->states;
  unsigned int states_padded = partition->states_padded;

  double *eigenvecs_imag = NULL;
  double *inv_eigenvecs_imag = NULL;
  double *eigenvals_imag = NULL;

  unsigned int inew, jnew;
  unsigned int new_states;
  double *new_freqs = NULL;

  a = create_ratematrix(subst_params, freqs, states,
                        partition->attributes & PLL_ATTRIB_NONREV);
  if (!a) {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg, 200, "Unable to allocate enough memory.");
    return PLL_FAILURE;
  }

  if (partition->attributes & PLL_ATTRIB_NONREV) {
    int status = 0;
    eigenvecs_imag = partition->eigenvecs_imag[params_index];
    inv_eigenvecs_imag = partition->inv_eigenvecs_imag[params_index];
    eigenvals_imag = partition->eigenvals_imag[params_index];
    result_no = pll_nonsym_eigen(a, states, states_padded, eigenvals,
                                 eigenvals_imag, eigenvecs, eigenvecs_imag,
                                 inv_eigenvecs, inv_eigenvecs_imag, &status);
    if (result_no == PLL_FAILURE) {
      return PLL_FAILURE;
    }
    if (status == PLL_NONREV_EIGEN_NONINVERTABLE){
      partition->eigen_decomp_valid[params_index] |= 0x1 | PLL_NONREV_EIGEN_FALLBACK;
    }
  } else {
    d = (double *)malloc(states * sizeof(double));
    e = (double *)malloc(states * sizeof(double));
    new_freqs = (double *)malloc(states * sizeof(double));
    if (!d || !e || !new_freqs) {
      if (d)
        free(d);
      if (e)
        free(e);
      if (new_freqs)
        free(new_freqs);
      for (i = 0; i < states; ++i)
        free(a[i]);
      free(a);
      pll_errno = PLL_ERROR_MEM_ALLOC;
      snprintf(pll_errmsg, 200, "Unable to allocate enough memory.");
      return PLL_FAILURE;
    }

    /* Here we use a technical trick to reduce rate matrix if some states
     * have (near) zero frequencies. Code adapted from IQTree, see:
     * https://github.com/Cibiv/IQ-TREE/commit/f222d317af46cf6abf8bcdb70d4db22475e9a7d2
     */
    new_states = eliminate_zero_states(a, freqs, states, new_freqs);

    mytred2(a, new_states, d, e);
    mytqli(d, e, new_states, a);

    for (i = 0, inew = 0; i < states; i++)
      eigenvals[i] = (freqs[i] > PLL_EIGEN_MINFREQ) ? d[inew++] : 0;

    assert(inew == new_states);

    /* pre-compute square roots of frequencies */
    for (i = 0; i < new_states; i++)
      new_freqs[i] = sqrt(new_freqs[i]);

    if (new_states < states) {
      /* initialize eigenvecs and inv_eigenvecs with diagonal matrix */
      memset(eigenvecs, 0, states_padded * states * sizeof(double));
      memset(inv_eigenvecs, 0, states_padded * states * sizeof(double));

      for (i = 0; i < states; i++) {
        eigenvecs[i * states_padded + i] = 1.;
        inv_eigenvecs[i * states_padded + i] = 1.;
      }

      for (i = 0, inew = 0; i < states; i++) {
        if (freqs[i] > PLL_EIGEN_MINFREQ) {
          for (j = 0, jnew = 0; j < states; j++) {
            if (freqs[j] > PLL_EIGEN_MINFREQ) {
              /* multiply the eigen vectors from the right with sqrt(pi) */
              eigenvecs[i * states_padded + j] =
                  a[inew][jnew] * new_freqs[jnew];
              /* multiply the inverse eigen vectors from the left with
               * sqrt(pi)^-1 */
              inv_eigenvecs[i * states_padded + j] =
                  a[jnew][inew] / new_freqs[inew];
              jnew++;
            }
          }
          inew++;
        }
      }
    } else {
      for (i = 0; i < states; i++) {
        for (j = 0; j < states; j++) {
          /* multiply the eigen vectors from the right with sqrt(pi) */
          eigenvecs[i * states_padded + j] = a[i][j] * new_freqs[j];
          /* multiply the inverse eigen vectors from the left with sqrt(pi)^-1
           */
          inv_eigenvecs[i * states_padded + j] = a[j][i] / new_freqs[i];
        }
      }
    }
  }

  partition->eigen_decomp_valid[params_index] = 1;

  if (d) {
    free(d);
  }
  if (e) {
    free(e);
  }
  if (new_freqs) {
    free(new_freqs);
  }
  for (i = 0; i < states; ++i)
    free(a[i]);
  free(a);

  return PLL_SUCCESS;
}

PLL_EXPORT int pll_update_prob_matrices(pll_partition_t * partition,
                                        const unsigned int * params_indices,
                                        const unsigned int * matrix_indices,
                                        const double * branch_lengths,
                                        unsigned int count)
{
  unsigned int n;

  /* check whether we have cached an eigen decomposition. If not, compute it */
  for (n = 0; n < partition->rate_cats; ++n)
  {
    if (!partition->eigen_decomp_valid[params_indices[n]])
    {
      if (!pll_update_eigen(partition, params_indices[n]))
        return PLL_FAILURE;
    }
  }

  if (partition->attributes & PLL_ATTRIB_NONREV){
    return pll_core_update_pmatrix_nonrev(partition,
                                          params_indices,
                                          matrix_indices,
                                          branch_lengths,
                                          count);

  }
  else{
    return pll_core_update_pmatrix(partition->pmatrix,
                                   partition->states,
                                   partition->rate_cats,
                                   partition->rates,
                                   branch_lengths,
                                   matrix_indices,
                                   params_indices,
                                   partition->prop_invar,
                                   partition->eigenvals,
                                   partition->eigenvecs,
                                   partition->inv_eigenvecs,
                                   count,
                                   partition->attributes);
  }
}

PLL_EXPORT void pll_set_frequencies(pll_partition_t * partition,
                                    unsigned int freqs_index,
                                    const double * frequencies)
{
  unsigned int i;
  double sum = 0.;

  memcpy(partition->frequencies[freqs_index],
         frequencies,
         partition->states*sizeof(double));

  /* make sure frequencies sum up to 1.0 */
  for (i = 0; i < partition->states; ++i)
    sum += partition->frequencies[freqs_index][i];

  if (fabs(sum - 1.0) > PLL_MISC_EPSILON)
  {
    for (i = 0; i < partition->states; ++i)
      partition->frequencies[freqs_index][i] /= sum;
  }

  partition->eigen_decomp_valid[freqs_index] = 0;
}

PLL_EXPORT void pll_set_category_rates(pll_partition_t * partition,
                                       const double * rates)
{
  memcpy(partition->rates, rates, partition->rate_cats*sizeof(double));
}

PLL_EXPORT void pll_set_category_weights(pll_partition_t * partition,
                                         const double * rate_weights)
{
  memcpy(partition->rate_weights, rate_weights,
         partition->rate_cats*sizeof(double));
}

PLL_EXPORT void pll_set_subst_params(pll_partition_t * partition,
                                     unsigned int params_index,
                                     const double * params)
{
  unsigned int count = 0;
  if (partition->attributes & PLL_ATTRIB_NONREV)
  {
    count = partition->states * (partition->states - 1);
  }
  else
  {
    count = partition->states * (partition->states-1) / 2;
  }

  memcpy(partition->subst_params[params_index],
         params, count*sizeof(double));
  partition->eigen_decomp_valid[params_index] = 0;

  /* NOTE: For protein models PLL/RAxML do a rate scaling by 10.0/max_rate */
}

PLL_EXPORT int pll_update_invariant_sites_proportion(pll_partition_t * partition,
                                                     unsigned int params_index,
                                                     double prop_invar)
{

  /* check that there is no ascertainment bias correction */
  if (prop_invar != 0.0 && (partition->attributes & PLL_ATTRIB_AB_MASK))
  {
    pll_errno = PLL_ERROR_INVAR_INCOMPAT;
    snprintf(pll_errmsg,
             200,
             "Invariant sites are not compatible with asc bias correction");
    return PLL_FAILURE;
  }

  /* validate new invariant sites proportion */
  if (prop_invar < 0 || prop_invar >= 1)
  {
    pll_errno = PLL_ERROR_INVAR_PROPORTION;
    snprintf(pll_errmsg,
             200,
             "Invalid proportion of invariant sites (%f)", prop_invar);
    return PLL_FAILURE;
  }

  if (params_index > partition->rate_matrices)
  {
    pll_errno = PLL_ERROR_INVAR_PARAMINDEX;
    snprintf(pll_errmsg,
             200,
             "Invalid params index (%u)", params_index);
    return PLL_FAILURE;
  }

  if (prop_invar > 0.0 && !partition->invariant)
  {
    if (!pll_update_invariant_sites(partition))
    {
      pll_errno = PLL_ERROR_INVAR_NONEFOUND;
      snprintf(pll_errmsg,
               200,
               "No invariant sites found");
      return PLL_FAILURE;
    }
  }

  partition->prop_invar[params_index] = prop_invar;

  return PLL_SUCCESS;
}

PLL_EXPORT unsigned int pll_count_invariant_sites(pll_partition_t * partition,
                                                  unsigned int * state_inv_count)
{
  unsigned int i,j,k;
  unsigned int invariant_count = 0;
  unsigned int tips = partition->tips;
  unsigned int sites = partition->sites;
  unsigned int states = partition->states;
  pll_state_t gap_state = 0;
  pll_state_t cur_state;
  int * invariant = partition->invariant;
  double * tipclv;

  /* gap state has always all bits set to one */
  for (i = 0; i < states; ++i)
  {
    gap_state <<= 1;
    gap_state |= 1;
  }

  if (state_inv_count)
    memset(state_inv_count, 0, states*sizeof(unsigned int));

  if (invariant)
  {
    /* count the invariant sites for each state */
    for (i=0; i<sites; ++i)
    {
      if (invariant[i] > -1)
      {
        cur_state = (pll_state_t) invariant[i];
        /* since the invariant sites array is generated in the library,
           it should not contain invalid values */
        assert (cur_state < states);

        /* increase the counter and per-state count */
        invariant_count += partition->pattern_weights[i];
        if (state_inv_count)
          state_inv_count[cur_state]++;
      }
    }
  }
  else
  {
    if (partition->attributes & PLL_ATTRIB_PATTERN_TIP)
    {
      for (j = 0; j < sites; ++j)
      {
        cur_state = gap_state;
        for (i = 0; i < tips; ++i)
        {
          cur_state &= ((unsigned int)(partition->tipchars[i][j]));
          if  (!cur_state)
          {
            break;
          }
        }
        if (PLL_STATE_POPCNT(cur_state) == 1)
        {
          invariant_count += partition->pattern_weights[j];
          if (state_inv_count)
            state_inv_count[PLL_STATE_CTZ(cur_state)]++;
        }
      }
    }
    else
    {
      /* warning: note that this operation traverses the clvs by columns, and
         hence it may be slow. If PLL_ATTRIB_PATTERN_TIP is not set, I suggest
         to call pll_update_invariant_sites() before calling this function in
         order to populate partition->invariant beforehand. It can be freed
         afterwards. */
      unsigned int span_padded = partition->rate_cats * partition->states_padded;

      for (j = 0; j < sites; ++j)
      {
        unsigned int clv_shift = j*span_padded;
        tipclv = partition->clv[0] + clv_shift;
        pll_state_t state = gap_state;
        for (i = 0; i < tips; ++i)
        {
          tipclv = partition->clv[i] + clv_shift;
          cur_state = 0;
          for (k = 0; k < states; ++k)
          {
            cur_state |= ((pll_state_t)tipclv[k] << k);
          }
          state &= cur_state;
          if (!state)
          {
            break;
          }
        }
        if (PLL_STATE_POPCNT(state) == 1)
        {
          invariant_count += partition->pattern_weights[j];
          if (state_inv_count)
            state_inv_count[PLL_STATE_CTZ(state)]++;
        }
      }
    }
  }
  return invariant_count;
}

PLL_EXPORT int pll_update_invariant_sites(pll_partition_t * partition)
{
  unsigned int i,j,k;
  pll_state_t state;
  unsigned int states = partition->states;
  unsigned int states_padded = partition->states_padded;
  unsigned int sites = partition->sites;
  unsigned int tips = partition->tips;
  unsigned int rate_cats = partition->rate_cats;
  pll_state_t gap_state = 0;
  pll_state_t * invariant;
  double * tipclv;

  /* gap state has always all bits set to one */
  for (i = 0; i < states; ++i)
  {
    gap_state <<= 1;
    gap_state |= 1;
  }

  /* allocate array (on first call) denoting the frequency index for invariant
     sites, or -1 for variant sites */
  if (!partition->invariant)
  {
    partition->invariant = (int *)malloc(sites * sizeof(int));
  }

  invariant = (pll_state_t *)malloc(sites * sizeof(pll_state_t));

  if (!invariant || !partition->invariant)
  {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg, 200,
        "Cannot allocate charmap for invariant sites array.");
    return PLL_FAILURE;
  }

  /* initialize all elements to the gap state */
  for (i = 0; i < partition->sites; ++i)
    invariant[i] = gap_state;

  /* depending on the attribute flag, fill each element of the invariant array
     with the bitwise AND of gap and all states in the corresponding site */
  if (partition->attributes & PLL_ATTRIB_PATTERN_TIP)
  {
    if (states == 4)
    {
      for (i = 0; i < tips; ++i)
        for (j = 0; j < sites; ++j)
        {
          state = (unsigned int)(partition->tipchars[i][j]);
          invariant[j] &= state;
        }
    }
    else
    {
      for (i = 0; i < tips; ++i)
        for (j = 0; j < sites; ++j)
        {
          state = partition->tipmap[(int)(partition->tipchars[i][j])];
          invariant[j] &= state;
        }
    }
  }
  else
  {
    unsigned int span_padded = rate_cats * states_padded;
    for (i = 0; i < tips; ++i)
    {
      const unsigned int * site_id = NULL;
      if (partition->repeats && partition->repeats->pernode_ids[i])
      {
        site_id = partition->repeats->pernode_site_id[i];
      }
      for (j = 0; j < sites; ++j)
      {
        unsigned int site = site_id ? site_id[j] : j;
        tipclv = partition->clv[i] + span_padded * site;
        state = 0;
        for (k = 0; k < states; ++k)
        {
          state |= ((pll_state_t)tipclv[k] << k);
        }
        invariant[j] &= state;
      }
    }
  }

  /* if all basecalls at current site are the same and not degenerate set the
     index in invariant to the frequency index of the basecall, otherwise -1 */
  for (i = 0; i < partition->sites; ++i)
  {
    if (invariant[i] == 0 || PLL_STATE_POPCNT(invariant[i]) > 1)
      partition->invariant[i] = -1;
    else
      partition->invariant[i] = PLL_STATE_CTZ(invariant[i]);
  }

  free(invariant);

  return PLL_SUCCESS;
}
