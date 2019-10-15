/*
    Copyright (C) 2015 Tomas Flouri

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
#include <gsl/gsl_complex.h>
#include <gsl/gsl_linalg.h>

PLL_EXPORT int pll_core_update_pmatrix_nonrev_diag(double ** pmatrix,
                                              unsigned int states,
                                              unsigned int states_padded,
                                              unsigned int rate_cats,
                                              const double * rates,
                                              const double * branch_lengths,
                                              const unsigned int * matrix_indices,
                                              const unsigned int * params_indices,
                                              const double * prop_invar,
                                              double * const * eigenvals,
                                              double * const * eigenvals_imag,
                                              double * const * eigenvecs,
                                              double * const * eigenvecs_imag,
                                              double * const * inv_eigenvecs,
                                              double * const * inv_eigenvecs_imag,
                                              unsigned int count,
                                              unsigned int attributes)
{
  double real, imag;
  double * expd;
  double * expd_imag;
  double * tempd;
  double * tempd_imag;
  double * pmat;
  double * cur_evals;
  double * cur_evals_imag;
  double * cur_evecs;
  double * cur_evecs_imag;
  double * cur_inv_evecs;
  double * cur_inv_evecs_imag;
  double cur_pinv;
  expd = (double*)malloc(states * sizeof(double));
  expd_imag = (double*)malloc(states * sizeof(double));
  tempd = (double*)malloc(states * states * sizeof(double));
  tempd_imag = (double*)malloc(states * states * sizeof(double));

  for (unsigned int branch_index = 0; branch_index < count; ++branch_index) {
    for (unsigned int rate_cat = 0; rate_cat < rate_cats; ++rate_cat) {
      pmat = pmatrix[matrix_indices[branch_index]] + rate_cat*states*states_padded;
      cur_evals = eigenvals[params_indices[rate_cat]];
      cur_evals_imag = eigenvals_imag[params_indices[rate_cat]];
      cur_evecs = eigenvecs[params_indices[rate_cat]];
      cur_evecs_imag = eigenvecs_imag[params_indices[rate_cat]];
      cur_inv_evecs = inv_eigenvecs[params_indices[rate_cat]];
      cur_inv_evecs_imag = inv_eigenvecs_imag[params_indices[rate_cat]];
      cur_pinv = prop_invar[params_indices[rate_cat]];

      if (branch_lengths[branch_index] == 0.0){
        for (unsigned int i = 0; i < states; ++i) {
          for (unsigned int j = 0; j < states; ++j) {
            if (i == j) pmat[i*states_padded + j] = 1.0;
            else pmat[i*states_padded + j] = 0.0;
          }
        }
        continue;
      }

      /*
       * compute the diagonal matrix using euler's formula:
       *
       * e^(a+bi) = e^a * (cos(b) + i*sin(b))
       *
       */
      if (cur_pinv > 0){
        for (unsigned int i = 0; i < states; ++i) {
          real = cur_evals[i] * rates[rate_cat] *
            branch_lengths[branch_index] / (1.0 - cur_pinv);
          imag = cur_evals_imag[i] * rates[rate_cat] *
            branch_lengths[branch_index] / (1.0 - cur_pinv);

          expd[i] = exp(real) * cos(imag);
          expd_imag[i] = exp(real) * sin(imag);
        }
      }
      else{
        for (unsigned int i = 0; i < states; ++i) {
          real = cur_evals[i] * rates[rate_cat] * branch_lengths[branch_index];
          imag = cur_evals_imag[i] * rates[rate_cat] * branch_lengths[branch_index];

          expd[i] = exp(real) * cos(imag);
          expd_imag[i] = exp(real) * sin(imag);
        }
      }

      /* compute T = E * D, where D is the diagonal matrix calculated above */
      for (unsigned int i = 0; i < states; ++i) {
        for (unsigned int j = 0; j < states; ++j) {
          real = cur_evecs[i * states + j];
          imag = cur_evecs_imag[i * states + j];
          tempd[i * states + j] = expd[j] * real - expd_imag[j] * imag;
          tempd_imag[i * states + j] = real * expd_imag[j] + imag * expd[j];
        }
      }

      /* compute P = T * E^-1 */
      for (unsigned int i = 0; i < states; ++i) {
        for (unsigned int j = 0; j < states; ++j) {
          pmat[i*states_padded+j] = 0.0;
          for (unsigned int k = 0; k < states; ++k) {
            pmat[i * states_padded + j] += tempd[i * states + k] *
              cur_inv_evecs[k * states_padded + j] - tempd_imag[i * states + k] *
              cur_inv_evecs_imag[k * states_padded + j];
          }
          if(pmat[i * states_padded + j] < 0.0){
            assert(-pmat[i * states_padded +j] < PLL_MISC_EPSILON);
            pmat[i * states_padded + j] = 0.0;
          }
        }
      }
    }
  }
  free(expd);
  free(expd_imag);
  free(tempd);
  free(tempd_imag);
  return PLL_SUCCESS;
}

PLL_EXPORT int pll_core_update_pmatrix_nonrev_nondiag(double** pmatrix,
                                                      unsigned int states,
                                                      unsigned int states_padded,
                                                      unsigned int rate_cats,
                                                      const double* rates,
                                                      const double* branch_lengths,
                                                      const unsigned int * matrix_indices,
                                                      const unsigned int * params_indices,
                                                      const double * prop_invar,
                                                      double * const * rate_matrices,
                                                      unsigned int count,
                                                      unsigned int attrib)
{
  gsl_matrix* tmp_rm = gsl_matrix_alloc(states, states);
  gsl_matrix* tmp_pmatrix = gsl_matrix_alloc(states, states);

  for (unsigned int branch_index = 0; branch_index < count; ++branch_index) {
    for (unsigned int rate_cat = 0; rate_cat < rate_cats; ++rate_cat) {
      double* pmat = pmatrix[matrix_indices[branch_index]] +
             rate_cat * states * states_padded;
      double const * cur_rate_matrix = rate_matrices[params_indices[rate_cat]];
      double cur_pinv = prop_invar[params_indices[rate_cat]];

      if (cur_pinv > 0) {
        unsigned int k = 0;
        for (unsigned int i = 0; i < states; ++i) {
          double row_sum = 0.0;
          for (unsigned int j = 0; j < states; ++j) {
            if (i == j) continue;
            double entry = rates[rate_cat] * branch_lengths[branch_index] *
                               cur_rate_matrix[k++] /
                               (1.0 - cur_pinv);
            gsl_matrix_set(tmp_rm, i, j,entry);
            row_sum += entry;
          }
          gsl_matrix_set(tmp_rm, i, i, -row_sum);
        }
      } else {
        unsigned int k = 0;
        for (unsigned int i = 0; i < states; ++i) {
          double row_sum = 0.0;
          for (unsigned int j = 0; j < states; ++j) {
            if (i == j) continue;
            double entry = rates[rate_cat] * branch_lengths[branch_index] *
                               cur_rate_matrix[k++] /
                               (1.0 - cur_pinv);
            gsl_matrix_set(tmp_rm, i, j,entry);
            row_sum += entry;
          }
          gsl_matrix_set(tmp_rm, i, i, -row_sum);
        }
      }

      gsl_linalg_exponential_ss(tmp_rm, tmp_pmatrix, GSL_MODE_DEFAULT);

      for (unsigned int i = 0; i < states; ++i) {
        for (unsigned int j = 0; j < states; ++j) {
          pmat[i * states_padded + j] = gsl_matrix_get(tmp_pmatrix, i, j);
          assert(pmat[i*states_padded + j] <= 1.0);
        }
      }
    }
  }
  gsl_matrix_free(tmp_rm);
  gsl_matrix_free(tmp_pmatrix);
  return PLL_SUCCESS;
}

PLL_EXPORT int pll_core_update_pmatrix_nonrev(pll_partition_t * partition,
                                              const unsigned int * params_indices,
                                              const unsigned int * matrix_indices,
                                              const double * branch_lengths,
                                              unsigned int count){
  int return_no;
  for(unsigned int i=0; i < partition->rate_matrices; ++i){
    if(partition->eigen_decomp_valid[i] & PLL_NONREV_EIGEN_FALLBACK){
      return_no = pll_core_update_pmatrix_nonrev_nondiag(partition->pmatrix,
                                             partition->states,
                                             partition->states_padded,
                                             partition->rate_cats,
                                             partition->rates,
                                             branch_lengths,
                                             matrix_indices,
                                             params_indices,
                                             partition->prop_invar,
                                             partition->subst_params,
                                             count,
                                             partition->attributes);
      if (return_no == PLL_FAILURE){
        return PLL_FAILURE;
      }
    } else {
      return_no = pll_core_update_pmatrix_nonrev_diag(partition->pmatrix,
                                                      partition->states,
                                                      partition->states_padded,
                                                      partition->rate_cats,
                                                      partition->rates,
                                                      branch_lengths,
                                                      matrix_indices,
                                                      params_indices,
                                                      partition->prop_invar,
                                                      partition->eigenvals,
                                                      partition->eigenvals_imag,
                                                      partition->eigenvecs,
                                                      partition->eigenvecs_imag,
                                                      partition->inv_eigenvecs,
                                                      partition->inv_eigenvecs_imag,
                                                      count,
                                                      partition->attributes);
      if (return_no == PLL_FAILURE){
        return PLL_FAILURE;
      }
    }
  }
  return PLL_SUCCESS;
}

PLL_EXPORT int pll_core_update_pmatrix(double ** pmatrix,
                                       unsigned int states,
                                       unsigned int rate_cats,
                                       const double * rates,
                                       const double * branch_lengths,
                                       const unsigned int * matrix_indices,
                                       const unsigned int * params_indices,
                                       const double * prop_invar,
                                       double * const * eigenvals,
                                       double * const * eigenvecs,
                                       double * const * inv_eigenvecs,
                                       unsigned int count,
                                       unsigned int attrib)
{
  unsigned int i,n,j,k,m;
  unsigned int states_padded = states;
  double * expd;
  double * temp;

  double pinvar;
  double * evecs;
  double * inv_evecs;
  double * evals;
  double * pmat;


  #ifdef HAVE_SSE3
  if (attrib & PLL_ATTRIB_ARCH_SSE && PLL_STAT(sse3_present))
  {
    if (states == 4)
    {
      return pll_core_update_pmatrix_4x4_sse(pmatrix,
                                             rate_cats,
                                             rates,
                                             branch_lengths,
                                             matrix_indices,
                                             params_indices,
                                             prop_invar,
                                             eigenvals,
                                             eigenvecs,
                                             inv_eigenvecs,
                                             count);
    }
    else if (states == 20)
    {
      return pll_core_update_pmatrix_20x20_sse(pmatrix,
                                               rate_cats,
                                               rates,
                                               branch_lengths,
                                               matrix_indices,
                                               params_indices,
                                               prop_invar,
                                               eigenvals,
                                               eigenvecs,
                                               inv_eigenvecs,
                                               count);
    }
    /* this line is never called, but should we disable the else case above,
       then states_padded must be set to this value */
    states_padded = (states+1) & 0xFFFFFFFE;
  }
  #endif
  #ifdef HAVE_AVX
  if (attrib & PLL_ATTRIB_ARCH_AVX && PLL_STAT(avx_present))
  {
    if (states == 4)
    {
      return pll_core_update_pmatrix_4x4_avx(pmatrix,
                                             rate_cats,
                                             rates,
                                             branch_lengths,
                                             matrix_indices,
                                             params_indices,
                                             prop_invar,
                                             eigenvals,
                                             eigenvecs,
                                             inv_eigenvecs,
                                             count);
    }
    if (states == 20)
    {
      return pll_core_update_pmatrix_20x20_avx(pmatrix,
                                             rate_cats,
                                             rates,
                                             branch_lengths,
                                             matrix_indices,
                                             params_indices,
                                             prop_invar,
                                             eigenvals,
                                             eigenvecs,
                                             inv_eigenvecs,
                                             count);
    }
    /* this line is never called, but should we disable the else case above,
       then states_padded must be set to this value */
    states_padded = (states+3) & 0xFFFFFFFC;
  }
  #endif
  #ifdef HAVE_AVX2
  if (attrib & PLL_ATTRIB_ARCH_AVX2 && PLL_STAT(avx2_present))
  {
    if (states == 4)
    {
      /* use AVX version here since FMA doesn't make much sense */
      return pll_core_update_pmatrix_4x4_avx(pmatrix,
                                             rate_cats,
                                             rates,
                                             branch_lengths,
                                             matrix_indices,
                                             params_indices,
                                             prop_invar,
                                             eigenvals,
                                             eigenvecs,
                                             inv_eigenvecs,
                                             count);
    }
    if (states == 20)
    {
      return pll_core_update_pmatrix_20x20_avx2(pmatrix,
                                             rate_cats,
                                             rates,
                                             branch_lengths,
                                             matrix_indices,
                                             params_indices,
                                             prop_invar,
                                             eigenvals,
                                             eigenvecs,
                                             inv_eigenvecs,
                                             count);
    }
    /* this line is never called, but should we disable the else case above,
       then states_padded must be set to this value */
    states_padded = (states+3) & 0xFFFFFFFC;
  }
  #endif

  expd = (double *)malloc(states * sizeof(double));
  temp = (double *)malloc(states*states*sizeof(double));

  if (!expd || !temp)
  {
    if (expd) free(expd);
    if (temp) free(temp);

    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg, 200, "Unable to allocate enough memory.");
    return PLL_FAILURE;
  }

  for (i = 0; i < count; ++i)
  {
    assert(branch_lengths[i] >= 0);

    /* compute effective pmatrix location */
    for (n = 0; n < rate_cats; ++n)
    {
      pmat = pmatrix[matrix_indices[i]] + n*states*states_padded;

      pinvar = prop_invar[params_indices[n]];
      evecs = eigenvecs[params_indices[n]];
      inv_evecs = inv_eigenvecs[params_indices[n]];
      evals = eigenvals[params_indices[n]];

      if (branch_lengths[i] > 0.)
      {
        /* NOTE: in order to deal with numerical issues in cases when Qt -> 0, we
         * use a trick suggested by Ben Redelings and explained here:
         * https://github.com/xflouris/libpll/issues/129#issuecomment-304004005
         * In short, we use expm1() to compute (exp(Qt) - I), and then correct
         * for this by adding an identity matrix I in the very end */

        /* exponentiate eigenvalues */
        if (pinvar > PLL_MISC_EPSILON)
        {
          for (j = 0; j < states; ++j)
            expd[j] = expm1(evals[j] * rates[n] * branch_lengths[i]
                                       / (1.0 - pinvar));
        }
        else
        {
          for (j = 0; j < states; ++j)
           expd[j] = expm1(evals[j] * rates[n] * branch_lengths[i]);
        }

        for (j = 0; j < states; ++j)
          for (k = 0; k < states; ++k)
            temp[j*states+k] = inv_evecs[j*states_padded+k] * expd[k];

        for (j = 0; j < states; ++j)
        {
          for (k = 0; k < states; ++k)
          {
            pmat[j*states_padded+k] = (j==k) ? 1.0 : 0;
            for (m = 0; m < states; ++m)
            {
              pmat[j*states_padded+k] +=
                  temp[j*states+m] * evecs[m*states_padded+k];
            }
          }
        }
      }
      else
      {
        /* if branch length is zero then set the p-matrix to identity matrix */
        for (j = 0; j < states; ++j)
          for (k = 0; k < states; ++k)
            pmat[j*states_padded + k] = (j == k) ? 1 : 0;
      }

      #ifdef DEBUG
      for (j = 0; j < states; ++j)
        for (k = 0; k < states; ++k)
          assert(pmat[j*states_padded+k] >= 0);
      #endif
    }
  }

  free(expd);
  free(temp);
  return PLL_SUCCESS;
}
