#include <iostream>
#include <fstream>
#include <catch2/catch_all.hpp>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_test.h>
#include <gsl/gsl_ieee_utils.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_vector.h>

struct counter_params {
  gsl_function * f;
  int neval;
} ;

double 
counter (double x, void * params)
{
  struct counter_params * p = (struct counter_params *) params;
  p->neval++ ; /* increment counter */
  return GSL_FN_EVAL(p->f, x);
}


TEST_CASE("GSL", "[gsl]") 
{
    double result = 0, abserr = 0;
	gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
	gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(1000);

    gsl_function f;
	f.function = [](double x, void * p) -> double {
		double result_inner = 0, abserr_inner = 0;

		gsl_function f_inner;
		f_inner.function = [](double y, void * p_inner) -> double {
			double x = *(double*)p_inner;
			return y * y + x * x + x * y * 2;
		};
		f_inner.params = &x;
		gsl_integration_workspace* w_inner = (gsl_integration_workspace*)p;
		int status = gsl_integration_qag(
			&f_inner, 0., 1., 
			0., 1.e-10, w_inner->limit,
			GSL_INTEG_GAUSS15, w_inner,
			&result_inner,
			&abserr_inner);

		return result_inner;
	};
	f.params = w_inner;

	int status = gsl_integration_qag(
		&f, 0., 1., 
		0., 1.e-10, w->limit,
		GSL_INTEG_GAUSS15, w,
		&result,
		&abserr);
	
	gsl_integration_workspace_free(w);
	gsl_integration_workspace_free(w_inner);
	CHECK(result == Catch::Approx(7./6.).epsilon(1e-12));
}
