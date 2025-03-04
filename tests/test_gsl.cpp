#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <catch2/catch_all.hpp>

#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_test.h>
#include <gsl/gsl_ieee_utils.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_vector.h>


TEST_CASE("GSL-double-integral", "[gsl]") 
{
    double result = 0, abserr = 0;
	gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
	// gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(1000);

	constexpr double epsilon = 1e-3;
	constexpr int quad_rule = GSL_INTEG_GAUSS61;

    gsl_function f;
	f.function = [](double x, void * p) -> double {
		double result_inner = 0, abserr_inner = 0;

		gsl_function f_inner;
		f_inner.function = [](double y, void * p_inner) -> double {
			double x = *(double*)p_inner;
			
			Eigen::Vector2d v00, v01, v10, v11;
			v00 << 0, 0;
			v01 << 1, 0;
			v11 << 0, 0.01;
			v10 << 1, 0.011;

			Eigen::Vector2d n0, n1;
			n0 << v00(1) - v01(1), v01(0) - v00(0);
			n1 << v10(1) - v11(1), v11(0) - v10(0);
			n0.normalize();
			n1.normalize();

			Eigen::Vector2d a = v00 + (v01 - v00) * x;
			Eigen::Vector2d b = v10 + (v11 - v10) * y;

			Eigen::Vector2d direc = a - b;
			double d = direc.norm();
			direc /= d;

			auto delta = [](double x) -> double {
				x = 5 * abs(x);
				if (x >= 1)
					return 0.;
				if (x >= 0.5)
				{
					double tmp = 1 - x;
					return tmp * tmp * tmp * (4. / 3.);
				}
				else
					return 2. / 3. - 4. * (x * x) * (1 - x);
			};

			return delta(1+direc.dot(n0)) * delta(1-direc.dot(n1)) / (d * d);
		};
		f_inner.params = &x;
		gsl_integration_workspace* w_inner = gsl_integration_workspace_alloc(1000); // (gsl_integration_workspace*)p;
		int status = gsl_integration_qag(
			&f_inner, 0., 1., 
			0., epsilon, w_inner->limit,
			quad_rule, w_inner,
			&result_inner,
			&abserr_inner);
		gsl_integration_workspace_free(w_inner);

		return result_inner;
	};
	// f.params = w_inner;

	int status = gsl_integration_qag(
		&f, 0., 1., 
		0., epsilon, w->limit,
		quad_rule, w,
		&result,
		&abserr);
	
	gsl_integration_workspace_free(w);
}

TEST_CASE("GSL-basic", "[gsl]") 
{
    double result = 0, abserr = 0;
	gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);

	constexpr double epsilon = 1e-12;
	constexpr int quad_rule = GSL_INTEG_GAUSS61;

    gsl_function f;
	f.function = [](double x, void * p) -> double {
		return x * x * sin(x);
	};

	int status = gsl_integration_qag(
		&f, 0., 1., 
		0., epsilon, w->limit,
		quad_rule, w,
		&result,
		&abserr);
	
	gsl_integration_workspace_free(w);
	CHECK(result == Catch::Approx(-2 + 2 * sin(1.) + cos(1.)).epsilon(1e-8));
}