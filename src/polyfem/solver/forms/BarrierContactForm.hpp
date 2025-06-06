#pragma once

#include "ContactForm.hpp"
#include <ipc/potentials/barrier_potential.hpp>

namespace polyfem::solver
{
    class BarrierContactForm : public ContactForm
    {
    public:
		BarrierContactForm(const ipc::CollisionMesh &collision_mesh,
					const double dhat,
					const double avg_mass,
					const bool use_convergent_formulation,
					const bool use_adaptive_barrier_stiffness,
					const bool is_time_dependent,
					const bool enable_shape_derivatives,
					const ipc::BroadPhaseMethod broad_phase_method,
					const double ccd_tolerance,
					const int ccd_max_iterations);

		virtual std::string name() const override { return "barrier-contact"; }

        virtual void update_barrier_stiffness(const Eigen::VectorXd &x, const Eigen::MatrixXd &grad_energy) override;

		const ipc::BarrierPotential &barrier_potential() const { return *std::dynamic_pointer_cast<ipc::BarrierPotential>(contact_potential_); }

		/// @brief Update fields after a step in the optimization
		/// @param iter_num Optimization iteration number
		/// @param x Current solution
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		bool use_convergent_formulation() const override { return collision_set_->use_convergent_formulation(); }

		void force_shape_derivative(ipc::CollisionsBase *collision_set, const Eigen::MatrixXd &solution, const Eigen::VectorXd &adjoint_sol, Eigen::VectorXd &term) override;

		ipc::Collisions &get_barrier_collision_set() { return *std::dynamic_pointer_cast<ipc::Collisions>(collision_set_); }
		const ipc::Collisions &get_barrier_collision_set() const { return *std::dynamic_pointer_cast<ipc::Collisions>(collision_set_); }
		const ipc::BarrierPotential &get_potential() const { return *contact_potential_; }

	protected:
		/// @brief Compute the contact barrier potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		virtual double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the value of the form multiplied per element
		/// @param x Current solution
		/// @return Computed value
		Eigen::VectorXd value_per_element_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		virtual void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		virtual void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

		void update_collision_set(const Eigen::MatrixXd &displaced_surface) override;

		/// @brief Contact potential
		std::shared_ptr<ipc::BarrierPotential> contact_potential_;
    };
}