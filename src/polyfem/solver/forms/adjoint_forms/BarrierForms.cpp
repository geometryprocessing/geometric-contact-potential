#include "BarrierForms.hpp"
#include <polyfem/State.hpp>
#include <polyfem/utils/BoundarySampler.hpp>
#include <igl/writeOBJ.h>

namespace polyfem::solver
{
	namespace
	{
		class QuadraticBarrier : public ipc::Barrier
		{
		public:
			QuadraticBarrier(const double weight = 1) : weight_(weight) {}

			double operator()(const double d, const double dhat) const override
			{
				if (d > dhat)
					return 0;
				else
					return weight_ * (d - dhat) * (d - dhat);
			}
			double first_derivative(const double d, const double dhat) const override
			{
				if (d > dhat)
					return 0;
				else
					return 2 * weight_ * (d - dhat);
			}
			double second_derivative(const double d, const double dhat) const override
			{
				if (d > dhat)
					return 0;
				else
					return 2 * weight_;
			}
			double units(const double dhat) const override
			{
				return dhat * dhat;
			}

		private:
			const double weight_;
		};

	} // namespace

	CollisionBarrierForm::CollisionBarrierForm(const VariableToSimulationGroup &variable_to_simulation, const State &state, const double dhat, const double dmin)
		: AdjointForm(variable_to_simulation), state_(state), dhat_(dhat), dmin_(dmin), barrier_potential_(dhat)
	{
		State::build_collision_mesh(
			*state_.mesh, state_.n_geom_bases, state_.geom_bases(), state_.geom_bases(),
			state_.total_local_boundary, state_.obstacle, state_.args,
			[this](const std::string &p) { return this->state_.resolve_input_path(p); },
			state_.in_node_to_node, state_.node_to_body_id, collision_mesh_);

		Eigen::MatrixXd V;
		state_.get_vertices(V);
		X_init = utils::flatten(V);

		broad_phase_method_ = BroadPhaseMethod::HASH_GRID;
	}

	double CollisionBarrierForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		return barrier_potential_(collision_set, collision_mesh_, displaced_surface);
	}

	void CollisionBarrierForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, &x]() {
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
			const Eigen::VectorXd grad = collision_mesh_.to_full_dof(barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface));
			return AdjointTools::map_node_to_primitive_order(state_, grad);
		});
	}

	void CollisionBarrierForm::solution_changed(const Eigen::VectorXd &x)
	{
		AdjointForm::solution_changed(x);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		build_collision_set(displaced_surface);
	}

	bool CollisionBarrierForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
		const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

		// Skip CCD if the displacement is zero.
		if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0.0)
			return true;

		const ipc::TightInclusionCCD tight_inclusion_ccd(1e-6, 1e6);
		bool is_valid = ipc::is_step_collision_free(
			collision_mesh_,
			collision_mesh_.vertices(V0),
			collision_mesh_.vertices(V1),
			dmin_,
			build_broad_phase(broad_phase_method_),
			tight_inclusion_ccd);

		return is_valid;
	}

	double CollisionBarrierForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
		const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

		const ipc::TightInclusionCCD tight_inclusion_ccd(1e-6, 1e6);
		double max_step = ipc::compute_collision_free_stepsize(
			collision_mesh_,
			collision_mesh_.vertices(V0),
			collision_mesh_.vertices(V1),
			dmin_,
			build_broad_phase(broad_phase_method_),
			tight_inclusion_ccd);

		adjoint_logger().info("Objective {}: max step size is {}.", name(), max_step);

		return max_step;
	}

	void CollisionBarrierForm::build_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		collision_set.build(collision_mesh_, displaced_surface, dhat_, dmin_, build_broad_phase(broad_phase_method_));

		cached_displaced_surface = displaced_surface;
	}

	Eigen::VectorXd CollisionBarrierForm::get_updated_mesh_nodes(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd X = X_init;
		variable_to_simulations_.compute_state_variable(ParameterType::Shape, &state_, x, X);
		return AdjointTools::map_primitive_to_node_order(state_, X);
	}

	LayerThicknessForm::LayerThicknessForm(const VariableToSimulationGroup &variable_to_simulations,
										   const State &state,
										   const std::vector<int> &boundary_ids,
										   const double dhat,
										   const bool use_log_barrier,
										   const double dmin)
		: CollisionBarrierForm(variable_to_simulations, state, dhat, dmin),
		  boundary_ids_(boundary_ids)
	{
		for (const auto &id : boundary_ids_)
			boundary_ids_to_dof_[id] = std::set<int>();

		build_collision_mesh();

		if (!use_log_barrier)
			barrier_potential_.set_barrier(std::make_shared<QuadraticBarrier>());
	}

	void LayerThicknessForm::build_collision_mesh()
	{
		Eigen::MatrixXd node_positions;
		Eigen::MatrixXi boundary_edges, boundary_triangles;
		std::vector<Eigen::Triplet<double>> displacement_map_entries;
		io::OutGeometryData::extract_boundary_mesh(*state_.mesh, state_.n_geom_bases, state_.geom_bases(), state_.total_local_boundary,
												   node_positions, boundary_edges, boundary_triangles, displacement_map_entries);

		std::vector<bool> is_on_surface;
		is_on_surface.resize(node_positions.rows(), false);

		assembler::ElementAssemblyValues vals;
		Eigen::MatrixXd points, uv, normals;
		Eigen::VectorXd weights;
		Eigen::VectorXi global_primitive_ids;
		for (const auto &lb : state_.total_local_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, state_.n_boundary_samples(), *state_.mesh, false, uv, points, normals, weights, global_primitive_ids);

			if (!has_samples)
				continue;

			const basis::ElementBases &gbs = state_.geom_bases()[e];

			vals.compute(e, state_.mesh->is_volume(), points, gbs, gbs);

			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = gbs.local_nodes_for_primitive(primitive_global_id, *state_.mesh);
				const int boundary_id = state_.mesh->get_boundary_id(primitive_global_id);

				if (!std::count(boundary_ids_.begin(), boundary_ids_.end(), boundary_id))
					continue;

				for (long n = 0; n < nodes.size(); ++n)
				{
					const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
					is_on_surface[v.global[0].index] = true;
					assert(v.global[0].index < node_positions.rows());
					boundary_ids_to_dof_[boundary_id].insert(v.global[0].index);
				}
			}
		}

		Eigen::SparseMatrix<double> displacement_map;
		if (!displacement_map_entries.empty())
		{
			displacement_map.resize(node_positions.rows(), state_.n_geom_bases);
			displacement_map.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
		}

		// Fix boundary edges and boundary triangles to exclude vertices not on triangles
		Eigen::MatrixXi boundary_edges_alt(0, 2), boundary_triangles_alt(0, 3);
		{
			for (int i = 0; i < boundary_edges.rows(); ++i)
			{
				bool on_surface = true;
				for (int j = 0; j < boundary_edges.cols(); ++j)
					on_surface &= is_on_surface[boundary_edges(i, j)];
				if (on_surface)
				{
					boundary_edges_alt.conservativeResize(boundary_edges_alt.rows() + 1, 2);
					boundary_edges_alt.row(boundary_edges_alt.rows() - 1) = boundary_edges.row(i);
				}
			}

			if (state_.mesh->is_volume())
			{
				for (int i = 0; i < boundary_triangles.rows(); ++i)
				{
					bool on_surface = true;
					for (int j = 0; j < boundary_triangles.cols(); ++j)
						on_surface &= is_on_surface[boundary_triangles(i, j)];
					if (on_surface)
					{
						boundary_triangles_alt.conservativeResize(boundary_triangles_alt.rows() + 1, 3);
						boundary_triangles_alt.row(boundary_triangles_alt.rows() - 1) = boundary_triangles.row(i);
					}
				}
			}
			else
				boundary_triangles_alt.resize(0, 0);
		}

		collision_mesh_ = ipc::CollisionMesh(is_on_surface,
											 std::vector<bool>(is_on_surface.size(), false),
											 node_positions,
											 boundary_edges_alt,
											 boundary_triangles_alt,
											 displacement_map);

		can_collide_cache_.resize(collision_mesh_.num_vertices(), collision_mesh_.num_vertices());
		for (int i = 0; i < can_collide_cache_.rows(); ++i)
		{
			int dof_idx_i = collision_mesh_.to_full_vertex_id(i);
			if (!is_on_surface[dof_idx_i])
				continue;
			for (int j = 0; j < can_collide_cache_.cols(); ++j)
			{
				int dof_idx_j = collision_mesh_.to_full_vertex_id(j);
				if (!is_on_surface[dof_idx_j])
					continue;

				bool collision_allowed = true;
				for (const auto &id : boundary_ids_)
					if (boundary_ids_to_dof_[id].count(dof_idx_i) && boundary_ids_to_dof_[id].count(dof_idx_j))
						collision_allowed = false;
				can_collide_cache_(i, j) = collision_allowed;
			}
		}

		collision_mesh_.can_collide = [&](size_t vi, size_t vj) {
			return (bool)can_collide_cache_(vi, vj);
		};

		collision_mesh_.init_area_jacobians();
	}

	DeformedCollisionBarrierForm::DeformedCollisionBarrierForm(const VariableToSimulationGroup &variable_to_simulation, const State &state, const double dhat)
		: AdjointForm(variable_to_simulation), state_(state), dhat_(dhat), barrier_potential_(dhat)
	{
		if (state_.n_bases != state_.n_geom_bases)
			log_and_throw_adjoint_error("[{}] Should use linear FE basis!", name());

		State::build_collision_mesh(
			*state_.mesh, state_.n_geom_bases, state_.geom_bases(), state_.geom_bases(),
			state_.total_local_boundary, state_.obstacle, state_.args,
			[this](const std::string &p) { return this->state_.resolve_input_path(p); },
			state_.in_node_to_node, state_.node_to_body_id, collision_mesh_);

		Eigen::MatrixXd V;
		state_.get_vertices(V);
		X_init = utils::flatten(V);

		broad_phase_method_ = BroadPhaseMethod::HASH_GRID;
	}

	double DeformedCollisionBarrierForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));

		return barrier_potential_(collision_set, collision_mesh_, displaced_surface);
	}

	void DeformedCollisionBarrierForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, &x]() {
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
			const Eigen::VectorXd grad = collision_mesh_.to_full_dof(barrier_potential_.gradient(collision_set, collision_mesh_, displaced_surface));
			return AdjointTools::map_node_to_primitive_order(state_, grad);
		});
	}

	void DeformedCollisionBarrierForm::solution_changed(const Eigen::VectorXd &x)
	{
		AdjointForm::solution_changed(x);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		build_collision_set(displaced_surface);
	}

	bool DeformedCollisionBarrierForm::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
		// const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

		// // Skip CCD if the displacement is zero.
		// if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0.0)
		//     return true;

		// bool is_valid = ipc::is_step_collision_free(
		//     collision_mesh_,
		//     collision_mesh_.vertices(V0),
		//     collision_mesh_.vertices(V1),
		//     broad_phase_method_,
		//     1e-6, 1e6);

		return true; // is_valid;
	}

	double DeformedCollisionBarrierForm::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		// const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
		// const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

		// double max_step = ipc::compute_collision_free_stepsize(
		//     collision_mesh_,
		//     collision_mesh_.vertices(V0),
		//     collision_mesh_.vertices(V1),
		//     broad_phase_method_, 1e-6, 1e6);

		return 1; // max_step;
	}

	void DeformedCollisionBarrierForm::build_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		static Eigen::MatrixXd cached_displaced_surface;
		if (cached_displaced_surface.size() == displaced_surface.size() && cached_displaced_surface == displaced_surface)
			return;

		collision_set.build(collision_mesh_, displaced_surface, dhat_, 0, build_broad_phase(broad_phase_method_));

		cached_displaced_surface = displaced_surface;
	}

	Eigen::VectorXd DeformedCollisionBarrierForm::get_updated_mesh_nodes(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd X = X_init;
		variable_to_simulations_.compute_state_variable(ParameterType::Shape, &state_, x, X);
		return AdjointTools::map_primitive_to_node_order(state_, X) + state_.diff_cached.u(0);
	}

	template <int dim>
	SmoothContactForceForm<dim>::SmoothContactForceForm(
		const VariableToSimulationGroup &variable_to_simulations,
		const State &state,
		const json &args)
		: StaticForm(variable_to_simulations),
		  state_(state),
		  params_(state.args["contact"]["dhat"], state.args["contact"]["alpha_t"], state.args["contact"]["beta_t"], state.args["contact"]["alpha_n"], state.args["contact"]["beta_n"], state.mesh->is_volume() ? 2 : 1),
		  potential_(params_)
	{
		assert(dim == state.mesh->dimension());

		auto tmp_ids = args["surface_selection"].get<std::vector<int>>();
		boundary_ids_ = std::set(tmp_ids.begin(), tmp_ids.end());

		build_collision_mesh();
	}

	template <int dim>
	void SmoothContactForceForm<dim>::build_collision_mesh()
	{
		// Deep copy and change the can_collide() function
		collision_mesh_ = state_.collision_mesh;

		// const int num_fe_nodes = state_.n_bases - state_.obstacle.n_vertices();

		// collision_mesh_.can_collide = [this, num_fe_nodes](size_t vi, size_t vj) {
		// 	return this->collision_mesh_.to_full_vertex_id(vi) >= num_fe_nodes || this->collision_mesh_.to_full_vertex_id(vj) >= num_fe_nodes;
		// };

		std::vector<int> is_obstacle(state_.n_bases);
		for (int e = 0; e < state_.bases.size(); e++)
		{
			const auto &b = state_.bases[e];
			if (state_.mesh->get_body_id(e) == 1)
				for (const auto &bs : b.bases)
				{
					for (const auto &g : bs.global())
					{
						is_obstacle[g.index] = true;
					}
				}
		}

		collision_mesh_.can_collide = [this, is_obstacle](size_t vi, size_t vj) {
			return is_obstacle[this->collision_mesh_.to_full_vertex_id(vi)] || is_obstacle[this->collision_mesh_.to_full_vertex_id(vj)];
		};
	}

	template <int dim>
	ipc::SmoothCollisions<dim> SmoothContactForceForm<dim>::get_smooth_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		ipc::SmoothCollisions<dim> collisions;
		const auto smooth_contact = dynamic_cast<const SmoothContactForm<dim> *>(state_.solve_data.contact_form.get());
		collisions.build(collision_mesh_, displaced_surface, smooth_contact->get_params(), smooth_contact->using_adaptive_dhat(), smooth_contact->get_broad_phase());
		return collisions;
	}

	template <int dim>
	double SmoothContactForceForm<dim>::value_unweighted_step(const int time_step, const Eigen::VectorXd &x) const
	{
		assert(state_.solve_data.contact_form != nullptr);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));

		Eigen::VectorXd forces = collision_mesh_.to_full_dof(potential_.gradient(collisions_, collision_mesh_, displaced_surface));

		// return forces.squaredNorm();

		Eigen::VectorXd coeff(forces.size());
		coeff.setZero();
		coeff(Eigen::seq(1, coeff.size(), collision_mesh_.dim())).array() = 1;
		return (coeff.array() * forces.array()).matrix().squaredNorm() / 2;
	}

	template <int dim>
	Eigen::VectorXd SmoothContactForceForm<dim>::compute_adjoint_rhs_step(const int time_step, const Eigen::VectorXd &x, const State &state) const
	{
		assert(state_.solve_data.contact_form != nullptr);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));

		Eigen::VectorXd forces = potential_.gradient(collisions_, collision_mesh_, displaced_surface);
		forces = collision_mesh_.to_full_dof(forces);

		StiffnessMatrix hessian = potential_.hessian(collisions_, collision_mesh_, displaced_surface, ipc::PSDProjectionMethod::NONE);
		hessian = collision_mesh_.to_full_dof(hessian);

		Eigen::VectorXd coeff(forces.size());
		coeff.setZero();
		coeff(Eigen::seq(1, coeff.size(), collision_mesh_.dim())).array() = 1;
		return weight() * (hessian * (coeff.array() * forces.array()).matrix());
	}

	template <int dim>
	void SmoothContactForceForm<dim>::compute_partial_gradient_step(const int time_step, const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		assert(state_.solve_data.contact_form != nullptr);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));

		Eigen::VectorXd forces = potential_.gradient(collisions_, collision_mesh_, displaced_surface);
		forces = collision_mesh_.to_full_dof(forces);

		StiffnessMatrix hessian = potential_.hessian(collisions_, collision_mesh_, displaced_surface, ipc::PSDProjectionMethod::NONE);
		hessian = collision_mesh_.to_full_dof(hessian);

		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, &x, &forces, &hessian]() {
			// Eigen::VectorXd grads = 2 * hessian.transpose() * forces;

			Eigen::VectorXd coeff(forces.size());
			coeff.setZero();
			coeff(Eigen::seq(1, coeff.size(), collision_mesh_.dim())).array() = 1;
			Eigen::VectorXd grads = (hessian * (coeff.array() * forces.array()).matrix());

			grads = state_.basis_nodes_to_gbasis_nodes * grads;
			return AdjointTools::map_node_to_primitive_order(state_, grads);
		});
	}

	template <int dim>
	void SmoothContactForceForm<dim>::solution_changed_step(const int time_step, const Eigen::VectorXd &x)
	{
		build_collision_mesh();
		const Eigen::MatrixXd displaced_surface = collision_mesh_.displace_vertices(utils::unflatten(state_.diff_cached.u(time_step), collision_mesh_.dim()));
		collisions_ = get_smooth_collision_set(displaced_surface);
	}

	template class SmoothContactForceForm<2>;
	template class SmoothContactForceForm<3>;

	template <int dim>
	SmoothLayerThicknessForm<dim>::SmoothLayerThicknessForm(
		const VariableToSimulationGroup &variable_to_simulations,
		const State &state,
		const std::vector<int> &boundary_ids,
		const double dhat)
		: AdjointForm(variable_to_simulations),
		  state_(state),
		  dhat_(dhat),
		  boundary_ids_(boundary_ids),
		  params_(dhat,
				  0.5, // state.args["contact"]["alpha_t"],
				  0,   // state.args["contact"]["beta_t"],
				  0.5, // state.args["contact"]["alpha_n"],
				  0,   // state.args["contact"]["beta_n"],
				  state.mesh->is_volume() ? 2 : 1),
		  barrier_potential_(params_)
	{
		assert(dim == state.mesh->dimension());

		Eigen::MatrixXd V;
		state_.get_vertices(V);
		X_init = utils::flatten(V);

		broad_phase_method_ = BroadPhaseMethod::HASH_GRID;

		build_collision_mesh();
	}

	template <int dim>
	void SmoothLayerThicknessForm<dim>::build_collision_mesh()
	{
		Eigen::MatrixXd node_positions;
		Eigen::MatrixXi boundary_edges, boundary_triangles;
		std::vector<Eigen::Triplet<double>> displacement_map_entries;
		io::OutGeometryData::extract_boundary_mesh(*state_.mesh, state_.n_geom_bases, state_.geom_bases(), state_.total_local_boundary,
												   node_positions, boundary_edges, boundary_triangles, displacement_map_entries);

		std::vector<bool> is_on_surface;
		is_on_surface.resize(node_positions.rows(), false);

		assembler::ElementAssemblyValues vals;
		Eigen::MatrixXd points, uv, normals;
		Eigen::VectorXd weights;
		Eigen::VectorXi global_primitive_ids;
		for (const auto &lb : state_.total_local_boundary)
		{
			const int e = lb.element_id();
			bool has_samples = utils::BoundarySampler::boundary_quadrature(lb, state_.n_boundary_samples(), *state_.mesh, false, uv, points, normals, weights, global_primitive_ids);

			if (!has_samples)
				continue;

			const basis::ElementBases &gbs = state_.geom_bases()[e];

			vals.compute(e, state_.mesh->is_volume(), points, gbs, gbs);

			for (int i = 0; i < lb.size(); ++i)
			{
				const int primitive_global_id = lb.global_primitive_id(i);
				const auto nodes = gbs.local_nodes_for_primitive(primitive_global_id, *state_.mesh);
				const int boundary_id = state_.mesh->get_boundary_id(primitive_global_id);

				// if (!std::count(boundary_ids_.begin(), boundary_ids_.end(), boundary_id))
				// 	continue;

				for (long n = 0; n < nodes.size(); ++n)
				{
					const assembler::AssemblyValues &v = vals.basis_values[nodes(n)];
					is_on_surface[v.global[0].index] = true;
					assert(v.global[0].index < node_positions.rows());
					boundary_ids_to_dof_[boundary_id].insert(v.global[0].index);
				}
			}
		}

		Eigen::SparseMatrix<double> displacement_map;
		if (!displacement_map_entries.empty())
		{
			displacement_map.resize(node_positions.rows(), state_.n_geom_bases);
			displacement_map.setFromTriplets(displacement_map_entries.begin(), displacement_map_entries.end());
		}

		// Fix boundary edges and boundary triangles to exclude vertices not on triangles
		// Eigen::MatrixXi boundary_edges_alt(0, 2), boundary_triangles_alt(0, 3);
		// {
		// 	for (int i = 0; i < boundary_edges.rows(); ++i)
		// 	{
		// 		bool on_surface = true;
		// 		for (int j = 0; j < boundary_edges.cols(); ++j)
		// 			on_surface &= is_on_surface[boundary_edges(i, j)];
		// 		if (on_surface)
		// 		{
		// 			boundary_edges_alt.conservativeResize(boundary_edges_alt.rows() + 1, 2);
		// 			boundary_edges_alt.row(boundary_edges_alt.rows() - 1) = boundary_edges.row(i);
		// 		}
		// 	}

		// 	if (state_.mesh->is_volume())
		// 	{
		// 		for (int i = 0; i < boundary_triangles.rows(); ++i)
		// 		{
		// 			bool on_surface = true;
		// 			for (int j = 0; j < boundary_triangles.cols(); ++j)
		// 				on_surface &= is_on_surface[boundary_triangles(i, j)];
		// 			if (on_surface)
		// 			{
		// 				boundary_triangles_alt.conservativeResize(boundary_triangles_alt.rows() + 1, 3);
		// 				boundary_triangles_alt.row(boundary_triangles_alt.rows() - 1) = boundary_triangles.row(i);
		// 			}
		// 		}
		// 	}
		// 	else
		// 		boundary_triangles_alt.resize(0, 0);
		// }

		collision_mesh_ = ipc::CollisionMesh(is_on_surface,
											 std::vector<bool>(is_on_surface.size(), false),
											 node_positions,
											 boundary_edges,     // boundary_edges_alt,
											 boundary_triangles, // boundary_triangles_alt,
											 displacement_map);

		can_collide_cache_.resize(collision_mesh_.num_vertices(), collision_mesh_.num_vertices());
		for (int i = 0; i < can_collide_cache_.rows(); ++i)
		{
			int dof_idx_i = collision_mesh_.to_full_vertex_id(i);
			if (!is_on_surface[dof_idx_i])
				continue;
			for (int j = 0; j < can_collide_cache_.cols(); ++j)
			{
				int dof_idx_j = collision_mesh_.to_full_vertex_id(j);
				if (!is_on_surface[dof_idx_j])
					continue;

				bool i_can_collide = false;
				bool j_can_collide = false;
				for (const auto &id : boundary_ids_)
				{
					if (boundary_ids_to_dof_[id].count(dof_idx_i))
						i_can_collide = true;
					if (boundary_ids_to_dof_[id].count(dof_idx_j))
						j_can_collide = true;
				}
				can_collide_cache_(i, j) = i_can_collide && j_can_collide;
			}
		}

		collision_mesh_.can_collide = [&](size_t vi, size_t vj) {
			// return true;
			return (bool)can_collide_cache_(vi, vj);
		};

		collision_mesh_.init_area_jacobians();
	}

	template <int dim>
	Eigen::VectorXd SmoothLayerThicknessForm<dim>::get_updated_mesh_nodes(const Eigen::VectorXd &x) const
	{
		Eigen::VectorXd X = X_init;
		variable_to_simulations_.compute_state_variable(ParameterType::Shape, &state_, x, X);
		return AdjointTools::map_primitive_to_node_order(state_, X);
	}

	template <int dim>
	void SmoothLayerThicknessForm<dim>::build_smooth_collision_set(const Eigen::MatrixXd &displaced_surface)
	{
		collisions_.build(collision_mesh_, displaced_surface, params_, false, build_broad_phase(broad_phase_method_));
	}

	template <int dim>
	double SmoothLayerThicknessForm<dim>::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		return barrier_potential_(collisions_, collision_mesh_, displaced_surface);
	}

	template <int dim>
	void SmoothLayerThicknessForm<dim>::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv = weight() * variable_to_simulations_.apply_parametrization_jacobian(ParameterType::Shape, &state_, x, [this, &x]() {
			const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
			const Eigen::VectorXd grad = collision_mesh_.to_full_dof(barrier_potential_.gradient(collisions_, collision_mesh_, displaced_surface));
			return AdjointTools::map_node_to_primitive_order(state_, grad);
		});
	}

	template <int dim>
	void SmoothLayerThicknessForm<dim>::solution_changed(const Eigen::VectorXd &x)
	{
		AdjointForm::solution_changed(x);

		const Eigen::MatrixXd displaced_surface = collision_mesh_.vertices(utils::unflatten(get_updated_mesh_nodes(x), state_.mesh->dimension()));
		build_smooth_collision_set(displaced_surface);
	}

	template <int dim>
	double SmoothLayerThicknessForm<dim>::max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
		const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

		const ipc::TightInclusionCCD tight_inclusion_ccd(1e-6, 1e6);
		double max_step = ipc::compute_collision_free_stepsize(
			collision_mesh_,
			collision_mesh_.vertices(V0),
			collision_mesh_.vertices(V1),
			0,
			build_broad_phase(broad_phase_method_),
			tight_inclusion_ccd);

		adjoint_logger().info("Objective {}: max step size is {}.", name(), max_step);

		return max_step;
	}

	template <int dim>
	bool SmoothLayerThicknessForm<dim>::is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) const
	{
		const Eigen::MatrixXd V0 = utils::unflatten(get_updated_mesh_nodes(x0), state_.mesh->dimension());
		const Eigen::MatrixXd V1 = utils::unflatten(get_updated_mesh_nodes(x1), state_.mesh->dimension());

		// Skip CCD if the displacement is zero.
		if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0.0)
			return true;

		const ipc::TightInclusionCCD tight_inclusion_ccd(1e-6, 1e6);
		bool is_valid = ipc::is_step_collision_free(
			collision_mesh_,
			collision_mesh_.vertices(V0),
			collision_mesh_.vertices(V1),
			0,
			build_broad_phase(broad_phase_method_),
			tight_inclusion_ccd);

		return is_valid;
	}

	template class SmoothLayerThicknessForm<2>;
	template class SmoothLayerThicknessForm<3>;
} // namespace polyfem::solver