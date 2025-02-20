#include "CollisionMesh.hpp"
#include <polyfem/utils/Logger.hpp>

namespace polyfem::curved_ipc {
    CurvedCollisionMesh::CurvedCollisionMesh(
        const mesh::Mesh &mesh,
        const std::vector<basis::ElementBases> &geom_bases,
        const std::vector<basis::ElementBases> &bases,
        const int n_bases,
        const std::vector<mesh::LocalBoundary> &total_local_boundary,
        const Eigen::SparseMatrix<double>& displacement_map)
            : geom_bases_(geom_bases), bases_(bases), total_local_boundary_(total_local_boundary)
    {
        m_dim = mesh.dimension();

        m_full_vertex_to_vertex.resize(n_bases);
        m_full_vertex_to_vertex.array() = -1;
        m_is_vertex_on_boundary.assign(n_bases, false);
        m_full_rest_positions.setZero(n_bases, m_dim);

        int cur_id = 0;
        for (const auto &lb : total_local_boundary)
        {
            const int e = lb.element_id();
            const basis::ElementBases &gbs = geom_bases[e];
            const basis::ElementBases &bs = bases[e];

            for (int i = 0; i < gbs.bases.size(); i++)
                if (gbs.bases[i].order() != 1)
                    log_and_throw_error("Only linear geometric elements are supported for collision mesh!");

            for (int i = 0; i < lb.size(); ++i)
            {
                const int primitive_g_id = lb.global_primitive_id(i);

                const auto nodes = bs.local_nodes_for_primitive(primitive_g_id, mesh);

                for (long n = 0; n < nodes.size(); ++n)
                {
                    if (bs.bases[nodes(n)].global().size() != 1)
                        log_and_throw_error("Only one global index per node in FE basis are supported for collision mesh!");
                    m_is_vertex_on_boundary[bs.bases[nodes(n)].global()[0].index] = true;
                    int &id = m_full_vertex_to_vertex(bs.bases[nodes(n)].global()[0].index);
                    if (id < 0)
                        id = cur_id++;
                }
            }
        }

        m_vertex_to_full_vertex.resize(cur_id);
        for (int i = 0; i < m_full_vertex_to_vertex.size(); i++)
            if (m_full_vertex_to_vertex(i) >= 0)
                m_vertex_to_full_vertex(m_full_vertex_to_vertex(i)) = i;

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(num_vertices());
        for (int vi = 0; vi < num_vertices(); vi++) {
            triplets.emplace_back(vi, m_vertex_to_full_vertex(vi), 1.0);
        }

        m_select_vertices.resize(num_vertices(), full_num_vertices());
        m_select_vertices.setFromTriplets(triplets.begin(), triplets.end());
        m_select_vertices.makeCompressed();

        m_select_dof = vertex_matrix_to_dof_matrix(m_select_vertices, dim());

        if (displacement_map.size() == 0) {
            m_displacement_map = m_select_vertices;
            m_displacement_dof_map = m_select_dof;
        } else {
            assert(displacement_map.rows() == full_num_vertices());
            // assert(displacement_map.cols() == full_num_vertices());

            m_displacement_map = m_select_vertices * displacement_map;
            m_displacement_map.makeCompressed();

            m_displacement_dof_map =
                m_select_dof * vertex_matrix_to_dof_matrix(displacement_map, dim());
            m_displacement_dof_map.makeCompressed();
        }
    }

    Eigen::SparseMatrix<double> CurvedCollisionMesh::vertex_matrix_to_dof_matrix(
        const Eigen::SparseMatrix<double>& M_V, int dim)
    {
        std::vector<Eigen::Triplet<double>> triplets;
        using InnerIterator = Eigen::SparseMatrix<double>::InnerIterator;
        for (int k = 0; k < M_V.outerSize(); ++k) {
            for (InnerIterator it(M_V, k); it; ++it) {
                for (int d = 0; d < dim; d++) {
                    triplets.emplace_back(
                        dim * it.row() + d, dim * it.col() + d, it.value());
                }
            }
        }

        Eigen::SparseMatrix<double> M_dof(M_V.rows() * dim, M_V.cols() * dim);
        M_dof.setFromTriplets(triplets.begin(), triplets.end());
        M_dof.makeCompressed();
        return M_dof;
    }

    Eigen::MatrixXd CurvedCollisionMesh::vertices(const Eigen::MatrixXd& full_positions) const
    {
        assert(full_positions.rows() == full_num_vertices());
        assert(full_positions.cols() == dim());
        Eigen::MatrixXd out(num_vertices(), dim());
        out = full_positions(m_vertex_to_full_vertex, Eigen::all);
        return out;
    }

    Eigen::MatrixXd CurvedCollisionMesh::displace_vertices(const Eigen::MatrixXd& full_displacements) const
    {
        return m_rest_positions + map_displacements(full_displacements);
    }

    Eigen::MatrixXd CurvedCollisionMesh::map_displacements(
        const Eigen::MatrixXd& full_displacements) const
    {
        assert(m_displacement_map.cols() == full_displacements.rows());
        assert(full_displacements.cols() == dim());
        return m_displacement_map * full_displacements;
    }

    Eigen::VectorXd CurvedCollisionMesh::to_full_dof(const Eigen::VectorXd& x) const
    {
        return m_displacement_dof_map.transpose() * x;
    }

    Eigen::SparseMatrix<double> CurvedCollisionMesh::to_full_dof(const Eigen::SparseMatrix<double>& X) const
    {
        return m_displacement_dof_map.transpose() * X * m_displacement_dof_map;
    }
} // namespace polyfem::curved_ipc
