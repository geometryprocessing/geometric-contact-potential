#pragma once
#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <ipc/collision_mesh.hpp>

namespace polyfem::curved_ipc {

class CurvedCollisionMesh {
public:
    CurvedCollisionMesh(
        const mesh::Mesh &mesh,
        const std::vector<basis::ElementBases> &geom_bases,
        const std::vector<basis::ElementBases> &bases,
        const int n_bases,
        const std::vector<mesh::LocalBoundary> &total_local_boundary,
        const Eigen::SparseMatrix<double>& displacement_map =
            Eigen::SparseMatrix<double>());

    ~CurvedCollisionMesh() = default;

    /// @brief Get the number of vertices in the collision mesh.
    size_t num_vertices() const { return m_vertex_to_full_vertex.size(); }

    /// @brief Get the dimension of the mesh.
    size_t dim() const { return m_dim; }

    /// @brief Get the number of degrees of freedom in the collision mesh.
    size_t ndof() const { return num_vertices() * dim(); }

    /// @brief Get the number of vertices in the full mesh.
    size_t full_num_vertices() const { return m_full_vertex_to_vertex.size(); }

    /// @brief Get the number of degrees of freedom in the full mesh.
    size_t full_ndof() const { return full_num_vertices() * dim(); }

    /// @brief Compute the vertex positions from the positions of the full mesh.
    /// @param full_positions The vertex positions of the full mesh (#FV × dim).
    /// @return The vertex positions of the collision mesh (#V × dim).
    Eigen::MatrixXd vertices(const Eigen::MatrixXd& full_positions) const;

    /// @brief Compute the vertex positions from vertex displacements on the full mesh.
    /// @param full_displacements The vertex displacements on the full mesh (#FV × dim).
    /// @return The vertex positions of the collision mesh (#V × dim).
    Eigen::MatrixXd
    displace_vertices(const Eigen::MatrixXd& full_displacements) const;

    /// @brief Map vertex displacements on the full mesh to vertex displacements on the collision mesh.
    /// @param full_displacements The vertex displacements on the full mesh (#FV × dim).
    /// @return The vertex displacements on the collision mesh (#V × dim).
    Eigen::MatrixXd
    map_displacements(const Eigen::MatrixXd& full_displacements) const;

    /// @brief Map a vertex ID to the corresponding vertex ID in the full mesh.
    /// @param id Vertex ID in the collision mesh.
    /// @return Vertex ID in the full mesh.
    size_t to_full_vertex_id(const size_t id) const
    {
        assert(id < num_vertices());
        return m_vertex_to_full_vertex[id];
    }

    /// @brief Map a vector quantity on the collision mesh to the full mesh.
    /// This is useful for mapping gradients from the collision mesh to the full
    /// mesh (i.e., applies the chain-rule).
    /// @param x Vector quantity on the collision mesh with size equal to ndof().
    /// @return Vector quantity on the full mesh with size equal to full_ndof().
    Eigen::VectorXd to_full_dof(const Eigen::VectorXd& x) const;

    /// @brief Map a matrix quantity on the collision mesh to the full mesh.
    /// This is useful for mapping Hessians from the collision mesh to the full
    /// mesh (i.e., applies the chain-rule).
    /// @param X Matrix quantity on the collision mesh with size equal to ndof() × ndof().
    /// @return Matrix quantity on the full mesh with size equal to full_ndof() × full_ndof().
    Eigen::SparseMatrix<double>
    to_full_dof(const Eigen::SparseMatrix<double>& X) const;

    bool is_vertex_on_boundary(const int vi) const { return m_is_vertex_on_boundary[vi]; }

    /// A function that takes two vertex IDs and returns true if the vertices
    /// (and faces or edges containing the vertices) can collide. By default all
    /// primitives can collide with all other primitives.
    std::function<bool(size_t, size_t)> can_collide = default_can_collide;

protected:

    /// @brief Convert a matrix meant for M_V * vertices to M_dof * x by duplicating the entries dim times.
    static Eigen::SparseMatrix<double> vertex_matrix_to_dof_matrix(
        const Eigen::SparseMatrix<double>& M_V, int dim);
    
    const std::vector<basis::ElementBases> &geom_bases_;
    const std::vector<basis::ElementBases> &bases_;
    const std::vector<mesh::LocalBoundary> &total_local_boundary_;

    /// @brief The full vertex positions at rest (#FV × dim).
    Eigen::MatrixXd m_full_rest_positions;
    /// @brief The vertex positions at rest (#V × dim).
    Eigen::MatrixXd m_rest_positions;

    /// @brief Map from full vertices to collision vertices.
    /// @note Negative values indicate full vertex is dropped.
    Eigen::VectorXi m_full_vertex_to_vertex;
    /// @brief Map from collision vertices to full vertices.
    Eigen::VectorXi m_vertex_to_full_vertex;

    int m_dim;

    /// @brief Selection matrix S ∈ ℝ^{collision×full} for vertices
    Eigen::SparseMatrix<double> m_select_vertices;
    /// @brief Selection matrix S ∈ ℝ^{(dim*collision)×(dim*full)} for DOF
    Eigen::SparseMatrix<double> m_select_dof;
    /// @brief Mapping from full displacements to collision displacements
    /// @note this is premultiplied by m_select_vertices
    Eigen::SparseMatrix<double> m_displacement_map;
    /// @brief Mapping from full displacements DOF to collision displacements DOF
    /// @note this is premultiplied by m_select_dof
    Eigen::SparseMatrix<double> m_displacement_dof_map;

    /// @brief Is vertex on the boundary of the triangle mesh in 3D or polyline in 2D?
    std::vector<bool> m_is_vertex_on_boundary;

private:
    /// @brief By default all primitives can collide with all other primitives.
    static int default_can_collide(size_t, size_t) { return true; }
};
} // namespace polyfem::curved_ipc
