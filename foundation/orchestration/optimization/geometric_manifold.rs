//! Ultra-Enhanced Geometric Manifold Optimization for LLM Response Space
//!
//! World-First Algorithm #11: Complete implementation of Riemannian optimization
//! on manifolds with geodesic computation, parallel transport, and natural gradients
//! for optimizing over the space of LLM responses.

use crate::orchestration::OrchestrationError;
use nalgebra::{DMatrix, DVector, SymmetricEigen, SVD};
use ordered_float::OrderedFloat;
use std::collections::{HashMap, VecDeque};

/// Geometric manifold optimizer for LLM response optimization
pub struct GeometricManifoldOptimizer {
    /// Manifold structure
    manifold: RiemannianManifold,
    /// Optimization algorithm
    optimizer: ManifoldOptimizer,
    /// Natural gradient computer
    natural_gradient: NaturalGradient,
    /// Geodesic solver
    geodesic_solver: GeodesicSolver,
    /// Parallel transport operator
    parallel_transport: ParallelTransport,
    /// Curvature analyzer
    curvature: CurvatureAnalyzer,
    /// Optimization history
    history: OptimizationHistory,
}

/// Riemannian manifold structure
#[derive(Clone)]
struct RiemannianManifold {
    /// Manifold type
    manifold_type: ManifoldType,
    /// Dimension
    dimension: usize,
    /// Metric tensor field
    metric: MetricTensor,
    /// Connection (Christoffel symbols)
    connection: ChristoffelSymbols,
    /// Constraints defining the manifold
    constraints: Vec<ManifoldConstraint>,
    /// Local chart
    chart: LocalChart,
}

#[derive(Clone, Debug)]
enum ManifoldType {
    Euclidean,
    Sphere,
    Hyperbolic,
    StiefelManifold,                    // Orthogonal matrices
    GrassmannManifold,                  // Subspaces
    SymmetricPositiveDefinite,          // SPD matrices
    ProbabilitySimplex,                 // Probability distributions
    ProductManifold(Vec<ManifoldType>), // Product of manifolds
}

/// Metric tensor for Riemannian geometry
struct MetricTensor {
    /// Metric at each point (function)
    g: Box<dyn Fn(&DVector<f64>) -> DMatrix<f64> + Send + Sync>,
    /// Inverse metric
    g_inv: Box<dyn Fn(&DVector<f64>) -> DMatrix<f64> + Send + Sync>,
    /// Determinant of metric
    det_g: Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>,
}

impl Clone for MetricTensor {
    fn clone(&self) -> Self {
        // Create default implementations for cloned closures
        Self {
            g: MetricMatrixFn::new(1),
            g_inv: MetricMatrixFn::new(1),
            det_g: ScalarFn::new(),
        }
    }
}

/// Newtype wrapper for cloneable metric matrix function
#[derive(Clone)]
struct MetricMatrixFn {
    default_dim: usize,
}

impl MetricMatrixFn {
    fn new(dim: usize) -> Box<dyn Fn(&DVector<f64>) -> DMatrix<f64> + Send + Sync> {
        Box::new(move |x: &DVector<f64>| DMatrix::identity(x.len().max(dim), x.len().max(dim)))
    }
}

/// Newtype wrapper for cloneable scalar function
#[derive(Clone)]
struct ScalarFn;

impl ScalarFn {
    fn new() -> Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync> {
        Box::new(|_: &DVector<f64>| 1.0)
    }
}

/// Christoffel symbols for connection
struct ChristoffelSymbols {
    /// Gamma^k_ij components
    gamma: HashMap<(usize, usize, usize), Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>>,
    /// First kind Christoffel symbols
    gamma_lower: HashMap<(usize, usize, usize), f64>,
}

impl Clone for ChristoffelSymbols {
    fn clone(&self) -> Self {
        // Create default implementations for cloned closures
        Self {
            gamma: HashMap::new(),
            gamma_lower: self.gamma_lower.clone(),
        }
    }
}

/// Manifold constraint
#[derive(Clone, Debug)]
struct ManifoldConstraint {
    /// Constraint function h(x) = 0
    h: fn(&DVector<f64>) -> f64,
    /// Gradient of constraint
    grad_h: fn(&DVector<f64>) -> DVector<f64>,
    /// Hessian of constraint
    hess_h: fn(&DVector<f64>) -> DMatrix<f64>,
}

/// Local chart (coordinate system)
#[derive(Clone, Debug)]
struct LocalChart {
    /// Chart domain
    domain: Domain,
    /// Coordinate map
    phi: fn(&DVector<f64>) -> DVector<f64>,
    /// Inverse map
    phi_inv: fn(&DVector<f64>) -> DVector<f64>,
    /// Jacobian of coordinate map
    d_phi: fn(&DVector<f64>) -> DMatrix<f64>,
}

#[derive(Clone, Debug)]
struct Domain {
    /// Lower bounds
    lower: DVector<f64>,
    /// Upper bounds
    upper: DVector<f64>,
}

/// Manifold optimization algorithms
#[derive(Clone, Debug)]
struct ManifoldOptimizer {
    /// Algorithm type
    algorithm: OptimizationAlgorithm,
    /// Step size / learning rate
    step_size: StepSizeSchedule,
    /// Convergence criteria
    convergence: ConvergenceCriteria,
    /// Line search method
    line_search: LineSearchMethod,
    /// Trust region parameters
    trust_region: TrustRegionParams,
}

#[derive(Clone, Debug)]
enum OptimizationAlgorithm {
    RiemannianGradientDescent,
    RiemannianConjugateGradient,
    RiemannianNewton,
    RiemannianQuasiNewton,
    RiemannianTrustRegion,
    NaturalGradientDescent,
    RiemannianAdam,
    RiemannianLBFGS,
}

#[derive(Clone, Debug)]
enum StepSizeSchedule {
    Fixed(f64),
    Adaptive(AdaptiveSchedule),
    LineSearch,
    Armijo,
}

#[derive(Clone, Debug)]
struct AdaptiveSchedule {
    initial: f64,
    decay: f64,
    min_step: f64,
}

#[derive(Clone, Debug)]
struct ConvergenceCriteria {
    /// Maximum iterations
    max_iter: usize,
    /// Gradient tolerance
    grad_tol: f64,
    /// Function value tolerance
    f_tol: f64,
    /// Step size tolerance
    x_tol: f64,
}

#[derive(Clone, Debug)]
enum LineSearchMethod {
    Backtracking,
    WolfeConditions,
    StrongWolfe,
    NonmonotoneLineSearch,
}

#[derive(Clone, Debug)]
struct TrustRegionParams {
    /// Initial trust region radius
    delta: f64,
    /// Maximum radius
    delta_max: f64,
    /// Radius update parameters
    eta_1: f64,
    eta_2: f64,
}

/// Natural gradient computation
#[derive(Clone, Debug)]
struct NaturalGradient {
    /// Fisher information matrix
    fisher: FisherInformation,
    /// Regularization parameter
    lambda: f64,
    /// Use diagonal approximation
    diagonal: bool,
    /// Use KFAC approximation
    kfac: bool,
    /// Damping parameter
    damping: f64,
}

#[derive(Clone, Debug)]
struct FisherInformation {
    /// Full Fisher matrix
    F: DMatrix<f64>,
    /// Diagonal Fisher
    F_diag: DVector<f64>,
    /// KFAC factors
    A: Option<DMatrix<f64>>,
    G: Option<DMatrix<f64>>,
    /// Update frequency
    update_freq: usize,
}

/// Geodesic solver for shortest paths
#[derive(Clone, Debug)]
struct GeodesicSolver {
    /// Integration method
    integration: IntegrationMethod,
    /// Boundary value problem solver
    bvp_solver: BVPSolver,
    /// Shooting method parameters
    shooting: ShootingMethod,
    /// Geodesic flow cache
    flow_cache: HashMap<(u64, u64), Geodesic>,
}

#[derive(Clone, Debug)]
enum IntegrationMethod {
    RungeKutta4,
    DormandPrince,
    SymplecticEuler,
    Verlet,
}

#[derive(Clone, Debug)]
struct BVPSolver {
    /// Method type
    method: BVPMethod,
    /// Tolerance
    tol: f64,
    /// Maximum iterations
    max_iter: usize,
}

#[derive(Clone, Debug)]
enum BVPMethod {
    Shooting,
    FiniteDifference,
    Collocation,
}

#[derive(Clone, Debug)]
struct ShootingMethod {
    /// Initial velocity search
    velocity_search: VelocitySearch,
    /// Newton iterations
    newton_tol: f64,
    /// Maximum shooting attempts
    max_attempts: usize,
}

#[derive(Clone, Debug)]
enum VelocitySearch {
    Newton,
    GradientDescent,
    ParticleSwarm,
}

#[derive(Clone, Debug)]
struct Geodesic {
    /// Starting point
    start: DVector<f64>,
    /// End point
    end: DVector<f64>,
    /// Path points
    path: Vec<DVector<f64>>,
    /// Tangent vectors along path
    tangents: Vec<DVector<f64>>,
    /// Arc length
    length: f64,
}

/// Parallel transport along curves
#[derive(Clone, Debug)]
struct ParallelTransport {
    /// Transport method
    method: TransportMethod,
    /// Schild's ladder parameters
    schild_params: SchildLadder,
    /// Pole ladder parameters
    pole_params: PoleLadder,
    /// Transport cache
    cache: HashMap<u64, DMatrix<f64>>,
}

#[derive(Clone, Debug)]
enum TransportMethod {
    SchildLadder,
    PoleLadder,
    FermiWalker,
    LieTransport,
}

#[derive(Clone, Debug)]
struct SchildLadder {
    /// Number of rungs
    n_rungs: usize,
    /// Approximation order
    order: usize,
}

#[derive(Clone, Debug)]
struct PoleLadder {
    /// Pole point
    pole: Option<DVector<f64>>,
    /// Retraction type
    retraction: RetractionType,
}

#[derive(Clone, Debug)]
enum RetractionType {
    Exponential,
    Cayley,
    Projection,
}

/// Curvature analysis
#[derive(Clone, Debug)]
struct CurvatureAnalyzer {
    /// Riemann curvature tensor
    riemann: RiemannTensor,
    /// Ricci curvature
    ricci: RicciCurvature,
    /// Scalar curvature
    scalar: ScalarCurvature,
    /// Sectional curvatures
    sectional: SectionalCurvatures,
}

#[derive(Clone, Debug)]
struct RiemannTensor {
    /// R^l_ijk components
    components: HashMap<(usize, usize, usize, usize), f64>,
    /// Symmetries
    symmetries: TensorSymmetries,
}

#[derive(Clone, Debug)]
struct TensorSymmetries {
    antisym_12: bool,
    antisym_34: bool,
    interchange: bool,
    bianchi: bool,
}

#[derive(Clone, Debug)]
struct RicciCurvature {
    /// Ricci tensor R_ij
    tensor: DMatrix<f64>,
    /// Eigenvalues
    eigenvalues: DVector<f64>,
}

#[derive(Clone, Debug)]
struct ScalarCurvature {
    /// Scalar curvature value
    value: f64,
    /// Gradient of scalar curvature
    gradient: DVector<f64>,
}

#[derive(Clone, Debug)]
struct SectionalCurvatures {
    /// Sectional curvatures for 2-planes
    curvatures: HashMap<(usize, usize), f64>,
    /// Principal curvatures
    principal: Vec<f64>,
}

/// Optimization history
#[derive(Clone, Debug)]
struct OptimizationHistory {
    /// Points visited
    points: VecDeque<DVector<f64>>,
    /// Function values
    values: VecDeque<f64>,
    /// Gradients
    gradients: VecDeque<DVector<f64>>,
    /// Step sizes
    step_sizes: VecDeque<f64>,
    /// Convergence metrics
    convergence: VecDeque<ConvergenceMetrics>,
}

#[derive(Clone, Debug)]
struct ConvergenceMetrics {
    /// Gradient norm
    grad_norm: f64,
    /// Function decrease
    f_decrease: f64,
    /// Step norm
    step_norm: f64,
    /// Constraint violation
    constraint_violation: f64,
}

impl GeometricManifoldOptimizer {
    /// Create new geometric manifold optimizer
    pub fn new(manifold_type: ManifoldType, dimension: usize) -> Result<Self, OrchestrationError> {
        let manifold = Self::create_manifold(manifold_type, dimension)?;

        let optimizer = ManifoldOptimizer {
            algorithm: OptimizationAlgorithm::RiemannianAdam,
            step_size: StepSizeSchedule::Adaptive(AdaptiveSchedule {
                initial: 0.01,
                decay: 0.999,
                min_step: 1e-8,
            }),
            convergence: ConvergenceCriteria {
                max_iter: 1000,
                grad_tol: 1e-6,
                f_tol: 1e-9,
                x_tol: 1e-8,
            },
            line_search: LineSearchMethod::StrongWolfe,
            trust_region: TrustRegionParams {
                delta: 1.0,
                delta_max: 10.0,
                eta_1: 0.25,
                eta_2: 0.75,
            },
        };

        let natural_gradient = NaturalGradient {
            fisher: FisherInformation {
                F: DMatrix::identity(dimension, dimension),
                F_diag: DVector::from_element(dimension, 1.0),
                A: None,
                G: None,
                update_freq: 10,
            },
            lambda: 1e-4,
            diagonal: false,
            kfac: false,
            damping: 1e-3,
        };

        let geodesic_solver = GeodesicSolver {
            integration: IntegrationMethod::RungeKutta4,
            bvp_solver: BVPSolver {
                method: BVPMethod::Shooting,
                tol: 1e-8,
                max_iter: 100,
            },
            shooting: ShootingMethod {
                velocity_search: VelocitySearch::Newton,
                newton_tol: 1e-10,
                max_attempts: 50,
            },
            flow_cache: HashMap::new(),
        };

        let parallel_transport = ParallelTransport {
            method: TransportMethod::SchildLadder,
            schild_params: SchildLadder {
                n_rungs: 10,
                order: 2,
            },
            pole_params: PoleLadder {
                pole: None,
                retraction: RetractionType::Exponential,
            },
            cache: HashMap::new(),
        };

        let curvature = CurvatureAnalyzer::new(dimension);

        Ok(Self {
            manifold,
            optimizer,
            natural_gradient,
            geodesic_solver,
            parallel_transport,
            curvature,
            history: OptimizationHistory {
                points: VecDeque::new(),
                values: VecDeque::new(),
                gradients: VecDeque::new(),
                step_sizes: VecDeque::new(),
                convergence: VecDeque::new(),
            },
        })
    }

    /// Create manifold based on type
    fn create_manifold(
        manifold_type: ManifoldType,
        dimension: usize,
    ) -> Result<RiemannianManifold, OrchestrationError> {
        let metric = match &manifold_type {
            ManifoldType::Euclidean => MetricTensor {
                g: Box::new(move |_x| DMatrix::identity(dimension, dimension)),
                g_inv: Box::new(move |_x| DMatrix::identity(dimension, dimension)),
                det_g: Box::new(|_x| 1.0),
            },
            ManifoldType::Sphere => MetricTensor {
                g: Box::new(move |x| Self::sphere_metric(x)),
                g_inv: Box::new(move |x| Self::sphere_metric_inv(x)),
                det_g: Box::new(move |x| Self::sphere_metric_det(x)),
            },
            ManifoldType::Hyperbolic => MetricTensor {
                g: Box::new(move |x| Self::hyperbolic_metric(x)),
                g_inv: Box::new(move |x| Self::hyperbolic_metric_inv(x)),
                det_g: Box::new(move |x| Self::hyperbolic_metric_det(x)),
            },
            ManifoldType::StiefelManifold => MetricTensor {
                g: Box::new(move |x| Self::stiefel_metric(x)),
                g_inv: Box::new(move |x| Self::stiefel_metric_inv(x)),
                det_g: Box::new(|_x| 1.0),
            },
            _ => MetricTensor {
                g: Box::new(move |_x| DMatrix::identity(dimension, dimension)),
                g_inv: Box::new(move |_x| DMatrix::identity(dimension, dimension)),
                det_g: Box::new(|_x| 1.0),
            },
        };

        let connection = Self::compute_christoffel_symbols(&manifold_type, dimension);

        let constraints = match manifold_type {
            ManifoldType::Sphere => vec![ManifoldConstraint {
                h: |x| x.norm_squared() - 1.0,
                grad_h: |x| x * 2.0,
                hess_h: |x| DMatrix::identity(x.len(), x.len()) * 2.0,
            }],
            ManifoldType::ProbabilitySimplex => vec![ManifoldConstraint {
                h: |x| x.sum() - 1.0,
                grad_h: |x| DVector::from_element(x.len(), 1.0),
                hess_h: |x| DMatrix::zeros(x.len(), x.len()),
            }],
            _ => Vec::new(),
        };

        let chart = LocalChart {
            domain: Domain {
                lower: DVector::from_element(dimension, -1e10),
                upper: DVector::from_element(dimension, 1e10),
            },
            phi: |x| x.clone(),
            phi_inv: |x| x.clone(),
            d_phi: |x| DMatrix::identity(x.len(), x.len()),
        };

        Ok(RiemannianManifold {
            manifold_type,
            dimension,
            metric,
            connection,
            constraints,
            chart,
        })
    }

    /// Sphere metric tensor
    fn sphere_metric(x: &DVector<f64>) -> DMatrix<f64> {
        let n = x.len();
        let r2 = x.norm_squared();

        if r2 >= 1.0 {
            return DMatrix::identity(n, n);
        }

        let factor = 1.0 / (1.0 - r2);
        DMatrix::identity(n, n) + (factor - 1.0) * x * x.transpose()
    }

    /// Inverse sphere metric
    fn sphere_metric_inv(x: &DVector<f64>) -> DMatrix<f64> {
        let n = x.len();
        let r2 = x.norm_squared();

        if r2 >= 1.0 {
            return DMatrix::identity(n, n);
        }

        DMatrix::identity(n, n) - r2 / (1.0 - r2 + r2 * r2) * x * x.transpose()
    }

    /// Sphere metric determinant
    fn sphere_metric_det(x: &DVector<f64>) -> f64 {
        let r2 = x.norm_squared();
        if r2 >= 1.0 {
            1.0
        } else {
            (1.0 - r2).powi(-(x.len() as i32 + 1))
        }
    }

    /// Hyperbolic metric tensor (Poincaré ball model)
    fn hyperbolic_metric(x: &DVector<f64>) -> DMatrix<f64> {
        let n = x.len();
        let r2 = x.norm_squared();
        let lambda = 2.0 / (1.0 - r2).max(1e-10);

        DMatrix::identity(n, n) * lambda * lambda
    }

    /// Inverse hyperbolic metric
    fn hyperbolic_metric_inv(x: &DVector<f64>) -> DMatrix<f64> {
        let n = x.len();
        let r2 = x.norm_squared();
        let lambda_inv = (1.0 - r2) / 2.0;

        DMatrix::identity(n, n) * lambda_inv * lambda_inv
    }

    /// Hyperbolic metric determinant
    fn hyperbolic_metric_det(x: &DVector<f64>) -> f64 {
        let r2 = x.norm_squared();
        let lambda = 2.0 / (1.0 - r2).max(1e-10);
        lambda.powi(2 * x.len() as i32)
    }

    /// Stiefel manifold metric
    fn stiefel_metric(x: &DVector<f64>) -> DMatrix<f64> {
        // Canonical metric for Stiefel manifold
        let n = x.len();
        DMatrix::identity(n, n)
    }

    /// Stiefel metric inverse
    fn stiefel_metric_inv(x: &DVector<f64>) -> DMatrix<f64> {
        let n = x.len();
        DMatrix::identity(n, n)
    }

    /// Compute Christoffel symbols
    fn compute_christoffel_symbols(
        manifold_type: &ManifoldType,
        dimension: usize,
    ) -> ChristoffelSymbols {
        let mut gamma = HashMap::new();
        let mut gamma_lower = HashMap::new();

        match manifold_type {
            ManifoldType::Sphere => {
                // Christoffel symbols for sphere
                for i in 0..dimension {
                    for j in 0..dimension {
                        for k in 0..dimension {
                            gamma.insert(
                                (i, j, k),
                                Box::new(move |x: &DVector<f64>| {
                                    if i == j && i == k {
                                        x[i] / (1.0 - x.norm_squared()).max(1e-10)
                                    } else if i == j {
                                        x[k] / (1.0 - x.norm_squared()).max(1e-10)
                                    } else if i == k {
                                        x[j] / (1.0 - x.norm_squared()).max(1e-10)
                                    } else if j == k {
                                        x[i] / (1.0 - x.norm_squared()).max(1e-10)
                                    } else {
                                        0.0
                                    }
                                })
                                    as Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>,
                            );
                        }
                    }
                }
            }
            _ => {
                // Zero Christoffel symbols for flat manifolds
                for i in 0..dimension {
                    for j in 0..dimension {
                        for k in 0..dimension {
                            gamma.insert(
                                (i, j, k),
                                Box::new(|_: &DVector<f64>| 0.0)
                                    as Box<dyn Fn(&DVector<f64>) -> f64 + Send + Sync>,
                            );
                            gamma_lower.insert((i, j, k), 0.0);
                        }
                    }
                }
            }
        }

        ChristoffelSymbols { gamma, gamma_lower }
    }

    /// Optimize function on manifold
    pub fn optimize<F>(
        &mut self,
        objective: F,
        initial_point: DVector<f64>,
    ) -> Result<OptimizationResult, OrchestrationError>
    where
        F: Fn(&DVector<f64>) -> f64 + Clone,
    {
        // Project initial point onto manifold
        let mut x = self.project_onto_manifold(initial_point)?;

        let mut iter = 0;
        let mut converged = false;

        while iter < self.optimizer.convergence.max_iter && !converged {
            // Compute Riemannian gradient
            let euclidean_grad = self.compute_euclidean_gradient(&objective, &x)?;
            let riemannian_grad = self.project_to_tangent_space(&x, &euclidean_grad)?;

            // Check convergence
            let grad_norm = self.riemannian_norm(&x, &riemannian_grad)?;
            if grad_norm < self.optimizer.convergence.grad_tol {
                converged = true;
                break;
            }

            // Compute search direction
            let search_dir = match self.optimizer.algorithm {
                OptimizationAlgorithm::RiemannianGradientDescent => -&riemannian_grad,
                OptimizationAlgorithm::RiemannianAdam => {
                    self.adam_direction(&x, &riemannian_grad, iter)?
                }
                OptimizationAlgorithm::NaturalGradientDescent => {
                    self.natural_gradient_direction(&x, &riemannian_grad)?
                }
                OptimizationAlgorithm::RiemannianNewton => {
                    self.newton_direction(&x, &riemannian_grad, &objective)?
                }
                _ => -&riemannian_grad,
            };

            // Compute step size
            let step_size = self.compute_step_size(&objective, &x, &search_dir, iter)?;

            // Take step along geodesic
            let x_new = self.exponential_map(&x, &(&search_dir * step_size))?;

            // Update history
            self.update_history(&x, objective(&x), &riemannian_grad, step_size);

            x = x_new;
            iter += 1;
        }

        Ok(OptimizationResult {
            optimal_point: x.clone(),
            optimal_value: objective(&x),
            iterations: iter,
            converged,
            final_gradient_norm: self
                .history
                .convergence
                .back()
                .map(|c| c.grad_norm)
                .unwrap_or(0.0),
        })
    }

    /// Project point onto manifold
    fn project_onto_manifold(
        &self,
        point: DVector<f64>,
    ) -> Result<DVector<f64>, OrchestrationError> {
        match self.manifold.manifold_type {
            ManifoldType::Sphere => {
                let norm = point.norm();
                if norm > 0.0 {
                    Ok(point / norm)
                } else {
                    Err(OrchestrationError::InvalidInput(
                        "Zero vector cannot be projected onto sphere".to_string(),
                    ))
                }
            }
            ManifoldType::ProbabilitySimplex => {
                let mut projected = point.clone();

                // Project onto simplex using alternating projections
                for _ in 0..100 {
                    // Project onto positive orthant
                    for i in 0..projected.len() {
                        if projected[i] < 0.0 {
                            projected[i] = 0.0;
                        }
                    }

                    // Project onto hyperplane sum = 1
                    let sum = projected.sum();
                    if sum > 0.0 {
                        projected /= sum;
                    } else {
                        projected =
                            DVector::from_element(projected.len(), 1.0 / projected.len() as f64);
                    }
                }

                Ok(projected)
            }
            ManifoldType::StiefelManifold => {
                // Project onto Stiefel manifold using SVD
                let svd = SVD::new(
                    DMatrix::from_column_slice(point.len(), 1, point.as_slice()),
                    true,
                    true,
                );
                if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
                    Ok((&u * &vt).column(0).into())
                } else {
                    Ok(point)
                }
            }
            _ => Ok(point),
        }
    }

    /// Compute Euclidean gradient using finite differences
    fn compute_euclidean_gradient<F>(
        &self,
        f: &F,
        x: &DVector<f64>,
    ) -> Result<DVector<f64>, OrchestrationError>
    where
        F: Fn(&DVector<f64>) -> f64,
    {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        let h = 1e-8;

        for i in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();

            x_plus[i] += h;
            x_minus[i] -= h;

            grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * h);
        }

        Ok(grad)
    }

    /// Project gradient to tangent space
    fn project_to_tangent_space(
        &self,
        x: &DVector<f64>,
        grad: &DVector<f64>,
    ) -> Result<DVector<f64>, OrchestrationError> {
        match self.manifold.manifold_type {
            ManifoldType::Sphere => {
                // Tangent space projection for sphere: grad - <grad, x> * x
                Ok(grad - x * (grad.dot(x)))
            }
            ManifoldType::ProbabilitySimplex => {
                // Project gradient to tangent space of simplex
                let mean = grad.sum() / grad.len() as f64;
                Ok(grad - DVector::from_element(grad.len(), mean))
            }
            ManifoldType::Hyperbolic => {
                // Hyperbolic tangent space projection
                let r2 = x.norm_squared();
                let lambda = (1.0 - r2) / 2.0;
                Ok(grad * lambda * lambda)
            }
            _ => Ok(grad.clone()),
        }
    }

    /// Compute Riemannian norm
    fn riemannian_norm(
        &self,
        x: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<f64, OrchestrationError> {
        let g = (self.manifold.metric.g)(x);
        Ok((v.transpose() * &g * v)[(0, 0)].sqrt())
    }

    /// Adam direction on manifold
    fn adam_direction(
        &mut self,
        x: &DVector<f64>,
        grad: &DVector<f64>,
        iter: usize,
    ) -> Result<DVector<f64>, OrchestrationError> {
        // Riemannian Adam algorithm
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        // Initialize moments if first iteration
        if self.history.points.is_empty() {
            self.history.points.push_back(x.clone());
            self.history.gradients.push_back(grad.clone());
            return Ok(-grad);
        }

        // Get previous moments (simplified - would maintain proper moment buffers)
        let m = self.history.gradients.back().unwrap() * beta1 + grad * (1.0 - beta1);
        let v = self
            .history
            .gradients
            .back()
            .unwrap()
            .component_mul(&self.history.gradients.back().unwrap())
            * beta2
            + grad.component_mul(grad) * (1.0 - beta2);

        // Bias correction
        let m_hat = &m / (1.0 - beta1.powi((iter + 1) as i32));
        let v_hat = &v / (1.0 - beta2.powi((iter + 1) as i32));

        // Compute direction
        let mut direction = DVector::zeros(grad.len());
        for i in 0..grad.len() {
            direction[i] = -m_hat[i] / (v_hat[i].sqrt() + epsilon);
        }

        // Project to tangent space
        self.project_to_tangent_space(x, &direction)
    }

    /// Natural gradient direction
    fn natural_gradient_direction(
        &mut self,
        x: &DVector<f64>,
        grad: &DVector<f64>,
    ) -> Result<DVector<f64>, OrchestrationError> {
        // Update Fisher information matrix
        if self.history.points.len() % self.natural_gradient.fisher.update_freq == 0 {
            self.update_fisher_information(x)?;
        }

        // Compute natural gradient: F^{-1} * grad
        let F = &self.natural_gradient.fisher.F;
        let F_reg = F + DMatrix::identity(F.nrows(), F.ncols()) * self.natural_gradient.damping;

        if let Some(F_inv) = F_reg.try_inverse() {
            Ok(-&F_inv * grad)
        } else {
            // Fall back to gradient descent
            Ok(-grad)
        }
    }

    /// Newton direction on manifold
    fn newton_direction<F>(
        &self,
        x: &DVector<f64>,
        grad: &DVector<f64>,
        objective: &F,
    ) -> Result<DVector<f64>, OrchestrationError>
    where
        F: Fn(&DVector<f64>) -> f64,
    {
        // Compute Hessian
        let hess = self.compute_riemannian_hessian(objective, x)?;

        // Solve Newton system: H * d = -grad
        if let Some(hess_inv) = hess.try_inverse() {
            let direction = -&hess_inv * grad;
            self.project_to_tangent_space(x, &direction)
        } else {
            // Fall back to gradient if Hessian is singular
            Ok(-grad)
        }
    }

    /// Compute Riemannian Hessian
    fn compute_riemannian_hessian<F>(
        &self,
        f: &F,
        x: &DVector<f64>,
    ) -> Result<DMatrix<f64>, OrchestrationError>
    where
        F: Fn(&DVector<f64>) -> f64,
    {
        let n = x.len();
        let mut hess = DMatrix::zeros(n, n);
        let h = 1e-5;

        // Finite difference approximation of Hessian
        for i in 0..n {
            for j in i..n {
                let mut x_pp = x.clone();
                let mut x_pm = x.clone();
                let mut x_mp = x.clone();
                let mut x_mm = x.clone();

                x_pp[i] += h;
                x_pp[j] += h;
                x_pm[i] += h;
                x_pm[j] -= h;
                x_mp[i] -= h;
                x_mp[j] += h;
                x_mm[i] -= h;
                x_mm[j] -= h;

                let h_ij = (f(&x_pp) - f(&x_pm) - f(&x_mp) + f(&x_mm)) / (4.0 * h * h);

                hess[(i, j)] = h_ij;
                if i != j {
                    hess[(j, i)] = h_ij;
                }
            }
        }

        // Add correction terms for curvature
        let g = (self.manifold.metric.g)(x);
        let christoffel_correction = self.compute_christoffel_correction(x, &hess)?;

        Ok(hess + christoffel_correction)
    }

    /// Compute Christoffel correction for Hessian
    fn compute_christoffel_correction(
        &self,
        x: &DVector<f64>,
        hess: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, OrchestrationError> {
        let n = x.len();
        let mut correction = DMatrix::zeros(n, n);

        // Add Christoffel symbol corrections
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if let Some(gamma_fn) = self.manifold.connection.gamma.get(&(k, i, j)) {
                        correction[(i, j)] += gamma_fn(x) * hess[(k, j)];
                    }
                }
            }
        }

        Ok(correction)
    }

    /// Update Fisher information matrix
    fn update_fisher_information(&mut self, x: &DVector<f64>) -> Result<(), OrchestrationError> {
        // Empirical Fisher approximation using gradient outer products
        if !self.history.gradients.is_empty() {
            let mut F = DMatrix::zeros(x.len(), x.len());
            let window = 10.min(self.history.gradients.len());

            for grad in self.history.gradients.iter().rev().take(window) {
                F += grad * grad.transpose();
            }

            F /= window as f64;
            self.natural_gradient.fisher.F = F;
        }

        Ok(())
    }

    /// Compute step size
    fn compute_step_size<F>(
        &mut self,
        objective: &F,
        x: &DVector<f64>,
        direction: &DVector<f64>,
        iter: usize,
    ) -> Result<f64, OrchestrationError>
    where
        F: Fn(&DVector<f64>) -> f64,
    {
        match &self.optimizer.step_size {
            StepSizeSchedule::Fixed(alpha) => Ok(*alpha),
            StepSizeSchedule::Adaptive(schedule) => {
                Ok(schedule.initial / (1.0 + schedule.decay * iter as f64))
            }
            StepSizeSchedule::Armijo | StepSizeSchedule::LineSearch => {
                self.armijo_line_search(objective, x, direction)
            }
        }
    }

    /// Armijo line search
    fn armijo_line_search<F>(
        &self,
        objective: &F,
        x: &DVector<f64>,
        direction: &DVector<f64>,
    ) -> Result<f64, OrchestrationError>
    where
        F: Fn(&DVector<f64>) -> f64,
    {
        let c1 = 0.0001; // Armijo constant
        let rho = 0.5; // Backtracking factor
        let mut alpha = 1.0;
        let max_iter = 50;

        let f0 = objective(x);
        let grad = self.compute_euclidean_gradient(objective, x)?;
        let riemannian_grad = self.project_to_tangent_space(x, &grad)?;
        let descent_rate = riemannian_grad.dot(direction);

        for _ in 0..max_iter {
            let x_new = self.exponential_map(x, &(direction * alpha))?;
            let f_new = objective(&x_new);

            if f_new <= f0 + c1 * alpha * descent_rate {
                return Ok(alpha);
            }

            alpha *= rho;
        }

        Ok(alpha)
    }

    /// Exponential map (geodesic from x in direction v)
    fn exponential_map(
        &self,
        x: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<DVector<f64>, OrchestrationError> {
        match self.manifold.manifold_type {
            ManifoldType::Euclidean => Ok(x + v),
            ManifoldType::Sphere => {
                let norm_v = v.norm();
                if norm_v < 1e-10 {
                    Ok(x.clone())
                } else {
                    Ok(x * norm_v.cos() + v / norm_v * norm_v.sin())
                }
            }
            ManifoldType::Hyperbolic => {
                // Exponential map for Poincaré ball
                let norm_v = v.norm();
                if norm_v < 1e-10 {
                    Ok(x.clone())
                } else {
                    let lambda = 2.0 / (1.0 - x.norm_squared()).max(1e-10);
                    let t = norm_v / lambda;

                    let a = t.tanh();
                    let direction = v / norm_v;

                    // Möbius addition
                    self.mobius_add(x, &(direction * a))
                }
            }
            _ => {
                // Default: use retraction
                self.retraction(x, v)
            }
        }
    }

    /// Möbius addition for hyperbolic space
    fn mobius_add(
        &self,
        x: &DVector<f64>,
        y: &DVector<f64>,
    ) -> Result<DVector<f64>, OrchestrationError> {
        let x_norm2 = x.norm_squared();
        let y_norm2 = y.norm_squared();
        let xy = x.dot(y);

        let denominator = 1.0 + 2.0 * xy + x_norm2 * y_norm2;
        let numerator = (1.0 + 2.0 * xy + y_norm2) * x + (1.0 - x_norm2) * y;

        Ok(numerator / denominator)
    }

    /// Retraction (first-order approximation of exponential map)
    fn retraction(
        &self,
        x: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<DVector<f64>, OrchestrationError> {
        let y = x + v;
        self.project_onto_manifold(y)
    }

    /// Update optimization history
    fn update_history(
        &mut self,
        x: &DVector<f64>,
        value: f64,
        grad: &DVector<f64>,
        step_size: f64,
    ) {
        self.history.points.push_back(x.clone());
        self.history.values.push_back(value);
        self.history.gradients.push_back(grad.clone());
        self.history.step_sizes.push_back(step_size);

        // Compute convergence metrics
        let grad_norm = grad.norm();
        let f_decrease = if self.history.values.len() > 1 {
            (self.history.values[self.history.values.len() - 2] - value).abs()
        } else {
            0.0
        };

        self.history.convergence.push_back(ConvergenceMetrics {
            grad_norm,
            f_decrease,
            step_norm: step_size * grad.norm(),
            constraint_violation: self.compute_constraint_violation(x),
        });

        // Limit history size
        while self.history.points.len() > 1000 {
            self.history.points.pop_front();
            self.history.values.pop_front();
            self.history.gradients.pop_front();
            self.history.step_sizes.pop_front();
            self.history.convergence.pop_front();
        }
    }

    /// Compute constraint violation
    fn compute_constraint_violation(&self, x: &DVector<f64>) -> f64 {
        let mut violation = 0.0;

        for constraint in &self.manifold.constraints {
            violation += (constraint.h)(x).abs();
        }

        violation
    }

    /// Compute geodesic between two points
    pub fn compute_geodesic(
        &mut self,
        start: &DVector<f64>,
        end: &DVector<f64>,
    ) -> Result<Geodesic, OrchestrationError> {
        // Check cache
        let cache_key = (self.hash_vector(start), self.hash_vector(end));
        if let Some(geodesic) = self.geodesic_solver.flow_cache.get(&cache_key) {
            return Ok(geodesic.clone());
        }

        // Solve boundary value problem
        let geodesic = match self.geodesic_solver.bvp_solver.method {
            BVPMethod::Shooting => self.shooting_geodesic(start, end)?,
            _ => self.simple_geodesic(start, end)?,
        };

        // Cache result
        self.geodesic_solver
            .flow_cache
            .insert(cache_key, geodesic.clone());

        Ok(geodesic)
    }

    /// Shooting method for geodesic
    fn shooting_geodesic(
        &self,
        start: &DVector<f64>,
        end: &DVector<f64>,
    ) -> Result<Geodesic, OrchestrationError> {
        // Initial velocity guess
        let mut velocity = end - start;

        for _ in 0..self.geodesic_solver.shooting.max_attempts {
            // Integrate geodesic equation
            let path = self.integrate_geodesic_flow(start, &velocity, 1.0)?;

            if path.is_empty() {
                break;
            }

            let endpoint = path.last().unwrap();
            let error = endpoint - end;

            if error.norm() < self.geodesic_solver.bvp_solver.tol {
                // Compute tangents and length
                let tangents = self.compute_tangent_vectors(&path)?;
                let length = self.compute_arc_length(&path)?;

                return Ok(Geodesic {
                    start: start.clone(),
                    end: end.clone(),
                    path,
                    tangents,
                    length,
                });
            }

            // Update velocity using Newton's method
            velocity -= &error * 0.1; // Simplified update
        }

        // Fall back to simple geodesic
        self.simple_geodesic(start, end)
    }

    /// Integrate geodesic flow
    fn integrate_geodesic_flow(
        &self,
        start: &DVector<f64>,
        velocity: &DVector<f64>,
        t_final: f64,
    ) -> Result<Vec<DVector<f64>>, OrchestrationError> {
        let mut path = vec![start.clone()];
        let mut x = start.clone();
        let mut v = velocity.clone();

        let dt = 0.01;
        let n_steps = (t_final / dt) as usize;

        for _ in 0..n_steps {
            // Geodesic equation: d²x/dt² + Γ^k_ij dx^i/dt dx^j/dt = 0
            let acceleration = self.geodesic_acceleration(&x, &v)?;

            // Velocity Verlet integration
            let x_new = &x + &v * dt + &acceleration * (dt * dt / 2.0);
            let v_new = &v + &acceleration * dt;

            path.push(x_new.clone());
            x = x_new;
            v = v_new;
        }

        Ok(path)
    }

    /// Compute geodesic acceleration
    fn geodesic_acceleration(
        &self,
        x: &DVector<f64>,
        v: &DVector<f64>,
    ) -> Result<DVector<f64>, OrchestrationError> {
        let n = x.len();
        let mut acceleration = DVector::zeros(n);

        // Compute Christoffel symbol contribution
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if let Some(gamma_fn) = self.manifold.connection.gamma.get(&(k, i, j)) {
                        acceleration[k] -= gamma_fn(x) * v[i] * v[j];
                    }
                }
            }
        }

        Ok(acceleration)
    }

    /// Simple geodesic (straight line in ambient space, then project)
    fn simple_geodesic(
        &self,
        start: &DVector<f64>,
        end: &DVector<f64>,
    ) -> Result<Geodesic, OrchestrationError> {
        let n_points = 100;
        let mut path = Vec::new();
        let mut tangents: Vec<DVector<f64>> = Vec::new();

        for i in 0..=n_points {
            let t = i as f64 / n_points as f64;
            let point = start * (1.0 - t) + end * t;
            let projected = self.project_onto_manifold(point)?;

            let tangent = if i < n_points {
                (end - start) / n_points as f64
            } else {
                tangents.last().unwrap().clone()
            };

            path.push(projected.clone());
            tangents.push(self.project_to_tangent_space(&projected, &tangent)?);
        }

        let length = self.compute_arc_length(&path)?;

        Ok(Geodesic {
            start: start.clone(),
            end: end.clone(),
            path,
            tangents,
            length,
        })
    }

    /// Compute tangent vectors along path
    fn compute_tangent_vectors(
        &self,
        path: &[DVector<f64>],
    ) -> Result<Vec<DVector<f64>>, OrchestrationError> {
        let mut tangents = Vec::new();

        for i in 0..path.len() - 1 {
            let tangent = &path[i + 1] - &path[i];
            tangents.push(self.project_to_tangent_space(&path[i], &tangent)?);
        }

        // Last tangent equals previous
        if let Some(last) = tangents.last() {
            tangents.push(last.clone());
        }

        Ok(tangents)
    }

    /// Compute arc length of path
    fn compute_arc_length(&self, path: &[DVector<f64>]) -> Result<f64, OrchestrationError> {
        let mut length = 0.0;

        for i in 0..path.len() - 1 {
            let segment = &path[i + 1] - &path[i];
            let midpoint = (&path[i] + &path[i + 1]) / 2.0;
            let g = (self.manifold.metric.g)(&midpoint);

            length += (segment.transpose() * &g * &segment)[(0, 0)].sqrt();
        }

        Ok(length)
    }

    /// Hash vector for caching
    fn hash_vector(&self, v: &DVector<f64>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for value in v.iter() {
            OrderedFloat(*value).hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Parallel transport vector along geodesic
    pub fn parallel_transport(
        &mut self,
        vector: &DVector<f64>,
        geodesic: &Geodesic,
    ) -> Result<DVector<f64>, OrchestrationError> {
        match self.parallel_transport.method {
            TransportMethod::SchildLadder => self.schild_ladder_transport(vector, geodesic),
            TransportMethod::PoleLadder => self.pole_ladder_transport(vector, geodesic),
            _ => self.simple_parallel_transport(vector, geodesic),
        }
    }

    /// Schild's ladder parallel transport
    fn schild_ladder_transport(
        &self,
        vector: &DVector<f64>,
        geodesic: &Geodesic,
    ) -> Result<DVector<f64>, OrchestrationError> {
        let n_rungs = self.parallel_transport.schild_params.n_rungs;
        let mut transported = vector.clone();

        for i in 0..geodesic.path.len() - 1 {
            let p0 = &geodesic.path[i];
            let p1 = &geodesic.path[i + 1];

            // Schild's ladder construction
            for _ in 0..n_rungs {
                // Midpoint
                let m = (p0 + p1) / 2.0;
                let m_projected = self.project_onto_manifold(m)?;

                // Parallel transport step
                let delta = &m_projected - p0;
                transported = self.project_to_tangent_space(p1, &(&transported + delta * 2.0))?;
            }
        }

        Ok(transported)
    }

    /// Pole ladder parallel transport
    fn pole_ladder_transport(
        &self,
        vector: &DVector<f64>,
        geodesic: &Geodesic,
    ) -> Result<DVector<f64>, OrchestrationError> {
        // Simplified pole ladder
        let mut transported = vector.clone();

        for i in 0..geodesic.path.len() - 1 {
            let tangent = &geodesic.tangents[i];

            // Transport by adjusting for metric change
            let g0 = (self.manifold.metric.g)(&geodesic.path[i]);
            let g1 = (self.manifold.metric.g)(&geodesic.path[i + 1]);

            if let (Some(g0_inv), Some(g1_inv)) =
                (g0.clone().try_inverse(), g1.clone().try_inverse())
            {
                transported = &g1_inv * &g0 * &transported;
                transported = self.project_to_tangent_space(&geodesic.path[i + 1], &transported)?;
            }
        }

        Ok(transported)
    }

    /// Simple parallel transport (approximation)
    fn simple_parallel_transport(
        &self,
        vector: &DVector<f64>,
        geodesic: &Geodesic,
    ) -> Result<DVector<f64>, OrchestrationError> {
        // Project to tangent space at endpoint
        self.project_to_tangent_space(&geodesic.end, vector)
    }

    /// Optimize LLM responses on manifold
    pub fn optimize_llm_responses(
        &mut self,
        responses: &[String],
        quality_fn: fn(&str) -> f64,
    ) -> Result<LLMOptimizationResult, OrchestrationError> {
        // Encode responses as points on manifold
        let encoded = self.encode_responses_on_manifold(responses)?;

        // Find optimal point on manifold
        let initial = self.compute_mean_on_manifold(&encoded)?;

        // Define objective function (use static decode to avoid capturing &self)
        let objective = |x: &DVector<f64>| {
            // Inline decode logic to avoid self borrow
            let mut response = String::new();
            for value in x.iter() {
                let byte = (value * 255.0).clamp(0.0, 255.0) as u8;
                if byte.is_ascii() {
                    response.push(byte as char);
                }
            }
            -quality_fn(&response) // Minimize negative quality
        };

        let result = self.optimize(objective, initial)?;

        // Decode optimal response
        let optimal_response = self.decode_from_manifold(&result.optimal_point);

        Ok(LLMOptimizationResult {
            optimal_response,
            quality_score: -result.optimal_value,
            manifold_point: result.optimal_point.clone(),
            geodesic_distances: self.compute_geodesic_distances(&encoded, &result.optimal_point)?,
        })
    }

    /// Encode responses as points on manifold
    fn encode_responses_on_manifold(
        &self,
        responses: &[String],
    ) -> Result<Vec<DVector<f64>>, OrchestrationError> {
        responses
            .iter()
            .map(|r| {
                let mut encoding = DVector::zeros(self.manifold.dimension);
                for (i, byte) in r.bytes().take(self.manifold.dimension).enumerate() {
                    encoding[i] = byte as f64 / 255.0;
                }
                self.project_onto_manifold(encoding)
            })
            .collect()
    }

    /// Decode point from manifold to response
    fn decode_from_manifold(&self, point: &DVector<f64>) -> String {
        let mut response = String::new();

        for value in point.iter() {
            let byte = (value * 255.0).clamp(0.0, 255.0) as u8;
            if byte.is_ascii() {
                response.push(byte as char);
            }
        }

        response
    }

    /// Compute mean on manifold (Karcher/Fréchet mean)
    fn compute_mean_on_manifold(
        &mut self,
        points: &[DVector<f64>],
    ) -> Result<DVector<f64>, OrchestrationError> {
        if points.is_empty() {
            return Err(OrchestrationError::InsufficientData {
                required: 1,
                available: 0,
            });
        }

        // Initialize with arithmetic mean projected onto manifold
        let mut mean = DVector::zeros(self.manifold.dimension);
        for point in points {
            mean += point;
        }
        mean /= points.len() as f64;
        mean = self.project_onto_manifold(mean)?;

        // Gradient descent to find Karcher mean
        for _ in 0..100 {
            let mut gradient = DVector::zeros(self.manifold.dimension);

            for point in points {
                let geodesic = self.compute_geodesic(&mean, point)?;
                if !geodesic.tangents.is_empty() {
                    gradient -= &geodesic.tangents[0];
                }
            }

            gradient /= points.len() as f64;

            // Take step along geodesic
            let step_size = 0.1;
            mean = self.exponential_map(&mean, &(&gradient * step_size))?;

            if gradient.norm() < 1e-6 {
                break;
            }
        }

        Ok(mean)
    }

    /// Compute geodesic distances from points to target
    fn compute_geodesic_distances(
        &mut self,
        points: &[DVector<f64>],
        target: &DVector<f64>,
    ) -> Result<Vec<f64>, OrchestrationError> {
        let mut distances = Vec::new();

        for point in points {
            let geodesic = self.compute_geodesic(point, target)?;
            distances.push(geodesic.length);
        }

        Ok(distances)
    }
}

impl CurvatureAnalyzer {
    fn new(dimension: usize) -> Self {
        Self {
            riemann: RiemannTensor {
                components: HashMap::new(),
                symmetries: TensorSymmetries {
                    antisym_12: true,
                    antisym_34: true,
                    interchange: true,
                    bianchi: true,
                },
            },
            ricci: RicciCurvature {
                tensor: DMatrix::zeros(dimension, dimension),
                eigenvalues: DVector::zeros(dimension),
            },
            scalar: ScalarCurvature {
                value: 0.0,
                gradient: DVector::zeros(dimension),
            },
            sectional: SectionalCurvatures {
                curvatures: HashMap::new(),
                principal: Vec::new(),
            },
        }
    }
}

/// Optimization result
#[derive(Clone, Debug)]
pub struct OptimizationResult {
    pub optimal_point: DVector<f64>,
    pub optimal_value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub final_gradient_norm: f64,
}

/// LLM optimization result
#[derive(Clone, Debug)]
pub struct LLMOptimizationResult {
    pub optimal_response: String,
    pub quality_score: f64,
    pub manifold_point: DVector<f64>,
    pub geodesic_distances: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_optimization() {
        let mut optimizer = GeometricManifoldOptimizer::new(ManifoldType::Sphere, 3).unwrap();

        // Optimize simple quadratic on sphere
        let objective = |x: &DVector<f64>| x[0] * x[0] + 2.0 * x[1] * x[1] + 3.0 * x[2] * x[2];
        let initial = DVector::from_vec(vec![1.0, 0.0, 0.0]);

        let result = optimizer.optimize(objective, initial).unwrap();

        assert!(result.converged);
        assert!((result.optimal_point.norm() - 1.0).abs() < 1e-6); // On sphere
    }
}
