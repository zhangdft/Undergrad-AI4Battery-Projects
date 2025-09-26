"""
Closed-Loop Bayesian Informed Optimization with BayesGap Algorithm.

Implements a Bayesian optimization loop using confidence bounds and gap-based arm selection.
Supports early prediction from simulation data and iterative policy refinement.
"""

import argparse
import csv
import os
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import pickle


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the BayesGap optimization.

    Returns:
        Parsed arguments as an argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Closed-Loop Optimization with early prediction and Bayes Gap."
    )

    parser.add_argument(
        "--policy_file",
        nargs="?",
        default="policies_all.csv",
        help="CSV file containing all policies for optimization.",
    )
    parser.add_argument(
        "--sim_policy_file",
        nargs="?",
        default="policies_sim.csv",
        help="Simulated policy performance file.",
    )
    parser.add_argument(
        "--data_folder",
        nargs="?",
        default="hyperdata_for_BO/",
        help="Folder to store hyperparameter logs.",
    )
    parser.add_argument(
        "--data_dir",
        nargs="?",
        default="hyperdata/",
        help="Base directory for data storage.",
    )
    parser.add_argument(
        "--arm_bounds_dir",
        nargs="?",
        default="bounds/",
        help="Directory to save arm bounds (confidence intervals).",
    )
    parser.add_argument(
        "--early_pred_dir",
        nargs="?",
        default="pred/",
        help="Directory to save early predictions per round.",
    )
    parser.add_argument(
        "--round_idx", type=int, default=3, help="Current optimization round index."
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--budget", type=int, default=4, help="Total number of optimization rounds."
    )
    parser.add_argument(
        "--bsize", type=int, default=50, help="Batch size (number of arms to select per round)."
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Kernel bandwidth for RBF/Nystroem kernel approximation.",
    )
    parser.add_argument(
        "--init_beta",
        type=float,
        default=5.0,
        help="Initial exploration constant (beta) in posterior bound calculation.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.5,
        help="Decay factor for exploration constant beta across rounds.",
    )

    parser.add_argument(
        "--likelihood_std",
        type=float,
        default=1.0,
        help="Standard deviation of likelihood noise in Gaussian process model.",
    )
    parser.add_argument(
        "--standardization_mean",
        type=float,
        default=28.0,
        help="Mean value used for standardizing observed rewards.",
    )
    parser.add_argument(
        "--standardization_std",
        type=float,
        default=1.0,
        help="Standard deviation used for standardizing observed rewards.",
    )

    return parser.parse_args()


class Bayes:
    """Bayesian optimization agent implementing BayesGap algorithm with early prediction.

    This class performs closed-loop optimization by selecting batches of policies based on
    upper confidence bounds and gap statistics. It maintains posterior estimates over policy
    performance using kernel approximation and updates beliefs iteratively.

    Attributes:
        policy_file (str): Path to CSV file with all candidate policies.
        policy_file_sim (str): Path to simulated policy evaluation results.
        prev_arm_bounds_file (str): Checkpoint file from previous round (bounds).
        prev_early_pred_file (str): Early prediction file from previous round.
        arm_bounds_file (str): Output path for current round's arm bounds.
        early_pred_file (str): Output path for selected policies in this round.
        param_space (np.ndarray): Design matrix of policy parameters (first 6 columns).
        num_arms (int): Number of candidate arms (policies).
        X (np.ndarray): Kernel-transformed design matrix (Nystroem features).
        num_dims (int): Dimensionality of feature space.
        batch_size (int): Number of arms to select per round.
        budget (int): Total allowed optimization rounds.
        round_idx (int): Current round index (0-indexed).
        sigma (float): Likelihood noise standard deviation.
        beta (float): Exploration coefficient (decays over time).
        epsilon (float): Decay rate for beta.
        standardization_mean (float): Mean for reward standardization.
        standardization_std (float): Std for reward standardization.
        eta (float): Prior scale parameter (equal to standardization_std).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize Bayes agent with given arguments.

        Args:
            args: Command-line arguments parsed by argparse.
        """
        self.policy_file = args.policy_file
        self.policy_file_sim = args.sim_policy_file
        self.prev_arm_bounds_file = os.path.join(
            args.data_dir, args.arm_bounds_dir, f"{args.round_idx - 1}.pkl"
        )
        self.prev_early_pred_file = os.path.join(
            args.data_dir, args.early_pred_dir, f"{args.round_idx - 1}.csv"
        )
        self.arm_bounds_file = os.path.join(
            args.data_dir, args.arm_bounds_dir, f"{args.round_idx}.pkl"
        )
        self.early_pred_file = os.path.join(
            args.data_dir, args.early_pred_dir, f"{args.round_idx}.csv"
        )

        self.param_space = self.get_parameter_space(self.policy_file)
        self.num_arms = self.param_space.shape[0]
        self.X = self.get_design_matrix(args.gamma)
        self.num_dims = self.X.shape[1]

        self.batch_size = args.bsize
        self.budget = args.budget
        self.round_idx = args.round_idx

        self.sigma = args.likelihood_std
        self.beta = args.init_beta
        self.epsilon = args.epsilon

        self.standardization_mean = args.standardization_mean
        self.standardization_std = args.standardization_std
        self.eta = self.standardization_std

        # Ensure output directories exist
        os.makedirs(os.path.dirname(self.arm_bounds_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.early_pred_file), exist_ok=True)

    def get_parameter_space(self, policy_file: str) -> np.ndarray:
        """Load parameter space from policy CSV file.

        Expects first row to be header; uses first 6 columns as parameters.

        Args:
            policy_file: Path to CSV file with policies.

        Returns:
            Array of shape (n_policies, 6) containing policy parameters.
        """
        policies = np.genfromtxt(policy_file, delimiter=",", skip_header=1)
        return policies[:, :6]

    def get_design_matrix(self, gamma: float) -> np.ndarray:
        """Generate kernel-based feature representation using Nystroem approximation.

        Transforms raw policy parameters into a higher-dimensional space via RBF kernel.

        Args:
            gamma: Kernel bandwidth for RBF kernel.

        Returns:
            Transformed design matrix of shape (num_arms, num_arms).
        """
        from sklearn.kernel_approximation import Nystroem

        feature_map = Nystroem(gamma=gamma, n_components=self.num_arms, random_state=1)
        return feature_map.fit_transform(self.param_space)

    def run(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Execute one round of Bayesian optimization.

        Selects a batch of policies using BayesGap logic, saves state, and returns best
        known policy so far.

        Returns:
            A tuple of:
                - Best policy parameters found so far (or None if round 0).
                - Simulated reward of best policy (or None if round 0).
        """
        sim = np.genfromtxt(self.policy_file_sim, delimiter=",", skip_header=0)
        sim_params = sim[:, :6]

        num_arms = self.num_arms
        batch_size = self.batch_size
        budget = self.budget
        epsilon = self.epsilon
        mean_offset = self.standardization_mean
        X = self.X
        round_idx = self.round_idx
        param_space = self.param_space

        def find_J_t(candidate_arms: List[int]) -> Tuple[int, float]:
            """Find index J_t maximizing gap between second-best upper bound and own lower bound.

            Args:
                candidate_arms: List of eligible arm indices.

            Returns:
                Index of selected arm and its gap value.
            """
            B_k_ts = []
            for k in range(num_arms):
                if k in candidate_arms:
                    temp_upper = np.delete(upper_bounds, k)
                    B_k_t = np.max(temp_upper) if temp_upper.size > 0 else -np.inf
                    B_k_ts.append(B_k_t - lower_bounds[k])
                else:
                    B_k_ts.append(np.inf)
            B_k_ts = np.array(B_k_ts)
            J_t = int(np.argmin(B_k_ts))
            min_B_k_t = float(np.min(B_k_ts))
            return J_t, min_B_k_t

        def find_j_t(candidate_arms: List[int], preselected_arm: int) -> int:
            """Find j_t as the candidate with highest upper bound excluding preselected arm.

            Args:
                candidate_arms: Eligible arm indices.
                preselected_arm: Arm already chosen (J_t), to be excluded.

            Returns:
                Index of arm with highest upper bound among others.
            """
            U_k_ts = [
                upper_bounds[k] if (k in candidate_arms and k != preselected_arm) else -np.inf
                for k in range(num_arms)
            ]
            return int(np.argmax(U_k_ts))

        def get_confidence_diameter(k: int) -> float:
            """Get diameter of confidence interval for arm k.

            Args:
                k: Arm index.

            Returns:
                Difference between upper and lower bounds.
            """
            return float(upper_bounds[k] - lower_bounds[k])

        # Initialization for round 0
        if round_idx == 0:
            X_t: List[np.ndarray] = []
            Y_t: List[np.ndarray] = []
            proposal_arms: List[int] = []
            proposal_gaps: List[float] = []
            beta = self.beta
            upper_bounds, lower_bounds = self.get_posterior_bounds(beta)
            best_arm_params = None
            best_arm_sim = None

        else:
            # Load history from previous round
            with open(self.prev_arm_bounds_file, "rb") as f:
                proposal_arms, proposal_gaps, X_t, Y_t, beta = pickle.load(f)

            # Decay exploration constant
            beta = round(beta * epsilon, 4)

            # Load early predictions from last round
            df = pd.read_csv(self.prev_early_pred_file)
            early_pred = df.to_numpy(dtype=float)
            batch_policies = early_pred[:, :6]
            batch_rewards = early_pred[:, 6].reshape(-1, 1)

            # Standardize rewards
            standardized_rewards = (batch_rewards + mean_offset) / 10 + 6 - mean_offset

            # Update design and observation matrices
            for policy, reward in zip(batch_policies, standardized_rewards):
                try:
                    arm_idx = param_space.tolist().index(policy.tolist())
                    X_t.append(X[arm_idx : arm_idx + 1])
                    Y_t.append(reward.reshape(1, 1))
                except ValueError as e:
                    print(f"Warning: Policy {policy} not found in param_space. Skipping.")
                    continue

            # Compute updated posterior bounds
            np_X_t = np.vstack(X_t) if X_t else np.empty((0, X.shape[1]))
            np_Y_t = np.vstack(Y_t) if Y_t else np.empty((0, 1))
            upper_bounds, lower_bounds = self.get_posterior_bounds(beta, np_X_t, np_Y_t)

            # Update best arm gap using J_prev_round
            J_prev_round = proposal_arms[round_idx - 1]
            temp_upper = np.delete(upper_bounds, J_prev_round)
            B_k_t = np.max(temp_upper) - lower_bounds[J_prev_round] if temp_upper.size > 0 else 0.0
            proposal_gaps.append(B_k_t)

            best_arm_idx = int(np.argmin(proposal_gaps))
            best_arm = proposal_arms[best_arm_idx]
            best_arm_params = param_space[best_arm]
            try:
                b_arm_sim_idx = sim_params.tolist().index(best_arm_params.tolist())
                best_arm_sim = float(sim[b_arm_sim_idx, 6] / 10 + 6)
            except ValueError:
                best_arm_sim = None

        # Transform bounds back to original scale
        nonstd_upper_bounds = upper_bounds + mean_offset
        nonstd_lower_bounds = lower_bounds + mean_offset

        # Select batch of arms using BayesGap strategy
        batch_arms: List[int] = []
        candidate_arms = list(range(num_arms))
        for _ in range(batch_size):
            J_t, _ = find_J_t(candidate_arms)
            j_t = find_j_t(candidate_arms, J_t)
            s_J_t = get_confidence_diameter(J_t)
            s_j_t = get_confidence_diameter(j_t)
            a_t = J_t if s_J_t >= s_j_t else j_t

            if len(batch_arms) == 0:
                proposal_arms.append(J_t)
            batch_arms.append(a_t)
            candidate_arms.remove(a_t)

        # Save state for current round
        with open(self.arm_bounds_file, "wb") as f:
            pickle.dump([proposal_arms, proposal_gaps, X_t, Y_t, beta], f)

        with open(self.arm_bounds_file[:-4] + "_bounds.pkl", "wb") as f:
            pickle.dump(
                [
                    self.param_space,
                    nonstd_upper_bounds,
                    nonstd_lower_bounds,
                    (nonstd_upper_bounds + nonstd_lower_bounds) / 2,
                ],
                f,
            )

        # Save selected policies for next simulation (unless final round)
        if round_idx < budget:
            selected_rows = [sim[arm] for arm in batch_arms]
            with open(self.early_pred_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(selected_rows)

        return best_arm_params, best_arm_sim

    def get_posterior_bounds(
        self,
        beta: float,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute upper and lower posterior confidence bounds for all arms.

        Uses Bayesian linear regression in kernel space.

        Args:
            beta: Exploration multiplier.
            X: Design matrix of observed arms (shape: [T, D]).
            Y: Observed rewards (shape: [T, 1]).

        Returns:
            Tuple of upper and lower confidence bounds (each shape: [num_arms,]).
        """
        posterior_theta_params = self.posterior_theta(X, Y)
        marginal_mu_params = self.marginal_mu(posterior_theta_params)
        marginal_mean, marginal_var = marginal_mu_params

        std_dev = np.sqrt(marginal_var)
        upper_bounds = np.around(marginal_mean + beta * std_dev, 4)
        lower_bounds = np.around(marginal_mean - beta * std_dev, 4)

        return upper_bounds.flatten(), lower_bounds.flatten()

    def posterior_theta(
        self,
        X_t: Optional[np.ndarray],
        Y_t: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior distribution over weights theta.

        Assumes a Gaussian prior and likelihood.

        Args:
            X_t: Observed design matrix (T x D), or None.
            Y_t: Observed outputs (T x 1), or None.

        Returns:
            Tuple of posterior mean (D,) and covariance (D, D).
        """
        num_dims = self.num_dims
        sigma = self.sigma
        eta = self.eta
        prior_mean = np.zeros(num_dims)
        prior_cov = (eta * eta) * np.eye(num_dims)

        if X_t is None or len(X_t) == 0:
            return prior_mean, prior_cov

        precision_prior = np.eye(num_dims) / (eta * eta)
        precision_data = np.dot(X_t.T, X_t) / (sigma * sigma)
        posterior_covar = np.linalg.inv(precision_data + precision_prior)

        posterior_mean = np.linalg.multi_dot([posterior_covar, X_t.T, Y_t]) / (sigma * sigma)
        posterior_mean = np.squeeze(posterior_mean)

        return posterior_mean, posterior_covar

    def marginal_mu(
        self, posterior_theta_params: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute marginal predictive mean and variance for all arms.

        Args:
            posterior_theta_params: Tuple of posterior mean and covariance of theta.

        Returns:
            Tuple of marginal means (num_arms,) and variances (num_arms,).
        """
        X = self.X
        post_mean, post_cov = posterior_theta_params

        marginal_mean = X.dot(post_mean)
        # Efficient diagonal computation: diag(X @ cov @ X^T)
        marginal_var = np.sum((X @ post_cov) * X, axis=1)

        return marginal_mean, marginal_var


def main() -> None:
    """Main entry point for BayesGap optimization loop."""
    args = parse_args()
    np.random.seed(args.seed)

    for _ in range(args.budget):
        agent = Bayes(args)
        best_arm, best_arm_sim = agent.run()
        if args.round_idx > 0:
            print(
                f"Best arm until round {args.round_idx - 1} is {best_arm}, "
                f"reward is {best_arm_sim}"
            )
        args.round_idx += 1


if __name__ == "__main__":
    main()