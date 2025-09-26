"""
Bayesian Informed Optimization in LHCE with BayesGap Algorithm and XML Export.

Implements a closed-loop Bayesian optimization algorithm using gap-based arm selection.
Supports kernel approximation, confidence bounds, batch policy selection,
and automatic XML test file generation via external Tools.XML module.
"""

import argparse
import csv
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import Tools.XML


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for BayesGap optimization.

    Returns:
        Parsed arguments as an argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Bayesian Informed Optimization in LHCE with BayesGap Algorithm and XML Export."
    )

    parser.add_argument(
        "--policy_file",
        type=str,
        default="policies_all.csv",
        help="CSV file containing all candidate policies for optimization.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="LHCE/data",
        help="Base directory for storing logs and intermediate data.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="log.csv",
        help="Filename to log best arms across rounds.",
    )
    parser.add_argument(
        "--arm_bounds_dir",
        type=str,
        default="bounds/",
        help="Subdirectory to save posterior bounds (pickle files).",
    )
    parser.add_argument(
        "--early_pred_dir",
        type=str,
        default="pred/",
        help="Directory containing early predictions from previous round.",
    )
    parser.add_argument(
        "--next_batch_dir",
        type=str,
        default="batch/",
        help="Directory to save selected policies for next round.",
    )
    parser.add_argument(
        "--round_idx",
        type=int,
        default=1,
        help="Current round index (1-indexed in usage, but logic is 0/1-based).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2,
        help="Total number of optimization rounds.",
    )
    parser.add_argument(
        "--bsize",
        type=int,
        default=50,
        help="Number of policies to select per round (batch size).",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Bandwidth parameter for RBF kernel in Nystroem approximation.",
    )
    parser.add_argument(
        "--init_beta",
        type=float,
        default=5.0 * 0.5**2,
        help="Initial exploration coefficient (e.g., scaled by decay factor).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.5,
        help="Decay factor for exploration coefficient beta.",
    )
    parser.add_argument(
        "--likelihood_std",
        type=float,
        default=0.89,
        help="Standard deviation of observation noise in likelihood model.",
    )
    parser.add_argument(
        "--standardization_mean",
        type=float,
        default=54.0,
        help="Mean value used to standardize observed battery lifetime.",
    )
    parser.add_argument(
        "--standardization_std",
        type=float,
        default=1.12,
        help="Std dev used to standardize observed battery lifetime.",
    )

    return parser.parse_args()


class BayesGap:
    """Bayesian optimization agent implementing the BayesGap algorithm.

    This class selects batches of charging protocols based on upper confidence bounds
    and gap statistics. It maintains a posterior Gaussian process model using kernel
    approximation and updates beliefs iteratively from observed results.

    Attributes:
        policy_file (str): Path to CSV with all candidate policies.
        prev_arm_bounds_file (str): Checkpoint file from previous round (bounds).
        prev_early_pred_file (str): Observed results from last round (CSV).
        arm_bounds_file (str): Output path for current state (pkl).
        next_batch_file (str): Output path for selected policies (CSV).
        param_space (np.ndarray): Policy parameters (first 6 columns).
        num_arms (int): Total number of candidate policies.
        X (np.ndarray): Kernel-transformed design matrix (via Nystroem).
        num_dims (int): Dimensionality after transformation.
        batch_size (int): Number of arms to select per round.
        budget (int): Maximum number of rounds.
        round_idx (int): Current round index.
        sigma (float): Observation noise std.
        beta (float): Exploration constant (decays over time).
        epsilon (float): Decay rate for beta.
        standardization_mean (float): Mean for reward standardization.
        standardization_std (float): Std for reward standardization.
        eta (float): Prior scale (same as standardization_std).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize BayesGap agent.

        Args:
            args: Parsed command-line arguments.
        """
        self.policy_file = args.policy_file
        self.prev_arm_bounds_file = os.path.join(
            args.data_dir, args.arm_bounds_dir, f"{args.round_idx - 1}.pkl"
        )
        self.prev_early_pred_file = os.path.join(
            args.data_dir, args.early_pred_dir, f"{args.round_idx}.csv"
        )
        self.arm_bounds_file = os.path.join(
            args.data_dir, args.arm_bounds_dir, f"{args.round_idx}.pkl"
        )
        self.next_batch_file = os.path.join(
            args.data_dir, args.next_batch_dir, f"{args.round_idx + 1}.csv"
        )

        self.param_space = self.get_parameter_space()
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

    def get_parameter_space(self) -> np.ndarray:
        """Load policy parameter space from CSV.

        Expects header row; uses first 6 columns as features.

        Returns:
            Array of shape (n_policies, 6).
        """
        policies = np.genfromtxt(self.policy_file, delimiter=",", skip_header=1)
        return policies[:, :6]

    def get_design_matrix(self, gamma: float) -> np.ndarray:
        """Generate kernel-based feature representation using Nystroem approximation.

        Args:
            gamma: RBF kernel bandwidth.

        Returns:
            Transformed design matrix of shape (num_arms, num_arms).
        """
        from sklearn.kernel_approximation import Nystroem

        transformer = Nystroem(gamma=gamma, n_components=self.num_arms, random_state=1)
        return transformer.fit_transform(self.param_space)

    def run(self) -> Optional[np.ndarray]:
        """Run one round of BayesGap optimization.

        Loads history, updates belief, selects next batch, saves state, generates XMLs.

        Returns:
            Best policy parameters found so far, or None if initializing.
        """
        if self.round_idx == 1:
            X_t: List[np.ndarray] = []
            Y_t: List[np.ndarray] = []
            proposal_arms: List[int] = []
            proposal_gaps: List[float] = []
            beta = self.beta
        else:
            # Load previous state
            try:
                with open(self.prev_arm_bounds_file, "rb") as f:
                    proposal_arms, proposal_gaps, X_t, Y_t, beta = pickle.load(f)
                beta = round(beta * self.epsilon, 4)
            except FileNotFoundError:
                print(f"Warning: Previous bounds file not found: {self.prev_arm_bounds_file}")
                X_t, Y_t, proposal_arms, proposal_gaps = [], [], [], []
                beta = self.beta

        # Load observed results from last round
        try:
            with open(self.prev_early_pred_file, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                early_pred = np.array([[float(x) for x in row] for row in reader])
        except FileNotFoundError:
            print(f"Warning: No early prediction file found: {self.prev_early_pred_file}")
            return None

        # Update design matrix X_t with executed policies
        batch_policies = early_pred[:, :6]
        for policy in batch_policies:
            try:
                idx = self.param_space.tolist().index(policy.tolist())
                X_t.append(self.X[idx : idx + 1])
            except ValueError:
                print(f"Warning: Policy {policy} not found in parameter space.")

        # Standardize observed rewards
        rewards = early_pred[:, 6]
        mean_val = np.nanmean(rewards)
        rewards = np.where(np.isnan(rewards), mean_val, rewards)
        standardized_rewards = (rewards - self.standardization_mean).reshape(-1, 1)
        Y_t.append(standardized_rewards)

        # Stack data and compute posterior
        np_X_t = np.vstack(X_t) if X_t else np.empty((0, self.num_dims))
        np_Y_t = np.vstack(Y_t) if Y_t else np.empty((0, 1))

        upper_bounds, lower_bounds = self.get_posterior_bounds(beta, np_X_t, np_Y_t)
        proposal_gaps, best_arm_params = self.get_proposal_solution(
            proposal_gaps, proposal_arms, upper_bounds, lower_bounds
        )

        # Denormalize bounds
        nonstd_upper = upper_bounds + self.standardization_mean
        nonstd_lower = lower_bounds + self.standardization_mean

        print(f"Round {self.round_idx}")
        print(f"Current beta: {beta}")

        # Select next batch
        proposal_arms, batch_arms = self.get_proposal_arms(
            proposal_arms, upper_bounds, lower_bounds
        )
        batch_policies = self.get_batch_policies(batch_arms)

        # Save state
        with open(self.arm_bounds_file, "wb") as f:
            pickle.dump([proposal_arms, proposal_gaps, X_t, Y_t, beta], f)

        with open(f"{self.arm_bounds_file[:-4]}_bounds.pkl", "wb") as f:
            pickle.dump(
                [self.param_space, nonstd_upper, nonstd_lower, (nonstd_upper + nonstd_lower) / 2],
                f,
            )

        # Generate XML test files
        for idx, arm_idx in enumerate(batch_arms):
            Tools.XML.XMLfile(
                arm=arm_idx,
                round_idx=6,
                stop_volt1=4.3,
                stop_volt2=2.8
            ).to_xml(index=idx, LOOP=100)

        # Save selected policies
        with open(self.next_batch_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(batch_policies)

        return best_arm_params

    def get_batch_policies(self, batch_arms: List[int]) -> List[np.ndarray]:
        """Retrieve parameter vectors for selected arms.

        Args:
            batch_arms: List of arm indices.

        Returns:
            List of corresponding policy parameter arrays.
        """
        return [self.param_space[arm] for arm in batch_arms]

    def get_proposal_arms(
        self,
        proposal_arms: List[int],
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """Select next batch using BayesGap strategy.

        Balances exploration (J_t) and exploitation (j_t).

        Args:
            proposal_arms: History of top arms.
            upper_bounds: Current UCB values.
            lower_bounds: Current LCB values.

        Returns:
            Updated proposal_arms and list of selected batch_arms.
        """
        batch_arms: List[int] = []
        candidate_arms = list(range(self.num_arms))

        def find_J_t(carms: List[int]) -> int:
            """Find J_t: maximizes gap between second-best UCB and own LCB."""
            B_k_ts = []
            for k in range(self.num_arms):
                if k in carms:
                    temp_ub = np.delete(upper_bounds, k)
                    max_other = np.max(temp_ub) if len(temp_ub) > 0 else -np.inf
                    B_k_t = max_other - lower_bounds[k]
                    B_k_ts.append(B_k_t)
                else:
                    B_k_ts.append(np.inf)
            return int(np.argmin(B_k_ts))

        def find_j_t(carms: List[int], preselected: int) -> int:
            """Find j_t: highest UCB excluding preselected arm."""
            U_k_ts = [
                upper_bounds[k] if (k in carms and k != preselected) else -np.inf
                for k in range(self.num_arms)
            ]
            return int(np.argmax(U_k_ts))

        def get_diameter(k: int) -> float:
            """Get confidence interval width for arm k."""
            return float(upper_bounds[k] - lower_bounds[k])

        for _ in range(self.batch_size):
            J_t = find_J_t(candidate_arms)
            j_t = find_j_t(candidate_arms, J_t)
            s_J = get_diameter(J_t)
            s_j = get_diameter(j_t)

            chosen = J_t if s_J >= s_j else j_t

            if not batch_arms:
                proposal_arms.append(J_t)
            batch_arms.append(chosen)
            candidate_arms.remove(chosen)

        print(f"Policy indices selected for this round: {batch_arms}")
        return proposal_arms, batch_arms

    def get_proposal_solution(
        self,
        proposal_gaps: List[float],
        proposal_arms: List[int],
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
    ) -> Tuple[List[float], np.ndarray]:
        """Update solution history using gap statistic.

        Args:
            proposal_gaps: Historical gap values.
            proposal_arms: Previously proposed best arms.
            upper_bounds: Current upper bounds.
            lower_bounds: Current lower bounds.

        Returns:
            Updated gaps and best policy parameters.
        """
        if not proposal_arms:
            best_arm = int(np.argmax(upper_bounds))
            gap = upper_bounds[best_arm] - lower_bounds[best_arm]
            proposal_gaps.append(gap)
        else:
            # Use round_idx - 2 because proposal_arms stores past best arms
            J_prev = proposal_arms[self.round_idx - 2]
            temp_ub = np.delete(upper_bounds, J_prev)
            max_other = np.max(temp_ub) if len(temp_ub) > 0 else 0.0
            gap = max_other - lower_bounds[J_prev]
            proposal_gaps.append(gap)
            best_arm = int(np.argmax(upper_bounds))  # Always pick current best

        best_params = self.param_space[best_arm]
        return proposal_gaps, best_params

    def get_posterior_bounds(
        self,
        beta: float,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior upper and lower confidence bounds.

        Args:
            beta: Exploration multiplier.
            X: Design matrix (T x D), or None.
            Y: Observations (T x 1), or None.

        Returns:
            Upper and lower bounds (each shape: (num_arms,)).
        """
        post_mean, post_cov = self.posterior_theta(X, Y)
        marginal_mean, marginal_var = self.marginal_mu((post_mean, post_cov))
        std = np.sqrt(marginal_var)
        upper = np.around(marginal_mean + beta * std, 4).flatten()
        lower = np.around(marginal_mean - beta * std, 4).flatten()
        return upper, lower

    def posterior_theta(
        self,
        X_t: Optional[np.ndarray],
        Y_t: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute posterior distribution over weights theta.

        Args:
            X_t: Observed features (T x D), or None.
            Y_t: Observed outputs (T x 1), or None.

        Returns:
            Posterior mean (D,) and covariance (D, D).
        """
        D = self.num_dims
        sigma = self.sigma
        eta = self.eta
        prior_mean = np.zeros(D)
        prior_cov = (eta * eta) * np.eye(D)

        if X_t is None or len(X_t) == 0:
            return prior_mean, prior_cov

        precision_data = X_t.T @ X_t / (sigma * sigma)
        precision_prior = np.eye(D) / (eta * eta)
        posterior_cov = np.linalg.inv(precision_data + precision_prior)
        posterior_mean = posterior_cov @ X_t.T @ Y_t / (sigma * sigma)
        return np.squeeze(posterior_mean), posterior_cov

    def marginal_mu(
        self,
        posterior_theta_params: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute predictive mean and variance for all arms.

        Args:
            posterior_theta_params: Tuple of posterior mean and covariance.

        Returns:
            Predictive mean (N,) and variance (N,).
        """
        X = self.X
        post_mean, post_cov = posterior_theta_params
        pred_mean = X @ post_mean
        pred_var = np.sum((X @ post_cov) * X, axis=1)
        return pred_mean, pred_var


def main() -> None:
    """Main entry point for BayesGap optimization loop."""
    args = parse_args()
    np.random.seed(args.seed)
    np.set_printoptions(threshold=np.inf)

    # Create required directories
    dirs = [
        os.path.join(args.data_dir, d)
        for d in [args.arm_bounds_dir, args.next_batch_dir, args.early_pred_dir]
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    agent = BayesGap(args)
    best_arm_params = agent.run()

    if args.round_idx > 0:
        print(f"Best arm until round {args.round_idx} is {best_arm_params}")

    # Log result
    log_path = os.path.join(args.data_dir, args.log_file)
    with open(log_path, "a", newline="") as f:
        print(f"Logged data for round {args.round_idx} in {args.log_file}")
        if args.round_idx == 0:
            f.write(f"{args.init_beta},{args.gamma},{args.epsilon},{args.seed}\n")
        else:
            f.write(f"{best_arm_params}\n")


if __name__ == "__main__":
    main()