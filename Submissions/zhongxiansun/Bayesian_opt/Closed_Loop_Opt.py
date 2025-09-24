import numpy as np
import argparse
import pickle
import os
import csv
import Tools.XML

def parse_args():

    parser = argparse.ArgumentParser(description='Closed-Loop Optimization with early prediction and Bayes Gap.')

    parser.add_argument('--policy_file', nargs='?', default='policies_all.csv', help='All policies for optimazition')
    parser.add_argument('--data_dir', nargs='?', default='data/')
    parser.add_argument('--log_file', nargs='?', default='log.csv')
    parser.add_argument('--arm_bounds_dir', nargs='?', default='bounds/')
    parser.add_argument('--early_pred_dir', nargs='?', default='pred/') 
    parser.add_argument('--next_batch_dir', nargs='?', default='batch/')
    parser.add_argument('--file_dir', nargs='?', default='Test_file/')
    parser.add_argument('--round_idx', default=4, type=int)

    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--budget', default=4, type=int, help='Time budget')
    parser.add_argument('--bsize', default=50, type=int, help='batch size')

    parser.add_argument('--gamma', default=1.05, type=float,help='kernel bandwidth for Gaussian kernel')
    parser.add_argument('--init_beta', default=5.0, type=float, help='initial exploration constant in Thm 1')
    parser.add_argument('--epsilon', default=0.5, type=float, help='decay constant for exploration')

    parser.add_argument('--likelihood_std', default=0.89, type=float, help='standard deviation for the likelihood std')
    parser.add_argument('--standardization_mean', default=29.00, type=float, help='mean lifetime from batch')
    parser.add_argument('--standardization_std', default=0.89, type=float, help='std lifetime from batch')

    return parser.parse_args()

class BayesGap(object):

    def __init__(self, args):
        self.policy_file = os.path.join(args.policy_file)
        self.prev_arm_bounds_file = os.path.join(args.data_dir, args.arm_bounds_dir, str(args.round_idx-1) + '.pkl') # note this is for previous round
        self.prev_early_pred_file = os.path.join(args.data_dir, args.early_pred_dir, str(args.round_idx-1) + '.csv') # note this is for previous round
        self.arm_bounds_file = os.path.join(args.data_dir, args.arm_bounds_dir, str(args.round_idx) + '.pkl')
        self.next_batch_file = os.path.join(args.data_dir, args.next_batch_dir, str(args.round_idx) + '.csv')
        
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

    def get_design_matrix(self, gamma):
        from sklearn.kernel_approximation import (RBFSampler, Nystroem)
        param_space = self.param_space

        feature_map_nystroem = Nystroem(gamma=gamma, n_components=self.num_arms, random_state=1)
        X = feature_map_nystroem.fit_transform(param_space)
        return X

    def run(self):

        if self.round_idx == 0:
            X_t = []
            Y_t = []
            proposal_arms = [] 
            proposal_gaps = []
            beta = self.beta
            upper_bounds, lower_bounds = self.get_posterior_bounds(beta)
            best_arm_params = None
        
        else:
            with open(self.prev_arm_bounds_file, 'rb') as infile: 
                proposal_arms, proposal_gaps, X_t, Y_t, beta = pickle.load(infile)

            beta = np.around(beta * self.epsilon, 4)

            # 获取上一轮次的 PC 协议和对应的结果
            with open(self.prev_early_pred_file, 'r', encoding='utf-8-sig') as infile:
                reader = csv.reader(infile, delimiter=',')
                early_pred = np.asarray([list(map(float, row)) for row in reader])

            # 获取已经执行过的协议的高斯核
            batch_policies = early_pred[:, :6]
            for policy in batch_policies:
                batch_arms = [self.param_space.tolist().index(policy.tolist())]
                X_t.append(self.X[batch_arms])
            np_X_t = np.vstack(X_t)
            
            # 获取已经执行过的协议的标准结果
            data = early_pred[:, -1]
            mean_value = np.nanmean(data)
            data[np.isnan(data)] = mean_value
            early_pred[:, -1] = data
            early_pred[:, -1] = early_pred[:, -1] - self.standardization_mean
            
            batch_rewards = early_pred[:, 6].reshape(-1, 1)
            Y_t.append(batch_rewards)
            np_Y_t = np.vstack(Y_t)

            upper_bounds, lower_bounds = self.get_posterior_bounds(beta, np_X_t, np_Y_t)

            proposal_gaps, best_arm_params = self.get_proposal_solution(proposal_gaps, proposal_arms, upper_bounds, lower_bounds)

        nonstd_upper_bounds = upper_bounds + self.standardization_mean
        nonstd_lower_bounds = lower_bounds + self.standardization_mean

        print('Round', self.round_idx)
        print('Current beta', beta)

        proposal_arms, batch_arms = self.get_proposal_arms(proposal_arms, upper_bounds, lower_bounds)

        batch_policies = self.get_batch_policies(batch_arms)

        # save proposal_arms, proposal_gaps, X_t, Y_t, beta for current round in bounds/<round_idx>.pkl
        with open(self.arm_bounds_file, 'wb') as outfile:
            pickle.dump([proposal_arms, proposal_gaps, X_t, Y_t, beta], outfile)
        
        # save results of bounds in optimazition for current round in bounds/<round_idx>_bounds.pkl
        with open(self.arm_bounds_file[:-4]+'_bounds.pkl', 'wb') as outfile:
            pickle.dump([self.param_space, nonstd_upper_bounds, nonstd_lower_bounds, (nonstd_upper_bounds + nonstd_lower_bounds) / 2], outfile)

        # save policies as [.xml] test files in File4test\budget<round_idx>
        for index,arm in enumerate(batch_arms):
            Tools.XML.XMLfile(arm, self.round_idx).to_xml(index)

        # save policies corresponding to batch_arms in batch/<round_idx>.csv
        with open(self.next_batch_file, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(batch_policies)
        
        return best_arm_params
    
    def get_batch_policies(self, batch_arms):

        return [self.param_space[arm] for arm in batch_arms]

    def get_proposal_arms(self, proposal_arms, upper_bounds, lower_bounds):
        batch_arms = []
        candidate_arms = list(range(self.num_arms))
        def find_J_t(carms):
            """
            找到在当前候选策略中，最适合探索的策略 J_t。
            """
            B_k_ts = [np.amax(np.delete(upper_bounds, k)) if k in carms else np.inf for k in range(self.num_arms)]
            B_k_ts = np.array(B_k_ts) - np.array(lower_bounds)
            J_t = np.argmin(B_k_ts)
            min_B_k_t = np.amin(B_k_ts)
            return J_t, min_B_k_t
        def find_j_t(carms, preselected_arm):
            """
            找到在当前候选策略中，最适合利用的策略 j_t。
            """
            U_k_ts = [upper_bounds[k] if k in carms and k != preselected_arm else -np.inf for k in range(self.num_arms)]
            j_t = np.argmax(np.array(U_k_ts))
            return j_t
        def get_confidence_diameter(k):
            """
            获得第 k 个策略的置信区间
            """
            return upper_bounds[k] - lower_bounds[k]
        for batch_elem in range(self.batch_size):
            J_t, _ = find_J_t(candidate_arms)
            j_t = find_j_t(candidate_arms, J_t)
            s_J_t = get_confidence_diameter(J_t)
            s_j_t = get_confidence_diameter(j_t)
            a_t = J_t if s_J_t >= s_j_t else j_t
            if batch_elem == 0:
                proposal_arms.append(J_t)
            batch_arms.append(a_t)
            candidate_arms.remove(a_t)
        
        print('Policy indices selected for this round:', batch_arms)

        return proposal_arms, batch_arms

    def get_proposal_solution(self, proposal_gaps, proposal_arms, upper_bounds, lower_bounds):
        """
        推荐上一轮的最优和次优解
        """
        J_prev_round = proposal_arms[self.round_idx-1]
        temp_upper_bounds = np.delete(upper_bounds, J_prev_round)
        proposal_gaps.append(np.amax(temp_upper_bounds) - lower_bounds[J_prev_round])

        best_arm = proposal_arms[np.argmin(np.array(proposal_gaps))]
        best_arm_params = self.param_space[best_arm]

        return proposal_gaps, best_arm_params

    def get_posterior_bounds(self, beta, X=None, Y=None):
        """
        计算每个策略的后验置信区间。
        """
        posterior_theta_params = self.posterior_theta(X, Y)
        marginal_mu_params = self.marginal_mu(posterior_theta_params)
        marginal_mean, marginal_var = marginal_mu_params

        upper_bounds = marginal_mean + beta * np.sqrt(marginal_var)
        lower_bounds = marginal_mean - beta * np.sqrt(marginal_var)

        upper_bounds = np.around(upper_bounds, 4)
        lower_bounds = np.around(lower_bounds, 4)

        return (upper_bounds, lower_bounds)
    def posterior_theta(self, X_t, Y_t):
        num_dims = self.num_dims
        sigma = self.sigma
        eta = self.eta
        prior_mean = np.zeros(num_dims)

        prior_theta_params = (prior_mean, eta * eta * np.identity(num_dims))

        if X_t is None:
            return prior_theta_params

        posterior_covar = np.linalg.inv(np.dot(X_t.T, X_t) / (sigma * sigma) + np.identity(num_dims) / (eta * eta))
        posterior_mean = np.linalg.multi_dot((posterior_covar, X_t.T, Y_t)) / (sigma * sigma)
        posterior_theta_params = (np.squeeze(posterior_mean), posterior_covar)
        return posterior_theta_params
    def marginal_mu(self, posterior_theta_params):
        X = self.X
        posterior_mean, posterior_covar = posterior_theta_params

        marginal_mean = np.dot(X, posterior_mean)
        marginal_var = np.sum(np.multiply(np.dot(X, posterior_covar), X), 1)
        marginal_mu_params = (marginal_mean, marginal_var)

        return marginal_mu_params

    def get_parameter_space(self):
        policies = np.genfromtxt(self.policy_file, delimiter=',', skip_header=1)
        # np.random.shuffle(policies)
        return policies[:, :6]

def main():

    args = parse_args()

    np.random.seed(args.seed)
    np.set_printoptions(threshold=np.inf)
    
    arm_bounds_file = os.path.join(args.data_dir, args.arm_bounds_dir)
    next_batch_flie = os.path.join(args.data_dir, args.next_batch_dir)
    early_pred_file = os.path.join(args.data_dir, args.early_pred_dir)
    
    os.makedirs(arm_bounds_file, exist_ok=True)
    os.makedirs(next_batch_flie, exist_ok=True)
    os.makedirs(early_pred_file, exist_ok=True)

    agent = BayesGap(args)
    best_arm_params = agent.run()

    if args.round_idx != 0:
        print('Best arm until round', args.round_idx-1, 'is', best_arm_params)

    # save the current round's data in log.csv
    log_path = os.path.join(args.data_dir, args.log_file)
    with open(log_path, "a") as log_file:
        print('have Logged data for round '+ str(args.round_idx) + ' in log.csv')
        if args.round_idx == 0:
            log_file.write(str(args.init_beta) + ',' +
                          str(args.gamma)      + ',' +
                          str(args.epsilon)    + ',' +
                          str(args.seed)       + '\n')
        elif args.round_idx != args.budget:
            log_file.write(str(best_arm_params) + '\n')
        elif args.round_idx == args.budget:
            log_file.write(str(best_arm_params) + '\n')


if __name__ == '__main__':
    main()
