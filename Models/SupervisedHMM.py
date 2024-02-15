import numpy as np


class SupervisedHMM:
    def __init__(self, num_states, num_features):
        self.num_states = num_states
        self.num_features = num_features
        self.initial_prob = np.zeros(num_states)
        self.transition_prob = np.zeros((num_states, num_states))
        self.emission_mean = np.zeros((num_states, num_features))
        self.emission_covariance = np.zeros((num_states, num_features, num_features))

    def viterbiTraining(self, sequences, states, num_iterations=100, learning_rate=0.01):
        prev_outputs = np.zeros_like(np.concatenate(states))
        for _ in range(num_iterations):

            most_likely_paths = []

            # E-step: Compute the most likely state sequence for each observation sequence
            for obs_seq, state_seq in zip(sequences, states):
                # Calculate the Viterbi path using the current model parameters
                best_path = self.predict(obs_seq)
                most_likely_paths.append(best_path)

            outputs = np.concatenate(most_likely_paths)
            loss = np.linalg.norm(prev_outputs - outputs)
            print(loss)
            if loss == 0:
                break

            prev_outputs = np.copy(outputs)
            # M-step: Update HMM parameters based on the most likely state sequences
            state_counts = np.zeros(self.num_states)
            transition_counts = np.zeros((self.num_states, self.num_states))
            emission_sum = np.zeros((self.num_states, self.num_features))
            emission_cov_sum = np.zeros((self.num_states, self.num_features, self.num_features))

            for i in range(len(sequences)):
                obs = sequences[i]
                true_state = states[i]
                path_i = most_likely_paths[i]
                # diff_label = np.average(np.abs(path_i - true_state))
                T = len(obs)

                state_counts += np.bincount(path_i, minlength=self.num_states)
                for t in range(T - 1):
                    # if (path_i[t] != true_state[t]):
                    transition_counts[path_i[t], path_i[t + 1]] += 1

                for t in range(T):
                    # if (path_i[t] != true_state[t]):
                    emission_sum[path_i[t]] += obs[t]
                    obs_diff = (obs[t] - self.emission_mean[path_i[t]])
                    emission_cov_sum[path_i[t]] += np.outer(obs_diff, obs_diff)

            self.initial_state_prob = state_counts / np.sum(state_counts)
            self.transition_prob = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
            for s in range(self.num_states):
                self.emission_mean[s] = emission_sum[s] / state_counts[s]
                self.emission_covariance[s] = emission_cov_sum[s] / state_counts[s]

        # print(self.transition_prob)
        # print(self.initial_prob)


    def initialize(self, sequences, states):
        # Calculate initial state probabilities
        self.initial_prob[0] = np.sum(states == 0, axis=0) / len(states)
        self.initial_prob[1] = np.sum(states == 1, axis=0) / len(states)

        # Calculate transition probabilities
        for i in range(self.num_states):
            for j in range(self.num_states):
                num_transitions = np.sum(np.logical_and(states == i, np.roll(states, -1, axis=0) == j))
                total_state_occurrences = np.sum(states == i)
                self.transition_prob[i, j] = num_transitions / total_state_occurrences

        # Calculate emission parameters (mean and covariance)
        for i in range(self.num_states):
            state_mask = (states == i)
            state_sequences = sequences[state_mask]
            state_sequence_count = np.sum(state_mask)

            if state_sequence_count > 0:
                self.emission_mean[i] = np.mean(state_sequences, axis=0)
                self.emission_covariance[i] = np.cov(state_sequences, rowvar=False)


    def predict(self, sequence):
        # predict the probability using Viterbi algorithm
        T = len(sequence)
        path = np.zeros(T, dtype=int)

        # Initialize the Viterbi algorithm matrices
        V = np.zeros((self.num_states, T))
        backpointer = np.zeros((self.num_states, T), dtype=int)

        # Initialization step
        V[:, 0] = self.initial_prob * self.multivariate_normal_pdf(sequence[0])

        # Recursion step
        for t in range(1, T):
            for s in range(self.num_states):
                trans_prob = V[:, t - 1] * self.transition_prob[:, s]
                max_trans_prob = np.max(trans_prob)
                V[s, t] = max_trans_prob * self.multivariate_normal_pdf(sequence[t], s)
                backpointer[s, t] = np.argmax(trans_prob)

        # Termination step
        best_path_prob = np.max(V[:, -1])
        path[-1] = np.argmax(V[:, -1])

        # Backtrack to find the best path
        for t in range(T - 2, -1, -1):
            path[t] = backpointer[path[t + 1], t + 1]

        return path


    def multivariate_normal_pdf(self, x, state=None):
        if state is None:
            # Calculate the PDF for all states
            pdf_values = np.zeros(self.num_states)
            for i in range(self.num_states):
                pdf_values[i] = self.multivariate_normal_pdf(x, state=i)
            return pdf_values

        # Calculate the PDF for a specific state
        mean = self.emission_mean[state]
        cov = self.emission_covariance[state]
        exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
        det_cov = np.abs(np.linalg.det(cov))
        normalization = 1 / np.sqrt((2 * np.pi) ** self.num_features * det_cov)
        return normalization * np.exp(exponent)


# Example usage

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from Conf import x_episode_columns
    from FeaturesReader import SingleFeaturesReader
    import matplotlib.pyplot as plt

    # case we have a sequence of ACGT

    # load features
    path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\summary\\single_episode_features.pkl"
    experiment_data_path = "/\\Experiment\\"
    train_list = pd.read_csv(experiment_data_path + "single_train_0.csv")
    test_list = pd.read_csv(experiment_data_path + "single_test_0.csv")

    train_subject = train_list.loc[:, "Subject1"].values
    test_subject = test_list.loc[:, "Subject1"].values
    features_reader_train = SingleFeaturesReader(path, include_subjects=train_subject)
    features_reader_test = SingleFeaturesReader(path, include_subjects=test_subject)

    X_train, y_train = features_reader_train.getAllData(train=True)

    X_test, y_test = features_reader_test.getAllData(train=False)

    num_states = 2
    n_features = len(x_episode_columns)


    hmm = SupervisedHMM(num_states, n_features)
    hmm.initialize(np.vstack(X_train), np.concatenate(y_train))
    hmm.viterbiTraining(X_train, y_train, num_iterations=50)

    from sklearn.metrics import accuracy_score
    idx_test = 2
    test_sequence = X_test[idx_test]
    predicted_states = hmm.predict(test_sequence)
    print("Predicted States:", predicted_states)
    print("GT States:", y_test[idx_test])
    print("ACC:", accuracy_score(predicted_states, y_test[idx_test]))


    # np.random.seed(1970)
    # num_states = 2
    # num_features = 34
    #
    # # Generate example sequences and states
    # sequences = [np.random.randn(np.random.randint(5, 20), num_features) for _ in range(100)]
    # states = [np.random.randint(0, num_states, len(seq)) for seq in sequences]
    #
    # hmm = SupervisedHMM(num_states, num_features)
    # hmm.initialize(np.vstack(sequences), np.concatenate(states))
    # hmm.viterbiTraining(sequences, states, num_iterations=50)
    #
    # from sklearn.metrics import accuracy_score
    #
    # idx_test = 2
    # test_sequence = np.random.randn(7, num_features)
    # predicted_states = hmm.predict(sequences[idx_test])
    # print("Predicted States:", predicted_states)
    # print("GT States:", states[idx_test])
    # print("ACC:", accuracy_score(predicted_states, states[idx_test]))
