# class BayesianOptimizer:
#     def __init__(self, model, param_space, n_iterations=50):
#         self.model = model
#         self.param_space = param_space
#         self.n_iterations = n_iterations
#         self.results = []

#     def optimize(self):
#         for i in range(self.n_iterations):
#             params = self.suggest_next_params()
#             score = self.evaluate_model(params)
#             self.results.append((params, score))
#             self.update_bayesian_model(params, score)

#     def suggest_next_params(self):
#         # Implement logic to suggest the next set of hyperparameters
#         pass

#     def evaluate_model(self, params):
#         # Implement logic to evaluate the model with the given hyperparameters
#         pass

#     def update_bayesian_model(self, params, score):
#         # Implement logic to update the Bayesian model with new results
#         pass

#     def get_best_params(self):
#         # Return the best hyperparameters found during optimization
#         return max(self.results, key=lambda x: x[1])[0]