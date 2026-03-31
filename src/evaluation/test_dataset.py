"""Test Dataset for Evaluation.

30+ ground truth question-answer pairs across 5 categories.
Used by the evaluation pipeline to measure system quality.
"""

TEST_QUESTIONS = [
    # === CONCEPT (direct route) ===
    {
        "question": "What is the difference between user-based and item-based collaborative filtering?",
        "ground_truth": "User-based collaborative filtering recommends items by finding similar users and using their preferences. Item-based collaborative filtering finds items similar to what the user already likes based on co-rating patterns. Item-based is typically more scalable.",
        "expected_route": "direct",
        "category": "concept",
    },
    {
        "question": "What is matrix factorization in the context of recommender systems?",
        "ground_truth": "Matrix factorization decomposes the user-item interaction matrix into two lower-rank matrices (user factors and item factors). The product of these matrices approximates the original matrix, enabling prediction of missing ratings.",
        "expected_route": "direct",
        "category": "concept",
    },
    {
        "question": "Explain the cold start problem in recommendation systems.",
        "ground_truth": "The cold start problem occurs when a system cannot make reliable recommendations due to insufficient data about new users (user cold start) or new items (item cold start). Common solutions include content-based features, demographic data, and hybrid approaches.",
        "expected_route": "direct",
        "category": "concept",
    },
    {
        "question": "What is differential privacy and how is it applied to machine learning?",
        "ground_truth": "Differential privacy is a mathematical framework that provides formal privacy guarantees by adding calibrated noise to computations. In ML, it's applied via DP-SGD (noisy gradients), output perturbation, or objective perturbation to prevent models from memorizing individual training examples.",
        "expected_route": "direct",
        "category": "concept",
    },
    {
        "question": "What is federated learning?",
        "ground_truth": "Federated learning is a distributed ML paradigm where multiple clients collaboratively train a model while keeping their data local. A central server coordinates the process by aggregating model updates rather than raw data, preserving privacy.",
        "expected_route": "direct",
        "category": "concept",
    },
    {
        "question": "What is the nuclear norm and why is it used in matrix completion?",
        "ground_truth": "The nuclear norm (sum of singular values) is the convex relaxation of the rank function. It is used in matrix completion as a regularizer to encourage low-rank solutions, making the optimization problem convex and tractable.",
        "expected_route": "direct",
        "category": "concept",
    },

    # === FACTUAL (vectorstore route) ===
    {
        "question": "What datasets are commonly used to benchmark federated recommendation systems?",
        "ground_truth": "Common benchmarks include MovieLens (100K, 1M), FilmTrust, BookCrossing, Amazon Reviews, Yelp, Jester, and Netflix Prize dataset.",
        "expected_route": "vectorstore",
        "category": "factual",
    },
    {
        "question": "What are the main challenges in privacy-preserving collaborative filtering?",
        "ground_truth": "Key challenges include balancing privacy guarantees with recommendation accuracy, communication efficiency in federated settings, handling heterogeneous data distributions across clients, and defending against inference attacks on shared gradients.",
        "expected_route": "vectorstore",
        "category": "factual",
    },
    {
        "question": "What evaluation metrics are used for recommender systems?",
        "ground_truth": "Common metrics include RMSE, MAE for rating prediction; Precision@K, Recall@K, NDCG@K, MAP for ranking quality; coverage, diversity, and novelty for beyond-accuracy evaluation.",
        "expected_route": "vectorstore",
        "category": "factual",
    },
    {
        "question": "How does alternating least squares work for matrix factorization?",
        "ground_truth": "ALS alternates between fixing user factors and solving for item factors, then fixing item factors and solving for user factors. Each sub-problem becomes a regularized least squares problem with a closed-form solution.",
        "expected_route": "vectorstore",
        "category": "factual",
    },
    {
        "question": "What privacy attacks exist against federated recommender systems?",
        "ground_truth": "Key attacks include gradient inversion (reconstructing training data from shared gradients), membership inference (determining if a user's data was used), and property inference (inferring sensitive attributes from model updates).",
        "expected_route": "vectorstore",
        "category": "factual",
    },
    {
        "question": "What is the role of regularization in matrix completion?",
        "ground_truth": "Regularization prevents overfitting to observed entries and controls the complexity of the learned factors. Common forms include L2 regularization on factor matrices (Frobenius norm), nuclear norm regularization, and implicit regularization through early stopping.",
        "expected_route": "vectorstore",
        "category": "factual",
    },

    # === DISCOVERY (web_search route) ===
    {
        "question": "What are the latest papers on privacy-preserving matrix factorization from 2025-2026?",
        "ground_truth": "Recent work includes advances in differentially private federated matrix factorization, secure aggregation protocols for recommendation, and randomized sketching approaches for privacy-preserving low-rank approximation.",
        "expected_route": "web_search",
        "category": "discovery",
    },
    {
        "question": "What new federated recommendation methods have been published recently?",
        "ground_truth": "Recent methods include personalized federated recommendation with heterogeneous data, federated graph neural networks for recommendation, and communication-efficient federated collaborative filtering.",
        "expected_route": "web_search",
        "category": "discovery",
    },
    {
        "question": "Find recent papers on LLM-based recommender systems.",
        "ground_truth": "Recent work explores using large language models for recommendation through prompt-based approaches, LLM-enhanced collaborative filtering, and conversational recommendation systems powered by foundation models.",
        "expected_route": "web_search",
        "category": "discovery",
    },
    {
        "question": "What are the most cited papers on federated learning from 2024?",
        "ground_truth": "Highly cited recent works cover personalization in federated learning, communication efficiency improvements, heterogeneity-aware aggregation, and applications of federated learning to recommendation and NLP.",
        "expected_route": "web_search",
        "category": "discovery",
    },
    {
        "question": "Are there recent surveys on matrix completion methods?",
        "ground_truth": "Recent surveys cover low-rank matrix completion theory and algorithms, applications to recommender systems, connections to compressed sensing, and modern deep learning approaches to matrix completion.",
        "expected_route": "web_search",
        "category": "discovery",
    },
    {
        "question": "What are the latest advances in randomized algorithms for linear algebra?",
        "ground_truth": "Recent advances include improved randomized SVD algorithms, sketching-based methods for least squares and regression, and streaming algorithms for low-rank approximation with theoretical guarantees.",
        "expected_route": "web_search",
        "category": "discovery",
    },

    # === COMPARISON (compare route) ===
    {
        "question": "Compare ALS and SGD for matrix factorization. Which is better?",
        "ground_truth": "ALS has closed-form sub-problems making it stable and parallelizable, but scales poorly with matrix size. SGD is memory-efficient and handles sparse data well, but requires careful tuning of learning rate and is sequential. ALS converges faster per iteration; SGD is faster per epoch on large sparse matrices.",
        "expected_route": "compare",
        "category": "comparison",
    },
    {
        "question": "What are the trade-offs between centralized and federated recommendation?",
        "ground_truth": "Centralized approaches achieve higher accuracy with direct data access but raise privacy concerns. Federated approaches preserve data locality and privacy but face challenges in communication cost, data heterogeneity, and typically lower accuracy. Federated methods also complicate model debugging and evaluation.",
        "expected_route": "compare",
        "category": "comparison",
    },
    {
        "question": "Compare differential privacy and secure multi-party computation for privacy in ML.",
        "ground_truth": "Differential privacy adds noise for statistical guarantees and is computationally cheap but degrades utility. Secure MPC preserves exact computation through cryptographic protocols but has high communication overhead. DP protects against arbitrary future attacks; MPC protects data during computation but not the output.",
        "expected_route": "compare",
        "category": "comparison",
    },
    {
        "question": "Compare nuclear norm minimization with randomized SVD for matrix completion.",
        "ground_truth": "Nuclear norm minimization provides theoretical recovery guarantees under incoherence conditions but is computationally expensive (full SVD per iteration). Randomized SVD is much faster and scalable but lacks the same theoretical guarantees. In practice, randomized methods often achieve comparable accuracy at a fraction of the cost.",
        "expected_route": "compare",
        "category": "comparison",
    },
    {
        "question": "How do content-based and collaborative filtering compare?",
        "ground_truth": "Content-based filtering uses item/user features and doesn't suffer from cold start for items but requires feature engineering and has limited novelty. Collaborative filtering leverages user behavior patterns and discovers unexpected preferences but requires sufficient interaction data and suffers from cold start and sparsity.",
        "expected_route": "compare",
        "category": "comparison",
    },
    {
        "question": "Compare FedAvg and FedProx for federated optimization.",
        "ground_truth": "FedAvg performs local SGD steps and averages models, working well with IID data but struggling with heterogeneity. FedProx adds a proximal term to the local objective, improving convergence on non-IID data at the cost of an additional hyperparameter (mu). FedProx is more robust but slightly slower per round.",
        "expected_route": "compare",
        "category": "comparison",
    },

    # === LITERATURE REVIEW (literature_review route) ===
    {
        "question": "Give me a literature review on federated matrix factorization.",
        "ground_truth": "Federated matrix factorization research spans privacy-preserving approaches using differential privacy and secure aggregation, communication-efficient protocols reducing upload costs, handling non-IID rating distributions across clients, and achieving convergence guarantees in decentralized settings.",
        "expected_route": "literature_review",
        "category": "literature_review",
    },
    {
        "question": "Summarize the research landscape on privacy in recommender systems.",
        "ground_truth": "Research on privacy in recommender systems covers differential privacy mechanisms for rating protection, federated approaches keeping data local, cryptographic methods for secure computation, and attacks demonstrating information leakage from recommendations and model updates.",
        "expected_route": "literature_review",
        "category": "literature_review",
    },
    {
        "question": "Write a literature review section on randomized methods for low-rank approximation.",
        "ground_truth": "Randomized low-rank approximation methods include random projection (Johnson-Lindenstrauss), randomized SVD (Halko-Martinsson-Tropp), sketching-based approaches, and random sampling methods. These achieve near-optimal approximation with significantly reduced computational cost.",
        "expected_route": "literature_review",
        "category": "literature_review",
    },
    {
        "question": "Overview the evolution of collaborative filtering from 2010 to present.",
        "ground_truth": "Collaborative filtering evolved from neighborhood methods and matrix factorization (2010s), through deep learning approaches like autoencoders and NCF (mid-2010s), to graph neural network methods and transformer-based models (2020s), and most recently to federated and privacy-preserving variants.",
        "expected_route": "literature_review",
        "category": "literature_review",
    },
    {
        "question": "Provide a research overview of communication efficiency in federated learning.",
        "ground_truth": "Communication efficiency in FL has been addressed through gradient compression and quantization, sparsification of model updates, local computation strategies (multiple local epochs), knowledge distillation, and one-shot or few-round protocols.",
        "expected_route": "literature_review",
        "category": "literature_review",
    },
    {
        "question": "Survey the use of sketching techniques in machine learning.",
        "ground_truth": "Sketching in ML includes random projections for dimensionality reduction, count sketches for streaming data, sketched SGD for distributed optimization, and sketch-based kernel approximation. These techniques trade small accuracy loss for major computational and memory savings.",
        "expected_route": "literature_review",
        "category": "literature_review",
    },
]
