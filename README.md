## A Hybrid GNN Approach for Improved Molecular Property PredictionLeveraging Graph Neural Networks for Enhanced Molecular Toxicity Prediction in the Tox21 Framework


In this work, we propose a hybrid approach that incorporates different graph-based methods to combine their strengths and mitigate their limitations to accurately predict molecular properties. The proposed approach consists in a multi-layered hybrid GNN architecture that integrates multiple GNN frameworks to compute graph embeddings for molecular property prediction. 

Furthermore, we conduct extensive experiments on multiple benchmark datasets to demonstrate that our hybrid approach significantly outperforms the state-of-the-art graph-based models.
We developed a hybrid Message-Passing Graph Neural Network (GNN) model to predict molecular toxicity across the Tox21 benchmark SR-ARE pathway. The proposed architecture achieves a peak test ROC-AUC of 0.8126. Building upon foundational multi-task GNN frameworks, our work integrates graph isomorphism mappings with local spatial aggregation and post-hoc explainability to map learned graph representations directly to known structural alerts.
![ScreenShot](Figures/Model_Architecture.PNG?raw=true)

This work is a fine-tuned impovement on the [Quesado et al. (2024)](https://doi.org/10.1089/cmb.2023.0452) model (SAGE-GIN-GIN)


## System Requirements

We used the following packages for code implementation:


## References

[1]Quesado, P., Torres, L. H. M., Ribeiro, B., & Arrais, J. P. (2024). A Hybrid GNN Approach for Improved Molecular Property Prediction. Journal of Computational Biology, 31, 1146–1157. https://doi.org/10.1089/cmb.2023.0452
