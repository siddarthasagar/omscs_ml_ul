This report was graded by Radhika.



Great job finishing the UL Report, Siddartha, just one more to go for the semester!

Continuing from the previous assignment, please see the section-by-section feedback below, highlighting 'What you did well' and 'What needs improvement.' Please create a private Ed post if you'd like clarification on any grading or feedback we provide and we'll happily follow up there!



====================



A. Requirements (8/10 points)

{

What you did well:

+ A1-1. Overleaf and LaTeX: Both a read-only link to your Overleaf is provided and your report is written in LaTeX.

+ A1-2. GitHub Hash: Your final commit hash is provided and your work is accessible.

+ A1-3. Run Instructions: Appropriate run instructions are provided in your REPRO document (not Canvas/GitHub/elsewhere)

+ A1-4. Exploratory Data Analysis (EDA): You provide appropriate EDA in your REPRO document for the datasets used for this report.

+ A2. Page Length: You adhered to the stated 8-page length max rule.

+ A4. Proper References: You provide references outside the scope of course materials, demonstrating initiative in filling in contextual gaps, and importantly your references are consistently, appropriately cited in the References section in your report.



What needs improvement:

x A3. Graph/Text Legibility: We deemed your visualizations and/or writing (graphs, tables, typeface, etc.) to not comply with the IEEE standard and not pass our legibility requirements (without relying on zoom).



General Feedback:

You did a good job in this section! You have satisfied all requirements. You submitted a properly formatted PDF using Overleaf and provided a valid GitHub commit hash. You also included valid references from outside the course material and adhered to the expected format. Some of the plot sizes are quite small, making legends, axis labels and plot values illegible. All plots and results should be easily readable so that the analysis can be verified.



}



====================



B. Hypothesis (8/8 points)

{

What you did well:

+ B1-1. Hypothesis Declaration: At least one context-relevant hypothesis is provided.

+ B1-2. Citing Evidence: Each hypothesis provided appropriately draws support from theory derived from a class lecture, paper or theoretical proof.

+ B2-1. Follow-Through: You provide follow-up to reason whether each hypothesis is proven or disproven through your experiments.

+ B2-2. Conclusions Defended: You further defend your observed evidence of each hypothesis' being proven or disproven by citing evidence with interpretation.



What needs improvement:

No suggestions



General Feedback:

You provided valid initial hypotheses and supported them with reasoning. Good job revisiting them after the experiments and correlating with the results.

}



====================



C. Step 1: Clustering (both datasets) (20/20 points)

{

What you did well:

+ C1-1. EM Hyperparameter Selection: You provide rationale for your selection of cluster count, among other disclosures.

+ C1-2. EM Results Visualization: You provide visualization to demonstrate your results to justify your written claims.

+ C1-3. EM Results Interpretation: You provide meaningful interpretation of your observed cluster shape/distribution and call to attention with reasoning and inter-algorithm comparison noteworthy results.

+ C2-1. KMC Hyperparameter Selection: You provide rationale for your selection of cluster count, among other disclosures.

+ C2-2. KMC Results Visualization: You provide visualization to demonstrate your results to justify your written claims.

+ C2-3. KMC Results Interpretation: You provide meaningful interpretation of your observed cluster shape/distribution and call to attention with reasoning and inter-algorithm comparison noteworthy results.



What needs improvement:

No suggestions



General Feedback:

You did a good job on describing the clustering results and evaluating them based on the nature of the datasets and the algorithms. Analysis could discuss shapes of the resulting clusters from both methods- Did the shapes of the clusters match the expectations?

}



====================



D. Step 2: Dimensionality Reduction (both datasets) (24/24 points)

{

What you did well:

+ D1-1. RP Component Selection: You appropriately justify the number of components settled upon by citing a suitable per-algorithm metric.

+ D1-2. RP Results Visualization: You provide appropriate visualization of your results with proper citation of evidence.

+ D1-3. RP Results Interpretation: You provide meaningful analysis of your results including some of the expected interpretations like loading scores, delving into the transformed/reduced feature space structure and including comparison with the original dataset's structure.

+ D2-1. PCA Component Selection: You appropriately justify the number of components settled upon by citing a suitable per-algorithm metric.

+ D2-2. PCA Results Visualization: You provide appropriate visualization of your results with proper citation of evidence.

+ D2-3.PCA Results Interpretation: You provide meaningful analysis of your results including some of the expected interpretations like loading scores, delving into the transformed/reduced feature space structure and including comparison with the original dataset's structure.

+ D3-1. ICA Component Selection: You appropriately justify the number of components settled upon by citing a suitable per-algorithm metric.

+ D3-2. ICA Results Visualization: You provide appropriate visualization of your results with proper citation of evidence.

+ D3-3. ICA Results Interpretation: You provide meaningful analysis of your results including some of the expected interpretations like loading scores, delving into the transformed/reduced feature space structure and including comparison with the original dataset's structure.



What needs improvement:

No suggestions



General Feedback:

Great job at correctly identifying the reduced components for each algorithm using the appropriate metrics and interpreting what the components represent.



}



====================



E. Step 3: Dimensionality Reduction then Clustering (both datasets) (6/14 points)

{

What you did well:

+ E1-2. Wine Pairwise Comparisons: You unpack your results meaningfully and interpret the distinct and combined effects of DR and Clustering approaches with interpretation and thoughtful justification of your results for the Wine dataset.

+ E2-2. Adult Pairwise Comparisons: You unpack your results meaningfully and interpret the distinct and combined effects of DR and Clustering approaches with interpretation and thoughtful justification of your results for the Adult dataset.



What needs improvement:

x E1-1. Wine Performance Visualization: You must provide an appropriate and orderly demonstration/visualization of your results produced using combined DR then Clustering across the 6 pairwise combinations for the Wine dataset.

x E2-1. Adult Performance Visualization: You must provide an appropriate and orderly demonstration/visualization of your results produced using combined DR then Clustering across the 6 pairwise combinations for the Adult dataset.



General Feedback:

You presented results for how clustering has changed after applying DR and provided good analysis of how clustering changed after DR. It is not clear how the new number of clusters were selected. In this step, you need to re-select the optimal number of clusters. Plots of Silhouette, AIC/BIC should be provided to justify the selection of optimal number of clusters.

}



====================



F. Step 4: Dimensionality Reduction on NN (one dataset) (12/12 points)

{

What you did well:

+ F1-1. RP Performance Commentary: You provide suitable evidence citing any observed performance improvements between your baseline NN and your Dimensionality Reduced NN for RP.

+ F1-2. RP Wall Clock Difference: You address whether your RP treatment on the baseline NN yielded a speed up in runtime and provide your results as justification.

+ F2-1. PCA Performance Commentary: You provide suitable evidence citing any observed performance improvements between your baseline NN and your Dimensionality Reduced NN for PCA.

+ F2-2. PCA Wall Clock Difference: You address whether your PCA treatment on the baseline NN yielded a speed up in runtime and provide your results as justification.

+ F3-1. ICA Performance Commentary: You provide suitable evidence citing any observed performance improvements between your baseline NN and your Dimensionality Reduced NN for ICA.

+ F3-2. ICA Wall Clock Difference: You address whether your ICA treatment on the baseline NN yielded a speed up in runtime and provide your results as justification.



What needs improvement:

You could include the wall clock times for each method in the table.



General Feedback:

Good job reporting results and analysis for how performance and wall clock time changed.

}



====================



G. Step 5: Clustering on NN (one dataset) (6/12 points)

{

What you did well:

+ G1-1. EM Performance Commentary: You provide suitable evidence citing any observed performance improvements between the baseline NN and provision of your clusters as features to the CL treatment of your NN and follow this demonstration up with thoughtful justification.

+ G2-1. KMC Performance Commentary: You provide suitable evidence citing any observed performance improvements between the baseline NN and provision of your clusters as features to the CL treatment of your NN and follow this demonstration up with thoughtful justification.



What needs improvement:

x G1-2. EM Wall Clock Difference: You must appropriately relate any observed speed up in wall clock time on your EM-clustered data with the baseline NN to the effect of EM clustering and make mention of or descriptively allude to the filtering effect.

x G2-2. KMC Wall Clock Difference: You must appropriately relate any observed speed up in wall clock time on your KM-clustered data with the baseline NN to the effect of KM clustering and make mention of or descriptively allude to the filtering effect.



General Feedback:

Good job reporting results and analysis for performance. Analysis could cover results and reasoning for wall clock time differences as well.

}



====================



Extra Credit: Nonlinear Manifold Learning (5/5 points)

{

+ 5pts. You implement a nonlinear manifold learning algorithm and situate its use alongside the required algorithms. In particular, you provide full explanation of its implementation and its observed effects and demonstrate this with appropriate visuals to avail comparison with the required Step 1: Clustering or Step 3: DR then Clustering results.

}



====================



Deductions:

{

None

}



====================



Overall Summary:

{

Overall, your report is clear and well-organized. You did a good job tying the various parts of the assignment together and justifying your choices. In some sections, you can improve by including relevant plots and results for the experiments and refer to them in the analysis. All analysis and reasoning should be based on empirical results which should be clearly readable and referenced from the analysis. Good work and good luck with the rest of the course!

}