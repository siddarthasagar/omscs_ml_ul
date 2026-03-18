# Unsupervised Learning and Dimensionality Reduction Report
**CS7641: Machine Learning — Spring 2026**

---

## 1. Assignment Weight

The assignment is worth **12%** of the total points.

Read everything below carefully as this assignment has changed term-over-term.

---

## 2. Objective

Now it is time to explore unsupervised learning algorithms. This part of the assignment asks you to use some of the clustering and dimensionality reduction algorithms we've looked at in class and to revisit earlier assignments. The goal is for you to think about how these algorithms are the same as, different from, and interact with your earlier work.

---

## 3. Required Datasets for Spring 2026

You will reuse the same two datasets from your SL and OL Reports. **If these datasets are not used, you will receive a zero for the assignment.**

**Dataset A: Adult Income (Census Income)**
- Task type: Binary classification
- Target: `income` (≤ 50K vs > 50K)

**Dataset B: Wine Quality (Red and White)**
- Task type: Multiclass classification
- Target: `quality` (discrete rating/class)

---

## 4. Procedure

### 4.1 The Problems Given to You

You are to implement five algorithms.

The first two are **clustering algorithms**. You can choose your own measures of distance/similarity. Justify your choices.

- Expectation Maximization
- K-Means Clustering

The last three are **linear dimensionality reduction algorithms**:

- Randomized Projections
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)

You are to run several experiments with the goal of disseminating how dimensionality reduction affects your data. You will develop hypotheses based on your datasets and the following exploration. These hypotheses should be well-posed and grounded in theory from the lectures and readings.

> **Extra Credit Opportunity:** There is an opportunity to add 5 points of extra credit. In addition to the above algorithms, you will also implement a **Non-linear Manifold Learning Algorithm of Your Choice** as both a comparison and a visualization of your datasets. You will need to justify your choice and briefly describe whatever it is that you use. This is not mandatory and may require more time than allotted. If you need a place to get started: https://sites.gatech.edu/omscs7641/2024/03/10/no-straight-lines-here-the-wacky-world-of-non-linear-manifold-learning/

---

### 4.2 Exploration

For Steps 1–3, use **both datasets**. For Steps 4–5, use **only one** of the datasets (either Adult or Wine) for the neural network experiments.

**Step 1 — Clustering on raw data.** Apply your clustering algorithms (K-Means and EM/GMM) separately to each dataset, and report the results for each algorithm/dataset pairing. This should yield four clustering analyses in total. For each analysis, describe the preprocessing used, the key hyperparameters selected (such as number of clusters or components, initialization, and convergence settings), and the evaluation criteria used to judge quality (such as silhouette score, BIC/AIC, or stability across runs). Summarize what the clustering results suggest about the structure of each dataset and how well they align with your hypotheses.

**Step 2 — Dimensionality reduction on raw data.** Apply your dimensionality reduction methods (PCA, ICA, and RP) separately to each dataset, and report the results for every method/dataset pairing. This should yield six dimensionality reduction analyses in total. For each method, explain any scaling or whitening decisions, the dimensionality selected and why, and what structure or patterns became visible in the transformed space. Examples include variance explained, component interpretability, or evidence of separation. Briefly note how these transformed representations may affect later clustering or classification tasks.

**Step 3 — Clustering in reduced-dimensional spaces.** For each reduced representation created in Step 2, re-apply the clustering algorithms from Step 1. This will produce twelve total combinations across dataset, dimensionality reduction method, and clustering algorithm. Present a comparative summary of these results, preferably using tables or other concise visual summaries. Highlight cases where dimensionality reduction improves cluster quality, separation, stability, or interpretability, and identify the combinations that most strongly support or contradict your hypotheses. Focus on clear comparison rather than exhaustive narration.

**Step 4 — Neural networks on dimensionality-reduced data (one dataset only).** Using only the Wine dataset or the Adult dataset, retrain the best neural network model from your SL or OL report on three transformed versions of the input data: RP, PCA, and ICA. Compare these results against the same neural network trained on the original feature set, using the same train/test split and the same evaluation metrics. Your discussion should focus on whether dimensionality reduction changes predictive performance, generalization, or training behavior, and whether any reduced representation appears more useful than the original feature space.

**Step 5 — Neural networks with cluster-derived features (one dataset only).** Using only the Wine dataset or the Adult dataset, create cluster-derived features from the raw-data clustering models in Step 1. These features may include one-hot cluster assignments, EM posterior probabilities, or distances to K-Means centroids. Use these cluster-derived features either by themselves or by appending them to the original feature set, then retrain the same neural network model used in Step 4. Compare whether features derived from K-Means or EM/GMM are more helpful for prediction, and explain the trade-offs in terms of predictive value, interpretability, and how much additional information the clustering appears to provide.

---

## 5. Experiments and Analysis

You must contain a hypothesis about your analysis. This is open-ended as each of you will have a variety of perspectives on the features and attributes of the data that may or may not perform a certain way given the required algorithms. Whatever hypothesis you choose, you will need to back it up with experimentation and thorough discussion. **It is not enough to just show results.**

### External sources & literature (required)

- Include outside sources beyond course materials, with **at least two peer-reviewed references**.
- Use these to justify choices and to interpret results.
- Acceptable citation styles: MLA, APA, or IEEE — pick one and be consistent in-text and in the bibliography.

### Interpret, don't summarize

Every figure/table needs a takeaway tied to algorithm behavior and data. Avoid repeating the legend; explain what the figure/table means.

**Analysis writeup is limited to 8 pages.** The page limit includes your citations. Citations must be in IEEE, MLA, or APA format. Anything past 8 pages will not be read. Please keep your analysis concise while still covering the requirements of the assignment.

As a final check during your submission process, download the submission to double-check everything looks correct on Canvas. Try not to wait until the last minute to submit.

**Your report must be written in LaTeX on Overleaf.** You can create an account with your Georgia Tech email (e.g., `gburdell3@gatech.edu`). Georgia Tech provides a premium account where you can track history. Do not start a free-tier account with your personal email as this will not track history. When submitting your report, you are required to include a **READ-ONLY link** to the Overleaf Project. If a link is not provided in the report or Canvas submission comment, **5 points will be deducted** from your score. Do not share the project directly with the Instructor or TAs via email. For a starting template, please use the [IEEE Conference template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn).

### Your report should contain:

- Brief description of the datasets, and hypotheses you want to highlight in your report.
- Explanations of methods. This is your opportunity to demonstrate nuances needed to support your hypotheses.
- Grounded descriptions of resulting clusters. Support descriptions with data-driven evidence.
- Analyses of your results. Use the following questions as a starting point (not an exhaustive list):
  - Why did you get the clusters you did? Do they make "sense"?
  - What about the labels? Did the clusters line up with the labels? Do they otherwise line up naturally? Why or why not?
  - Compare and contrast the different algorithms. What sort of changes might you make to each to improve performance?
  - How much performance was due to the problems you chose?
  - Be creative and think of as many questions you can, and as many answers as you can. Justify your analysis with data explicitly.
- Descriptions of how the data looks in the new spaces created by the various dimensionality reduction algorithms:
  - For PCA: what is the distribution of eigenvalues?
  - For ICA: how kurtotic are the distributions? Do the projection axes seem to capture anything "meaningful"?
  - For RP: how well is the data reconstructed? How much variation did you get when you re-ran random projections several times?
  - How does noise affect each algorithm? What is the rank of your data? How collinear is your data, both qualitatively and quantitatively? How might specific properties of your data influence algorithm outputs?
- When you reproduced your clustering experiments on the datasets projected onto the new spaces created by ICA, PCA, and RP: did you get the same clusters as before? Different clusters? Why or why not?
- When you re-ran your neural network algorithms: were there any differences in performance? Speed? Consider how you might judge differences in performance and include these notes in your discussion.

> **Note on figures:** Figures should remain legible. Do not try to squish figures together in sections where axis labels become 8pt font or less. We are looking for clear and concise demonstration of knowledge and synthesis of results. Any paper that solely has figures without formal writing will not be graded. Be methodical with your space.

You may submit the assignment as many times as you wish up to the due date, but we will only consider your last submission for grading purposes.

> We need to be able to get to your code and your data. Providing entire libraries isn't necessary when a URL would suffice; however, you should at least provide any files you found necessary to change and enough support and explanation so we can reproduce your results on a standard Linux machine.

---

## 6. Acceptable Libraries

### Core ML (Python)

- `scikit-learn` — pipelines, CV, metrics, calibration.
- `imbalanced-learn` — resampling and class-weight tools for imbalanced targets.
- `scikit-learn-extra` — extra algorithms (e.g., k-medoids) when relevant.

### Deep Learning (for NNs)

- PyTorch (Lightning optional for cleaner loops).
- Keras / TensorFlow 2.

### Data & Visualization

- `pandas` / `polars` — tabular wrangling.
- `NumPy` — arrays and linear algebra.
- `matplotlib` / `plotly` / `altair` — plotting; use plotly/altair for interactive PR/ROC/residuals.

---

## 7. Submission Details

You will submit **two PDFs**:

1. `UL Report {GTusername}.pdf` — your report (Overleaf).
2. `REPRO UL {GTusername}.pdf` — your reproducibility sheet, which must include:
   - A READ-ONLY link to your Overleaf project.
   - A GitHub commit hash (single SHA) from the final push of your code.
   - Exact run instructions to reproduce results on a standard Linux machine (environment setup, commands, data paths, and random seeds).
   - EDA summary confirmation for both datasets (Adult and Wine). Note: EDA should match the SL Report unless changes are justified and explicitly disclosed.

- Include the READ-ONLY Overleaf link in the report or Canvas submission comment. Do not send email invitations.
- Use the GT Enterprise GitHub for course-related code and GT Enterprise Overleaf for writing your report.
- Provide sufficient instructions to retrieve code and data (Canvas paths and file names are sufficient).
- Only your latest submission will be graded. Please double-check that both PDFs are submitted.

### Formal writing requirement (bullets)

Your report is a formal technical paper. Bullet writing is not formal writing and will be treated as a draft. Use paragraphs and integrated prose. **In your final report, any list environment (`itemize`, `enumerate`, or `description`) with more than two items in any section will incur a 50% deduction on the overall report score.** Keep enumerations to ≤ 2 items or convert them to narrative prose.

---

## 8. Feedback Requests

When your assignment is scored, you will receive feedback explaining your errors and successes in some level of detail. This feedback is for your benefit, both on this assignment and for future assignments. It is considered a part of your learning goal to internalize this feedback.

If you are confused by a piece of feedback, please start a private thread on Ed and we will help clarify. When you make a private request, please use the **`Feedback Request - UL`** tag on Ed for this cycle. We will continue to proceed with the Reviewer Response with the UL Report after grades and feedback are posted.

---

## 9. Plagiarism and Proper Citation

The easiest way to fail this class is to plagiarize. Using the analysis, code, or graphs of others in this class is considered plagiarism. We care about your analysis: it must be original and grounded in your own experiments.

If you copy any amount of text from other students, websites, or any other source without proper attribution, that is plagiarism. Citing is required but does not permit copying large blocks of text. All citations must use a consistent style (IEEE, MLA, or APA).

We report all suspected cases of plagiarism to the Office of Student Integrity. Students who are under investigation are not allowed to drop from the course in question, and the consequences can be severe, ranging from a lowered grade to expulsion from the program.

### LLMs (disclosure required)

We treat AI-based assistance the same way we treat collaboration with people. You may discuss ideas and seek help from classmates, colleagues, and AI tools, but all submitted work must be your own. The goal of reports is synthesis of analysis, not merely getting an algorithm to run.

**Every submission must include an AI Use Statement.** List the tools used and what they assisted with, and confirm that you reviewed and understood all assisted content.

- **Allowed with disclosure:** brainstorming, outlining, grammar and clarity edits, code generation, code refactoring, and debugging.
- **Not allowed:** submitting AI-written analysis, conclusions, or figures as your own; fabricating results or citations; paraphrasing AI or prior work to evade checks.

**Example statement** (placed at the very end of the report before References):

> *"AI Use Statement. I used ChatGPT and Visual Studio Code Copilot to brainstorm and outline sections of the report, generate and refactor small code snippets, debug an indexing issue, and edit grammar and clarity throughout. I reviewed, verified, and understand all assisted content."*

### How to cite peer-reviewed sources in-text

You may use MLA, APA, or IEEE; pick one style and stay consistent across the paper (in-text and references). Examples using the hotel paper:

- **APA:** "… (António, Almeida, & Nunes, 2019)." or "António et al. (2019) …"
- **MLA:** "… (António, Almeida, and Nunes 2019)." or "António, Almeida, and Nunes argue …"
- **IEEE:** "… [1]." (numbered, bracketed citations; reference list ordered by first appearance)

Include the full reference entry in your bibliography. In LaTeX, use BibTeX/BibLaTeX with an appropriate style (e.g., `style=apa` or `style=ieee`). **Tip:** In Google Scholar, click "Cite" → "BibTeX" to copy a starter entry, then verify authors, capitalization, and DOI.

---

## 10. Version Control

| Version | Date | Notes |
|---------|------|-------|
| v2.0 | 03/11/2026 | TJL updated steps 4 and 5 to include use of either dataset, but only one. |
| v1.0 | 03/07/2026 | TJL finalized UL Report for Spring 2026 term. |

*Assignment description written by Theodore LaGrow.*
