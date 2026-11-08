# Run, review and deploy

This is a locally prepared working implementation. The official brief permits AI for learning support and disallows direct copy-paste submissions. Review the code, explain the choices, make your own justified changes, and execute your assessed work in BITS Virtual Lab. Do not fabricate execution evidence or commit history.

## 1. Files to keep together

Keep app.py, ml_core.py, train.py, requirements.txt, README.md, test_data.csv, model/, data/ and results/ in the same project folder. Saved models include their fitted preprocessing. requirements-report.txt and the two build scripts produce the report; verify_project.py checks the experiment and app.

The included results/training.log and execution_summary.json are actual local Windows execution records. They are not lab evidence.

## 2. Run locally or in the lab

Use Python 3.12. From the project folder in a terminal:

```text
python -m pip install -r requirements.txt
python train.py
python build_readme.py
python verify_project.py
python -m streamlit run app.py
```

For an isolated environment, create it with python -m venv .venv and activate it before installation. The pinned scikit-learn version must match the version used to save/load the models. scikit-learn 1.9.1 emits a deprecation warning for SVC probability=True, but it remains supported in the pinned version; it is not an execution failure.

The original UCI data conversion is bundled, with its checksum. Training reads it offline; internet is needed only to install dependencies or re-download a missing source file. Training overwrites saved models, metrics and the test split with a reproducible run. If you edit the experiment, regenerate the README and PDF afterwards.

## 3. Genuine BITS Virtual Lab execution

1. Sign in to the actual BITS Virtual Lab and upload/extract this project. Select a Python 3.12 environment, or arrange a compatible environment through lab support.
2. Install requirements.txt in that environment. In a Jupyter cell, use %pip install -r requirements.txt.
3. Run python train.py --execution-context bits-lab in the lab terminal. In Jupyter, use !python train.py --execution-context bits-lab after changing to the project directory.
4. Run python verify_project.py. Read the output and verify all five models, six metrics, dataset sizes and saved artifacts.
5. Capture one genuine screenshot with the BITS Lab interface and successful execution/metrics visible. Keep session tokens and unrelated private content outside the capture. The execution-context label alone is not proof.
6. Download the resulting project files back from the lab, including model/, results/, test_data.csv and any modified source. Regenerate README.md from those results. Preserve the screenshot separately from the public source repository.

The assignment PDF lists lab support addresses if access fails: csislabsupport@wilp.bits-pilani.ac.in and raj.rohan@wilp.bits-pilani.ac.in, subject "NSP1 ML Assignment 2: BITS Lab issue". No support email has been sent.

## 4. Publish the assignment project to GitHub

The existing M.Tech coursework repository is private and contains unrelated course material. Create a separate assignment repository containing this project only, so evaluators can access the source without exposing the entire coursework archive.

Initialize Git in a separate extracted copy (not inside the existing coursework repository), commit your actual development changes, and push to your chosen GitHub repository. Include all five model files, test_data.csv, data/, results/, the source code and requirements.txt. Do not upload .venv, passwords, secrets, session tokens or private lab captures.

Once the actual repository exists, run:

```text
python build_readme.py --github-url https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY
```

Replace the example URL with the real repository URL. Commit the regenerated README. Check the repository link from a signed-out browser. Do not claim that the private coursework repository is publicly accessible.

## 5. Deploy on Streamlit Community Cloud

Open https://streamlit.io/cloud and sign in using GitHub. Create an app, select the assignment repository and branch, use app.py as the entry point, and select Python 3.12 in advanced settings. Deploy and wait for the interactive page.

The app loads saved models; it does not train on the server. No secrets or API keys are required. If deployment fails, inspect the build log and confirm requirements.txt and all five model files are present.

Check both the bundled data and an upload of test_data.csv. Select every classifier and confirm six metrics, the comparison table and confusion matrix/report appear. Check the live URL in a signed-out browser.

## 6. Generate the single submission PDF

Install the report dependencies:

```text
python -m pip install -r requirements-report.txt
```

The included review copy was produced with python build_submission.py --draft. It is explicitly incomplete and must not be submitted.

After the lab run, GitHub publication and deployment, run the following as one command (replace all three example values):

```text
python build_submission.py --github-url https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY --app-url https://YOUR_APP.streamlit.app --lab-screenshot /path/to/actual-bits-lab-screenshot.png
```

The builder refreshes README.md from the measured results and creates output/pdf/2026AJ05692_ML_Assignment2.pdf in the required order: GitHub link, live app link, one lab screenshot, then the complete README content. The filename is descriptive; the supplied brief specifies one PDF but does not mandate a filename. Follow any additional filename/size rule displayed by your submission portal.

Review every PDF page for legibility and verify the links. If the README changed, commit that same version to GitHub so the PDF and repository agree. The builder checks supplied inputs, not the authenticity of a screenshot or public accessibility of your URLs.

## 7. Submit and preserve confirmation

Deadline: 8 November 2026, 11:59 PM as displayed in the supplied Taxila event. One attempt; no extensions; Draft submissions are not accepted. Upload the completed single PDF, submit/confirm, reopen the uploaded file and save the portal confirmation. Do not upload the project ZIP or review PDF as the assessed submission.

## Explain the work before submission

- Why are duplicate rows removed before splitting?
- What is fitted by StandardScaler, and why is fitting it on test data leakage?
- Why use macro F1 for seven uneven class sizes?
- Why use probability scores for AUC instead of predicted class labels?
- Why can Gaussian Naive Bayes struggle with correlated shape measurements?
- How are validation-based selection and final test evaluation separated?
- Where does the app load models, and why does it never call fit?

The implementation chooses fixed settings rather than tuning on the test set. A single split cannot establish statistical superiority or performance on future data.
