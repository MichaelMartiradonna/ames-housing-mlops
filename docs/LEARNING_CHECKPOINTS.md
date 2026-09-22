# Optional learning checkpoints

These exercises are optional study material. The supplied course instructions require working code, tests, documentation including a reflection, and a demo; they do not require these questions or an extra independently written test. Skipping these exercises does not leave a course deliverable incomplete. The required reflection is already included in [README.md](../README.md#reflection).

These activities are also separate from any broader independent skills assessment after the capstone. Neither this guide nor the submission materials claim that an independent assessment has been completed.

## 1. Data and evaluation — about 10 minutes

Explain the target and 14 inputs without opening the code. Then open `configs/model.yaml` and identify the train/validation/test split, five configurations and performance gates.

- Why fit imputation and scaling only on training data?
- Why retain a validation-selected model even when another candidate has lower test error?
- What does an MAE of about $17,765 mean? What does it not tell you about one house?
- Why does a Random Forest not need scaling, even though this project includes it?
- Why require four facts and show defaults for ten others? How do the separate validation results differ from the published full-input test error?

## 2. Follow one request — about 10 minutes

Run the example and trace it through `src/llm.py`, `src/interface.py`, `src/serving.py`, and `src/app.py`.

- Which component supplies the price?
- What happens to a missing value, unknown neighborhood, negative area, or ambiguous unit?
- Why is JSON validity insufficient to establish extraction accuracy?
- Why does the form still require confirmation?
- Which checks use mocked replies, and which invoke the real language model?

## 3. Make one small change yourself — about 10 minutes

On a new Git branch, add one original interface test for an edge case you think matters. Run that test, explain why it passes, and identify what a failure would reveal. Avoid merely copying a supplied test and changing its number.

Then run the complete tests and point to the CI result, model release manifest and experiment comparison.

## 4. Optional reflection practice — about 5 minutes

The README contains the required project reflection and discloses AI assistance. For additional practice, explain one decision, one difficulty, and one future improvement without relying on the prepared wording. This is a learning exercise, not an additional submission requirement or evidence of independent mastery.

## Later: independent skills assessment

If useful after submission, use a separate session to assess what you can do without generated solutions: inspect a dataset, explain a train/test split, diagnose a failing test, make a small API or UI change, and describe how to deploy it. Use the results to choose the next employment-focused portfolio task. This assessment has not been performed and is not required for the capstone submission.
