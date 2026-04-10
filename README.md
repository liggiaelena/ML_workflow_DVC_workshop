# ML Workflow with DVC

## Workshop Summary

This workshop introduces students to building a reproducible machine learning workflow using Data Version Control (DVC). The goal is to move beyond ad-hoc experimentation toward structured, traceable, and collaborative ML development practices.

Students begin by setting up a minimal project that combines Git and DVC. They learn how to separate code, data, and models while maintaining a unified workflow. The project uses a simple Convolutional Neural Network (CNN) trained on the MNIST dataset to keep the focus on workflow concepts rather than model complexity.

The workflow is divided into pipeline stages: data preparation, model training, and prediction. Each stage is defined in `dvc.yaml`, allowing DVC to track dependencies and determine when stages need to be re-executed. Students run the pipeline using `dvc repro`, inspect results using `dvc metrics show`, and compare experiments using `dvc metrics diff`.

A key component of the workshop is experimentation through `params.yaml`. Students modify hyperparameters such as learning rate, epochs, and batch size, and observe how these changes propagate through the pipeline. This reinforces the idea that experiments should be controlled, reproducible, and comparable.

The workshop also introduces the concept of extending pipelines. Students add a prediction stage that reuses the trained model, demonstrating how ML workflows evolve from training to inference. This stage highlights the importance of modular code and reinforces best practices such as separating model definitions from execution logic.

Finally, students learn how to share their work using both Git and DVC. Git is used to version code and metadata, while DVC manages large artifacts such as datasets and models. By configuring a remote storage location and using `dvc push`, students understand how teams collaborate on ML projects without storing large files in Git repositories.

## Learning Objectives

By the end of this workshop, students will be able to:

- Explain the role of DVC in an MLOps workflow
- Describe the limitations of Git for machine learning projects
- Build and run a multi-stage DVC pipeline
- Track datasets, models, and metrics using DVC
- Modify experiment parameters and compare results
- Understand dependency-aware pipeline execution
- Extend a pipeline to include inference (prediction)
- Apply best practices for modular ML code design
- Differentiate between `git push` and `dvc push`
- Reproduce a workflow on another machine using DVC

## Project Structure

```
project/
├── data/
├── src/
│   ├── prepare.py
│   ├── train.py
│   ├── predict.py
├── params.yaml
├── dvc.yaml
```

## Getting Started

```bash
pip install dvc torch torchvision scikit-learn pandas pyyaml
git init
dvc init
dvc repro
dvc metrics show
```

## Team Members

| Name | Student ID |
|---|---|
| Emmanuel Ihejiamaizu | 9080005 |
| Liggia Elena Taboada Cruz | 9085905 |
| Chao-Chung Liu | 9067679 |

## ML Pipelines in Machine Learning and CI/CD

Machine learning pipelines are structured sequences of steps that automate the journey from raw data to a deployed model. Each step — data preparation, feature engineering, training, evaluation, and inference — is defined as a discrete unit with explicit inputs and outputs. This structure makes workflows reproducible, auditable, and easier to debug compared to ad-hoc scripts.

In the context of CI/CD (Continuous Integration / Continuous Delivery), ML pipelines play the same role that automated build and test pipelines play in software engineering. Just as a code change triggers a CI pipeline that compiles, tests, and validates software, a change in data or model parameters can trigger an ML pipeline that retrains, evaluates, and validates the model before it reaches production.

DVC enables this by defining pipelines in `dvc.yaml` with explicit dependencies between stages. When a dependency changes — a script, a dataset, or a parameter — DVC knows exactly which stages need to rerun and which can be skipped. Combined with Git for code versioning, this gives teams a full audit trail of what ran, on what data, with what parameters, and what results were produced. This is the foundation of MLOps: applying DevOps principles to machine learning so that model development is as reliable, automated, and collaborative as software development.

## Collaboration Workflow

```bash
dvc push
git push
```

To reproduce on another machine:

```bash
git clone <repo>
cd <repo>
pip install dvc torch torchvision scikit-learn pandas pyyaml
dvc pull
dvc repro
```
